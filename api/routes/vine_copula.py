"""Vine copula endpoints — fit, simulate, portfolio VaR via pyvinecopulib."""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.models import (
    VineCopulaFitRequest,
    VineCopulaFitResponse,
    VineCopulaSimulateResponse,
    VineCopulaVaRResponse,
)
from api.stores import _copula_store, _get_portfolio_model, _PORTFOLIO_KEY

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vine-copula"])

# In-memory store for vine fit (contains non-serializable _vine_object)
_vine_store: dict[str, dict] = {}


def _fit_vine_sync(req: VineCopulaFitRequest) -> dict:
    from cortex.copula import fit_vine_copula

    model = _get_portfolio_model()
    returns_df = model["returns_df"]
    return fit_vine_copula(
        returns_df,
        structure=req.structure,
        family_set=req.family_set,
    )


@router.post("/portfolio/vine-copula/fit", response_model=VineCopulaFitResponse)
async def fit_vine(req: VineCopulaFitRequest):
    """Fit a vine copula to the calibrated portfolio returns.

    Requires a calibrated portfolio (POST /portfolio/calibrate).
    """
    try:
        result = await asyncio.to_thread(_fit_vine_sync, req)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    _vine_store[_PORTFOLIO_KEY] = result
    return VineCopulaFitResponse(
        engine=result["engine"],
        structure=result["structure"],
        n_obs=result["n_obs"],
        n_assets=result["n_assets"],
        families_used=result["families_used"],
        log_likelihood=result["log_likelihood"],
        n_params=result["n_params"],
        aic=result["aic"],
        bic=result["bic"],
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/portfolio/vine-copula/simulate", response_model=VineCopulaSimulateResponse)
async def simulate_vine(
    n_samples: int = Query(10_000, ge=100, le=100_000),
    seed: int = Query(42),
):
    """Generate samples from the fitted vine copula."""
    if _PORTFOLIO_KEY not in _vine_store:
        raise HTTPException(404, "No vine copula fit. Call POST /portfolio/vine-copula/fit first.")

    from cortex.copula import vine_copula_simulate
    import numpy as np

    vine_fit = _vine_store[_PORTFOLIO_KEY]
    try:
        samples = await asyncio.to_thread(vine_copula_simulate, vine_fit, n_samples, seed)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    return VineCopulaSimulateResponse(
        n_samples=int(samples.shape[0]),
        n_assets=int(samples.shape[1]),
        sample_mean=[float(x) for x in np.mean(samples, axis=0)],
        sample_std=[float(x) for x in np.std(samples, axis=0)],
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/portfolio/vine-copula/var", response_model=VineCopulaVaRResponse)
async def vine_copula_var(
    weights: dict[str, float] | None = None,
    alpha: float = Query(0.05, gt=0.0, lt=1.0),
    n_simulations: int = Query(10_000, ge=1000, le=100_000),
    seed: int = Query(42),
):
    """Portfolio VaR using vine copula Monte Carlo simulation.

    Uses the fitted vine copula for proper multivariate dependence
    instead of single-family bivariate approximation.
    """
    if _PORTFOLIO_KEY not in _vine_store:
        raise HTTPException(404, "No vine copula fit. Call POST /portfolio/vine-copula/fit first.")

    from cortex.copula import vine_copula_portfolio_var

    model = _get_portfolio_model()
    if not weights:
        assets = model["assets"]
        weights = {a: 1.0 / len(assets) for a in assets}

    vine_fit = _vine_store[_PORTFOLIO_KEY]
    try:
        result = await asyncio.to_thread(
            vine_copula_portfolio_var, model, weights, vine_fit,
            alpha, n_simulations, seed,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    return VineCopulaVaRResponse(
        vine_var=result["vine_var"],
        gaussian_var=result["gaussian_var"],
        var_ratio=result["var_ratio"],
        engine=result["engine"],
        structure=result["structure"],
        n_params=result["n_params"],
        n_simulations=result["n_simulations"],
        alpha=result["alpha"],
        timestamp=datetime.now(timezone.utc),
    )
