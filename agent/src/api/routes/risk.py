"""
Risk endpoints — A-LAMS-VaR model.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import structlog

from ...models.risk import ALAMSVaRModel, ALAMSConfig, LiquidityConfig

logger = structlog.get_logger()
router = APIRouter()

# Singleton model instance (lazy loaded on first fit)
_var_model: ALAMSVaRModel | None = None


def _get_or_create_model() -> ALAMSVaRModel:
    global _var_model
    if _var_model is None:
        _var_model = ALAMSVaRModel()
    return _var_model


# ---------- Request / Response models ----------


class VaRFitRequest(BaseModel):
    """Request to fit (calibrate) the A-LAMS-VaR model."""

    returns: list[float] = Field(
        ..., min_length=100, description="Log returns array (min 100 observations)"
    )
    n_regimes: int = Field(default=5, ge=2, le=10)
    asymmetry_prior: float = Field(default=0.15, ge=0.0, le=0.5)
    max_iter: int = Field(default=200, ge=10, le=1000)


class VaRFitResponse(BaseModel):
    """Response after model calibration."""

    log_likelihood: float
    delta: float
    n_obs: int
    n_regimes: int
    aic: float
    bic: float
    stage1_success: bool
    stage2_success: bool


class VaRCalculateRequest(BaseModel):
    """Request to calculate VaR with optional liquidity adjustment."""

    confidence: float = Field(default=0.95, ge=0.5, le=0.999)
    trade_size_usd: float = Field(default=0.0, ge=0.0)
    pool_depth_usd: float = Field(default=1e9, gt=0.0)
    returns: Optional[list[float]] = Field(
        default=None, description="Optional: new returns to filter before calculation"
    )


class VaRCalculateResponse(BaseModel):
    """Response with VaR calculation results."""

    var_pure: float
    slippage_component: float
    var_total: float
    confidence: float
    current_regime: int
    regime_probs: list[float]
    delta: float
    regime_means: list[float]
    regime_sigmas: list[float]


# ---------- Endpoints ----------


@router.post("/risk/var/fit", response_model=VaRFitResponse)
async def fit_var_model(request: VaRFitRequest) -> VaRFitResponse:
    """
    Calibrate the A-LAMS-VaR model on historical returns.

    Two-stage MLE:
    1. Regime means/variances and base transition matrix
    2. Asymmetry parameter δ
    """
    global _var_model

    config = ALAMSConfig(
        n_regimes=request.n_regimes,
        asymmetry_prior=request.asymmetry_prior,
        max_iter=request.max_iter,
    )
    model = ALAMSVaRModel(config=config)

    try:
        returns = np.array(request.returns, dtype=np.float64)
        diagnostics = model.fit(returns)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("var_fit_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Model fit failed: {e}")

    _var_model = model

    return VaRFitResponse(**diagnostics)


@router.post("/risk/var/calculate", response_model=VaRCalculateResponse)
async def calculate_var(request: VaRCalculateRequest) -> VaRCalculateResponse:
    """
    Calculate A-LAMS-VaR (regime-weighted VaR + liquidity adjustment).

    Requires model to be fitted first via /risk/var/fit.
    """
    model = _get_or_create_model()

    if not model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Model not fitted. Call /risk/var/fit first.",
        )

    try:
        returns_arr = (
            np.array(request.returns, dtype=np.float64)
            if request.returns is not None
            else None
        )

        result = model.calculate_liquidity_adjusted_var(
            confidence=request.confidence,
            trade_size_usd=request.trade_size_usd,
            pool_depth_usd=request.pool_depth_usd,
            returns=returns_arr,
        )
    except Exception as e:
        logger.error("var_calculate_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

    return VaRCalculateResponse(**result)


@router.get("/risk/var/summary")
async def var_summary() -> dict[str, Any]:
    """Get current A-LAMS-VaR model summary."""
    model = _get_or_create_model()
    return model.summary()
