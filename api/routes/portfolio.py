"""Portfolio VaR endpoints: calibrate, VaR, marginal-VaR, stress-VaR, copula."""

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from api.models import (
    AssetDecompositionItem,
    AssetStressItem,
    CopulaCompareItem,
    CopulaCompareResponse,
    CopulaDiagnosticsResponse,
    CopulaFitResult,
    CopulaPortfolioVaRResponse,
    MarginalVaRResponse,
    PortfolioCalibrateRequest,
    PortfolioVaRResponse,
    RegimeBreakdownItem,
    RegimeCopulaItem,
    RegimeDependentCopulaVaRResponse,
    RegimeTailDependenceItem,
    StressVaRResponse,
    TailDependence,
)
from api.stores import (
    _copula_store,
    _get_copula_fit,
    _get_portfolio_model,
    _portfolio_store,
    _PORTFOLIO_KEY,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["portfolio"])


def _load_portfolio_returns(req: PortfolioCalibrateRequest) -> pd.DataFrame:
    if req.data_source.value == "solana":
        from cortex.data.solana import get_token_ohlcv, ohlcv_to_returns

        period = req.period
        now = datetime.now(timezone.utc)
        if period.endswith("y"):
            days = int(period[:-1]) * 365
        elif period.endswith("mo"):
            days = int(period[:-2]) * 30
        else:
            days = int(period.rstrip("d"))
        start_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = now.strftime("%Y-%m-%d")

        frames = {}
        for ticker in req.tokens:
            df = get_token_ohlcv(ticker, start_date, end_date, "1D")
            rets = ohlcv_to_returns(df)
            if len(rets) < 30:
                raise HTTPException(400, f"Insufficient data for {ticker}")
            frames[ticker] = rets.rename(ticker)
        returns_df = pd.DataFrame(frames).dropna()
        if len(returns_df) < 30:
            raise HTTPException(400, f"Only {len(returns_df)} common observations after alignment")
        return returns_df

    import yfinance as yf

    frames = {}
    for ticker in req.tokens:
        df = yf.download(ticker, period=req.period, auto_adjust=True, progress=False)
        if df.empty or len(df) < 30:
            raise HTTPException(400, f"Insufficient data for {ticker}")
        close = df["Close"].squeeze()
        rets = 100.0 * np.diff(np.log(close.values))
        frames[ticker] = pd.Series(rets, index=close.index[1:], name=ticker)
    returns_df = pd.DataFrame(frames).dropna()
    if len(returns_df) < 30:
        raise HTTPException(400, f"Only {len(returns_df)} common observations after alignment")
    return returns_df


@router.post("/portfolio/calibrate", response_model=PortfolioVaRResponse)
def calibrate_portfolio(req: PortfolioCalibrateRequest):
    from cortex.portfolio import calibrate_multivariate, portfolio_var as pvar_fn

    returns_df = _load_portfolio_returns(req)
    model = calibrate_multivariate(returns_df, num_states=req.num_states, method=req.method)
    _portfolio_store[_PORTFOLIO_KEY] = model

    if req.copula_family:
        from cortex.copula import compare_copulas, fit_copula

        family = req.copula_family.lower()
        if family == "auto":
            ranking = compare_copulas(returns_df)
            copula_fit = ranking[0] if ranking else fit_copula(returns_df, "gaussian")
        else:
            copula_fit = fit_copula(returns_df, family=family)
        _copula_store[_PORTFOLIO_KEY] = copula_fit

    result = pvar_fn(model, req.weights, alpha=0.05)
    return PortfolioVaRResponse(
        portfolio_var=result["portfolio_var"],
        portfolio_sigma=result["portfolio_sigma"],
        z_alpha=result["z_alpha"],
        weights=result["weights"],
        regime_breakdown=[RegimeBreakdownItem(**rb) for rb in result["regime_breakdown"]],
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/portfolio/var", response_model=PortfolioVaRResponse)
def compute_portfolio_var(
    weights: dict[str, float],
    alpha: float = Query(0.05, gt=0.0, lt=1.0),
):
    from cortex.portfolio import portfolio_var as pvar_fn

    if _PORTFOLIO_KEY not in _portfolio_store:
        raise HTTPException(404, "No calibrated portfolio. Call POST /portfolio/calibrate first.")
    model = _portfolio_store[_PORTFOLIO_KEY]
    result = pvar_fn(model, weights, alpha=alpha)
    return PortfolioVaRResponse(
        portfolio_var=result["portfolio_var"],
        portfolio_sigma=result["portfolio_sigma"],
        z_alpha=result["z_alpha"],
        weights=result["weights"],
        regime_breakdown=[RegimeBreakdownItem(**rb) for rb in result["regime_breakdown"]],
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/portfolio/marginal-var", response_model=MarginalVaRResponse)
def compute_marginal_var(
    weights: dict[str, float],
    alpha: float = Query(0.05, gt=0.0, lt=1.0),
):
    from cortex.portfolio import marginal_var as mvar_fn

    if _PORTFOLIO_KEY not in _portfolio_store:
        raise HTTPException(404, "No calibrated portfolio. Call POST /portfolio/calibrate first.")
    model = _portfolio_store[_PORTFOLIO_KEY]
    result = mvar_fn(model, weights, alpha=alpha)
    return MarginalVaRResponse(
        portfolio_var=result["portfolio_var"],
        portfolio_sigma=result["portfolio_sigma"],
        decomposition=[AssetDecompositionItem(**d) for d in result["decomposition"]],
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/portfolio/stress-var", response_model=StressVaRResponse)
def compute_stress_var(
    weights: dict[str, float],
    forced_regime: int = Query(5, ge=1),
    alpha: float = Query(0.05, gt=0.0, lt=1.0),
):
    from cortex.portfolio import stress_var as svar_fn

    if _PORTFOLIO_KEY not in _portfolio_store:
        raise HTTPException(404, "No calibrated portfolio. Call POST /portfolio/calibrate first.")
    model = _portfolio_store[_PORTFOLIO_KEY]
    result = svar_fn(model, weights, forced_regime=forced_regime, alpha=alpha)
    return StressVaRResponse(
        forced_regime=result["forced_regime"],
        stressed_var=result["stressed_var"],
        stressed_sigma=result["stressed_sigma"],
        normal_var=result["normal_var"],
        normal_sigma=result["normal_sigma"],
        stress_multiplier=result["stress_multiplier"],
        regime_correlation=result["regime_correlation"],
        asset_stress=[AssetStressItem(**a) for a in result["asset_stress"]],
        timestamp=datetime.now(timezone.utc),
    )


def _copula_fit_to_model(fit: dict) -> CopulaFitResult:
    td = fit["tail_dependence"]
    return CopulaFitResult(
        family=fit["family"],
        params=fit["params"],
        log_likelihood=fit["log_likelihood"],
        aic=fit["aic"],
        bic=fit["bic"],
        n_obs=fit["n_obs"],
        n_assets=fit["n_assets"],
        n_params=fit["n_params"],
        tail_dependence=TailDependence(lambda_lower=td["lambda_lower"], lambda_upper=td["lambda_upper"]),
    )


@router.post("/portfolio/copula/var", response_model=CopulaPortfolioVaRResponse)
def compute_copula_portfolio_var(
    weights: dict[str, float],
    alpha: float = Query(0.05, gt=0.0, lt=1.0),
    n_simulations: int = Query(10_000, ge=1000, le=100_000),
):
    from cortex.copula import copula_portfolio_var as cpvar_fn

    model = _get_portfolio_model()
    copula_fit = _get_copula_fit()
    result = cpvar_fn(model, weights, copula_fit, alpha=alpha, n_simulations=n_simulations)
    td = result["tail_dependence"]
    return CopulaPortfolioVaRResponse(
        copula_var=result["copula_var"],
        gaussian_var=result["gaussian_var"],
        var_ratio=result["var_ratio"],
        copula_family=result["copula_family"],
        tail_dependence=TailDependence(lambda_lower=td["lambda_lower"], lambda_upper=td["lambda_upper"]),
        n_simulations=result["n_simulations"],
        alpha=result["alpha"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/portfolio/copula/diagnostics", response_model=CopulaDiagnosticsResponse)
def get_copula_diagnostics():
    from cortex.copula import regime_conditional_copulas

    model = _get_portfolio_model()
    copula_fit = _get_copula_fit()

    regime_copulas = regime_conditional_copulas(model, family=copula_fit["family"])

    return CopulaDiagnosticsResponse(
        portfolio_key=_PORTFOLIO_KEY,
        copula_family=copula_fit["family"],
        fit=_copula_fit_to_model(copula_fit),
        regime_copulas=[
            RegimeCopulaItem(
                regime=rc["regime"],
                n_obs=rc["n_obs"],
                copula=_copula_fit_to_model(rc["copula"]),
            )
            for rc in regime_copulas
        ],
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/portfolio/copula/compare", response_model=CopulaCompareResponse)
def compare_portfolio_copulas():
    from cortex.copula import compare_copulas

    model = _get_portfolio_model()
    returns_df = model["returns_df"]
    ranking = compare_copulas(returns_df)

    if ranking:
        _copula_store[_PORTFOLIO_KEY] = ranking[0]

    return CopulaCompareResponse(
        portfolio_key=_PORTFOLIO_KEY,
        results=[
            CopulaCompareItem(
                family=r["family"],
                log_likelihood=r["log_likelihood"],
                aic=r["aic"],
                bic=r["bic"],
                tail_dependence=TailDependence(
                    lambda_lower=r["tail_dependence"]["lambda_lower"],
                    lambda_upper=r["tail_dependence"]["lambda_upper"],
                ),
                rank=r["rank"],
                best=r["best"],
            )
            for r in ranking
        ],
        timestamp=datetime.now(timezone.utc),
    )



@router.post("/portfolio/copula/regime-var", response_model=RegimeDependentCopulaVaRResponse)
def compute_regime_dependent_copula_var(
    weights: dict[str, float] | None = None,
    alpha: float = Query(0.05, gt=0.0, lt=1.0),
    n_simulations: int = Query(10_000, ge=1000, le=100_000),
):
    from cortex.copula import regime_dependent_copula_var as rdcv_fn

    model = _get_portfolio_model()
    if not weights:
        assets = model["assets"]
        weights = {a: 1.0 / len(assets) for a in assets}
    result = rdcv_fn(model, weights, alpha=alpha, n_simulations=n_simulations)

    rc = result["current_regime_copula"]
    td_rc = rc["tail_dependence"]

    return RegimeDependentCopulaVaRResponse(
        regime_dependent_var=result["regime_dependent_var"],
        static_var=result["static_var"],
        var_difference_pct=result["var_difference_pct"],
        current_regime_copula=CopulaFitResult(
            family=rc["family"],
            params=rc["params"],
            log_likelihood=rc["log_likelihood"],
            aic=rc["aic"],
            bic=rc["bic"],
            n_obs=rc["n_obs"],
            n_assets=rc["n_assets"],
            n_params=rc["n_params"],
            tail_dependence=TailDependence(
                lambda_lower=td_rc["lambda_lower"],
                lambda_upper=td_rc["lambda_upper"],
            ),
        ),
        regime_tail_dependence=[
            RegimeTailDependenceItem(
                regime=rtd["regime"],
                family=rtd["family"],
                lambda_lower=rtd["lambda_lower"],
                lambda_upper=rtd["lambda_upper"],
            )
            for rtd in result["regime_tail_dependence"]
        ],
        dominant_regime=result["dominant_regime"],
        regime_probs=result["regime_probs"],
        n_simulations=result["n_simulations"],
        alpha=result["alpha"],
        timestamp=datetime.now(timezone.utc),
    )
