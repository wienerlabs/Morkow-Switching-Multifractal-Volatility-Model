import asyncio
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect

from api.models import (
    AssetDecompositionItem,
    AssetStressItem,
    BacktestSummaryResponse,
    CalibrateRequest,
    CalibrateResponse,
    CalibrationMetrics,
    CompareRequest,
    CompareResponse,
    ComparisonReportResponse,
    CopulaCompareItem,
    CopulaCompareResponse,
    CopulaDiagnosticsResponse,
    CopulaFitResult,
    CopulaPortfolioVaRResponse,
    ErrorResponse,
    GuardianAssessRequest,
    GuardianAssessResponse,
    GuardianComponentScore,
    RegimeDependentCopulaVaRResponse,
    RegimeTailDependenceItem,
    EVTBacktestResponse,
    EVTBacktestRow,
    EVTCalibrateRequest,
    EVTCalibrateResponse,
    EVTDiagnosticsResponse,
    EVTVaRResponse,
    HawkesCalibrateRequest,
    HawkesCalibrateResponse,
    HawkesClusterItem,
    HawkesClustersResponse,
    HawkesIntensityResponse,
    HawkesSimulateRequest,
    HawkesSimulateResponse,
    HawkesVaRRequest,
    HawkesVaRResponse,
    HurstResponse,
    FractalDiagnosticsResponse,
    MultifractalSpectrumResponse,
    RegimeHurstItem,
    RegimeHurstResponse,
    RoughCalibrateRequest,
    RoughCalibrateResponse,
    RoughCalibrationMetrics,
    RoughCompareMSMResponse,
    RoughDiagnosticsResponse,
    RoughForecastResponse,
    RoughModelMetrics,
    SVJCalibrateRequest,
    SVJCalibrateResponse,
    SVJClustering,
    SVJDiagnosticsResponse,
    SVJEVTTail,
    SVJHawkesParams,
    SVJJumpRiskResponse,
    SVJJumpStats,
    SVJMomentComparison,
    SVJParameterQuality,
    SVJVaRResponse,
    MarginalVaRResponse,
    ModelMetricsRow,
    NewsFeedResponse,
    NewsMarketSignalModel,
    PortfolioCalibrateRequest,
    PortfolioVaRResponse,
    RegimeBreakdownItem,
    RegimeCopulaItem,
    RegimeDurationsResponse,
    RegimeHistoryResponse,
    RegimePeriod,
    RegimeResponse,
    RegimeStatisticsResponse,
    RegimeStatRow,
    RegimeStreamMessage,
    StressVaRResponse,
    TailDependence,
    TailProbResponse,
    TransitionAlertResponse,
    VaRComparisonRow,
    VaRResponse,
    VolatilityForecastResponse,
    get_regime_name,
)

from api.middleware import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(verify_api_key)])

# In-memory model state (per-token)
_model_store: dict[str, dict] = {}
_portfolio_store: dict[str, dict] = {}
_evt_store: dict[str, dict] = {}  # EVT calibration results per token
_copula_store: dict[str, dict] = {}  # Copula fit results per portfolio key
_hawkes_store: dict[str, dict] = {}  # Hawkes calibration results per token
_rough_store: dict[str, dict] = {}  # Rough volatility calibration per token
_svj_store: dict[str, dict] = {}  # SVJ calibration results per token


def _get_model(token: str) -> dict:
    if token not in _model_store:
        raise HTTPException(
            status_code=404,
            detail=f"No calibrated model for '{token}'. Call POST /calibrate first.",
        )
    return _model_store[token]


def _load_returns(req: CalibrateRequest) -> pd.Series:
    """Fetch data and convert to log-returns in %."""
    if req.data_source.value == "solana":
        from cortex.data.solana import get_token_ohlcv, ohlcv_to_returns

        df = get_token_ohlcv(req.token, req.start_date, req.end_date, req.interval)
        return ohlcv_to_returns(df)

    import yfinance as yf

    df = yf.download(req.token, start=req.start_date, end=req.end_date, progress=False)
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No yfinance data for '{req.token}'")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    close = close.dropna()
    vals = close.values.flatten()
    rets = 100 * np.diff(np.log(vals))
    return pd.Series(rets, index=close.index[1:], name="r")


@router.post("/calibrate", response_model=CalibrateResponse)
def calibrate(req: CalibrateRequest):
    from cortex import msm

    try:
        returns = _load_returns(req)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if len(returns) < 30:
        raise HTTPException(status_code=400, detail="Need at least 30 data points")

    cal = msm.calibrate_msm_advanced(
        returns,
        num_states=req.num_states,
        method=req.method.value,
        target_var_breach=req.target_var_breach,
        verbose=False,
    )

    sigma_f, sigma_filt, fprobs, sigma_states, P = msm.msm_vol_forecast(
        returns,
        num_states=cal["num_states"],
        sigma_low=cal["sigma_low"],
        sigma_high=cal["sigma_high"],
        p_stay=cal["p_stay"],
    )

    _model_store[req.token] = {
        "calibration": cal,
        "returns": returns,
        "sigma_forecast": sigma_f,
        "sigma_filtered": sigma_filt,
        "filter_probs": fprobs,
        "sigma_states": sigma_states,
        "P_matrix": P,
        "use_student_t": req.use_student_t,
        "nu": req.nu,
        "calibrated_at": datetime.now(timezone.utc),
    }

    return CalibrateResponse(
        token=req.token,
        method=cal["method"],
        num_states=cal["num_states"],
        sigma_low=cal["sigma_low"],
        sigma_high=cal["sigma_high"],
        p_stay=cal["p_stay"],
        sigma_states=cal["sigma_states"].tolist(),
        metrics=CalibrationMetrics(**cal["metrics"]),
        calibrated_at=_model_store[req.token]["calibrated_at"],
    )


@router.get("/regime/current", response_model=RegimeResponse)
def get_current_regime(token: str = Query(...)):
    from cortex import msm
    m = _get_model(token)

    probs = np.asarray(m["filter_probs"].iloc[-1])
    state_idx = int(np.argmax(probs)) + 1
    num_states = m["calibration"]["num_states"]

    var_t1, sigma_t1, z_alpha, pi_t1 = msm.msm_var_forecast_next_day(
        m["filter_probs"], m["sigma_states"], m["P_matrix"], alpha=0.05
    )

    return RegimeResponse(
        timestamp=datetime.now(timezone.utc),
        regime_state=state_idx,
        regime_name=get_regime_name(state_idx, num_states),
        regime_probabilities=probs.tolist(),
        volatility_filtered=float(m["sigma_filtered"].iloc[-1]),
        volatility_forecast=sigma_t1,
        var_95=var_t1,
        transition_matrix=m["P_matrix"].tolist(),
    )




@router.get("/var/{confidence}", response_model=VaRResponse)
def get_var(
    confidence: float,
    token: str = Query(...),
    use_student_t: bool = Query(None, description="Override distribution. Defaults to calibration setting."),
    nu: float = Query(None, gt=2.0, description="Override Student-t df."),
):
    from cortex import msm
    m = _get_model(token)

    if confidence > 1.0:
        confidence = confidence / 100.0
    alpha = 1.0 - confidence if confidence > 0.5 else confidence
    st = use_student_t if use_student_t is not None else m.get("use_student_t", False)
    df = nu if nu is not None else m.get("nu", 5.0)

    var_t1, sigma_t1, z_alpha, pi_t1 = msm.msm_var_forecast_next_day(
        m["filter_probs"], m["sigma_states"], m["P_matrix"],
        alpha=alpha, use_student_t=st, nu=df,
    )

    return VaRResponse(
        timestamp=datetime.now(timezone.utc),
        confidence=confidence,
        var_value=var_t1,
        sigma_forecast=sigma_t1,
        z_alpha=z_alpha,
        regime_probabilities=pi_t1.tolist(),
        distribution="student_t" if st else "normal",
    )


@router.get("/volatility/forecast", response_model=VolatilityForecastResponse)
def get_volatility_forecast(token: str = Query(...)):
    from cortex import msm
    m = _get_model(token)

    _, sigma_t1, _, pi_t1 = msm.msm_var_forecast_next_day(
        m["filter_probs"], m["sigma_states"], m["P_matrix"]
    )

    return VolatilityForecastResponse(
        timestamp=datetime.now(timezone.utc),
        sigma_forecast=sigma_t1,
        sigma_filtered=float(m["sigma_filtered"].iloc[-1]),
        regime_probabilities=pi_t1.tolist(),
        sigma_states=m["sigma_states"].tolist(),
    )


@router.get("/backtest/summary", response_model=BacktestSummaryResponse)
def get_backtest_summary(token: str = Query(...), alpha: float = Query(0.05)):
    from scipy.stats import norm

    from cortex import msm
    m = _get_model(token)

    returns = m["returns"]
    sigma_forecast = m["sigma_forecast"]
    z = norm.ppf(alpha)
    var_series = z * sigma_forecast
    breaches = (returns < var_series).astype(int)

    kupiec_lr, kupiec_p, x, n = msm.kupiec_test(breaches, alpha=alpha)
    chris_lr, chris_p, _ = msm.christoffersen_independence_test(breaches)

    return BacktestSummaryResponse(
        token=token,
        num_observations=int(n),
        var_alpha=alpha,
        breach_count=int(x),
        breach_rate=float(x / n) if n > 0 else 0.0,
        kupiec_lr=None if np.isnan(kupiec_lr) else float(kupiec_lr),
        kupiec_pvalue=None if np.isnan(kupiec_p) else float(kupiec_p),
        kupiec_pass=bool(kupiec_p > 0.05) if not np.isnan(kupiec_p) else False,
        christoffersen_lr=None if np.isnan(chris_lr) else float(chris_lr),
        christoffersen_pvalue=None if np.isnan(chris_p) else float(chris_p),
        christoffersen_pass=bool(chris_p > 0.05) if not np.isnan(chris_p) else False,
    )


@router.get("/tail-probs", response_model=TailProbResponse)
def get_tail_probs(
    token: str = Query(...),
    alpha: float = Query(0.05),
    use_student_t: bool = Query(False),
    nu: float = Query(5.0),
):
    from cortex import msm
    m = _get_model(token)

    result = msm.msm_tail_probs(
        m["returns"],
        m["filter_probs"],
        m["sigma_states"],
        alpha=alpha,
        horizons=(1, 3, 5),
        use_student_t=use_student_t,
        nu=nu,
    )

    return TailProbResponse(
        l1_threshold=result["L1"],
        p1_day=result["p1"],
        horizon_probs={int(k): v for k, v in result["horizon_probs"].items()},
        distribution=result["distribution"],
    )


@router.websocket("/stream/regime")
async def stream_regime(ws: WebSocket, token: str = Query(...)):
    """Stream regime updates every 5 seconds for a calibrated token."""
    from cortex import msm
    await ws.accept()

    try:
        while True:
            if token not in _model_store:
                await ws.send_json({"error": f"No model for '{token}'"})
                await asyncio.sleep(5)
                continue

            m = _model_store[token]
            probs = np.asarray(m["filter_probs"].iloc[-1])
            state_idx = int(np.argmax(probs)) + 1
            num_states = m["calibration"]["num_states"]

            var_t1, sigma_t1, _, _ = msm.msm_var_forecast_next_day(
                m["filter_probs"], m["sigma_states"], m["P_matrix"], alpha=0.05
            )

            msg = RegimeStreamMessage(
                timestamp=datetime.now(timezone.utc),
                regime_state=state_idx,
                regime_name=get_regime_name(state_idx, num_states),
                regime_probabilities=probs.tolist(),
                volatility_forecast=sigma_t1,
                var_95=var_t1,
            )
            await ws.send_json(msg.model_dump(mode="json"))
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected for token=%s", token)
    except Exception as exc:
        logger.exception("WebSocket error for token=%s", token)
        await ws.close(code=1011, reason=str(exc)[:120])


# ── News Intelligence Endpoints ──


def _current_regime_state() -> int:
    """Get regime state from any calibrated model, default 3 (Normal)."""
    for m in _model_store.values():
        probs = np.asarray(m["filter_probs"].iloc[-1])
        return int(np.argmax(probs)) + 1
    return 3


@router.get("/news/feed", response_model=NewsFeedResponse)
def get_news_feed(
    regime_state: int = Query(None, ge=1, le=10, description="Override regime state"),
    max_items: int = Query(50, ge=1, le=200),
):
    """
    Full news intelligence feed: fetch from all sources, score, deduplicate, aggregate.
    If a model is calibrated, uses its regime state for impact amplification.
    """
    from cortex.news import fetch_news_intelligence

    rs = regime_state if regime_state is not None else _current_regime_state()
    try:
        result = fetch_news_intelligence(
            regime_state=rs, max_items=max_items,
        )
    except Exception as exc:
        logger.exception("News feed fetch failed")
        raise HTTPException(status_code=502, detail=f"News fetch error: {exc}")

    return NewsFeedResponse(**result)


@router.get("/news/sentiment", response_model=NewsFeedResponse)
def get_news_sentiment(
    regime_state: int = Query(None, ge=1, le=10),
    max_items: int = Query(20, ge=1, le=100),
):
    """
    Same as /news/feed but with smaller default page size — intended for
    quick sentiment checks without the full feed.
    """
    from cortex.news import fetch_news_intelligence

    rs = regime_state if regime_state is not None else _current_regime_state()
    try:
        result = fetch_news_intelligence(
            regime_state=rs, max_items=max_items,
        )
    except Exception as exc:
        logger.exception("News sentiment fetch failed")
        raise HTTPException(status_code=502, detail=f"News fetch error: {exc}")

    return NewsFeedResponse(**result)


@router.get("/news/signal", response_model=NewsMarketSignalModel)
def get_news_signal(
    regime_state: int = Query(None, ge=1, le=10),
):
    """
    Returns only the aggregate MarketSignal — direction, strength, EWMA,
    momentum, entropy, confidence. Lightweight endpoint for trading bots.
    """
    from cortex.news import fetch_news_intelligence

    rs = regime_state if regime_state is not None else _current_regime_state()
    try:
        result = fetch_news_intelligence(
            regime_state=rs, max_items=30,
        )
    except Exception as exc:
        logger.exception("News signal fetch failed")
        raise HTTPException(status_code=502, detail=f"News fetch error: {exc}")

    return NewsMarketSignalModel(**result["signal"])


# ── Regime Analytics Endpoints ──


@router.get("/regime/durations", response_model=RegimeDurationsResponse)
def get_regime_durations(token: str = Query(...)):
    """Expected duration (in days) for each regime state."""
    from cortex.regime import compute_expected_durations

    m = _get_model(token)
    cal = m["calibration"]
    durations = compute_expected_durations(cal["p_stay"], cal["num_states"])

    return RegimeDurationsResponse(
        token=token,
        p_stay=cal["p_stay"],
        num_states=cal["num_states"],
        durations=durations,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/regime/history", response_model=RegimeHistoryResponse)
def get_regime_history(token: str = Query(...)):
    """Historical timeline of consecutive regime periods."""
    from cortex.regime import extract_regime_history

    m = _get_model(token)
    df = extract_regime_history(m["filter_probs"], m["returns"], m["sigma_states"])

    periods = [
        RegimePeriod(
            start=row["start"],
            end=row["end"],
            regime=int(row["regime"]),
            duration=int(row["duration"]),
            cumulative_return=float(row["cumulative_return"]),
            volatility=float(row["volatility"]),
            max_drawdown=float(row["max_drawdown"]),
        )
        for _, row in df.iterrows()
    ]

    return RegimeHistoryResponse(
        token=token,
        num_periods=len(periods),
        periods=periods,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/regime/transition-alert", response_model=TransitionAlertResponse)
def get_transition_alert(
    token: str = Query(...),
    threshold: float = Query(0.3, gt=0.0, lt=1.0),
):
    """Alert when probability of leaving current regime exceeds threshold."""
    from cortex.regime import detect_regime_transition

    m = _get_model(token)
    result = detect_regime_transition(m["filter_probs"], threshold=threshold)

    return TransitionAlertResponse(
        token=token,
        alert=result["alert"],
        current_regime=result["current_regime"],
        transition_probability=result["transition_probability"],
        most_likely_next_regime=result["most_likely_next_regime"],
        next_regime_probability=result["next_regime_probability"],
        threshold=result["threshold"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/regime/statistics", response_model=RegimeStatisticsResponse)
def get_regime_statistics(token: str = Query(...)):
    """Per-regime conditional statistics (mean return, vol, Sharpe, drawdown)."""
    from cortex.regime import compute_regime_statistics

    m = _get_model(token)
    df = compute_regime_statistics(m["returns"], m["filter_probs"], m["sigma_states"])

    stats = [
        RegimeStatRow(
            regime=int(row["regime"]),
            mean_return=float(row["mean_return"]),
            volatility=float(row["volatility"]),
            sharpe_ratio=float(row["sharpe_ratio"]),
            max_drawdown=float(row["max_drawdown"]),
            days_in_regime=int(row["days_in_regime"]),
            frequency=float(row["frequency"]),
        )
        for _, row in df.iterrows()
    ]

    return RegimeStatisticsResponse(
        token=token,
        num_states=m["calibration"]["num_states"],
        total_observations=len(m["returns"]),
        statistics=stats,
        timestamp=datetime.now(timezone.utc),
    )


# ── Model Comparison Endpoints ──

# Cache comparison results per token for the report endpoint
_comparison_cache: dict[str, tuple[pd.DataFrame, float]] = {}


@router.post("/compare", response_model=CompareResponse)
def run_model_comparison(req: CompareRequest):
    """Run volatility model comparison on a calibrated token's returns."""
    from cortex.comparison import compare_models, _MODEL_REGISTRY

    m = _get_model(req.token)
    returns = m["returns"]

    if req.models:
        invalid = [k for k in req.models if k not in _MODEL_REGISTRY]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model keys: {invalid}. Valid: {list(_MODEL_REGISTRY.keys())}",
            )

    try:
        df = compare_models(returns, alpha=req.alpha, models=req.models)
    except Exception as exc:
        logger.exception("Model comparison failed for token=%s", req.token)
        raise HTTPException(status_code=500, detail=f"Comparison error: {exc}")

    _comparison_cache[req.token] = (df, req.alpha)

    results = [
        ModelMetricsRow(**row.to_dict())
        for _, row in df.iterrows()
    ]

    return CompareResponse(
        token=req.token,
        alpha=req.alpha,
        num_observations=len(returns),
        models_compared=[r.model for r in results],
        results=results,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/compare/report/{token}", response_model=ComparisonReportResponse)
def get_comparison_report(token: str, alpha: float = Query(0.05)):
    """Generate a structured report from a previous comparison run."""
    from cortex.comparison import generate_comparison_report

    if token not in _comparison_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No comparison results for '{token}'. Call POST /compare first.",
        )

    df, cached_alpha = _comparison_cache[token]
    report = generate_comparison_report(df, alpha=cached_alpha)

    return ComparisonReportResponse(
        token=token,
        alpha=cached_alpha,
        summary_table=report["summary_table"],
        winners=report["winners"],
        pass_fail=report["pass_fail"],
        ranking=report["ranking"],
        timestamp=datetime.now(timezone.utc),
    )


# =========================================================================
# Portfolio VaR Endpoints
# =========================================================================

_PORTFOLIO_KEY = "default"


def _load_portfolio_returns(req: PortfolioCalibrateRequest) -> pd.DataFrame:
    """Fetch multi-asset returns and return a DataFrame of log-returns in %."""
    if req.data_source.value == "solana":
        from datetime import timedelta
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
    """Calibrate multi-asset MSM and compute portfolio VaR. Optionally fit copula."""
    from cortex.portfolio import calibrate_multivariate, portfolio_var as pvar_fn

    returns_df = _load_portfolio_returns(req)
    model = calibrate_multivariate(returns_df, num_states=req.num_states, method=req.method)
    _portfolio_store[_PORTFOLIO_KEY] = model

    # Optionally fit copula during calibration
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
    """Compute portfolio VaR with custom weights/alpha on a previously calibrated model."""
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
    """Marginal VaR and Euler risk decomposition."""
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
    """Stressed VaR by forcing the model into a specific regime."""
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


# =========================================================================
# EVT (Extreme Value Theory) Endpoints
# =========================================================================


@router.post("/evt/calibrate", response_model=EVTCalibrateResponse)
def evt_calibrate(req: EVTCalibrateRequest):
    """Fit GPD to historical losses from a calibrated MSM model."""
    from cortex.evt import fit_gpd, select_threshold

    m = _get_model(req.token)
    returns = m["returns"]

    try:
        th_result = select_threshold(
            returns,
            method=req.threshold_method.value,
            min_exceedances=req.min_exceedances,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Threshold selection failed: {exc}")

    losses = -np.asarray(returns.values if hasattr(returns, "values") else returns, dtype=float)

    try:
        gpd = fit_gpd(losses, threshold=th_result["threshold"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"GPD fitting failed: {exc}")

    _evt_store[req.token] = {
        **gpd,
        "threshold_method": req.threshold_method.value,
    }

    return EVTCalibrateResponse(
        token=req.token,
        xi=gpd["xi"],
        beta=gpd["beta"],
        threshold=gpd["threshold"],
        n_total=gpd["n_total"],
        n_exceedances=gpd["n_exceedances"],
        log_likelihood=gpd["log_likelihood"],
        aic=gpd["aic"],
        bic=gpd["bic"],
        threshold_method=req.threshold_method.value,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/evt/var/{confidence}", response_model=EVTVaRResponse)
def get_evt_var(confidence: float, token: str = Query(...)):
    """Compute EVT-VaR and CVaR for extreme tail quantiles."""
    from cortex.evt import evt_cvar, evt_var

    if token not in _evt_store:
        raise HTTPException(404, f"No EVT calibration for '{token}'. Call POST /evt/calibrate first.")

    e = _evt_store[token]
    if confidence > 1.0:
        confidence = confidence / 100.0
    alpha = 1.0 - confidence if confidence > 0.5 else confidence

    var_loss = evt_var(
        xi=e["xi"], beta=e["beta"], threshold=e["threshold"],
        n_total=e["n_total"], n_exceedances=e["n_exceedances"], alpha=alpha,
    )
    cvar_loss = evt_cvar(
        xi=e["xi"], beta=e["beta"], threshold=e["threshold"],
        var_value=var_loss, alpha=alpha,
    )

    return EVTVaRResponse(
        timestamp=datetime.now(timezone.utc),
        confidence=1.0 - alpha,
        var_value=-var_loss,
        cvar_value=-cvar_loss,
        xi=e["xi"],
        beta=e["beta"],
        threshold=e["threshold"],
    )


@router.get("/evt/diagnostics", response_model=EVTDiagnosticsResponse)
def get_evt_diagnostics(token: str = Query(...)):
    """Return EVT backtest results and Normal/Student-t/EVT comparison."""
    from cortex.evt import compare_var_methods, evt_backtest

    m = _get_model(token)
    if token not in _evt_store:
        raise HTTPException(404, f"No EVT calibration for '{token}'. Call POST /evt/calibrate first.")

    e = _evt_store[token]
    returns = m["returns"]
    sigma_forecast = float(m["sigma_forecast"].iloc[-1])
    nu = m.get("nu", 5.0)

    bt = evt_backtest(
        returns, xi=e["xi"], beta=e["beta"], threshold=e["threshold"],
        n_total=e["n_total"], n_exceedances=e["n_exceedances"],
    )
    cmp = compare_var_methods(
        returns, sigma_forecast=sigma_forecast,
        xi=e["xi"], beta=e["beta"], threshold=e["threshold"],
        n_total=e["n_total"], n_exceedances=e["n_exceedances"], nu=nu,
    )

    return EVTDiagnosticsResponse(
        token=token,
        xi=e["xi"],
        beta=e["beta"],
        threshold=e["threshold"],
        threshold_method=e["threshold_method"],
        n_exceedances=e["n_exceedances"],
        backtest=[EVTBacktestRow(**row) for row in bt],
        comparison=[VaRComparisonRow(**row) for row in cmp],
        timestamp=datetime.now(timezone.utc),
    )


# =========================================================================
# Copula Portfolio VaR Endpoints
# =========================================================================


def _get_portfolio_model() -> dict:
    if _PORTFOLIO_KEY not in _portfolio_store:
        raise HTTPException(404, "No calibrated portfolio. Call POST /portfolio/calibrate first.")
    return _portfolio_store[_PORTFOLIO_KEY]


def _get_copula_fit() -> dict:
    if _PORTFOLIO_KEY not in _copula_store:
        raise HTTPException(
            404,
            "No copula fit. Call POST /portfolio/calibrate with copula_family "
            "or POST /portfolio/copula/compare first.",
        )
    return _copula_store[_PORTFOLIO_KEY]


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
    """Portfolio VaR using copula-based Monte Carlo simulation."""
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
    """Return copula fit details and regime-conditional copulas."""
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
    """Compare all copula families on the calibrated portfolio data."""
    from cortex.copula import compare_copulas

    model = _get_portfolio_model()
    returns_df = model["returns_df"]
    ranking = compare_copulas(returns_df)

    # Store the best copula fit
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
    """Portfolio VaR using regime-dependent copula mixture.

    Crisis regimes use Student-t copula (tail dependence),
    calm regimes use Gaussian copula (no tail dependence).
    Samples are blended proportionally to current regime probabilities.
    """
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


# ── Hawkes Process Endpoints ──────────────────────────────────────────


@router.post("/hawkes/calibrate", response_model=HawkesCalibrateResponse)
async def hawkes_calibrate(req: HawkesCalibrateRequest):
    """Fit Hawkes self-exciting process to extreme events from calibrated model."""
    from cortex.hawkes import extract_events, fit_hawkes

    m = _get_model(req.token)
    returns = m["returns"]

    ev = extract_events(
        returns,
        threshold_percentile=req.threshold_percentile,
        use_absolute=req.use_absolute,
    )

    if ev["n_events"] < 5:
        raise HTTPException(
            400,
            f"Only {ev['n_events']} extreme events detected. "
            "Lower threshold_percentile or use more data.",
        )

    params = fit_hawkes(ev["event_times"], ev["T"])

    _hawkes_store[req.token] = {
        **params,
        "event_times": ev["event_times"],
        "event_returns": ev["event_returns"],
        "threshold": ev["threshold"],
    }

    return HawkesCalibrateResponse(
        token=req.token,
        mu=params["mu"],
        alpha=params["alpha"],
        beta=params["beta"],
        branching_ratio=params["branching_ratio"],
        half_life=params["half_life"],
        stationary=params["stationary"],
        n_events=params["n_events"],
        log_likelihood=params["log_likelihood"],
        aic=params["aic"],
        bic=params["bic"],
        threshold=ev["threshold"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/hawkes/intensity", response_model=HawkesIntensityResponse)
async def hawkes_intensity_endpoint(token: str = Query(...)):
    """Get current Hawkes intensity, crash clustering metrics, and contagion risk score."""
    from cortex.hawkes import detect_flash_crash_risk, hawkes_intensity

    _get_model(token)
    if token not in _hawkes_store:
        raise HTTPException(404, f"No Hawkes calibration for '{token}'. Call POST /hawkes/calibrate first.")

    h = _hawkes_store[token]
    result = hawkes_intensity(h["event_times"], h)
    risk = detect_flash_crash_risk(h["event_times"], h)

    return HawkesIntensityResponse(
        token=token,
        current_intensity=result["current_intensity"],
        baseline=result["baseline"],
        intensity_ratio=result["intensity_ratio"],
        peak_intensity=result["peak_intensity"],
        mean_intensity=result["mean_intensity"],
        contagion_risk_score=risk["contagion_risk_score"],
        excitation_level=risk["excitation_level"],
        risk_level=risk["risk_level"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/hawkes/clusters", response_model=HawkesClustersResponse)
async def hawkes_clusters_endpoint(token: str = Query(...)):
    """Detect temporal clusters of extreme events."""
    from cortex.hawkes import detect_clusters

    _get_model(token)
    if token not in _hawkes_store:
        raise HTTPException(404, f"No Hawkes calibration for '{token}'. Call POST /hawkes/calibrate first.")

    h = _hawkes_store[token]
    clusters = detect_clusters(h["event_times"], h)

    return HawkesClustersResponse(
        token=token,
        clusters=[HawkesClusterItem(**c) for c in clusters],
        n_clusters=len(clusters),
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/hawkes/var", response_model=HawkesVaRResponse)
async def hawkes_var_endpoint(req: HawkesVaRRequest):
    """Compute Hawkes intensity-adjusted VaR."""
    from cortex.hawkes import hawkes_intensity, hawkes_var_adjustment

    m = _get_model(req.token)
    if req.token not in _hawkes_store:
        raise HTTPException(404, f"No Hawkes calibration for '{req.token}'. Call POST /hawkes/calibrate first.")

    h = _hawkes_store[req.token]

    import MSM_VaR_MODEL as msm
    alpha = 1.0 - req.confidence / 100.0
    st = m.get("use_student_t", False)
    df = m.get("nu", 5.0)

    var_t1, sigma_t1, z_alpha, pi_t1 = msm.msm_var_forecast_next_day(
        m["filter_probs"], m["sigma_states"], m["P_matrix"],
        alpha=alpha, use_student_t=st, nu=df,
    )

    intens = hawkes_intensity(h["event_times"], h)
    adj = hawkes_var_adjustment(
        var_t1, intens["current_intensity"], intens["baseline"],
        max_multiplier=req.max_multiplier,
    )

    from cortex.hawkes import detect_flash_crash_risk
    risk = detect_flash_crash_risk(h["event_times"], h)

    return HawkesVaRResponse(
        adjusted_var=adj["adjusted_var"],
        base_var=adj["base_var"],
        multiplier=adj["multiplier"],
        intensity_ratio=adj["intensity_ratio"],
        capped=adj["capped"],
        confidence=req.confidence,
        recent_events=risk["recent_event_count"],
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/hawkes/simulate", response_model=HawkesSimulateResponse)
async def hawkes_simulate_endpoint(req: HawkesSimulateRequest):
    """Simulate Hawkes process to generate synthetic crash scenarios."""
    from cortex.hawkes import simulate_hawkes

    if req.token is not None:
        if req.token not in _hawkes_store:
            raise HTTPException(404, f"No Hawkes calibration for '{req.token}'. Call POST /hawkes/calibrate first.")
        h = _hawkes_store[req.token]
        params = {"mu": h["mu"], "alpha": h["alpha"], "beta": h["beta"]}
    elif req.mu is not None and req.alpha is not None and req.beta is not None:
        params = {"mu": req.mu, "alpha": req.alpha, "beta": req.beta}
    else:
        raise HTTPException(400, "Provide either 'token' or all of (mu, alpha, beta)")

    if params["alpha"] / params["beta"] >= 1.0:
        raise HTTPException(400, f"Branching ratio α/β = {params['alpha']/params['beta']:.3f} ≥ 1 — process is non-stationary")

    sim = simulate_hawkes(params, T=req.T, seed=req.seed)

    intensity_arr = sim["intensity_path"]
    return HawkesSimulateResponse(
        n_events=sim["n_events"],
        T=sim["T"],
        mean_intensity=float(np.mean(intensity_arr)),
        peak_intensity=float(np.max(intensity_arr)) if intensity_arr else params["mu"],
        timestamp=datetime.now(timezone.utc),
    )


# ── Multifractal / Hurst endpoints ───────────────────────────────────


@router.get(
    "/fractal/hurst",
    response_model=HurstResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Hurst exponent (R/S or DFA)",
)
def get_fractal_hurst(
    token: str = Query(...),
    method: str = Query("rs", pattern="^(rs|dfa)$"),
):
    from cortex.multifractal import hurst_dfa, hurst_rs

    m = _get_model(token)
    ret = m["returns"]

    result = hurst_rs(ret) if method == "rs" else hurst_dfa(ret)

    return HurstResponse(
        token=token,
        H=result["H"],
        H_se=result["H_se"],
        r_squared=result["r_squared"],
        interpretation=result["interpretation"],
        method=result["method"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/fractal/spectrum",
    response_model=MultifractalSpectrumResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Multifractal spectrum f(α)",
)
def get_fractal_spectrum(token: str = Query(...)):
    from cortex.multifractal import multifractal_spectrum

    m = _get_model(token)
    spec = multifractal_spectrum(m["returns"])

    return MultifractalSpectrumResponse(
        token=token,
        width=spec["width"],
        peak_alpha=spec["peak_alpha"],
        is_multifractal=spec["is_multifractal"],
        q_values=spec["q_values"],
        tau_q=spec["tau_q"],
        H_q=spec["H_q"],
        alpha=spec["alpha"],
        f_alpha=spec["f_alpha"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/fractal/regime-hurst",
    response_model=RegimeHurstResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Per-regime Hurst analysis",
)
def get_fractal_regime_hurst(token: str = Query(...)):
    from cortex.multifractal import compare_fractal_regimes

    m = _get_model(token)
    result = compare_fractal_regimes(
        m["returns"], m["filter_probs"], m["sigma_states"]
    )

    items = [RegimeHurstItem(**r) for r in result["per_regime"]]

    return RegimeHurstResponse(
        token=token,
        per_regime=items,
        n_states=result["n_states"],
        summary=result["summary"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/fractal/diagnostics",
    response_model=FractalDiagnosticsResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Full fractal diagnostics",
)
def get_fractal_diagnostics(token: str = Query(...)):
    from cortex.multifractal import (
        hurst_dfa,
        hurst_rs,
        long_range_dependence_test,
        multifractal_spectrum,
    )

    m = _get_model(token)
    ret = m["returns"]

    rs = hurst_rs(ret)
    dfa = hurst_dfa(ret)
    spec = multifractal_spectrum(ret)
    lrd = long_range_dependence_test(ret)

    return FractalDiagnosticsResponse(
        token=token,
        H_rs=rs["H"],
        H_dfa=dfa["H"],
        spectrum_width=spec["width"],
        is_multifractal=spec["is_multifractal"],
        is_long_range_dependent=lrd["is_long_range_dependent"],
        confidence_z=lrd["confidence_z"],
        timestamp=datetime.now(timezone.utc),
    )



# ── Rough Volatility ─────────────────────────────────────────────────


@router.post(
    "/rough/calibrate",
    response_model=RoughCalibrateResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Calibrate rough volatility model (rBergomi or rHeston)",
)
def post_rough_calibrate(req: RoughCalibrateRequest):
    from cortex.rough_vol import calibrate_rough_bergomi, calibrate_rough_heston

    m = _get_model(req.token)

    if req.model.value == "rough_bergomi":
        cal = calibrate_rough_bergomi(m["returns"], window=req.window, max_lag=req.max_lag)
    else:
        cal = calibrate_rough_heston(m["returns"], window=req.window, max_lag=req.max_lag)

    _rough_store[req.token] = cal

    metrics = RoughCalibrationMetrics(**cal["metrics"])

    return RoughCalibrateResponse(
        token=req.token,
        model=cal["model"],
        H=cal["H"],
        nu=cal.get("nu"),
        lambda_=cal.get("lambda_"),
        theta=cal.get("theta"),
        xi=cal.get("xi"),
        V0=cal["V0"],
        metrics=metrics,
        method=cal["method"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/rough/forecast",
    response_model=RoughForecastResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Forecast volatility using calibrated rough model",
)
def get_rough_forecast(
    token: str = Query(...),
    horizon: int = Query(10, ge=1, le=252),
    n_paths: int = Query(500, ge=100, le=5000),
):
    from cortex.rough_vol import rough_vol_forecast

    m = _get_model(token)
    if token not in _rough_store:
        raise HTTPException(status_code=404, detail=f"No rough calibration for '{token}'. POST /rough/calibrate first.")

    cal = _rough_store[token]
    result = rough_vol_forecast(m["returns"], cal, horizon=horizon, n_paths=n_paths, seed=42)

    return RoughForecastResponse(
        token=token,
        model=result["model"],
        horizon=result["horizon"],
        current_vol=result["current_vol"],
        point_forecast=result["point_forecast"],
        lower_95=result["lower_95"],
        upper_95=result["upper_95"],
        mean_forecast=result["mean_forecast"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/rough/diagnostics",
    response_model=RoughDiagnosticsResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Roughness diagnostics for token's return series",
)
def get_rough_diagnostics(
    token: str = Query(...),
    window: int = Query(5, ge=2, le=60),
    max_lag: int = Query(50, ge=10, le=200),
):
    from cortex.rough_vol import estimate_roughness

    m = _get_model(token)
    result = estimate_roughness(m["returns"], window=window, max_lag=max_lag)

    return RoughDiagnosticsResponse(
        token=token,
        H_variogram=result["H"],
        H_se=result["H_se"],
        r_squared=result["r_squared"],
        is_rough=result["is_rough"],
        lags=result["lags"],
        variogram=result["variogram"],
        interpretation=result["interpretation"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/rough/compare-msm",
    response_model=RoughCompareMSMResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Compare Rough Bergomi vs MSM volatility models",
)
def get_rough_compare_msm(token: str = Query(...)):
    from cortex.rough_vol import compare_rough_vs_msm

    m = _get_model(token)
    result = compare_rough_vs_msm(m["returns"], m["calibration"])

    rb = result["rough_bergomi"]
    ms = result["msm"]
    cm = result["comparison_metrics"]

    return RoughCompareMSMResponse(
        token=token,
        rough_H=rb["H"],
        rough_nu=rb["nu"],
        rough_is_rough=rb["is_rough"],
        rough_metrics=RoughModelMetrics(mae=rb["mae"], rmse=rb["rmse"], correlation=rb["correlation"]),
        msm_num_states=ms["num_states"],
        msm_metrics=RoughModelMetrics(mae=ms["mae"], rmse=ms["rmse"], correlation=ms["correlation"]),
        winner=result["winner"],
        mae_ratio=cm["mae_ratio"],
        rmse_ratio=cm["rmse_ratio"],
        corr_diff=cm["corr_diff"],
        timestamp=datetime.now(timezone.utc),
    )


# ── SVJ (Stochastic Volatility with Jumps) ──────────────────────────


@router.post("/svj/calibrate", response_model=SVJCalibrateResponse)
def svj_calibrate(req: SVJCalibrateRequest):
    from cortex.svj import calibrate_svj

    m = _get_model(req.token)
    cal = calibrate_svj(
        m["returns"],
        use_hawkes=req.use_hawkes,
        jump_threshold_multiplier=req.jump_threshold_multiplier,
    )

    _svj_store[req.token] = {"calibration": cal, "returns": m["returns"]}

    hp = None
    if cal.get("hawkes_params"):
        hp = SVJHawkesParams(**cal["hawkes_params"])

    return SVJCalibrateResponse(
        token=req.token,
        kappa=cal["kappa"],
        theta=cal["theta"],
        sigma=cal["sigma"],
        rho=cal["rho"],
        lambda_=cal["lambda_"],
        mu_j=cal["mu_j"],
        sigma_j=cal["sigma_j"],
        feller_ratio=cal["feller_ratio"],
        feller_satisfied=cal["feller_satisfied"],
        log_likelihood=cal.get("log_likelihood"),
        aic=cal.get("aic"),
        bic=cal.get("bic"),
        n_obs=cal["n_obs"],
        n_jumps_detected=cal["n_jumps_detected"],
        jump_fraction=cal["jump_fraction"],
        bns_statistic=cal["bns_statistic"],
        bns_pvalue=cal["bns_pvalue"],
        optimization_success=cal["optimization_success"],
        use_hawkes=cal["use_hawkes"],
        hawkes_params=hp,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/svj/var", response_model=SVJVaRResponse)
def svj_var_endpoint(
    token: str = Query(...),
    alpha: float = Query(0.05, ge=0.001, le=0.5),
    n_simulations: int = Query(50000, ge=1000, le=500000),
):
    from cortex.svj import svj_var

    if token not in _svj_store:
        raise HTTPException(status_code=404, detail=f"SVJ not calibrated for {token}. POST /svj/calibrate first.")

    s = _svj_store[token]
    result = svj_var(s["returns"], s["calibration"], alpha=alpha, n_simulations=n_simulations)

    return SVJVaRResponse(token=token, **result, timestamp=datetime.now(timezone.utc))


@router.get("/svj/jump-risk", response_model=SVJJumpRiskResponse)
def svj_jump_risk(token: str = Query(...)):
    from cortex.svj import decompose_risk

    if token not in _svj_store:
        raise HTTPException(status_code=404, detail=f"SVJ not calibrated for {token}. POST /svj/calibrate first.")

    s = _svj_store[token]
    result = decompose_risk(s["returns"], s["calibration"])

    return SVJJumpRiskResponse(token=token, **result, timestamp=datetime.now(timezone.utc))


@router.get("/svj/diagnostics", response_model=SVJDiagnosticsResponse)
def svj_diagnostics_endpoint(token: str = Query(...)):
    from cortex.svj import svj_diagnostics

    if token not in _svj_store:
        raise HTTPException(status_code=404, detail=f"SVJ not calibrated for {token}. POST /svj/calibrate first.")

    s = _svj_store[token]
    result = svj_diagnostics(s["returns"], s["calibration"])

    js = SVJJumpStats(**result["jump_stats"])
    pq = SVJParameterQuality(**result["parameter_quality"])
    mc = SVJMomentComparison(**result["moment_comparison"])
    et = SVJEVTTail(**result["evt_tail"]) if result.get("evt_tail") else None
    cl = SVJClustering(**result["clustering"]) if result.get("clustering") else None

    return SVJDiagnosticsResponse(
        token=token,
        jump_stats=js,
        parameter_quality=pq,
        moment_comparison=mc,
        evt_tail=et,
        clustering=cl,
        timestamp=datetime.now(timezone.utc),
    )


# ── Guardian (Unified Risk Veto) ─────────────────────────────────────


@router.post("/guardian/assess", response_model=GuardianAssessResponse)
def guardian_assess(req: GuardianAssessRequest):
    """Unified risk veto endpoint for Cortex autonomous trading agents."""
    from cortex.guardian import _cache as guardian_cache
    from cortex.guardian import assess_trade

    # Urgency flag bypasses cache
    if req.urgency:
        cache_key = f"{req.token}:{req.direction}"
        guardian_cache.pop(cache_key, None)

    model_data = _model_store.get(req.token)
    evt_data = _evt_store.get(req.token)
    svj_data = _svj_store.get(req.token)
    hawkes_data = _hawkes_store.get(req.token)

    if not any([model_data, evt_data, svj_data, hawkes_data]):
        raise HTTPException(
            status_code=404,
            detail=f"No calibrated models for '{req.token}'. "
            "Calibrate at least one model first (MSM, EVT, SVJ, or Hawkes).",
        )

    # Fetch news intelligence signal (best-effort, non-blocking)
    news_data = None
    try:
        from cortex.news import fetch_news_intelligence
        regime_state = _current_regime_state()
        news_result = fetch_news_intelligence(
            regime_state=regime_state, max_items=30, timeout=10.0,
        )
        news_data = news_result.get("signal")
    except Exception:
        logger.debug("News intelligence unavailable for guardian assessment", exc_info=True)

    result = assess_trade(
        token=req.token,
        trade_size_usd=req.trade_size_usd,
        direction=req.direction,
        model_data=model_data,
        evt_data=evt_data,
        svj_data=svj_data,
        hawkes_data=hawkes_data,
        news_data=news_data,
    )

    return GuardianAssessResponse(
        approved=result["approved"],
        risk_score=result["risk_score"],
        veto_reasons=result["veto_reasons"],
        recommended_size=result["recommended_size"],
        regime_state=result["regime_state"],
        confidence=result["confidence"],
        expires_at=result["expires_at"],
        component_scores=[
            GuardianComponentScore(**s) for s in result["component_scores"]
        ],
        from_cache=result["from_cache"],
    )