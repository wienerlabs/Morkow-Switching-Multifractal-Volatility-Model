import asyncio
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

# Add project root to path so we can import the MSM model
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.models import (
    BacktestSummaryResponse,
    CalibrateRequest,
    CalibrateResponse,
    CalibrationMetrics,
    CompareRequest,
    CompareResponse,
    ComparisonReportResponse,
    ErrorResponse,
    ModelMetricsRow,
    NewsFeedResponse,
    NewsMarketSignalModel,
    RegimeDurationsResponse,
    RegimeHistoryResponse,
    RegimePeriod,
    RegimeResponse,
    RegimeStatisticsResponse,
    RegimeStatRow,
    RegimeStreamMessage,
    TailProbResponse,
    TransitionAlertResponse,
    VaRResponse,
    VolatilityForecastResponse,
    get_regime_name,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory model state (per-token)
_model_store: dict[str, dict] = {}


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
        from solana_data_adapter import get_token_ohlcv, ohlcv_to_returns

        df = get_token_ohlcv(req.token, req.start_date, req.end_date, req.interval)
        return ohlcv_to_returns(df)

    import yfinance as yf

    df = yf.download(req.token, start=req.start_date, end=req.end_date, progress=False)
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No yfinance data for '{req.token}'")
    close = df["Close"].dropna()
    rets = 100 * np.diff(np.log(close.values))
    return pd.Series(rets, index=close.index[1:], name="r")


@router.post("/calibrate", response_model=CalibrateResponse)
def calibrate(req: CalibrateRequest):
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")

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
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")
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
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")
    m = _get_model(token)

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
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")
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
    from importlib import import_module
    from scipy.stats import norm

    msm = import_module("MSM-VaR_MODEL")
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
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")
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
    from importlib import import_module

    msm = import_module("MSM-VaR_MODEL")
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
    from news_intelligence import fetch_news_intelligence

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
    from news_intelligence import fetch_news_intelligence

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
    from news_intelligence import fetch_news_intelligence

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
    from regime_analytics import compute_expected_durations

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
    from regime_analytics import extract_regime_history

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
    from regime_analytics import detect_regime_transition

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
    from regime_analytics import compute_regime_statistics

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
    from model_comparison import compare_models, _MODEL_REGISTRY

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
    from model_comparison import generate_comparison_report

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