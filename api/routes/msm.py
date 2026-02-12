"""MSM calibration, VaR, forecast, backtest, tail-probs, and streaming endpoints."""

import asyncio
import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

from api.models import (
    BacktestSummaryResponse,
    CalibrateRequest,
    CalibrateResponse,
    CalibrationMetrics,
    RegimeResponse,
    RegimeStreamMessage,
    TailProbResponse,
    VaRResponse,
    VolatilityForecastResponse,
    get_regime_name,
)
from api.stores import _get_model, _load_returns, _model_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["msm"])


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
        leverage_gamma=req.leverage_gamma,
    )

    leverage_gamma_val = cal.get("leverage_gamma", 0.0)

    sigma_f, sigma_filt, fprobs, sigma_states, P = msm.msm_vol_forecast(
        returns,
        num_states=cal["num_states"],
        sigma_low=cal["sigma_low"],
        sigma_high=cal["sigma_high"],
        p_stay=cal["p_stay"],
        leverage_gamma=leverage_gamma_val,
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
        "leverage_gamma": leverage_gamma_val,
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
        leverage_gamma=leverage_gamma_val,
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
        m["filter_probs"], m["sigma_states"], m["P_matrix"], alpha=0.05,
        leverage_gamma=m.get("leverage_gamma", 0.0),
        last_return=float(m["returns"].iloc[-1]),
        p_stay=m["calibration"]["p_stay"],
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
        leverage_gamma=m.get("leverage_gamma", 0.0),
        last_return=float(m["returns"].iloc[-1]),
        p_stay=m["calibration"]["p_stay"],
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
        m["filter_probs"], m["sigma_states"], m["P_matrix"],
        leverage_gamma=m.get("leverage_gamma", 0.0),
        last_return=float(m["returns"].iloc[-1]),
        p_stay=m["calibration"]["p_stay"],
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
                m["filter_probs"], m["sigma_states"], m["P_matrix"], alpha=0.05,
                leverage_gamma=m.get("leverage_gamma", 0.0),
                last_return=float(m["returns"].iloc[-1]),
                p_stay=m["calibration"]["p_stay"],
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

