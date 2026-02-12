"""Rough volatility endpoints: calibrate, forecast, diagnostics, compare-msm."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.models import (
    ErrorResponse,
    RoughCalibrateRequest,
    RoughCalibrateResponse,
    RoughCalibrationMetrics,
    RoughCompareMSMResponse,
    RoughDiagnosticsResponse,
    RoughForecastResponse,
    RoughModelMetrics,
)
from api.stores import _get_model, _rough_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rough"])


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

