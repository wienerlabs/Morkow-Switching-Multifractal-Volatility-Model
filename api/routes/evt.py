"""EVT (Extreme Value Theory) endpoints: calibrate, VaR, diagnostics."""

import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from api.models import (
    EVTBacktestRow,
    EVTCalibrateRequest,
    EVTCalibrateResponse,
    EVTDiagnosticsResponse,
    EVTVaRResponse,
    VaRComparisonRow,
)
from api.stores import _evt_store, _get_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evt"])


@router.post("/evt/calibrate", response_model=EVTCalibrateResponse)
def evt_calibrate(req: EVTCalibrateRequest):
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

