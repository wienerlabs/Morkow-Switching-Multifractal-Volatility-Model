"""SVJ (Stochastic Volatility with Jumps) endpoints."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.models import (
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
)
from api.stores import _get_model, _svj_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["svj"])


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

