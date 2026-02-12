"""Hawkes process endpoints: calibrate, intensity, clusters, VaR, simulate."""

import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from api.models import (
    HawkesCalibrateRequest,
    HawkesCalibrateResponse,
    HawkesClusterItem,
    HawkesClustersResponse,
    HawkesIntensityResponse,
    HawkesSimulateRequest,
    HawkesSimulateResponse,
    HawkesVaRRequest,
    HawkesVaRResponse,
)
from api.stores import _get_model, _hawkes_store, _model_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["hawkes"])


@router.post("/hawkes/calibrate", response_model=HawkesCalibrateResponse)
async def hawkes_calibrate(req: HawkesCalibrateRequest):
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
    from cortex.hawkes import detect_flash_crash_risk, hawkes_intensity, hawkes_var_adjustment

    m = _get_model(req.token)
    if req.token not in _hawkes_store:
        raise HTTPException(404, f"No Hawkes calibration for '{req.token}'. Call POST /hawkes/calibrate first.")

    h = _hawkes_store[req.token]

    from cortex import msm
    alpha = 1.0 - req.confidence / 100.0
    st = m.get("use_student_t", False)
    df = m.get("nu", 5.0)

    var_t1, sigma_t1, z_alpha, pi_t1 = msm.msm_var_forecast_next_day(
        m["filter_probs"], m["sigma_states"], m["P_matrix"],
        alpha=alpha, use_student_t=st, nu=df,
        leverage_gamma=m.get("leverage_gamma", 0.0),
        last_return=float(m["returns"].iloc[-1]),
        p_stay=m["calibration"]["p_stay"],
    )

    intens = hawkes_intensity(h["event_times"], h)
    adj = hawkes_var_adjustment(
        var_t1, intens["current_intensity"], intens["baseline"],
        max_multiplier=req.max_multiplier,
    )

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
