"""Regime analytics endpoints: durations, history, transitions, statistics."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query

from api.models import (
    RegimeDurationsResponse,
    RegimeHistoryResponse,
    RegimeParamsResponse,
    RegimePeriod,
    RegimeStatisticsResponse,
    RegimeStatRow,
    TransitionAlertResponse,
)
from api.stores import _get_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["regime"])


@router.get("/regime/durations", response_model=RegimeDurationsResponse)
def get_regime_durations(token: str = Query(...)):
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


@router.get("/regime/params", response_model=RegimeParamsResponse)
def get_regime_params(token: str = Query("SOL")):
    import numpy as np

    m = _get_model(token)
    cal = m["calibration"]
    sigma_states = np.asarray(m["sigma_states"])
    P_matrix = np.asarray(m["P_matrix"])
    num_states = cal["num_states"]
    leverage_gamma = m.get("leverage_gamma", 0.0)

    # Build approximate emission params from sigma_states.
    # Frontend HMM uses 3 features: [daily_return, volatility, volume_ratio].
    # The backend MSM has volatility levels directly — derive approximate params.
    emission_mean = []
    emission_std = []
    for k in range(num_states):
        sigma = float(sigma_states[k])
        # Leverage effect: low-vol states -> slight positive drift,
        # high-vol states -> negative drift
        frac = k / max(num_states - 1, 1)  # 0..1 from low to high vol
        ret_mean = 0.005 * (1 - 2 * frac)  # +0.005 to -0.005
        vol_ratio = 0.9 + 0.2 * frac  # 0.9 to 1.1

        emission_mean.append([ret_mean, sigma, vol_ratio])
        emission_std.append([
            sigma * 0.8,
            sigma * 0.3,
            0.3 + 0.05 * k,
        ])

    return RegimeParamsResponse(
        token=token,
        num_states=num_states,
        transition_matrix=P_matrix.tolist(),
        sigma_states=sigma_states.tolist(),
        emission_params={
            "mean": emission_mean,
            "std": emission_std,
        },
        calibration={
            "sigma_low": cal["sigma_low"],
            "sigma_high": cal["sigma_high"],
            "p_stay": cal["p_stay"],
            "leverage_gamma": leverage_gamma,
        },
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/regime/statistics", response_model=RegimeStatisticsResponse)
def get_regime_statistics(token: str = Query(...)):
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

