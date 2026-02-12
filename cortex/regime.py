"""
Regime analytics for the MSM-VaR model.

Provides temporal analysis, historical regime tracking, transition detection,
and conditional statistics on top of the Markov Switching Multifractal model.
"""

import math

import numpy as np
import pandas as pd


def compute_expected_durations(
    p_stay, num_states: int
) -> dict[int, float]:
    """
    Expected regime duration: E[Duration_k] = 1 / (1 - p_stay_k).

    p_stay: scalar (same for all regimes) or list/array of length K.
    """
    if num_states < 2:
        raise ValueError(f"num_states must be >= 2, got {num_states}")

    if isinstance(p_stay, (list, np.ndarray)):
        arr = np.asarray(p_stay, dtype=float)
        if arr.size == 1:
            arr = np.full(num_states, arr.flat[0])
    else:
        arr = np.full(num_states, float(p_stay))

    if np.any(arr <= 0) or np.any(arr >= 1):
        raise ValueError(f"All p_stay values must be in (0, 1), got {arr}")

    return {
        k: round(1.0 / (1.0 - arr[k - 1]), 2)
        for k in range(1, num_states + 1)
    }


def _max_drawdown_pct(returns: np.ndarray) -> float:
    """Peak-to-trough max drawdown from a return series (%)."""
    if len(returns) == 0:
        return 0.0
    cum = np.cumsum(returns)
    running_max = np.maximum.accumulate(cum)
    drawdowns = cum - running_max
    return float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0


def extract_regime_history(
    filter_probs: pd.DataFrame,
    returns: pd.Series,
    sigma_states: np.ndarray,
) -> pd.DataFrame:
    """
    Build a timeline of consecutive regime periods with per-period statistics.

    Args:
        filter_probs: DataFrame (T × K) of filtered regime probabilities.
                      Columns named ``state_1 .. state_K``, DatetimeIndex.
        returns: Series of daily log-returns in %, same DatetimeIndex.
        sigma_states: Array (K,) of volatility levels per regime.

    Returns:
        DataFrame with columns ``[start, end, regime, duration,
        cumulative_return, volatility, max_drawdown]`` sorted by start date.
    """
    probs = filter_probs.values
    regimes = np.argmax(probs, axis=1) + 1  # 1-based
    dates = filter_probs.index
    rets = returns.reindex(dates).fillna(0.0).values

    periods: list[dict] = []
    seg_start = 0

    for t in range(1, len(regimes)):
        if regimes[t] != regimes[seg_start]:
            _append_period(periods, dates, regimes, rets, sigma_states,
                           seg_start, t - 1)
            seg_start = t

    # last segment
    _append_period(periods, dates, regimes, rets, sigma_states,
                   seg_start, len(regimes) - 1)

    if not periods:
        return pd.DataFrame(
            columns=["start", "end", "regime", "duration",
                     "cumulative_return", "volatility", "max_drawdown"]
        )

    return pd.DataFrame(periods)


def _append_period(
    periods: list[dict],
    dates: pd.DatetimeIndex,
    regimes: np.ndarray,
    rets: np.ndarray,
    sigma_states: np.ndarray,
    i_start: int,
    i_end: int,
) -> None:
    """Append one regime period to the periods list."""
    regime = int(regimes[i_start])
    seg_rets = rets[i_start: i_end + 1]
    periods.append({
        "start": dates[i_start],
        "end": dates[i_end],
        "regime": regime,
        "duration": i_end - i_start + 1,
        "cumulative_return": round(float(np.sum(seg_rets)), 4),
        "volatility": round(float(sigma_states[regime - 1]), 4),
        "max_drawdown": round(_max_drawdown_pct(seg_rets), 4),
    })


def detect_regime_transition(
    filter_probs: pd.DataFrame,
    threshold: float = 0.3,
) -> dict:
    """
    Alert system for imminent regime transitions.

    Checks whether the probability of leaving the current regime exceeds
    *threshold* and identifies the most likely destination state.

    Args:
        filter_probs: DataFrame (T × K) of filtered regime probabilities.
        threshold: Probability above which a transition alert fires
                   (default 0.3 = 30%).

    Returns:
        Dict with keys ``alert``, ``current_regime``,
        ``transition_probability``, ``most_likely_next_regime``,
        ``next_regime_probability``, ``threshold``.
    """
    if filter_probs.empty:
        return {
            "alert": False,
            "current_regime": 0,
            "transition_probability": 0.0,
            "most_likely_next_regime": 0,
            "next_regime_probability": 0.0,
            "threshold": threshold,
        }

    last_probs = np.asarray(filter_probs.iloc[-1], dtype=float)
    current_idx = int(np.argmax(last_probs))          # 0-based
    current_regime = current_idx + 1                   # 1-based
    p_stay_now = float(last_probs[current_idx])
    p_transition = 1.0 - p_stay_now

    # Most likely *other* regime
    other_probs = last_probs.copy()
    other_probs[current_idx] = -1.0                    # exclude current
    next_idx = int(np.argmax(other_probs))
    next_regime = next_idx + 1
    next_prob = float(last_probs[next_idx])

    return {
        "alert": bool(p_transition > threshold),
        "current_regime": current_regime,
        "transition_probability": round(p_transition, 4),
        "most_likely_next_regime": next_regime,
        "next_regime_probability": round(next_prob, 4),
        "threshold": threshold,
    }


def compute_regime_statistics(
    returns: pd.Series,
    filter_probs: pd.DataFrame,
    sigma_states: np.ndarray,
) -> pd.DataFrame:
    """
    Conditional statistics grouped by the most-probable regime.

    For each regime k ∈ {1 .. K} computes mean return, volatility,
    annualised Sharpe ratio, max drawdown, day count, and frequency.

    Args:
        returns: Series of daily log-returns in %.
        filter_probs: DataFrame (T × K) of filtered regime probabilities.
        sigma_states: Array (K,) of volatility levels per regime.

    Returns:
        DataFrame indexed by regime (1-based) with columns
        ``[regime, mean_return, volatility, sharpe_ratio, max_drawdown,
        days_in_regime, frequency]``.
    """
    probs = filter_probs.values
    regimes = np.argmax(probs, axis=1) + 1  # 1-based
    rets = returns.reindex(filter_probs.index).fillna(0.0).values
    K = len(sigma_states)
    total_days = len(regimes)

    rows: list[dict] = []
    for k in range(1, K + 1):
        mask = regimes == k
        days = int(mask.sum())
        if days == 0:
            rows.append({
                "regime": k,
                "mean_return": 0.0,
                "volatility": round(float(sigma_states[k - 1]), 4),
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "days_in_regime": 0,
                "frequency": 0.0,
            })
            continue

        seg = rets[mask]
        mean_r = float(np.mean(seg))
        vol = float(sigma_states[k - 1])
        sharpe = (mean_r / vol * math.sqrt(252)) if vol > 1e-12 else 0.0
        mdd = _max_drawdown_pct(seg)

        rows.append({
            "regime": k,
            "mean_return": round(mean_r, 4),
            "volatility": round(vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "max_drawdown": round(mdd, 4),
            "days_in_regime": days,
            "frequency": round(days / total_days, 4),
        })

    df = pd.DataFrame(rows)
    df.index = df["regime"]
    return df
