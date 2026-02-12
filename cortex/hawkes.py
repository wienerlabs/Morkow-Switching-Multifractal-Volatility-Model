"""
Hawkes self-exciting point process for volatility clustering and flash crash contagion.

Models how extreme market events cluster in time — one crash increases the
probability of subsequent crashes. Integrates with MSM-VaR to provide
intensity-adjusted risk measures.

Mathematics:
  λ(t) = μ + Σ_{t_i < t} α·exp(-β·(t - t_i))

  μ = baseline intensity (background event rate)
  α = excitation magnitude (jump in intensity per event)
  β = decay rate (how fast excitation fades)
  α/β = branching ratio (must be < 1 for stationarity)

  Log-likelihood:
    ℓ = Σ_i log λ(t_i) - ∫_0^T λ(t) dt
      = Σ_i log λ(t_i) - μT - (α/β) Σ_i [1 - exp(-β(T - t_i))]
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def _compute_intensity(
    events: np.ndarray, params: tuple[float, float, float], t_eval: np.ndarray
) -> np.ndarray:
    """Compute Hawkes intensity λ(t) at evaluation points."""
    mu, alpha, beta = params
    intensity = np.full_like(t_eval, mu, dtype=float)
    for t_i in events:
        mask = t_eval > t_i
        intensity[mask] += alpha * np.exp(-beta * (t_eval[mask] - t_i))
    return intensity


def _log_likelihood(params: np.ndarray, events: np.ndarray, T: float) -> float:
    """Negative log-likelihood for Hawkes process (for minimization)."""
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0 or alpha / beta >= 1.0:
        return 1e15

    n = len(events)
    if n == 0:
        return mu * T

    # Recursive computation of intensity at each event time
    # A_i = Σ_{j<i} exp(-β(t_i - t_j)) = exp(-β(t_i - t_{i-1})) * (1 + A_{i-1})
    A = np.zeros(n)
    for i in range(1, n):
        A[i] = np.exp(-beta * (events[i] - events[i - 1])) * (1 + A[i - 1])

    lambdas = mu + alpha * A
    lambdas = np.maximum(lambdas, 1e-15)

    # ℓ = Σ log λ(t_i) - μT - (α/β) Σ [1 - exp(-β(T - t_i))]
    ll = np.sum(np.log(lambdas))
    ll -= mu * T
    ll -= (alpha / beta) * np.sum(1 - np.exp(-beta * (T - events)))

    return -ll  # negative for minimization


def extract_events(
    returns: pd.Series | np.ndarray,
    threshold_percentile: float = 5.0,
    use_absolute: bool = True,
) -> dict:
    """
    Extract extreme events from return series for Hawkes process fitting.

    Args:
        returns: Return series (percentage).
        threshold_percentile: Percentile for event detection (default 5% = large losses).
        use_absolute: If True, use |returns| > threshold (both tails).
                      If False, use returns < -threshold (left tail only).

    Returns:
        Dict with event_times (normalized to [0, T]), event_returns,
        threshold, n_events, T, dates (if available).
    """
    if isinstance(returns, pd.Series):
        values = returns.values.astype(float)
        dates = returns.index
    else:
        values = np.asarray(returns, dtype=float)
        dates = None

    n = len(values)
    T = float(n)

    if use_absolute:
        threshold = float(np.percentile(np.abs(values), 100 - threshold_percentile))
        mask = np.abs(values) > threshold
    else:
        threshold = float(np.percentile(values, threshold_percentile))
        mask = values < threshold

    event_indices = np.where(mask)[0]
    event_times = event_indices.astype(float)
    event_returns = values[event_indices]

    return {
        "event_times": event_times,
        "event_returns": event_returns,
        "event_indices": event_indices,
        "threshold": threshold,
        "n_events": len(event_indices),
        "T": T,
        "dates": dates[event_indices].tolist() if dates is not None else None,
    }


def fit_hawkes(
    events: np.ndarray,
    T: float,
    method: str = "mle",
) -> dict:
    """
    Fit Hawkes process parameters via MLE.

    Args:
        events: 1D array of event times in [0, T].
        T: Total observation window length.
        method: Estimation method ('mle').

    Returns:
        Dict with mu, alpha, beta, branching_ratio, log_likelihood,
        aic, bic, n_events, T, half_life, stationarity.
    """
    n = len(events)
    if n < 5:
        raise ValueError(f"Need ≥5 events for Hawkes fitting, got {n}")

    events = np.sort(events)

    # Initial guesses: mu ~ n/T * 0.5, alpha ~ 0.3, beta ~ 1.0
    mu0 = n / T * 0.5
    alpha0 = 0.3
    beta0 = 1.0

    best_result = None
    best_nll = float("inf")

    starts = [
        [mu0, alpha0, beta0],
        [mu0 * 0.3, 0.5, 2.0],
        [mu0 * 1.5, 0.1, 0.5],
        [mu0, 0.7, 3.0],
    ]

    for x0 in starts:
        try:
            result = minimize(
                _log_likelihood,
                x0=x0,
                args=(events, T),
                method="L-BFGS-B",
                bounds=[(1e-6, None), (1e-6, None), (1e-6, None)],
                options={"maxiter": 500},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception as e:
            logger.debug("Hawkes MLE attempt failed: %s", e)
            continue

    if best_result is None:
        raise RuntimeError("Hawkes MLE optimization failed for all starting points")

    mu, alpha, beta = best_result.x
    branching_ratio = alpha / beta
    ll = -best_nll
    n_params = 3

    return {
        "mu": float(mu),
        "alpha": float(alpha),
        "beta": float(beta),
        "branching_ratio": float(branching_ratio),
        "log_likelihood": float(ll),
        "aic": float(2 * n_params - 2 * ll),
        "bic": float(n_params * np.log(max(n, 1)) - 2 * ll),
        "n_events": n,
        "T": T,
        "half_life": float(np.log(2) / beta),
        "stationary": bool(branching_ratio < 1.0),
    }


def hawkes_intensity(
    events: np.ndarray,
    params: dict,
    t_eval: np.ndarray | None = None,
    T: float | None = None,
    n_points: int = 500,
) -> dict:
    """
    Compute Hawkes intensity function λ(t) over time.

    Args:
        events: Event times array.
        params: Output of fit_hawkes() (needs mu, alpha, beta).
        t_eval: Specific times to evaluate. If None, uses linspace(0, T, n_points).
        T: End of observation window (required if t_eval is None).
        n_points: Number of evaluation points if t_eval is None.

    Returns:
        Dict with t_eval, intensity, current_intensity, baseline,
        intensity_ratio, peak_intensity, mean_intensity.
    """
    mu, alpha, beta = params["mu"], params["alpha"], params["beta"]

    if t_eval is None:
        if T is None:
            T = params.get("T", float(events[-1]) + 1 if len(events) > 0 else 1.0)
        t_eval = np.linspace(0, T, n_points)

    intensity = _compute_intensity(events, (mu, alpha, beta), t_eval)
    current = float(intensity[-1]) if len(intensity) > 0 else mu

    return {
        "t_eval": t_eval.tolist(),
        "intensity": intensity.tolist(),
        "current_intensity": current,
        "baseline": mu,
        "intensity_ratio": current / mu if mu > 1e-12 else 1.0,
        "peak_intensity": float(np.max(intensity)),
        "mean_intensity": float(np.mean(intensity)),
    }


def hawkes_var_adjustment(
    base_var: float,
    current_intensity: float,
    baseline_intensity: float,
    max_multiplier: float = 3.0,
) -> dict:
    """
    Adjust VaR using Hawkes intensity ratio.

    When intensity is elevated (post-crash clustering), VaR should be wider.
    Multiplier = min(λ_current / λ_baseline, max_multiplier).

    Args:
        base_var: Base VaR from MSM model (negative number).
        current_intensity: Current Hawkes intensity λ(t_now).
        baseline_intensity: Baseline intensity μ.
        max_multiplier: Cap on the adjustment multiplier.

    Returns:
        Dict with adjusted_var, base_var, multiplier, intensity_ratio.
    """
    if baseline_intensity <= 1e-12:
        ratio = 1.0
    else:
        ratio = current_intensity / baseline_intensity

    multiplier = min(ratio, max_multiplier)
    # VaR is negative, so multiplying by >1 makes it more negative (wider)
    adjusted_var = base_var * multiplier

    return {
        "adjusted_var": float(adjusted_var),
        "base_var": float(base_var),
        "multiplier": float(multiplier),
        "intensity_ratio": float(ratio),
        "capped": bool(ratio > max_multiplier),
    }


def detect_clusters(
    events: np.ndarray,
    params: dict,
    gap_threshold: float | None = None,
) -> list[dict]:
    """
    Identify temporal clusters of extreme events.

    Events separated by less than gap_threshold are grouped into the same cluster.
    Default gap = 2 × half_life (events within two decay periods are related).

    Args:
        events: Sorted event times array.
        params: Output of fit_hawkes() (needs beta for half_life).
        gap_threshold: Max gap between events in same cluster.
                       If None, uses 2 × half_life.

    Returns:
        List of cluster dicts with start_time, end_time, n_events,
        duration, peak_intensity.
    """
    if len(events) == 0:
        return []

    events = np.sort(events)

    if gap_threshold is None:
        beta = params["beta"]
        gap_threshold = 2.0 * np.log(2) / beta

    clusters: list[dict] = []
    cluster_start = events[0]
    cluster_events = [events[0]]

    for i in range(1, len(events)):
        if events[i] - events[i - 1] <= gap_threshold:
            cluster_events.append(events[i])
        else:
            if len(cluster_events) >= 2:
                t_eval = np.array(cluster_events)
                intensity = _compute_intensity(
                    t_eval, (params["mu"], params["alpha"], params["beta"]), t_eval
                )
                clusters.append({
                    "cluster_id": len(clusters),
                    "start_time": float(cluster_events[0]),
                    "end_time": float(cluster_events[-1]),
                    "n_events": len(cluster_events),
                    "duration": float(cluster_events[-1] - cluster_events[0]),
                    "peak_intensity": float(np.max(intensity)),
                })
            cluster_start = events[i]
            cluster_events = [events[i]]

    # Handle last cluster
    if len(cluster_events) >= 2:
        t_eval = np.array(cluster_events)
        intensity = _compute_intensity(
            t_eval, (params["mu"], params["alpha"], params["beta"]), t_eval
        )
        clusters.append({
            "cluster_id": len(clusters),
            "start_time": float(cluster_events[0]),
            "end_time": float(cluster_events[-1]),
            "n_events": len(cluster_events),
            "duration": float(cluster_events[-1] - cluster_events[0]),
            "peak_intensity": float(np.max(intensity)),
        })

    return clusters


def simulate_hawkes(
    params: dict,
    T: float,
    seed: int = 42,
) -> dict:
    """
    Simulate Hawkes process via Ogata's thinning algorithm.

    Args:
        params: Dict with mu, alpha, beta.
        T: Simulation horizon.
        seed: Random seed for reproducibility.

    Returns:
        Dict with event_times, n_events, intensity_path (sampled).
    """
    rng = np.random.default_rng(seed)
    mu, alpha, beta = params["mu"], params["alpha"], params["beta"]

    event_times: list[float] = []
    t = 0.0

    # Upper bound on intensity starts at mu
    lambda_bar = mu

    while t < T:
        # Draw next candidate time from exponential with rate lambda_bar
        u = rng.uniform()
        dt = -np.log(u) / lambda_bar
        t += dt

        if t >= T:
            break

        # Compute actual intensity at candidate time
        lambda_t = mu
        for t_i in event_times:
            lambda_t += alpha * np.exp(-beta * (t - t_i))

        # Accept/reject
        if rng.uniform() <= lambda_t / lambda_bar:
            event_times.append(t)
            # Update upper bound: intensity just jumped by alpha
            lambda_bar = lambda_t + alpha
        else:
            lambda_bar = lambda_t

    events_arr = np.array(event_times) if event_times else np.array([])

    # Sample intensity at regular points for visualization
    n_sample = min(500, max(100, int(T)))
    t_sample = np.linspace(0, T, n_sample)
    intensity_path = _compute_intensity(events_arr, (mu, alpha, beta), t_sample)

    return {
        "event_times": events_arr,
        "n_events": len(event_times),
        "T": T,
        "intensity_t": t_sample.tolist(),
        "intensity_path": intensity_path.tolist(),
    }


def detect_flash_crash_risk(
    events: np.ndarray,
    params: dict,
    t_now: float | None = None,
    lookback_window: float | None = None,
) -> dict:
    """
    Real-time contagion risk score based on current Hawkes intensity.

    Maps the intensity ratio λ(t)/μ to a [0, 1] risk score and categorical
    risk level. Higher scores indicate elevated flash crash clustering risk.

    Args:
        events: Event times array.
        params: Output of fit_hawkes() (needs mu, alpha, beta).
        t_now: Current time. If None, uses max(events) or T.
        lookback_window: Window for counting recent events. Default: 5 × half_life.

    Returns:
        Dict with contagion_risk_score (0-1), current_intensity, baseline,
        excitation_level, intensity_ratio, recent_event_count,
        risk_level (low/medium/high/critical).
    """
    mu = params["mu"]
    alpha = params["alpha"]
    beta = params["beta"]
    half_life = np.log(2) / beta

    if t_now is None:
        t_now = float(events[-1]) if len(events) > 0 else params.get("T", 1.0)

    if lookback_window is None:
        lookback_window = 5.0 * half_life

    # Current intensity at t_now
    t_eval = np.array([t_now])
    intensity = _compute_intensity(events, (mu, alpha, beta), t_eval)
    current_intensity = float(intensity[0])

    # Excitation = how much above baseline
    excitation_level = current_intensity - mu

    # Intensity ratio
    intensity_ratio = current_intensity / mu if mu > 1e-12 else 1.0

    # Risk score: map intensity_ratio to [0, 1]
    # At baseline (ratio=1) → score=0, at critical (ratio=max_ratio) → score=1
    # max_ratio corresponds to when branching ratio approaches 1
    branching_ratio = alpha / beta
    max_ratio = max(1.0 / (1.0 - branching_ratio), 5.0) if branching_ratio < 1.0 else 5.0
    score = min(1.0, max(0.0, (intensity_ratio - 1.0) / (max_ratio - 1.0)))

    # Categorical risk level
    if score < 0.25:
        risk_level = "low"
    elif score < 0.50:
        risk_level = "medium"
    elif score < 0.75:
        risk_level = "high"
    else:
        risk_level = "critical"

    # Count recent events in lookback window
    window_start = t_now - lookback_window
    recent_event_count = int(np.sum(events >= window_start)) if len(events) > 0 else 0

    return {
        "contagion_risk_score": float(score),
        "current_intensity": current_intensity,
        "baseline": mu,
        "excitation_level": float(excitation_level),
        "intensity_ratio": float(intensity_ratio),
        "recent_event_count": recent_event_count,
        "risk_level": risk_level,
    }