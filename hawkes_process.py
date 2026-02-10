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
from typing import Optional

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
        except Exception:
            continue

    if best_result is None:
        raise RuntimeError("Hawkes MLE optimization failed for all starting points")

    mu, alpha, beta = best_result.x
    branching_ratio = alpha / beta
    ll = -best_nll
    n_params = 3

