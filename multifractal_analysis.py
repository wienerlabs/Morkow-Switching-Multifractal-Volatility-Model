"""
Multifractal analysis for financial time series.

Implements Hurst exponent estimation (R/S and DFA), multifractal spectrum
f(α) via structure functions, and long-range dependence testing. Validates
the MSM model's multifractal properties against empirical data.

Mathematics:
  Hurst exponent H:
    E[R(n)/S(n)] ~ c·n^H  (R/S analysis)
    F(n) ~ n^H             (DFA)
    H = 0.5 → random walk, H > 0.5 → persistent, H < 0.5 → anti-persistent

  Multifractal spectrum:
    τ(q) = lim_{n→∞} log(S_q(n)) / log(n)  where S_q(n) = Σ|X_i(n)|^q
    α = dτ/dq  (singularity strength)
    f(α) = q·α - τ(q)  (Legendre transform → fractal dimension)
    Width of f(α) = degree of multifractality
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import linregress

logger = logging.getLogger(__name__)


def _rs_statistic(x: np.ndarray) -> float:
    """Rescaled range R/S for a single segment."""
    n = len(x)
    if n < 2:
        return np.nan
    mean = np.mean(x)
    y = np.cumsum(x - mean)
    R = np.max(y) - np.min(y)
    S = np.std(x, ddof=1)
    if S < 1e-15:
        return np.nan
    return R / S


def hurst_rs(
    returns: pd.Series | np.ndarray,
    min_window: int = 10,
    max_window: int | None = None,
    n_windows: int = 20,
) -> dict:
    """
    Estimate Hurst exponent via R/S (Rescaled Range) analysis.

    Computes E[R/S] at multiple window sizes n, then fits log(E[R/S]) = H·log(n) + c.

    Args:
        returns: Return series.
        min_window: Smallest window size.
        max_window: Largest window size (default: len/4).
        n_windows: Number of window sizes to evaluate.

    Returns:
        Dict with H, H_se, r_squared, window_sizes, rs_values, interpretation.
    """
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n < 50:
        raise ValueError(f"Need ≥50 observations for R/S analysis, got {n}")

    if max_window is None:
        max_window = n // 4

    max_window = min(max_window, n // 2)
    min_window = max(min_window, 4)

    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), n_windows).astype(int)
    )
    window_sizes = window_sizes[window_sizes >= 4]

    rs_means = []
    valid_sizes = []

    for w in window_sizes:
        n_segments = n // w
        if n_segments < 1:
            continue
        rs_vals = []
        for i in range(n_segments):
            segment = x[i * w : (i + 1) * w]
            rs = _rs_statistic(segment)
            if np.isfinite(rs):
                rs_vals.append(rs)
        if len(rs_vals) >= 1:
            rs_means.append(np.mean(rs_vals))
            valid_sizes.append(w)

    if len(valid_sizes) < 3:
        raise ValueError("Not enough valid window sizes for regression")

    log_n = np.log(np.array(valid_sizes, dtype=float))
    log_rs = np.log(np.array(rs_means, dtype=float))

    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_rs)

    if slope < 0.5:
        interp = "anti-persistent (mean-reverting)"
    elif slope < 0.55:
        interp = "random walk (no long-range dependence)"
    elif slope < 0.75:
        interp = "persistent (trending / long memory)"
    else:
        interp = "strongly persistent (strong long-range dependence)"

    return {
        "H": float(slope),
        "H_se": float(std_err),
        "r_squared": float(r_value ** 2),
        "intercept": float(intercept),
        "p_value": float(p_value),
        "window_sizes": [int(w) for w in valid_sizes],
        "rs_values": [float(v) for v in rs_means],
        "interpretation": interp,
        "method": "rs",
    }


def hurst_dfa(
    returns: pd.Series | np.ndarray,
    order: int = 1,
    min_window: int = 10,
    max_window: int | None = None,
    n_windows: int = 20,
) -> dict:
    """
    Estimate Hurst exponent via DFA (Detrended Fluctuation Analysis).

    More robust than R/S for non-stationary series. Removes polynomial
    trends of given order before computing fluctuation function.

    Args:
        returns: Return series.
        order: Polynomial detrending order (1=linear, 2=quadratic).
        min_window: Smallest window size.
        max_window: Largest window size (default: len/4).
        n_windows: Number of window sizes.

    Returns:
        Dict with H, H_se, r_squared, window_sizes, fluctuations, interpretation.
    """
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n < 50:
        raise ValueError(f"Need ≥50 observations for DFA, got {n}")

    # Cumulative sum (profile)
    profile = np.cumsum(x - np.mean(x))

    if max_window is None:
        max_window = n // 4
    max_window = min(max_window, n // 2)
    min_window = max(min_window, order + 2)

    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), n_windows).astype(int)
    )
    window_sizes = window_sizes[window_sizes >= order + 2]

    fluctuations = []
    valid_sizes = []

    for w in window_sizes:
        n_segments = n // w
        if n_segments < 1:
            continue
        rms_list = []
        for i in range(n_segments):
            segment = profile[i * w : (i + 1) * w]
            t = np.arange(w, dtype=float)
            coeffs = np.polyfit(t, segment, order)
            trend = np.polyval(coeffs, t)
            residual = segment - trend
            rms = np.sqrt(np.mean(residual ** 2))
            rms_list.append(rms)
        # Also walk backwards for better coverage
        for i in range(n_segments):
            segment = profile[n - (i + 1) * w : n - i * w]
            t = np.arange(w, dtype=float)
            coeffs = np.polyfit(t, segment, order)
            trend = np.polyval(coeffs, t)
            residual = segment - trend
            rms = np.sqrt(np.mean(residual ** 2))
            rms_list.append(rms)

        if rms_list:
            F_n = np.sqrt(np.mean(np.array(rms_list) ** 2))
            if F_n > 1e-15:
                fluctuations.append(F_n)
                valid_sizes.append(w)

    if len(valid_sizes) < 3:
        raise ValueError("Not enough valid window sizes for DFA regression")

    log_n = np.log(np.array(valid_sizes, dtype=float))
    log_f = np.log(np.array(fluctuations, dtype=float))

    slope, intercept, r_value, p_value, std_err = linregress(log_n, log_f)

    if slope < 0.5:
        interp = "anti-persistent (mean-reverting)"
    elif slope < 0.55:
        interp = "random walk (no long-range dependence)"
    elif slope < 0.75:
        interp = "persistent (trending / long memory)"
    else:
        interp = "strongly persistent (strong long-range dependence)"

    return {
        "H": float(slope),
        "H_se": float(std_err),
        "r_squared": float(r_value ** 2),
        "intercept": float(intercept),
        "p_value": float(p_value),
        "order": order,
        "window_sizes": [int(w) for w in valid_sizes],
        "fluctuations": [float(f) for f in fluctuations],
        "interpretation": interp,
        "method": "dfa",
    }
        "intercept": float(intercept),
        "p_value": float(p_value),
        "window_sizes": [int(w) for w in valid_sizes],
        "rs_values": [float(v) for v in rs_means],
        "interpretation": interp,
        "method": "rs",
    }

