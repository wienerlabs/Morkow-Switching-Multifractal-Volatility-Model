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


def multifractal_spectrum(
    returns: pd.Series | np.ndarray,
    q_range: tuple[float, float] | None = None,
    n_q: int = 41,
    min_window: int = 10,
    max_window: int | None = None,
    n_windows: int = 20,
) -> dict:
    """
    Compute multifractal spectrum f(α) via structure functions.

    For each moment order q, computes scaling exponent τ(q) from
    S_q(n) = Σ|X_i(n)|^q ~ n^τ(q), then applies Legendre transform.

    Args:
        returns: Return series.
        q_range: (q_min, q_max) range of moment orders.
        n_q: Number of q values.
        min_window: Smallest window size.
        max_window: Largest window size.
        n_windows: Number of window sizes.

    Returns:
        Dict with q_values, tau_q, H_q, alpha, f_alpha, width, peak_alpha.
    """
    x = np.asarray(returns, dtype=float)
    n = len(x)
    if n < 50:
        raise ValueError(f"Need ≥50 observations for spectrum, got {n}")

    if q_range is None:
        q_range = (-5.0, 5.0)

    q_values = np.linspace(q_range[0], q_range[1], n_q)
    # Remove q=0 (undefined for structure function)
    q_values = q_values[np.abs(q_values) > 0.05]

    if max_window is None:
        max_window = n // 4
    max_window = min(max_window, n // 2)
    min_window = max(min_window, 4)

    window_sizes = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), n_windows).astype(int)
    )
    window_sizes = window_sizes[window_sizes >= 4]

    # Compute structure functions S_q(n) for each q and window size
    tau_q = []
    valid_q = []

    for q in q_values:
        log_sq = []
        log_n_vals = []
        for w in window_sizes:
            n_segments = n // w
            if n_segments < 2:
                continue
            # Segment sums
            seg_sums = []
            for i in range(n_segments):
                seg = x[i * w : (i + 1) * w]
                seg_abs = np.abs(np.sum(seg))
                if seg_abs > 1e-15:
                    seg_sums.append(seg_abs)
            if len(seg_sums) < 2:
                continue
            seg_arr = np.array(seg_sums)
            s_q = np.mean(seg_arr ** q)
            if s_q > 0 and np.isfinite(s_q):
                log_sq.append(np.log(s_q))
                log_n_vals.append(np.log(w))

        if len(log_sq) >= 3:
            slope, _, _, _, _ = linregress(log_n_vals, log_sq)
            tau_q.append(slope)
            valid_q.append(q)

    if len(valid_q) < 5:
        raise ValueError("Not enough valid q values for spectrum computation")

    q_arr = np.array(valid_q)
    tau_arr = np.array(tau_q)

    # Generalized Hurst exponent: H(q) = (τ(q)/q + 1) for q ≠ 0
    H_q = (tau_arr / q_arr + 1.0)

    # Legendre transform: α = dτ/dq, f(α) = q·α - τ(q)
    # Numerical derivative
    alpha = np.gradient(tau_arr, q_arr)
    f_alpha = q_arr * alpha - tau_arr

    width = float(np.max(alpha) - np.min(alpha))
    peak_idx = np.argmax(f_alpha)
    peak_alpha = float(alpha[peak_idx])

    return {
        "q_values": [float(v) for v in q_arr],
        "tau_q": [float(v) for v in tau_arr],
        "H_q": [float(v) for v in H_q],
        "alpha": [float(v) for v in alpha],
        "f_alpha": [float(v) for v in f_alpha],
        "width": width,
        "peak_alpha": peak_alpha,
        "is_multifractal": width > 0.1,
    }


def multifractal_width(returns: pd.Series | np.ndarray) -> dict:
    """
    Convenience wrapper: compute degree of multifractality.

    Returns spectrum width (α_max - α_min) and whether the series
    is genuinely multifractal (width > 0.1).
    """
    spec = multifractal_spectrum(returns)
    alpha_arr = np.array(spec["alpha"])
    return {
        "width": spec["width"],
        "alpha_min": float(np.min(alpha_arr)),
        "alpha_max": float(np.max(alpha_arr)),
        "peak_alpha": spec["peak_alpha"],
        "is_multifractal": spec["is_multifractal"],
    }


def long_range_dependence_test(returns: pd.Series | np.ndarray) -> dict:
    """
    Test for long-range dependence using both R/S and DFA.

    Compares Hurst estimates from both methods. Series has long-range
    dependence if both H estimates are significantly above 0.5.
    """
    rs = hurst_rs(returns)
    dfa = hurst_dfa(returns)

    h_rs = rs["H"]
    h_dfa = dfa["H"]
    se_rs = rs["H_se"]
    se_dfa = dfa["H_se"]

    # Both methods must show H > 0.5 with at least 1 SE margin
    rs_significant = (h_rs - se_rs) > 0.5
    dfa_significant = (h_dfa - se_dfa) > 0.5
    is_lrd = rs_significant and dfa_significant

    # Confidence: average of how many SEs above 0.5
    z_rs = (h_rs - 0.5) / max(se_rs, 1e-10)
    z_dfa = (h_dfa - 0.5) / max(se_dfa, 1e-10)
    confidence = float(np.clip((z_rs + z_dfa) / 2.0, -5.0, 5.0))

    return {
        "H_rs": float(h_rs),
        "H_rs_se": float(se_rs),
        "H_dfa": float(h_dfa),
        "H_dfa_se": float(se_dfa),
        "is_long_range_dependent": bool(is_lrd),
        "confidence_z": confidence,
        "interpretation": (
            "Long-range dependence detected"
            if is_lrd
            else "No significant long-range dependence"
        ),
    }


def compare_fractal_regimes(
    returns: pd.Series,
    filter_probs: pd.DataFrame,
    sigma_states: np.ndarray,
) -> dict:
    """
    Per-regime fractal analysis using MSM filter probabilities.

    Assigns each observation to its most likely regime, then computes
    Hurst exponent for each regime's subseries.

    Args:
        returns: Return series with DatetimeIndex.
        filter_probs: (T × K) DataFrame from MSM model.
        sigma_states: (K,) array of regime volatilities.

    Returns:
        Dict with per_regime list and overall comparison.
    """
    n_states = len(sigma_states)
    T = min(len(returns), len(filter_probs))

    ret_arr = np.asarray(returns[:T], dtype=float)
    prob_arr = filter_probs.iloc[:T].values

    # Assign each observation to most likely regime
    regime_labels = np.argmax(prob_arr, axis=1)

    per_regime = []
    for k in range(n_states):
        mask = regime_labels == k
        n_obs = int(np.sum(mask))

        regime_info = {
            "regime": k + 1,
            "sigma": float(sigma_states[k]),
            "n_obs": n_obs,
            "fraction": float(n_obs / T) if T > 0 else 0.0,
        }

        if n_obs >= 50:
            try:
                rs_result = hurst_rs(ret_arr[mask])
                regime_info["H"] = rs_result["H"]
                regime_info["H_se"] = rs_result["H_se"]
                regime_info["interpretation"] = rs_result["interpretation"]
            except (ValueError, Exception) as e:
                logger.warning("Hurst failed for regime %d: %s", k + 1, e)
                regime_info["H"] = None
                regime_info["H_se"] = None
                regime_info["interpretation"] = f"insufficient data ({n_obs} obs)"
        else:
            regime_info["H"] = None
            regime_info["H_se"] = None
            regime_info["interpretation"] = f"insufficient data ({n_obs} obs)"

        per_regime.append(regime_info)

    # Summary: which regimes have valid Hurst estimates
    valid = [r for r in per_regime if r["H"] is not None]
    if len(valid) >= 2:
        h_values = [r["H"] for r in valid]
        h_spread = max(h_values) - min(h_values)
        summary = (
            f"Hurst spread across regimes: {h_spread:.3f}. "
            f"{'Fractal properties vary by regime' if h_spread > 0.1 else 'Similar fractal properties across regimes'}"
        )
    elif len(valid) == 1:
        summary = f"Only regime {valid[0]['regime']} has enough data for Hurst estimation"
    else:
        summary = "No regime has enough observations for Hurst estimation"

    return {
        "per_regime": per_regime,
        "n_states": n_states,
        "summary": summary,
    }