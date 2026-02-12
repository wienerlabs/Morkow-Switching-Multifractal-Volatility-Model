"""
Extreme Value Theory (EVT) module — Peaks-Over-Threshold with GPD.

Provides tail risk measurement beyond what Normal or Student-t distributions
can capture. Uses the Generalized Pareto Distribution for exceedances above
a high threshold, yielding accurate VaR/CVaR at extreme quantiles (99%+).

Mathematical foundation:
    F_u(y) = 1 - (1 + ξy/β)^(-1/ξ)   for ξ ≠ 0
    F_u(y) = 1 - exp(-y/β)             for ξ = 0
where y = loss - u (exceedance), ξ = shape, β = scale, u = threshold.
"""

from __future__ import annotations

import math
from typing import Literal

import logging

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def fit_gpd(
    losses: np.ndarray,
    threshold: float,
) -> dict:
    """Fit GPD to exceedances above threshold via MLE.

    Args:
        losses: Array of positive losses (absolute returns).
        threshold: The threshold u; only losses > u are used.

    Returns:
        Dict with xi, beta, threshold, n_total, n_exceedances,
        log_likelihood, aic, bic.
    """
    losses = np.asarray(losses, dtype=float)
    exceedances = losses[losses > threshold] - threshold
    n_exc = len(exceedances)
    n_total = len(losses)

    if n_exc < 10:
        raise ValueError(f"Only {n_exc} exceedances above threshold {threshold:.4f}. Need ≥10.")

    # scipy.stats.genpareto: CDF = 1 - (1 + c*x/scale)^(-1/c)  where c = xi
    c, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    xi = float(c)
    beta = float(scale)

    if beta <= 0:
        raise ValueError(f"GPD fit produced invalid scale β={beta:.6f}")

    ll = float(stats.genpareto.logpdf(exceedances, c=xi, loc=0, scale=beta).sum())
    k_params = 2  # xi, beta
    aic = 2 * k_params - 2 * ll
    bic = k_params * math.log(n_exc) - 2 * ll

    return {
        "xi": round(xi, 6),
        "beta": round(beta, 6),
        "threshold": round(threshold, 6),
        "n_total": n_total,
        "n_exceedances": n_exc,
        "log_likelihood": round(ll, 4),
        "aic": round(aic, 4),
        "bic": round(bic, 4),
    }


def select_threshold(
    returns: np.ndarray | pd.Series,
    method: Literal["percentile", "mean_excess", "variance_stability"] = "variance_stability",
    min_exceedances: int = 50,
) -> dict:
    """Automated threshold selection for POT.

    Args:
        returns: Raw returns (will be negated to get losses).
        method: Selection algorithm.
        min_exceedances: Minimum exceedances required.

    Returns:
        Dict with threshold, method, n_exceedances, diagnostics.
    """
    r = np.asarray(returns if not isinstance(returns, pd.Series) else returns.values, dtype=float)
    losses = -r  # positive = loss

    if method == "percentile":
        return _threshold_percentile(losses, min_exceedances)
    elif method == "mean_excess":
        return _threshold_mean_excess(losses, min_exceedances)
    elif method == "variance_stability":
        return _threshold_variance_stability(losses, min_exceedances)
    else:
        raise ValueError(f"Unknown threshold method: {method}")


def _threshold_percentile(
    losses: np.ndarray, min_exc: int
) -> dict:
    """Use 90th percentile, adjusting upward if too few exceedances."""
    for pct in [90, 85, 80, 75]:
        u = float(np.percentile(losses, pct))
        n_exc = int((losses > u).sum())
        if n_exc >= min_exc:
            return {
                "threshold": round(u, 6),
                "method": "percentile",
                "percentile_used": pct,
                "n_exceedances": n_exc,
            }
    u = float(np.percentile(losses, 75))
    return {
        "threshold": round(u, 6),
        "method": "percentile",
        "percentile_used": 75,
        "n_exceedances": int((losses > u).sum()),
    }


def _threshold_mean_excess(
    losses: np.ndarray, min_exc: int
) -> dict:
    """Mean Excess Function: find where E[X-u | X>u] becomes linear in u."""
    sorted_losses = np.sort(losses)
    n = len(sorted_losses)
    candidates = sorted_losses[max(0, n - min_exc * 5): n - min_exc]
    if len(candidates) < 5:
        candidates = sorted_losses[int(n * 0.7): int(n * 0.95)]

    thresholds = []
    mean_excesses = []
    for u in candidates:
        exc = losses[losses > u] - u
        if len(exc) >= min_exc:
            thresholds.append(float(u))
            mean_excesses.append(float(np.mean(exc)))

    if not thresholds:
        return _threshold_percentile(losses, min_exc)

    # Find the point where mean excess starts increasing linearly
    # (GPD property: E[X-u|X>u] = β/(1-ξ) + ξu/(1-ξ) — linear in u for ξ>0)
    # Pick the threshold where the slope stabilizes
    me = np.array(mean_excesses)
    th = np.array(thresholds)
    if len(th) < 3:
        best_idx = len(th) // 2
    else:
        diffs = np.abs(np.diff(me) / np.diff(th))
        d2 = np.abs(np.diff(diffs))
        best_idx = int(np.argmin(d2)) + 1

    return {
        "threshold": round(thresholds[best_idx], 6),
        "method": "mean_excess",
        "n_exceedances": int((losses > thresholds[best_idx]).sum()),
        "diagnostics": {
            "thresholds": [round(t, 4) for t in thresholds[:20]],
            "mean_excesses": [round(m, 4) for m in mean_excesses[:20]],
        },
    }


def _threshold_variance_stability(
    losses: np.ndarray, min_exc: int
) -> dict:
    """Scan thresholds and pick where GPD shape ξ stabilizes."""
    sorted_losses = np.sort(losses)
    n = len(sorted_losses)
    lo = max(0, n - min_exc * 6)
    hi = n - min_exc
    if hi <= lo:
        return _threshold_percentile(losses, min_exc)

    step = max(1, (hi - lo) // 40)
    indices = range(lo, hi, step)

    thresholds: list[float] = []
    xis: list[float] = []

    for idx in indices:
        u = float(sorted_losses[idx])
        exc = losses[losses > u] - u
        if len(exc) < min_exc:
            continue
        try:
            c, _, sc = stats.genpareto.fit(exc, floc=0)
            if sc > 0:
                thresholds.append(u)
                xis.append(float(c))
        except Exception as e:
            logger.debug("GPD fit failed at threshold %.4f: %s", u, e)
            continue

    if len(thresholds) < 3:
        return _threshold_percentile(losses, min_exc)

    # Pick threshold where ξ is most stable (smallest local variance)
    xi_arr = np.array(xis)
    window = max(3, len(xi_arr) // 5)
    rolling_var = pd.Series(xi_arr).rolling(window, center=True).var().values
    valid = ~np.isnan(rolling_var)
    if not valid.any():
        best_idx = len(thresholds) // 2
    else:
        best_idx = int(np.nanargmin(rolling_var))

    return {
        "threshold": round(thresholds[best_idx], 6),
        "method": "variance_stability",
        "n_exceedances": int((losses > thresholds[best_idx]).sum()),
        "diagnostics": {
            "thresholds": [round(t, 4) for t in thresholds],
            "xi_estimates": [round(x, 4) for x in xis],
        },
    }


def evt_var(
    xi: float,
    beta: float,
    threshold: float,
    n_total: int,
    n_exceedances: int,
    alpha: float = 0.01,
) -> float:
    """EVT-based VaR using the GPD tail estimator.

    Formula: VaR_p = u + (β/ξ) * [(n/N_u * (1-p))^(-ξ) - 1]   for ξ ≠ 0
             VaR_p = u + β * ln(n / (N_u * (1-p)))              for ξ ≈ 0

    Args:
        xi: GPD shape parameter.
        beta: GPD scale parameter.
        threshold: The threshold u.
        n_total: Total number of observations.
        n_exceedances: Number of exceedances above u.
        alpha: Tail probability (e.g. 0.01 for 99% VaR).

    Returns:
        VaR as a positive loss value (negate for return space).
    """
    if beta <= 0:
        raise ValueError(f"β must be > 0, got {beta}")
    if n_exceedances <= 0:
        raise ValueError("n_exceedances must be > 0")
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in (0,1), got {alpha}")

    p = 1.0 - alpha  # confidence level
    ratio = n_total / n_exceedances * (1.0 - p)

    if abs(xi) < 1e-8:
        # Exponential limit
        var_val = threshold + beta * math.log(1.0 / ratio)
    else:
        var_val = threshold + (beta / xi) * (ratio ** (-xi) - 1.0)

    return float(var_val)


def evt_cvar(
    xi: float,
    beta: float,
    threshold: float,
    var_value: float,
    alpha: float = 0.01,
) -> float:
    """EVT-based CVaR (Expected Shortfall) beyond VaR.

    Formula: CVaR = VaR/(1-ξ) + (β - ξ*u)/(1-ξ)

    Args:
        xi: GPD shape parameter.
        beta: GPD scale parameter.
        threshold: The threshold u.
        var_value: Pre-computed VaR (positive loss).
        alpha: Tail probability.

    Returns:
        CVaR as a positive loss value.
    """
    if xi >= 1.0:
        raise ValueError(f"CVaR undefined for ξ ≥ 1 (infinite mean), got ξ={xi}")

    denom = 1.0 - xi
    cvar_val = var_value / denom + (beta - xi * threshold) / denom
    return float(cvar_val)


def evt_backtest(
    returns: np.ndarray | pd.Series,
    xi: float,
    beta: float,
    threshold: float,
    n_total: int,
    n_exceedances: int,
    alphas: list[float] | None = None,
) -> list[dict]:
    """Backtest EVT-VaR at multiple confidence levels.

    Returns a list of dicts, one per alpha, with breach_rate, expected_rate,
    kupiec_lr, kupiec_pvalue, kupiec_pass.
    """
    if alphas is None:
        alphas = [0.05, 0.01, 0.005, 0.001]

    r = np.asarray(returns if not isinstance(returns, pd.Series) else returns.values, dtype=float)
    losses = -r
    n = len(r)
    results = []

    for a in alphas:
        var_val = evt_var(xi, beta, threshold, n_total, n_exceedances, alpha=a)
        breaches = int((losses > var_val).sum())
        breach_rate = breaches / n if n > 0 else 0.0

        # Kupiec LR test
        kup_lr, kup_p = _kupiec_lr(breaches, n, a)

        results.append({
            "alpha": a,
            "confidence": round(1.0 - a, 4),
            "evt_var": round(-var_val, 4),  # negative in return space
            "breach_count": breaches,
            "breach_rate": round(breach_rate, 6),
            "expected_rate": a,
            "kupiec_lr": round(kup_lr, 4) if np.isfinite(kup_lr) else None,
            "kupiec_pvalue": round(kup_p, 4) if np.isfinite(kup_p) else None,
            "kupiec_pass": bool(kup_p > 0.05) if np.isfinite(kup_p) else None,
        })

    return results


def _kupiec_lr(x: int, n: int, alpha: float) -> tuple[float, float]:
    """Kupiec (1995) proportion-of-failures LR test."""
    if n == 0 or x == 0 or x == n:
        return float("nan"), float("nan")
    p_hat = x / n
    try:
        lr = 2 * (
            x * math.log(p_hat / alpha)
            + (n - x) * math.log((1 - p_hat) / (1 - alpha))
        )
    except (ValueError, ZeroDivisionError):
        return float("nan"), float("nan")
    p_value = float(1.0 - stats.chi2.cdf(lr, df=1))
    return float(lr), p_value


def compare_var_methods(
    returns: np.ndarray | pd.Series,
    sigma_forecast: float,
    xi: float,
    beta: float,
    threshold: float,
    n_total: int,
    n_exceedances: int,
    nu: float = 5.0,
    alphas: list[float] | None = None,
) -> list[dict]:
    """Compare Normal vs Student-t vs EVT VaR at multiple confidence levels.

    Args:
        returns: Historical returns for breach counting.
        sigma_forecast: Current MSM sigma forecast.
        xi, beta, threshold, n_total, n_exceedances: GPD parameters.
        nu: Student-t degrees of freedom.
        alphas: Confidence levels to compare.

    Returns:
        List of dicts with method, alpha, var_value, breach_rate.
    """
    from scipy.stats import norm, t as student_t

    if alphas is None:
        alphas = [0.05, 0.01, 0.005, 0.001]

    r = np.asarray(returns if not isinstance(returns, pd.Series) else returns.values, dtype=float)
    losses = -r
    n = len(r)
    results = []

    for a in alphas:
        # Normal VaR
        z_n = float(norm.ppf(a))
        var_normal = z_n * sigma_forecast

        # Student-t VaR
        z_t = float(student_t.ppf(a, df=nu))
        var_t = z_t * sigma_forecast

        # EVT VaR
        var_evt_loss = evt_var(xi, beta, threshold, n_total, n_exceedances, alpha=a)
        var_evt = -var_evt_loss  # return space (negative)

        for method, var_val in [("normal", var_normal), ("student_t", var_t), ("evt_gpd", var_evt)]:
            breach_count = int((r < var_val).sum())
            results.append({
                "method": method,
                "alpha": a,
                "confidence": round(1.0 - a, 4),
                "var_value": round(var_val, 4),
                "breach_count": breach_count,
                "breach_rate": round(breach_count / n, 6) if n > 0 else 0.0,
                "expected_rate": a,
            })

    return results

