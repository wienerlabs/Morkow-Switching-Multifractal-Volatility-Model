"""
Stochastic Volatility with Jumps (SVJ) — Bates (1996) model.

Combines Heston stochastic volatility with Merton jump-diffusion:

    Asset:    dS/S = μdt + √V dW₁ + (J-1)dN
    Variance: dV   = κ(θ - V)dt + σ√V dW₂
    Corr(dW₁, dW₂) = ρ

    Jump component: log(J) ~ N(μⱼ, σⱼ²), N ~ Poisson(λ)

Seven parameters:
    κ  — mean reversion speed of variance
    θ  — long-run variance level
    σ  — vol-of-vol (volatility of variance)
    ρ  — correlation between asset and variance Brownians
    λ  — jump intensity (expected jumps per unit time)
    μⱼ — mean jump size (in log-return space)
    σⱼ — jump size volatility

References:
    Bates, D. (1996). "Jumps and Stochastic Volatility."
    Duffie, Pan, Singleton (2000). "Transform Analysis and Asset Pricing."
"""

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def detect_jumps(
    returns: pd.Series | np.ndarray,
    threshold_multiplier: float = 3.0,
    window: int = 20,
) -> dict:
    """
    Detect jumps using Barndorff-Nielsen-Shephard (BNS) bipower variation test.

    Compares realized variance (RV) with bipower variation (BV). When RV >> BV,
    the excess is attributed to jumps. Individual jumps are flagged when
    |r_t| > threshold_multiplier × local_vol.

    Args:
        returns: Return series (percentage).
        threshold_multiplier: Sigma multiplier for individual jump detection.
        window: Rolling window for local volatility estimation.

    Returns:
        Dict with jump_dates, jump_returns, n_jumps, jump_fraction,
        avg_jump_size, jump_vol, bns_statistic, bns_pvalue.
    """
    if isinstance(returns, pd.Series):
        values = returns.values.astype(float)
        index = returns.index
    else:
        values = np.asarray(returns, dtype=float)
        index = np.arange(len(values))

    n = len(values)
    if n < window + 10:
        raise ValueError(f"Need at least {window + 10} observations, got {n}")

    rv = np.sum(values ** 2)
    abs_ret = np.abs(values)
    bv = (np.pi / 2.0) * np.sum(abs_ret[1:] * abs_ret[:-1])

    # BNS test statistic: (RV - BV) / RV ~ asymptotically standard normal under H0
    jump_variation = max(rv - bv, 0.0)
    tri_power = 0.0
    for i in range(2, n):
        tri_power += abs_ret[i] ** (2/3) * abs_ret[i-1] ** (2/3) * abs_ret[i-2] ** (2/3)
    mu_1 = math.sqrt(2.0 / math.pi)
    tri_power *= (mu_1 ** (-3)) * (2.0 ** (2/3)) * math.gamma(7/6) / math.gamma(0.5)

    iq_estimate = (n / 3.0) * tri_power if tri_power > 0 else rv ** 2 / n
    denom = max(math.sqrt(2.0 * iq_estimate / n), 1e-12)
    bns_stat = (rv - bv) / denom
    bns_pvalue = 1.0 - stats.norm.cdf(bns_stat)

    # Individual jump detection via rolling local vol
    rolling_std = pd.Series(values).rolling(window=window, min_periods=max(5, window // 2)).std().values
    rolling_std = np.where(np.isnan(rolling_std), np.nanstd(values), rolling_std)
    rolling_std = np.where(rolling_std < 1e-10, np.nanstd(values), rolling_std)

    jump_mask = np.abs(values) > threshold_multiplier * rolling_std
    jump_indices = np.where(jump_mask)[0]
    jump_returns_arr = values[jump_indices]

    return {
        "jump_dates": [index[i] for i in jump_indices],
        "jump_returns": jump_returns_arr.tolist(),
        "jump_indices": jump_indices.tolist(),
        "n_jumps": int(len(jump_indices)),
        "jump_fraction": round(len(jump_indices) / n, 6),
        "avg_jump_size": round(float(np.mean(np.abs(jump_returns_arr))), 6) if len(jump_indices) > 0 else 0.0,
        "jump_vol": round(float(np.std(jump_returns_arr)), 6) if len(jump_indices) > 1 else 0.0,
        "bns_statistic": round(float(bns_stat), 6),
        "bns_pvalue": round(float(bns_pvalue), 6),
        "jump_variation": round(float(jump_variation), 6),
        "realized_variance": round(float(rv), 6),
        "bipower_variation": round(float(bv), 6),
        "threshold_multiplier": threshold_multiplier,
    }


def calibrate_svj(
    returns: pd.Series | np.ndarray,
    use_hawkes: bool = False,
    jump_threshold_multiplier: float = 3.0,
) -> dict:
    """
    Calibrate the 7-parameter Bates SVJ model.

    Uses method of moments for initial estimates, then refines via
    quasi-MLE optimization. Optionally integrates Hawkes process for
    time-varying jump intensity.

    Args:
        returns: Return series (percentage).
        use_hawkes: If True, use Hawkes process for jump clustering.
        jump_threshold_multiplier: Multiplier for jump detection.

    Returns:
        Dict with kappa, theta, sigma, rho, lambda_, mu_j, sigma_j,
        plus diagnostics and optional hawkes_params.
    """
    if isinstance(returns, pd.Series):
        values = returns.values.astype(float)
    else:
        values = np.asarray(returns, dtype=float)

    n = len(values)
    if n < 50:
        raise ValueError(f"Need at least 50 observations for SVJ calibration, got {n}")

    # Scale returns to decimal (from percentage)
    r = values / 100.0

    # Step 1: Detect jumps for initial parameter estimation
    jumps = detect_jumps(returns, threshold_multiplier=jump_threshold_multiplier)
    jump_mask = np.zeros(n, dtype=bool)
    for idx in jumps["jump_indices"]:
        jump_mask[idx] = True

    # Separate diffusion and jump returns
    diffusion_returns = r[~jump_mask]
    jump_returns_raw = r[jump_mask]

    # Step 2: Method of moments initial estimates
    total_var = float(np.var(r))
    diffusion_var = float(np.var(diffusion_returns)) if len(diffusion_returns) > 5 else total_var * 0.8

    # Jump parameters from detected jumps
    n_jumps = len(jump_returns_raw)
    lambda_init = max(n_jumps / (n / 252.0), 0.5)  # annualized jump rate
    mu_j_init = float(np.mean(jump_returns_raw)) if n_jumps > 0 else -0.02
    sigma_j_init = float(np.std(jump_returns_raw)) if n_jumps > 1 else 0.03

    # Heston parameters from diffusion component
    theta_init = diffusion_var * 252.0  # annualized variance
    kappa_init = 5.0  # moderate mean reversion
    sigma_init = 0.5  # vol-of-vol

    # Leverage effect: correlation between returns and squared returns
    r_lag = r[:-1]
    r2_lead = r[1:] ** 2
    if len(r_lag) > 10:
        rho_init = float(np.corrcoef(r_lag, r2_lead)[0, 1])
        rho_init = np.clip(rho_init, -0.95, -0.05)
    else:
        rho_init = -0.5

    # Step 3: Quasi-MLE optimization
    # params = [kappa, theta, sigma, rho, lambda, mu_j, sigma_j]
    x0 = np.array([kappa_init, theta_init, sigma_init, rho_init,
                    lambda_init, mu_j_init, sigma_j_init])

    bounds = [
        (0.1, 50.0),      # kappa
        (1e-6, 1.0),      # theta (annualized variance)
        (0.01, 5.0),      # sigma (vol-of-vol)
        (-0.99, -0.01),   # rho (leverage, typically negative)
        (0.01, 100.0),    # lambda (jump intensity)
        (-0.20, 0.05),    # mu_j (jump mean, typically negative)
        (0.001, 0.30),    # sigma_j (jump vol)
    ]

    best_result = None
    best_nll = float("inf")

    starts = [
        x0,
        np.array([3.0, theta_init * 0.8, 0.3, -0.7, lambda_init * 0.5, -0.03, 0.04]),
        np.array([8.0, theta_init * 1.2, 0.8, -0.3, lambda_init * 1.5, -0.01, 0.02]),
        np.array([5.0, theta_init, 0.5, -0.5, 5.0, -0.02, 0.03]),
    ]

    for start in starts:
        start_clipped = np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            result = minimize(
                _svj_neg_log_likelihood,
                start_clipped,
                args=(r,),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_result = result
        except Exception as e:
            logger.debug("SVJ MLE attempt failed: %s", e)
            continue

    if best_result is None:
        logger.warning("All SVJ optimization attempts failed, using moment estimates")
        opt_params = x0
        opt_success = False
        opt_nit = 0
    else:
        opt_params = best_result.x
        opt_success = bool(best_result.success)
        opt_nit = int(best_result.nit)

    kappa, theta, sigma, rho, lambda_, mu_j, sigma_j = opt_params

    # Feller condition: 2κθ > σ² (ensures variance stays positive)
    feller_ratio = 2.0 * kappa * theta / (sigma ** 2) if sigma > 0 else float("inf")
    feller_satisfied = feller_ratio > 1.0

    # Step 4: Optional Hawkes integration for time-varying jump intensity
    hawkes_params = None
    if use_hawkes:
        try:
            from hawkes_process import extract_events, fit_hawkes, hawkes_intensity
            events = extract_events(returns, threshold_percentile=5.0, use_absolute=True)
            if events["n_events"] >= 5:
                h_fit = fit_hawkes(events["event_times"], events["T"])
                h_intens = hawkes_intensity(events["event_times"], h_fit)
                hawkes_params = {
                    "mu": round(h_fit["mu"], 6),
                    "alpha": round(h_fit["alpha"], 6),
                    "beta": round(h_fit["beta"], 6),
                    "branching_ratio": round(h_fit["branching_ratio"], 6),
                    "current_intensity": round(h_intens["current_intensity"], 6),
                    "baseline_intensity": round(h_intens["baseline"], 6),
                    "intensity_ratio": round(
                        h_intens["current_intensity"] / max(h_intens["baseline"], 1e-12), 4
                    ),
                }
        except Exception as e:
            logger.debug("Hawkes integration failed: %s", e)

    ll = -best_nll if best_result is not None else float("nan")
    k_params = 7
    aic = 2 * k_params - 2 * ll if np.isfinite(ll) else float("nan")
    bic = k_params * math.log(n) - 2 * ll if np.isfinite(ll) else float("nan")

    return {
        "kappa": round(float(kappa), 6),
        "theta": round(float(theta), 6),
        "sigma": round(float(sigma), 6),
        "rho": round(float(rho), 6),
        "lambda_": round(float(lambda_), 6),
        "mu_j": round(float(mu_j), 6),
        "sigma_j": round(float(sigma_j), 6),
        "feller_ratio": round(float(feller_ratio), 4),
        "feller_satisfied": feller_satisfied,
        "log_likelihood": round(float(ll), 4) if np.isfinite(ll) else None,
        "aic": round(float(aic), 4) if np.isfinite(aic) else None,
        "bic": round(float(bic), 4) if np.isfinite(bic) else None,
        "n_obs": n,
        "n_jumps_detected": jumps["n_jumps"],
        "jump_fraction": jumps["jump_fraction"],
        "bns_statistic": jumps["bns_statistic"],
        "bns_pvalue": jumps["bns_pvalue"],
        "optimization_success": opt_success,
        "optimization_nit": opt_nit,
        "use_hawkes": use_hawkes,
        "hawkes_params": hawkes_params,
    }



def _svj_neg_log_likelihood(params: np.ndarray, returns: np.ndarray) -> float:
    """
    Negative log-likelihood for the SVJ model using a mixture approximation.

    Approximates the SVJ density as a Poisson-weighted mixture of Gaussians:
    f(r) ≈ Σ_{j=0}^{J_max} P(N=j|λΔt) × N(r | μ + jμⱼ, V_t + jσⱼ²)

    where V_t follows a simplified discretized Heston process.
    """
    kappa, theta, sigma_v, rho, lambda_, mu_j, sigma_j = params

    dt = 1.0 / 252.0
    n = len(returns)
    j_max = 5  # truncate Poisson at 5 jumps per day

    # Poisson probabilities for 0..j_max jumps per day
    lam_dt = lambda_ * dt
    poisson_probs = np.array([
        np.exp(-lam_dt) * lam_dt**j / math.factorial(j) for j in range(j_max + 1)
    ])
    poisson_probs /= poisson_probs.sum()  # normalize truncation

    # Discretized variance process (Euler-Maruyama)
    V = np.empty(n)
    V[0] = theta  # start at long-run level

    total_ll = 0.0
    for t in range(n):
        v_t = max(V[t], 1e-10)
        sqrt_v = math.sqrt(v_t)

        # Jump-compensated drift
        drift = -0.5 * v_t * dt - lambda_ * (math.exp(mu_j + 0.5 * sigma_j**2) - 1.0) * dt

        # Mixture density: sum over possible jump counts
        log_density = -np.inf
        for j in range(j_max + 1):
            mean_j = drift + j * mu_j
            var_j = v_t * dt + j * sigma_j**2
            if var_j <= 0:
                continue
            std_j = math.sqrt(var_j)
            z = (returns[t] - mean_j) / std_j
            log_comp = math.log(max(poisson_probs[j], 1e-300)) - 0.5 * math.log(2 * math.pi) - math.log(std_j) - 0.5 * z**2
            if log_comp > log_density:
                # log-sum-exp trick
                log_density = log_comp + math.log(1.0 + math.exp(log_density - log_comp)) if np.isfinite(log_density) else log_comp
            else:
                log_density = log_density + math.log(1.0 + math.exp(log_comp - log_density)) if np.isfinite(log_comp) else log_density

        total_ll += log_density

        # Update variance for next step
        if t < n - 1:
            dV = kappa * (theta - v_t) * dt + sigma_v * sqrt_v * math.sqrt(dt) * (rho * returns[t] / sqrt_v + math.sqrt(1 - rho**2) * 0.0)
            V[t + 1] = max(v_t + dV, 1e-10)

    if not np.isfinite(total_ll):
        return 1e12

    return -total_ll



def svj_var(
    returns: pd.Series | np.ndarray,
    calibration: dict,
    alpha: float = 0.05,
    n_simulations: int = 50000,
    seed: int = 42,
) -> dict:
    """
    Jump-adjusted VaR via Monte Carlo simulation of the SVJ process.

    Simulates one-day-ahead returns from the calibrated SVJ model and
    computes VaR at the specified confidence level.

    Args:
        returns: Historical returns (percentage) for current state estimation.
        calibration: Output of calibrate_svj().
        alpha: Tail probability (e.g. 0.05 for 95% VaR).
        n_simulations: Number of Monte Carlo paths.
        seed: Random seed.

    Returns:
        Dict with var_svj, var_diffusion_only, var_jump_component,
        expected_shortfall, jump_contribution_pct.
    """
    if isinstance(returns, pd.Series):
        values = returns.values.astype(float)
    else:
        values = np.asarray(returns, dtype=float)

    rng = np.random.RandomState(seed)
    dt = 1.0 / 252.0

    kappa = calibration["kappa"]
    theta = calibration["theta"]
    sigma = calibration["sigma"]
    rho = calibration["rho"]
    lambda_ = calibration["lambda_"]
    mu_j = calibration["mu_j"]
    sigma_j = calibration["sigma_j"]

    # Current variance estimate from recent realized vol
    recent_var = float(np.var(values[-min(20, len(values)):] / 100.0)) * 252.0
    V_current = max(recent_var, theta * 0.5)

    # Simulate one-day returns with full SVJ
    Z1 = rng.randn(n_simulations)
    Z2 = rng.randn(n_simulations)
    W1 = Z1
    W2 = rho * Z1 + math.sqrt(1 - rho**2) * Z2

    sqrt_V = math.sqrt(max(V_current, 1e-10))

    # Diffusion component
    diffusion = -0.5 * V_current * dt + sqrt_V * math.sqrt(dt) * W1

    # Jump component: N ~ Poisson(λΔt), J ~ N(μⱼ, σⱼ²)
    n_jumps_sim = rng.poisson(lambda_ * dt, n_simulations)
    jump_component = np.zeros(n_simulations)
    for i in range(n_simulations):
        if n_jumps_sim[i] > 0:
            jump_sizes = rng.normal(mu_j, sigma_j, n_jumps_sim[i])
            jump_component[i] = np.sum(jump_sizes)

    # Jump-compensated drift
    jump_compensator = lambda_ * (math.exp(mu_j + 0.5 * sigma_j**2) - 1.0) * dt
    full_returns = (diffusion + jump_component - jump_compensator) * 100.0  # back to percentage

    # Diffusion-only returns for comparison
    diffusion_returns = diffusion * 100.0

    # VaR calculations
    var_svj = float(np.percentile(full_returns, alpha * 100))
    var_diffusion = float(np.percentile(diffusion_returns, alpha * 100))

    # Expected shortfall (CVaR)
    tail_returns = full_returns[full_returns <= var_svj]
    es = float(np.mean(tail_returns)) if len(tail_returns) > 0 else var_svj

    # Jump contribution
    var_jump_only = var_svj - var_diffusion
    jump_pct = abs(var_jump_only / var_svj) * 100 if abs(var_svj) > 1e-10 else 0.0

    return {
        "var_svj": round(float(var_svj), 6),
        "var_diffusion_only": round(float(var_diffusion), 6),
        "var_jump_component": round(float(var_jump_only), 6),
        "expected_shortfall": round(float(es), 6),
        "jump_contribution_pct": round(float(jump_pct), 4),
        "alpha": alpha,
        "confidence": round(1.0 - alpha, 4),
        "n_simulations": n_simulations,
        "current_variance": round(float(V_current), 6),
        "avg_jumps_per_day": round(float(np.mean(n_jumps_sim)), 4),
    }


def decompose_risk(
    returns: pd.Series | np.ndarray,
    calibration: dict,
) -> dict:
    """
    Decompose total return variance into diffusion and jump components.

    Total variance = diffusion variance + jump variance
    Diffusion variance = θ (long-run Heston variance)
    Jump variance = λ(μⱼ² + σⱼ²) (Poisson-compound-normal)

    Args:
        returns: Return series (percentage).
        calibration: Output of calibrate_svj().

    Returns:
        Dict with diffusion_var, jump_var, total_var, jump_share_pct,
        diffusion_share_pct, annualized metrics.
    """
    if isinstance(returns, pd.Series):
        values = returns.values.astype(float)
    else:
        values = np.asarray(returns, dtype=float)

    theta = calibration["theta"]
    lambda_ = calibration["lambda_"]
    mu_j = calibration["mu_j"]
    sigma_j = calibration["sigma_j"]

    # Annualized variance decomposition
    diffusion_var = theta  # long-run Heston variance (annualized)
    jump_var = lambda_ * (mu_j**2 + sigma_j**2)  # jump contribution (annualized)
    total_model_var = diffusion_var + jump_var

    # Empirical variance for comparison
    empirical_var = float(np.var(values / 100.0)) * 252.0

    jump_share = jump_var / total_model_var * 100 if total_model_var > 1e-12 else 0.0
    diffusion_share = diffusion_var / total_model_var * 100 if total_model_var > 1e-12 else 100.0

    # Daily metrics
    daily_diffusion_vol = math.sqrt(diffusion_var / 252.0) * 100  # percentage
    daily_jump_vol = math.sqrt(max(jump_var / 252.0, 0)) * 100
    daily_total_vol = math.sqrt(max(total_model_var / 252.0, 0)) * 100

    return {
        "diffusion_variance": round(float(diffusion_var), 6),
        "jump_variance": round(float(jump_var), 6),
        "total_model_variance": round(float(total_model_var), 6),
        "empirical_variance": round(float(empirical_var), 6),
        "jump_share_pct": round(float(jump_share), 4),
        "diffusion_share_pct": round(float(diffusion_share), 4),
        "daily_diffusion_vol": round(float(daily_diffusion_vol), 4),
        "daily_jump_vol": round(float(daily_jump_vol), 4),
        "daily_total_vol": round(float(daily_total_vol), 4),
        "annualized_diffusion_vol": round(float(math.sqrt(diffusion_var) * 100), 4),
        "annualized_jump_vol": round(float(math.sqrt(max(jump_var, 0)) * 100), 4),
        "annualized_total_vol": round(float(math.sqrt(max(total_model_var, 0)) * 100), 4),
    }


def svj_diagnostics(
    returns: pd.Series | np.ndarray,
    calibration: dict,
) -> dict:
    """
    Comprehensive diagnostics for the calibrated SVJ model.

    Includes jump detection statistics, parameter stability indicators,
    EVT tail analysis of jump sizes, and Hawkes clustering metrics.

    Args:
        returns: Return series (percentage).
        calibration: Output of calibrate_svj().

    Returns:
        Dict with jump_stats, parameter_quality, tail_analysis, clustering.
    """
    if isinstance(returns, pd.Series):
        values = returns.values.astype(float)
    else:
        values = np.asarray(returns, dtype=float)

    jumps = detect_jumps(returns)

    # Parameter quality checks
    kappa = calibration["kappa"]
    theta = calibration["theta"]
    sigma = calibration["sigma"]
    lambda_ = calibration["lambda_"]

    half_life = math.log(2) / kappa if kappa > 0 else float("inf")
    mean_reversion_days = half_life * 252

    # Tail analysis: kurtosis and skewness of returns
    r = values / 100.0
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r))

    # Model-implied moments
    model_var = theta + lambda_ * (calibration["mu_j"]**2 + calibration["sigma_j"]**2)
    model_skew_approx = lambda_ * calibration["mu_j"] * (3 * calibration["sigma_j"]**2 + calibration["mu_j"]**2)

    # EVT integration: fit GPD to jump sizes if enough jumps
    evt_tail = None
    if jumps["n_jumps"] >= 10:
        try:
            from extreme_value_theory import fit_gpd, select_threshold
            jump_losses = np.abs(np.array(jumps["jump_returns"]))
            thresh_info = select_threshold(jump_losses)
            gpd_fit = fit_gpd(jump_losses, thresh_info["threshold"])
            evt_tail = {
                "gpd_xi": gpd_fit["xi"],
                "gpd_beta": gpd_fit["beta"],
                "threshold": gpd_fit["threshold"],
                "n_exceedances": gpd_fit["n_exceedances"],
                "tail_index": gpd_fit["xi"],
            }
        except Exception as e:
            logger.debug("EVT tail analysis of jumps failed: %s", e)

    # Clustering analysis via Hawkes if enough jumps
    clustering = None
    if jumps["n_jumps"] >= 5:
        try:
            from hawkes_process import extract_events, fit_hawkes, detect_clusters
            events = extract_events(returns, threshold_percentile=5.0, use_absolute=True)
            if events["n_events"] >= 5:
                h_fit = fit_hawkes(events["event_times"], events["T"])
                clusters = detect_clusters(events["event_times"], h_fit)
                clustering = {
                    "branching_ratio": round(h_fit["branching_ratio"], 4),
                    "half_life_days": round(h_fit["half_life"] * 252, 2),
                    "n_clusters": len(clusters),
                    "avg_cluster_size": round(
                        float(np.mean([c["n_events"] for c in clusters])), 2
                    ) if clusters else 0.0,
                    "stationarity": h_fit["stationarity"],
                }
        except Exception as e:
            logger.debug("Hawkes clustering analysis failed: %s", e)

    return {
        "jump_stats": {
            "n_jumps": jumps["n_jumps"],
            "jump_fraction": jumps["jump_fraction"],
            "avg_jump_size": jumps["avg_jump_size"],
            "jump_vol": jumps["jump_vol"],
            "bns_statistic": jumps["bns_statistic"],
            "bns_pvalue": jumps["bns_pvalue"],
            "jumps_significant": jumps["bns_pvalue"] < 0.05,
        },
        "parameter_quality": {
            "feller_satisfied": calibration["feller_satisfied"],
            "feller_ratio": calibration["feller_ratio"],
            "half_life_years": round(float(half_life), 4),
            "mean_reversion_days": round(float(mean_reversion_days), 1),
            "optimization_success": calibration["optimization_success"],
        },
        "moment_comparison": {
            "empirical_skewness": round(float(skew), 6),
            "empirical_kurtosis": round(float(kurt), 6),
            "model_variance": round(float(model_var), 6),
            "model_skew_approx": round(float(model_skew_approx), 6),
        },
        "evt_tail": evt_tail,
        "clustering": clustering,
    }


def svj_vol_series(returns: pd.Series) -> pd.Series:
    """
    Generate SVJ-implied volatility series for model comparison.

    Calibrates SVJ and produces a time series of conditional volatility
    estimates using the discretized Heston variance process with jump
    contribution.

    Args:
        returns: Daily log-returns in percentage with DatetimeIndex.

    Returns:
        pd.Series of conditional volatility (same index as returns).
    """
    values = returns.values.astype(float)
    n = len(values)

    cal = calibrate_svj(returns)
    kappa = cal["kappa"]
    theta = cal["theta"]
    sigma = cal["sigma"]
    rho = cal["rho"]
    lambda_ = cal["lambda_"]
    mu_j = cal["mu_j"]
    sigma_j = cal["sigma_j"]

    dt = 1.0 / 252.0
    r = values / 100.0

    # Run discretized Heston variance filter
    V = np.empty(n)
    V[0] = theta

    for t in range(1, n):
        v_prev = max(V[t - 1], 1e-10)
        # Euler-Maruyama step for variance
        innovation = r[t - 1] + 0.5 * v_prev * dt  # approximate return innovation
        dV = kappa * (theta - v_prev) * dt + sigma * math.sqrt(v_prev * dt) * (
            rho * innovation / math.sqrt(max(v_prev * dt, 1e-12))
        )
        V[t] = max(v_prev + dV, 1e-10)

    # Total conditional vol = sqrt(V_t/252 + jump_var/252) in percentage
    jump_var_daily = lambda_ * (mu_j**2 + sigma_j**2) * dt
    total_vol = np.sqrt(V * dt + jump_var_daily) * 100.0

    return pd.Series(total_vol, index=returns.index, name="svj_vol")