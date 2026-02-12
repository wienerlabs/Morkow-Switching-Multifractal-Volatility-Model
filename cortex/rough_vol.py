"""
Rough Volatility Module — Fractional Brownian Motion (fBm) based volatility modeling.

Implements the Gatheral-Jaisson-Rosenbaum (2018) framework where realized volatility
exhibits rough paths with Hurst exponent H ≈ 0.1, contradicting the H = 0.5 assumption
of standard Brownian motion.

Models:
  - Rough Bergomi (rBergomi): log-normal forward variance with fBm driver
  - Rough Heston (rHeston): fractional mean-reverting variance process

Key references:
  - Gatheral, Jaisson, Rosenbaum (2018) "Volatility is rough"
  - Bayer, Friz, Gatheral (2016) "Pricing under rough volatility"
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def generate_fbm(n: int, H: float = 0.1, seed: int | None = None) -> np.ndarray:
    """
    Generate fractional Brownian motion increments via Davies-Harte (FFT) method.

    Uses circulant embedding of the autocovariance function for exact generation
    in O(n log n) time.

    Args:
        n: Number of increments to generate.
        H: Hurst exponent (0 < H < 1). H < 0.5 = rough, H = 0.5 = standard BM, H > 0.5 = smooth.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray of shape (n,) — fBm increments.
    """
    if not 0 < H < 1:
        raise ValueError(f"Hurst exponent H must be in (0, 1), got {H}")
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    rng = np.random.RandomState(seed)

    # Autocovariance of fBm increments: γ(k) = 0.5*(|k+1|^{2H} - 2|k|^{2H} + |k-1|^{2H})
    k = np.arange(n)
    two_h = 2.0 * H
    gamma = 0.5 * (np.abs(k + 1) ** two_h - 2.0 * np.abs(k) ** two_h + np.abs(k - 1) ** two_h)

    # Circulant embedding: mirror the autocovariance
    m = 2 * n
    row = np.zeros(m)
    row[:n] = gamma
    row[n:] = gamma[n - 1 :: -1][: n]  # mirror: γ(n-1), γ(n-2), ..., γ(1)

    # Eigenvalues via FFT (real since circulant is symmetric)
    eigenvalues = np.real(np.fft.fft(row))

    # Clamp tiny negative eigenvalues from numerical noise
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Generate complex Gaussian in spectral domain
    z_real = rng.randn(m)
    z_imag = rng.randn(m)
    z = z_real + 1j * z_imag

    # Multiply by sqrt of eigenvalues and inverse FFT
    w = np.fft.ifft(np.sqrt(eigenvalues) * z)

    # Take real part of first n elements
    increments = np.real(w[:n]) / np.sqrt(2.0)

    return increments


def estimate_roughness(
    returns: pd.Series | np.ndarray,
    window: int = 5,
    max_lag: int = 50,
) -> dict[str, Any]:
    """
    Estimate Hurst exponent H from realized volatility using the variogram method.

    The key insight from Gatheral et al. (2018): for log realized volatility X_t,
    E[|X_{t+Δ} - X_t|²] ~ C · Δ^{2H}, so regressing log(variogram) on log(lag)
    gives slope = 2H.

    Args:
        returns: Daily log-returns (% scale).
        window: Rolling window for realized volatility estimation.
        max_lag: Maximum lag for variogram computation.

    Returns:
        Dict with keys: H, H_se, r_squared, method, lags, variogram,
        is_rough (H < 0.3), interpretation.
    """
    r = np.asarray(returns, dtype=float)
    if len(r) < window + max_lag + 10:
        raise ValueError(
            f"Need at least {window + max_lag + 10} observations, got {len(r)}"
        )

    # Compute realized volatility (rolling std of returns)
    rv = pd.Series(r).rolling(window=window).std().dropna().values
    if len(rv) < max_lag + 10:
        raise ValueError("Not enough data after rolling window computation")

    # Log realized volatility
    rv_positive = np.maximum(rv, 1e-12)
    log_rv = np.log(rv_positive)

    # Compute variogram: V(Δ) = mean(|log_rv_{t+Δ} - log_rv_t|²)
    lags = np.arange(1, min(max_lag + 1, len(log_rv) // 2))
    variogram = np.zeros(len(lags))
    for i, lag in enumerate(lags):
        diffs = log_rv[lag:] - log_rv[:-lag]
        variogram[i] = np.mean(diffs ** 2)

    # Filter out zero/negative variogram values
    valid = variogram > 0
    if valid.sum() < 3:
        raise ValueError("Insufficient valid variogram points for regression")

    log_lags = np.log(lags[valid])
    log_var = np.log(variogram[valid])

    # OLS regression: log(V(Δ)) = 2H·log(Δ) + const
    X = np.column_stack([log_lags, np.ones(len(log_lags))])
    beta, residuals, _, _ = np.linalg.lstsq(X, log_var, rcond=None)
    slope = beta[0]
    intercept = beta[1]
    H = slope / 2.0

    # Clamp H to valid range
    H = float(np.clip(H, 0.001, 0.999))

    # R² and standard error
    ss_res = np.sum((log_var - X @ beta) ** 2)
    ss_tot = np.sum((log_var - np.mean(log_var)) ** 2)
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    n_pts = len(log_lags)
    se = float(np.sqrt(ss_res / max(n_pts - 2, 1)) / np.sqrt(np.sum((log_lags - np.mean(log_lags)) ** 2))) / 2.0

    if H < 0.3:
        interpretation = f"Rough volatility (H={H:.3f} < 0.3): anti-persistent, mean-reverting at all scales"
    elif H < 0.7:
        interpretation = f"Standard volatility (H={H:.3f} ≈ 0.5): consistent with Brownian motion"
    else:
        interpretation = f"Smooth volatility (H={H:.3f} > 0.7): persistent, trending behavior"

    return {
        "H": H,
        "H_se": se,
        "r_squared": r_squared,
        "method": "variogram",
        "lags": lags[valid].tolist(),
        "variogram": variogram[valid].tolist(),
        "is_rough": H < 0.3,
        "interpretation": interpretation,
    }


def calibrate_rough_bergomi(
    returns: pd.Series | np.ndarray,
    window: int = 5,
    max_lag: int = 50,
) -> dict[str, Any]:
    """
    Calibrate the Rough Bergomi model to observed returns.

    The rBergomi model:
        log(V_t / V_0) = -0.5 * ν² * t^{2H} + ν * W^H_t

    Parameters estimated:
        - H: Hurst exponent (from variogram of log realized vol)
        - nu (ν): vol-of-vol parameter (from variance of log realized vol increments)
        - V0: initial variance level (from first realized vol observation)

    Args:
        returns: Daily log-returns (% scale).
        window: Rolling window for realized volatility.
        max_lag: Maximum lag for variogram.

    Returns:
        Dict with keys: model, H, nu, V0, sigma0, metrics, method.
    """
    r = np.asarray(returns, dtype=float)

    roughness = estimate_roughness(r, window=window, max_lag=max_lag)
    H = roughness["H"]

    # Realized volatility
    rv = pd.Series(r).rolling(window=window).std().dropna().values
    rv_positive = np.maximum(rv, 1e-12)
    log_rv = np.log(rv_positive)

    V0 = float(rv_positive[0] ** 2)
    sigma0 = float(rv_positive[0])

    # Estimate ν from the variance of log-RV increments at lag 1
    # Under rBergomi: Var(log V_{t+1} - log V_t) = ν² · 1^{2H} = ν²
    dlog_rv = np.diff(log_rv)
    nu_squared = float(np.var(dlog_rv))
    nu = float(np.sqrt(max(nu_squared, 1e-12)))

    # Goodness of fit: simulate rBergomi paths and compare variogram
    n_sim = len(rv)
    fbm_inc = generate_fbm(n_sim, H=H, seed=42)
    t_grid = np.arange(1, n_sim + 1, dtype=float)
    log_v_sim = -0.5 * nu ** 2 * t_grid ** (2 * H) + nu * np.cumsum(fbm_inc)
    v_sim = V0 * np.exp(log_v_sim)
    sigma_sim = np.sqrt(np.maximum(v_sim, 1e-12))

    # Correlation between simulated and realized vol
    min_len = min(len(sigma_sim), len(rv_positive))
    corr = float(np.corrcoef(sigma_sim[:min_len], rv_positive[:min_len])[0, 1])

    # MAE
    mae = float(np.mean(np.abs(sigma_sim[:min_len] - rv_positive[:min_len])))

    return {
        "model": "rough_bergomi",
        "H": H,
        "nu": nu,
        "V0": V0,
        "sigma0": sigma0,
        "metrics": {
            "H_se": roughness["H_se"],
            "H_r_squared": roughness["r_squared"],
            "vol_correlation": corr if np.isfinite(corr) else 0.0,
            "mae": mae,
            "is_rough": roughness["is_rough"],
        },
        "method": "variogram_moments",
    }


def calibrate_rough_heston(
    returns: pd.Series | np.ndarray,
    window: int = 5,
    max_lag: int = 50,
) -> dict[str, Any]:
    """
    Calibrate the Rough Heston model to observed returns.

    The rHeston model:
        dV_t = λ(θ - V_t)dt + ξ√V_t dW^H_t

    Parameters estimated via method of moments + optimization:
        - H: Hurst exponent (from variogram)
        - λ (lambda_): mean-reversion speed
        - θ (theta): long-run variance
        - ξ (xi): vol-of-vol

    Args:
        returns: Daily log-returns (% scale).
        window: Rolling window for realized volatility.
        max_lag: Maximum lag for variogram.

    Returns:
        Dict with keys: model, H, lambda_, theta, xi, V0, metrics, method.
    """
    r = np.asarray(returns, dtype=float)

    roughness = estimate_roughness(r, window=window, max_lag=max_lag)
    H = roughness["H"]

    # Realized variance
    rv = pd.Series(r).rolling(window=window).std().dropna().values
    rv_var = rv ** 2
    rv_var_positive = np.maximum(rv_var, 1e-12)

    V0 = float(rv_var_positive[0])
    theta_init = float(np.mean(rv_var_positive))

    # Method of moments for initial estimates
    # E[V_t] = θ (stationary mean)
    # Var[V_t] ≈ ξ²θ / (2λ) (approximate for fractional case)
    # Autocorrelation decay gives λ
    var_of_var = float(np.var(rv_var_positive))
    mean_var = float(np.mean(rv_var_positive))

    # Estimate λ from autocorrelation of variance at lag 1
    if len(rv_var_positive) > 2:
        acf1 = float(np.corrcoef(rv_var_positive[:-1], rv_var_positive[1:])[0, 1])
        acf1 = max(min(acf1, 0.999), 0.001)
        lambda_init = -np.log(acf1)
    else:
        lambda_init = 0.5

    xi_init = float(np.sqrt(max(2.0 * lambda_init * var_of_var / max(mean_var, 1e-12), 1e-6)))

    def _rh_objective(params: np.ndarray) -> float:
        lam, theta, xi = params
        if lam <= 0 or theta <= 0 or xi <= 0:
            return 1e12

        # Euler-Maruyama simulation of rHeston with fBm
        n_sim = len(rv_var_positive)
        fbm_inc = generate_fbm(n_sim, H=H, seed=42)
        dt = 1.0
        v_sim = np.zeros(n_sim)
        v_sim[0] = V0

        for t in range(1, n_sim):
            v_prev = max(v_sim[t - 1], 1e-12)
            dv = lam * (theta - v_prev) * dt + xi * np.sqrt(v_prev) * fbm_inc[t]
            v_sim[t] = max(v_prev + dv, 1e-12)

        # Minimize MSE between simulated and realized variance
        mse = float(np.mean((v_sim - rv_var_positive) ** 2))
        return mse

    result = minimize(
        _rh_objective,
        x0=np.array([lambda_init, theta_init, xi_init]),
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-8},
    )

    lambda_opt, theta_opt, xi_opt = np.abs(result.x)

    # Compute fit metrics
    n_sim = len(rv_var_positive)
    fbm_inc = generate_fbm(n_sim, H=H, seed=42)
    v_sim = np.zeros(n_sim)
    v_sim[0] = V0
    for t in range(1, n_sim):
        v_prev = max(v_sim[t - 1], 1e-12)
        dv = lambda_opt * (theta_opt - v_prev) + xi_opt * np.sqrt(v_prev) * fbm_inc[t]
        v_sim[t] = max(v_prev + dv, 1e-12)

    sigma_sim = np.sqrt(v_sim)
    rv_sigma = np.sqrt(rv_var_positive)
    corr = float(np.corrcoef(sigma_sim, rv_sigma)[0, 1])
    mae = float(np.mean(np.abs(sigma_sim - rv_sigma)))

    return {
        "model": "rough_heston",
        "H": H,
        "lambda_": float(lambda_opt),
        "theta": float(theta_opt),
        "xi": float(xi_opt),
        "V0": V0,
        "metrics": {
            "H_se": roughness["H_se"],
            "H_r_squared": roughness["r_squared"],
            "vol_correlation": corr if np.isfinite(corr) else 0.0,
            "mae": mae,
            "optimization_success": result.success,
            "optimization_nit": int(result.nit),
            "is_rough": roughness["is_rough"],
        },
        "method": "moments_optimization",
    }



def rough_vol_forecast(
    returns: pd.Series | np.ndarray,
    calibration: dict[str, Any],
    horizon: int = 10,
    n_paths: int = 1000,
    seed: int | None = None,
) -> dict[str, Any]:
    """
    Forecast volatility using a calibrated rough volatility model.

    Uses Monte Carlo simulation with the calibrated parameters to generate
    forward-looking volatility paths and compute point forecasts with
    confidence intervals.

    Args:
        returns: Historical daily log-returns (% scale).
        calibration: Output of calibrate_rough_bergomi() or calibrate_rough_heston().
        horizon: Number of days to forecast.
        n_paths: Number of Monte Carlo simulation paths.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: model, horizon, point_forecast, lower_95, upper_95,
        mean_forecast, all_quantiles.
    """
    model = calibration["model"]
    H = calibration["H"]
    rng = np.random.RandomState(seed)

    r = np.asarray(returns, dtype=float)
    rv = pd.Series(r).rolling(window=5).std().dropna().values
    current_vol = float(rv[-1]) if len(rv) > 0 else float(np.std(r))
    current_var = current_vol ** 2

    paths = np.zeros((n_paths, horizon))

    if model == "rough_bergomi":
        nu = calibration["nu"]
        for i in range(n_paths):
            fbm_inc = generate_fbm(horizon, H=H, seed=rng.randint(0, 2**31))
            t_grid = np.arange(1, horizon + 1, dtype=float)
            log_v = -0.5 * nu ** 2 * t_grid ** (2 * H) + nu * np.cumsum(fbm_inc)
            paths[i] = current_var * np.exp(log_v)

    elif model == "rough_heston":
        lam = calibration["lambda_"]
        theta = calibration["theta"]
        xi = calibration["xi"]
        for i in range(n_paths):
            fbm_inc = generate_fbm(horizon, H=H, seed=rng.randint(0, 2**31))
            v = np.zeros(horizon)
            v[0] = current_var
            for t in range(1, horizon):
                v_prev = max(v[t - 1], 1e-12)
                dv = lam * (theta - v_prev) + xi * np.sqrt(v_prev) * fbm_inc[t]
                v[t] = max(v_prev + dv, 1e-12)
            paths[i] = v
    else:
        raise ValueError(f"Unknown model: {model}")

    # Convert variance paths to volatility
    vol_paths = np.sqrt(np.maximum(paths, 1e-12))

    # Point forecast: median across paths at each horizon
    point_forecast = np.median(vol_paths, axis=0).tolist()
    lower_95 = np.percentile(vol_paths, 2.5, axis=0).tolist()
    upper_95 = np.percentile(vol_paths, 97.5, axis=0).tolist()
    mean_forecast = np.mean(vol_paths, axis=0).tolist()

    quantiles = {
        "q05": np.percentile(vol_paths, 5, axis=0).tolist(),
        "q25": np.percentile(vol_paths, 25, axis=0).tolist(),
        "q50": point_forecast,
        "q75": np.percentile(vol_paths, 75, axis=0).tolist(),
        "q95": np.percentile(vol_paths, 95, axis=0).tolist(),
    }

    return {
        "model": model,
        "horizon": horizon,
        "current_vol": current_vol,
        "point_forecast": point_forecast,
        "lower_95": lower_95,
        "upper_95": upper_95,
        "mean_forecast": mean_forecast,
        "all_quantiles": quantiles,
    }


def compare_rough_vs_msm(
    returns: pd.Series | np.ndarray,
    msm_calibration: dict[str, Any],
) -> dict[str, Any]:
    """
    Head-to-head comparison of Rough Bergomi vs MSM volatility models.

    Compares in-sample fit, forecasting accuracy, and model characteristics.

    Args:
        returns: Daily log-returns (% scale).
        msm_calibration: Dict with keys from calibrate_msm_advanced() output,
            must include sigma_low, sigma_high, p_stay, num_states.

    Returns:
        Dict with keys: rough_bergomi, msm, winner, comparison_metrics.
    """
    import importlib
    msm_module = importlib.import_module("MSM-VaR_MODEL")

    r = np.asarray(returns, dtype=float)
    r_series = pd.Series(r) if not isinstance(returns, pd.Series) else returns

    # Calibrate rough Bergomi
    rough_cal = calibrate_rough_bergomi(r)

    # Get MSM forecasts
    sf_msm, _, _, _, _ = msm_module.msm_vol_forecast(
        r_series,
        num_states=msm_calibration["num_states"],
        sigma_low=msm_calibration["sigma_low"],
        sigma_high=msm_calibration["sigma_high"],
        p_stay=msm_calibration["p_stay"],
    )
    msm_sigma = sf_msm.values.astype(float)

    # Get rough Bergomi in-sample vol
    window = 5
    rv = pd.Series(r).rolling(window=window).std().dropna().values
    rv_positive = np.maximum(rv, 1e-12)

    H = rough_cal["H"]
    nu = rough_cal["nu"]
    V0 = rough_cal["V0"]
    n_sim = len(rv_positive)
    fbm_inc = generate_fbm(n_sim, H=H, seed=42)
    t_grid = np.arange(1, n_sim + 1, dtype=float)
    log_v_sim = -0.5 * nu ** 2 * t_grid ** (2 * H) + nu * np.cumsum(fbm_inc)
    rough_sigma = np.sqrt(np.maximum(V0 * np.exp(log_v_sim), 1e-12))

    # Align lengths for comparison
    realized = np.abs(r)
    min_len = min(len(realized), len(msm_sigma), len(rough_sigma))
    realized_aligned = realized[-min_len:]
    msm_aligned = msm_sigma[-min_len:]
    rough_aligned = rough_sigma[-min_len:]

    # Compute metrics for both
    msm_mae = float(np.mean(np.abs(realized_aligned - msm_aligned)))
    rough_mae = float(np.mean(np.abs(realized_aligned - rough_aligned)))

    msm_corr = float(np.corrcoef(realized_aligned, msm_aligned)[0, 1])
    rough_corr = float(np.corrcoef(realized_aligned, rough_aligned)[0, 1])

    msm_rmse = float(np.sqrt(np.mean((realized_aligned - msm_aligned) ** 2)))
    rough_rmse = float(np.sqrt(np.mean((realized_aligned - rough_aligned) ** 2)))

    # Winner determination (lower MAE = better)
    rough_wins = 0
    msm_wins = 0
    if rough_mae < msm_mae:
        rough_wins += 1
    else:
        msm_wins += 1
    if rough_corr > msm_corr:
        rough_wins += 1
    else:
        msm_wins += 1
    if rough_rmse < msm_rmse:
        rough_wins += 1
    else:
        msm_wins += 1

    winner = "rough_bergomi" if rough_wins > msm_wins else "msm"

    return {
        "rough_bergomi": {
            "H": rough_cal["H"],
            "nu": rough_cal["nu"],
            "mae": rough_mae,
            "rmse": rough_rmse,
            "correlation": rough_corr if np.isfinite(rough_corr) else 0.0,
            "is_rough": rough_cal["metrics"]["is_rough"],
        },
        "msm": {
            "num_states": msm_calibration["num_states"],
            "mae": msm_mae,
            "rmse": msm_rmse,
            "correlation": msm_corr if np.isfinite(msm_corr) else 0.0,
        },
        "winner": winner,
        "comparison_metrics": {
            "mae_ratio": rough_mae / max(msm_mae, 1e-12),
            "rmse_ratio": rough_rmse / max(msm_rmse, 1e-12),
            "corr_diff": rough_corr - msm_corr if np.isfinite(rough_corr) and np.isfinite(msm_corr) else 0.0,
        },
    }


def rough_bergomi_vol_series(returns: pd.Series) -> pd.Series:
    """
    Generate a volatility forecast series from Rough Bergomi for model_comparison.py integration.

    Returns a pd.Series with the same index as returns, containing the rBergomi
    in-sample volatility estimates.
    """
    r = np.asarray(returns, dtype=float)

    try:
        cal = calibrate_rough_bergomi(r)
    except ValueError:
        logger.debug("Rough Bergomi calibration failed, falling back to rolling vol")
        return pd.Series(np.abs(r), index=returns.index if isinstance(returns, pd.Series) else None)

    H = cal["H"]
    nu = cal["nu"]
    V0 = cal["V0"]

    window = 5
    rv = pd.Series(r).rolling(window=window).std().dropna().values
    n_sim = len(rv)
    fbm_inc = generate_fbm(n_sim, H=H, seed=42)
    t_grid = np.arange(1, n_sim + 1, dtype=float)
    log_v_sim = -0.5 * nu ** 2 * t_grid ** (2 * H) + nu * np.cumsum(fbm_inc)
    rough_sigma = np.sqrt(np.maximum(V0 * np.exp(log_v_sim), 1e-12))

    # Pad the beginning (rolling window warmup) with first valid value
    full_sigma = np.full(len(r), rough_sigma[0] if len(rough_sigma) > 0 else np.std(r))
    full_sigma[window - 1:window - 1 + len(rough_sigma)] = rough_sigma

    idx = returns.index if isinstance(returns, pd.Series) else None
    return pd.Series(full_sigma, index=idx)