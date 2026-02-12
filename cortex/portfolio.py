"""
Portfolio-level regime-switching VaR.

Extends the single-asset MSM model to multi-asset portfolios with
regime-conditional correlations (correlations spike in crisis regimes).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from cortex import msm

logger = logging.getLogger(__name__)


def _weighted_correlation(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute weighted Pearson correlation matrix.
    weights are per-observation (T,), returns are (T, N).
    """
    w = weights / weights.sum()
    mean = (w[:, None] * returns).sum(axis=0)
    centered = returns - mean
    cov = (centered * w[:, None]).T @ centered
    std = np.sqrt(np.diag(cov))
    std = np.where(std < 1e-12, 1e-12, std)
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def calibrate_multivariate(
    returns_df: pd.DataFrame,
    num_states: int = 5,
    method: str = "mle",
) -> dict:
    """
    Calibrate MSM per asset and estimate regime-conditional correlation matrices.

    Args:
        returns_df: DataFrame with columns = asset names, values = log-returns in %.
        num_states: Number of MSM regimes.
        method: Calibration method (mle, grid, empirical, hybrid).

    Returns:
        Dict with per-asset calibrations, regime covariance matrices,
        and current regime probabilities.
    """
    assets = list(returns_df.columns)
    n_assets = len(assets)
    common_idx = returns_df.dropna().index
    returns_clean = returns_df.loc[common_idx]
    T = len(returns_clean)

    if T < 30:
        raise ValueError(f"Need ≥30 common observations, got {T}")
    if n_assets < 2:
        raise ValueError("Need ≥2 assets for portfolio VaR")

    per_asset: dict[str, dict] = {}
    all_filter_probs = np.zeros((T, num_states))

    for asset in assets:
        rets = returns_clean[asset]
        cal = msm.calibrate_msm_advanced(
            rets, num_states=num_states, method=method, verbose=False
        )
        sigma_f, sigma_filt, fprobs, sigma_states, P = msm.msm_vol_forecast(
            rets,
            num_states=cal["num_states"],
            sigma_low=cal["sigma_low"],
            sigma_high=cal["sigma_high"],
            p_stay=cal["p_stay"],
        )
        per_asset[asset] = {
            "calibration": cal,
            "sigma_forecast": sigma_f,
            "sigma_filtered": sigma_filt,
            "filter_probs": fprobs,
            "sigma_states": sigma_states,
            "P_matrix": P,
        }
        all_filter_probs += fprobs.values

    # Average regime probabilities across assets
    avg_probs = all_filter_probs / n_assets
    current_probs = avg_probs[-1]
    current_probs /= current_probs.sum()

    # Build regime-conditional covariance matrices
    returns_arr = returns_clean.values  # (T, N)
    regime_corr = np.zeros((num_states, n_assets, n_assets))
    regime_cov = np.zeros((num_states, n_assets, n_assets))

    for k in range(num_states):
        w_k = avg_probs[:, k]
        if w_k.sum() < 1e-12:
            regime_corr[k] = np.eye(n_assets)
        else:
            regime_corr[k] = _weighted_correlation(returns_arr, w_k)

        # Per-asset volatility in regime k
        sigma_k = np.array([per_asset[a]["sigma_states"][k] for a in assets])
        regime_cov[k] = np.diag(sigma_k) @ regime_corr[k] @ np.diag(sigma_k)

    # Average transition matrix
    P_avg = np.mean([per_asset[a]["P_matrix"] for a in assets], axis=0)

    return {
        "assets": assets,
        "num_states": num_states,
        "per_asset": per_asset,
        "regime_corr": regime_corr,
        "regime_cov": regime_cov,
        "current_probs": current_probs,
        "P_matrix": P_avg,
        "returns_df": returns_clean,
    }




def _regime_weighted_cov(model: dict, probs: np.ndarray) -> np.ndarray:
    """Compute covariance matrix as probability-weighted sum across regimes."""
    cov = np.zeros_like(model["regime_cov"][0])
    for k in range(model["num_states"]):
        cov += probs[k] * model["regime_cov"][k]
    return cov


def portfolio_var(
    model: dict,
    weights: dict[str, float],
    alpha: float = 0.05,
) -> dict:
    """
    Portfolio VaR considering regime-switching diversification.

    Args:
        model: Output of calibrate_multivariate().
        weights: Asset weights, e.g. {"SOL": 0.4, "RAY": 0.3, "JUP": 0.3}.
        alpha: VaR confidence level (0.05 = 95% VaR).

    Returns:
        Dict with portfolio VaR, sigma, and per-regime breakdown.
    """
    assets = model["assets"]
    w = np.array([weights.get(a, 0.0) for a in assets])
    if abs(w.sum() - 1.0) > 0.01:
        logger.warning("Weights sum to %.3f, not 1.0", w.sum())

    probs = model["current_probs"]
    cov = _regime_weighted_cov(model, probs)

    sigma_p = float(np.sqrt(w @ cov @ w))
    z_alpha = float(norm.ppf(alpha))
    var_p = z_alpha * sigma_p

    regime_detail = []
    for k in range(model["num_states"]):
        sig_k = float(np.sqrt(w @ model["regime_cov"][k] @ w))
        regime_detail.append({
            "regime": k + 1,
            "probability": float(probs[k]),
            "portfolio_sigma": sig_k,
            "portfolio_var": z_alpha * sig_k,
        })

    return {
        "portfolio_var": var_p,
        "portfolio_sigma": sigma_p,
        "z_alpha": z_alpha,
        "weights": {a: float(w[i]) for i, a in enumerate(assets)},
        "regime_breakdown": regime_detail,
    }


def marginal_var(
    model: dict,
    weights: dict[str, float],
    alpha: float = 0.05,
) -> dict:
    """
    Marginal VaR and Component VaR (Euler risk decomposition).

    Marginal VaR_i = z × (Σw)_i / σ_p  (sensitivity to weight_i)
    Component VaR_i = w_i × Marginal VaR_i  (sums to portfolio VaR)
    """
    assets = model["assets"]
    w = np.array([weights.get(a, 0.0) for a in assets])

    probs = model["current_probs"]
    cov = _regime_weighted_cov(model, probs)

    sigma_p = float(np.sqrt(w @ cov @ w))
    z_alpha = float(norm.ppf(alpha))

    cov_w = cov @ w
    marginal = z_alpha * cov_w / sigma_p
    component = w * marginal
    total_cvar = component.sum()
    pct = component / total_cvar if abs(total_cvar) > 1e-12 else np.zeros_like(component)

    decomposition = []
    for i, a in enumerate(assets):
        decomposition.append({
            "asset": a,
            "weight": float(w[i]),
            "marginal_var": float(marginal[i]),
            "component_var": float(component[i]),
            "pct_contribution": float(pct[i]),
        })

    return {
        "portfolio_var": z_alpha * sigma_p,
        "portfolio_sigma": sigma_p,
        "decomposition": sorted(decomposition, key=lambda d: d["component_var"]),
    }


def stress_var(
    model: dict,
    weights: dict[str, float],
    forced_regime: int = 5,
    alpha: float = 0.05,
) -> dict:
    """
    Stressed VaR by forcing the model into a specific regime.

    Args:
        model: Output of calibrate_multivariate().
        weights: Asset weights dict.
        forced_regime: 1-based regime index (5 = Crisis for K=5).
        alpha: VaR confidence level.
    """
    K = model["num_states"]
    if forced_regime < 1 or forced_regime > K:
        raise ValueError(f"forced_regime must be 1..{K}, got {forced_regime}")

    assets = model["assets"]
    w = np.array([weights.get(a, 0.0) for a in assets])
    z_alpha = float(norm.ppf(alpha))

    # Normal VaR (current regime mix)
    cov_normal = _regime_weighted_cov(model, model["current_probs"])
    sigma_normal = float(np.sqrt(w @ cov_normal @ w))
    var_normal = z_alpha * sigma_normal

    # Stressed VaR (forced single regime)
    k = forced_regime - 1
    cov_stress = model["regime_cov"][k]
    sigma_stress = float(np.sqrt(w @ cov_stress @ w))
    var_stress = z_alpha * sigma_stress

    multiplier = sigma_stress / sigma_normal if sigma_normal > 1e-12 else float("inf")

    asset_stress = []
    for i, a in enumerate(assets):
        asset_stress.append({
            "asset": a,
            "normal_sigma": float(np.sqrt(cov_normal[i, i])),
            "stressed_sigma": float(np.sqrt(cov_stress[i, i])),
        })

    return {
        "forced_regime": forced_regime,
        "stressed_var": var_stress,
        "stressed_sigma": sigma_stress,
        "normal_var": var_normal,
        "normal_sigma": sigma_normal,
        "stress_multiplier": multiplier,
        "regime_correlation": model["regime_corr"][k].tolist(),
        "asset_stress": asset_stress,
    }