"""
Liquidity-Adjusted VaR (LVaR) — the "L" in A-LAMS-VaR.

Extends MSM-VaR with liquidity risk components:
  1. Bid-ask spread estimation via Roll (1984) and Amihud (2002)
  2. Liquidity-adjusted VaR per Bangia et al. (1999)
  3. Regime-conditional liquidity profiles
  4. Market impact via square-root law (Kyle 1985)

Mathematical formulas:
  Roll spread:   S = 2 * sqrt(max(0, -Cov(Δp_t, Δp_{t-1})))
  Amihud ILLIQ:  ILLIQ = (1/T) * Σ|r_t| / V_t
  LVaR:          LVaR = VaR + LC  where LC = 0.5 * S * position + z_α * σ_S * position
  Regime spread: S_k = S_base * (σ_k / σ_1)^δ  where δ ≈ 1.5
  Market impact: MI = σ * sqrt(Q / ADV)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


def estimate_spread(ohlcv_df: pd.DataFrame) -> dict:
    """
    Estimate effective bid-ask spread and illiquidity from OHLCV data.

    Roll (1984) estimator:
        S = 2 * sqrt(max(0, -Cov(Δp_t, Δp_{t-1})))

    Amihud (2002) illiquidity ratio:
        ILLIQ = (1/T) * Σ|r_t| / V_t

    Args:
        ohlcv_df: DataFrame with columns [Open, High, Low, Close, Volume].

    Returns:
        Dict with roll_spread_pct, amihud_illiq, serial_covariance, etc.
    """
    close = ohlcv_df["Close"].values.astype(float)
    volume = ohlcv_df["Volume"].values.astype(float)

    if len(close) < 10:
        raise ValueError(f"Need ≥10 observations for spread estimation, got {len(close)}")

    dp = np.diff(close)
    cov_val = float(np.cov(dp[:-1], dp[1:])[0, 1])
    spread_abs = 2.0 * np.sqrt(max(0.0, -cov_val))
    mean_price = float(np.mean(close))
    spread_pct = (spread_abs / mean_price * 100.0) if mean_price > 0 else 0.0

    log_returns = np.diff(np.log(close))
    safe_volume = np.where(volume[1:] > 0, volume[1:], np.nan)
    ratios = np.abs(log_returns) / safe_volume
    amihud_illiq = float(np.nanmean(ratios)) if np.any(np.isfinite(ratios)) else 0.0

    return {
        "roll_spread_pct": float(spread_pct),
        "roll_spread_abs": float(spread_abs),
        "amihud_illiq": amihud_illiq,
        "serial_covariance": cov_val,
        "mean_price": mean_price,
        "n_obs": len(close),
    }


def liquidity_adjusted_var(
    var_value: float,
    spread: float,
    position_value: float,
    spread_vol: float | None = None,
    alpha: float = 0.05,
) -> dict:
    """
    Compute Liquidity-Adjusted VaR (Bangia et al. 1999).

        LVaR = VaR + 0.5 * spread * position + z_α * σ_spread * position

    Args:
        var_value: Base VaR as a positive loss (e.g. 3.2 for 3.2% loss).
        spread: Bid-ask spread as fraction of mid-price (e.g. 0.005 for 0.5%).
        position_value: Position notional in USD.
        spread_vol: Volatility of spread (same units as spread). None = 0.
        alpha: VaR tail probability (default 0.05 for 95% VaR).

    Returns:
        Dict with lvar, liquidity_cost, base_var, and breakdown.
    """
    if spread_vol is None:
        spread_vol = 0.0

    z_alpha = float(norm.ppf(1.0 - alpha))
    lc_fixed = 0.5 * spread * position_value
    lc_variable = spread_vol * z_alpha * position_value if spread_vol > 0 else 0.0
    liquidity_cost = lc_fixed + lc_variable
    lvar = var_value + liquidity_cost / position_value if position_value > 0 else var_value

    return {
        "lvar": float(lvar),
        "base_var": float(var_value),
        "liquidity_cost": float(liquidity_cost),
        "liquidity_cost_pct": float(liquidity_cost / position_value * 100.0) if position_value > 0 else 0.0,
        "lc_fixed": float(lc_fixed),
        "lc_variable": float(lc_variable),
        "spread": float(spread),
        "spread_vol": float(spread_vol),
        "z_alpha": z_alpha,
        "alpha": float(alpha),
        "position_value": float(position_value),
    }


def regime_liquidity_profile(
    returns: pd.Series | np.ndarray,
    filter_probs: pd.DataFrame | np.ndarray,
    sigma_states: np.ndarray | list[float],
    volumes: np.ndarray | pd.Series | None = None,
    delta: float = 1.5,
) -> dict:
    """
    Compute per-regime spread estimates and liquidity metrics.

    Regime spread scaling:
        S_k = S_base * (σ_k / σ_1)^δ   where δ ≈ 1.5

    Spreads widen in crisis regimes because higher volatility drives
    wider bid-ask spreads via inventory risk and adverse selection.

    Args:
        returns: Return series (log-returns in %).
        filter_probs: MSM filter probabilities (T × K matrix).
        sigma_states: Per-regime volatility levels (length K).
        volumes: Optional volume series (same length as returns).
        delta: Spread scaling exponent (default 1.5).

    Returns:
        Dict with per-regime profiles, base spread, and weighted metrics.
    """
    returns_arr = np.asarray(returns, dtype=float)
    probs = np.asarray(filter_probs, dtype=float)
    sigmas = np.asarray(sigma_states, dtype=float)
    num_states = len(sigmas)

    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    regime_labels = np.argmax(probs, axis=1) + 1  # 1-based

    # Base spread from full sample
    dp = np.diff(returns_arr)
    if len(dp) >= 3:
        cov_val = float(np.cov(dp[:-1], dp[1:])[0, 1])
        base_spread = 2.0 * np.sqrt(max(0.0, -cov_val))
    else:
        base_spread = 0.0

    sigma_min = float(sigmas[0]) if sigmas[0] > 0 else 1.0

    profiles = []
    for k in range(1, num_states + 1):
        mask = regime_labels == k
        n_obs = int(mask.sum())

        sigma_k = float(sigmas[k - 1])
        spread_multiplier = (sigma_k / sigma_min) ** delta if sigma_min > 0 else 1.0
        regime_spread = base_spread * spread_multiplier

        amihud = None
        mean_vol = None
        if volumes is not None:
            vol_arr = np.asarray(volumes, dtype=float)
            if len(vol_arr) == len(regime_labels):
                regime_vols = vol_arr[mask]
                regime_rets = returns_arr[mask]
                safe_vol = np.where(regime_vols > 0, regime_vols, np.nan)
                ratios = np.abs(regime_rets) / safe_vol
                amihud = float(np.nanmean(ratios)) if np.any(np.isfinite(ratios)) else None
                mean_vol = float(np.mean(regime_vols))

        liquidity_score = 1.0 / (1.0 + regime_spread) * 100.0

        profiles.append({
            "regime": k,
            "sigma": float(sigma_k),
            "spread": float(regime_spread),
            "spread_multiplier": float(spread_multiplier),
            "amihud_illiq": amihud,
            "mean_volume": mean_vol,
            "liquidity_score": float(liquidity_score),
            "n_obs": n_obs,
        })

    current_probs = probs[-1] if len(probs) > 0 else np.ones(num_states) / num_states
    weighted_spread = sum(
        float(current_probs[k]) * profiles[k]["spread"] for k in range(num_states)
    )

    return {
        "num_states": num_states,
        "base_spread": float(base_spread),
        "delta": float(delta),
        "profiles": profiles,
        "current_weighted_spread": float(weighted_spread),
        "current_regime_probs": current_probs.tolist(),
    }


def estimate_market_impact(
    trade_size_usd: float,
    adv: float,
    sigma: float,
) -> dict:
    """
    Estimate price impact using the square-root law (Kyle 1985).

        MI = σ * sqrt(trade_size / ADV)

    Args:
        trade_size_usd: Trade notional in USD.
        adv: Average daily volume in USD.
        sigma: Daily volatility (decimal, e.g. 0.03 for 3%).

    Returns:
        Dict with impact_pct, impact_usd, participation_rate.
    """
    if adv <= 0:
        raise ValueError("ADV must be positive")
    if trade_size_usd <= 0:
        raise ValueError("Trade size must be positive")
    if sigma < 0:
        raise ValueError("Volatility must be non-negative")

    participation = trade_size_usd / adv
    impact_pct = float(sigma * np.sqrt(participation) * 100.0)
    impact_usd = float(impact_pct / 100.0 * trade_size_usd)

    return {
        "impact_pct": impact_pct,
        "impact_usd": impact_usd,
        "participation_rate": float(participation),
        "sigma_daily": float(sigma),
        "trade_size_usd": float(trade_size_usd),
        "adv_usd": float(adv),
    }
