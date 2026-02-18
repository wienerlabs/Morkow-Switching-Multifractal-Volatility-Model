"""
Liquidity-Adjusted VaR (LVaR) module for the A-LAMS-VaR framework.

Extends MSM-VaR with liquidity risk components:
  1. Bid-ask spread estimation (Roll 1984)
  2. Liquidity-adjusted VaR (Bangia et al. 1999)
  3. Regime-conditional liquidity profiles
  4. Market depth / price impact (Kyle 1985)

References:
  - Roll, R. (1984). "A Simple Implicit Measure of the Effective Bid-Ask Spread"
  - Bangia, A. et al. (1999). "Liquidity on the Outside" (Risk Magazine)
  - Kyle, A.S. (1985). "Continuous Auctions and Insider Trading"
"""
from __future__ import annotations

__all__ = [
    "estimate_spread",
    "market_impact_cost",
    "liquidity_adjusted_var",
    "regime_liquidity_profile",
    "compute_lvar_with_regime",
    "fetch_dexscreener_spread",
    "liquidity_adjusted_var_with_dexscreener",
    "fetch_onchain_spread",
    "liquidity_adjusted_var_with_onchain",
]

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


def estimate_spread(
    prices: np.ndarray | pd.Series,
    method: str = "roll",
    window: int | None = None,
    highs: np.ndarray | pd.Series | None = None,
    lows: np.ndarray | pd.Series | None = None,
) -> dict:
    """
    Estimate effective bid-ask spread from price series.

    Roll (1984) estimator:
        spread = 2 * sqrt(max(0, -Cov(Δp_t, Δp_{t-1})))

    Corwin-Schultz (2012) high-low estimator:
        Derives bid-ask spreads from daily high and low prices.
        High prices are buyer-initiated, lows are seller-initiated.
        The ratio of high-to-low reflects both volatility and spread.

    Args:
        prices: Price series (Close prices).
        method: Estimation method — "roll" (default) or "high_low" (Corwin-Schultz 2012).
        window: Rolling window size. None = full-sample estimate.
        highs: High price series (required for "high_low" method).
        lows: Low price series (required for "high_low" method).

    Returns:
        Dict with spread_pct (as % of price), spread_abs, method, and
        rolling series if window is specified.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 10:
        raise ValueError(f"Need ≥10 prices for spread estimation, got {len(prices)}")

    if method == "roll":
        return _roll_spread(prices, window)
    elif method == "high_low":
        if highs is None or lows is None:
            raise ValueError("high_low method requires 'highs' and 'lows' arrays")
        h = np.asarray(highs, dtype=float)
        l = np.asarray(lows, dtype=float)
        if len(h) != len(prices) or len(l) != len(prices):
            raise ValueError(
                f"highs ({len(h)}), lows ({len(l)}), and prices ({len(prices)}) "
                "must have the same length"
            )
        return _corwin_schultz_spread(prices, h, l, window)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'roll' or 'high_low'.")


def _roll_spread(prices: np.ndarray, window: int | None) -> dict:
    """Roll (1984) spread estimator implementation."""
    dp = np.diff(prices)

    if window is None:
        cov_val = np.cov(dp[:-1], dp[1:])[0, 1]
        spread_abs = 2.0 * np.sqrt(max(0.0, -cov_val))
        mean_price = np.mean(prices)
        spread_pct = (spread_abs / mean_price * 100.0) if mean_price > 0 else 0.0
        spread_vol = 0.0  # no rolling → no volatility estimate
        return {
            "spread_pct": float(spread_pct),
            "spread_abs": float(spread_abs),
            "spread_vol_pct": float(spread_vol),
            "method": "roll",
            "n_obs": len(prices),
            "serial_covariance": float(cov_val),
        }

    # Rolling estimation
    if window < 5:
        raise ValueError("Rolling window must be ≥5")

    n = len(dp)
    spreads = np.full(n - 1, np.nan)
    for i in range(window, n):
        seg = dp[i - window : i]
        cov_val = np.cov(seg[:-1], seg[1:])[0, 1]
        spreads[i - 1] = 2.0 * np.sqrt(max(0.0, -cov_val))

    valid = spreads[~np.isnan(spreads)]
    mean_price = np.mean(prices)
    mean_spread = float(np.nanmean(valid)) if len(valid) > 0 else 0.0
    spread_vol = float(np.nanstd(valid)) if len(valid) > 1 else 0.0

    return {
        "spread_pct": float(mean_spread / mean_price * 100.0) if mean_price > 0 else 0.0,
        "spread_abs": float(mean_spread),
        "spread_vol_pct": float(spread_vol / mean_price * 100.0) if mean_price > 0 else 0.0,
        "spread_vol_abs": float(spread_vol),
        "method": "roll",
        "window": window,
        "n_obs": len(prices),
        "rolling_spreads": spreads.tolist(),
    }


def _corwin_schultz_spread(
    prices: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    window: int | None,
) -> dict:
    """Corwin-Schultz (2012) high-low spread estimator.

    Formulas:
        β = Σ [ln(H_t / L_t)]²  for each day in a 2-day window
        γ = [ln(H_{t,t+1} / L_{t,t+1})]²  using 2-day high and low
        α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
        spread = 2(e^α - 1) / (1 + e^α)
    """
    n = len(prices)
    k1 = 3.0 - 2.0 * np.sqrt(2.0)  # ≈ 0.17157

    log_hl = np.log(highs / lows)
    beta = log_hl[:-1] ** 2 + log_hl[1:] ** 2
    high_2d = np.maximum(highs[:-1], highs[1:])
    low_2d = np.minimum(lows[:-1], lows[1:])
    gamma = np.log(high_2d / low_2d) ** 2

    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k1 - np.sqrt(gamma / k1)
    alpha_clamped = np.maximum(alpha, 0.0)
    spread_frac = 2.0 * (np.exp(alpha_clamped) - 1.0) / (1.0 + np.exp(alpha_clamped))

    if window is None:
        mean_spread_frac = float(np.nanmean(spread_frac))
        mean_price = float(np.mean(prices))
        return {
            "spread_pct": float(mean_spread_frac * 100.0),
            "spread_abs": float(mean_spread_frac * mean_price),
            "spread_vol_pct": 0.0,
            "method": "high_low",
            "n_obs": n,
            "n_pairs": len(spread_frac),
            "mean_alpha": float(np.nanmean(alpha)),
            "pct_negative_alpha": float(np.mean(alpha < 0) * 100.0),
        }

    if window < 5:
        raise ValueError("Rolling window must be ≥5")

    rolling = np.full(len(spread_frac), np.nan)
    for i in range(window - 1, len(spread_frac)):
        rolling[i] = np.nanmean(spread_frac[i - window + 1 : i + 1])

    valid = rolling[~np.isnan(rolling)]
    mean_price = float(np.mean(prices))
    mean_spread_frac = float(np.nanmean(valid)) if len(valid) > 0 else 0.0
    spread_vol_frac = float(np.nanstd(valid)) if len(valid) > 1 else 0.0

    return {
        "spread_pct": float(mean_spread_frac * 100.0),
        "spread_abs": float(mean_spread_frac * mean_price),
        "spread_vol_pct": float(spread_vol_frac * 100.0),
        "spread_vol_abs": float(spread_vol_frac * mean_price),
        "method": "high_low",
        "window": window,
        "n_obs": n,
        "n_pairs": len(spread_frac),
        "rolling_spreads": (rolling * 100.0).tolist(),
    }


def market_impact_cost(
    sigma: float,
    trade_size_usd: float,
    adv_usd: float,
    participation_rate: float = 0.10,
) -> dict:
    """
    Estimate price impact cost using the square-root model (Kyle 1985).

        impact = σ_daily * sqrt(trade_size / ADV)

    This captures the permanent price impact of a trade relative to
    average daily volume. For DeFi, ADV is the pool's 24h volume.

    Args:
        sigma: Daily volatility (as decimal, e.g. 0.03 for 3%).
        trade_size_usd: Trade notional in USD.
        adv_usd: Average daily volume in USD.
        participation_rate: Max fraction of ADV to trade (default 10%).

    Returns:
        Dict with impact_pct, impact_usd, participation, and risk metrics.
    """
    if adv_usd <= 0:
        raise ValueError("ADV must be positive")
    if trade_size_usd <= 0:
        raise ValueError("Trade size must be positive")
    if sigma < 0:
        raise ValueError("Volatility must be non-negative")

    participation = trade_size_usd / adv_usd
    impact_pct = sigma * np.sqrt(participation) * 100.0
    impact_usd = impact_pct / 100.0 * trade_size_usd

    return {
        "impact_pct": float(impact_pct),
        "impact_usd": float(impact_usd),
        "participation_rate": float(participation),
        "participation_warning": participation > participation_rate,
        "sigma_daily": float(sigma),
        "trade_size_usd": float(trade_size_usd),
        "adv_usd": float(adv_usd),
    }


def liquidity_adjusted_var(
    var_value: float,
    spread_pct: float,
    spread_vol_pct: float = 0.0,
    position_value: float = 1.0,
    alpha: float = 0.05,
    holding_period: int = 1,
) -> dict:
    """
    Compute Liquidity-Adjusted VaR (Bangia et al. 1999).

        LVaR = VaR + LC
        LC   = 0.5 * (μ_spread + z_α * σ_spread) * √(holding_period)

    The liquidity cost (LC) captures the expected cost of unwinding a
    position under adverse spread conditions. The z_α term accounts for
    spread widening at the VaR confidence level.

    Args:
        var_value: Base VaR (negative number, e.g. -3.2 for 3.2% loss).
        spread_pct: Mean bid-ask spread as % of mid-price.
        spread_vol_pct: Volatility of spread as % of mid-price.
        position_value: Position notional (for absolute LC calculation).
        alpha: VaR confidence level (0.05 = 95% VaR).
        holding_period: Holding period in days (for multi-day LVaR).

    Returns:
        Dict with lvar, liquidity_cost, base_var, and breakdown.
    """
    z_alpha = norm.ppf(alpha)
    sqrt_hp = np.sqrt(holding_period)

    # Liquidity cost: half-spread + confidence-scaled spread volatility
    lc_pct = 0.5 * (spread_pct + abs(z_alpha) * spread_vol_pct) * sqrt_hp
    lvar = var_value - lc_pct  # more negative = worse

    lc_abs = lc_pct / 100.0 * position_value
    lvar_abs = lvar / 100.0 * position_value

    return {
        "lvar": float(lvar),
        "base_var": float(var_value),
        "liquidity_cost_pct": float(lc_pct),
        "liquidity_cost_abs": float(lc_abs),
        "lvar_abs": float(lvar_abs),
        "spread_pct": float(spread_pct),
        "spread_vol_pct": float(spread_vol_pct),
        "alpha": float(alpha),
        "z_alpha": float(z_alpha),
        "holding_period": holding_period,
        "lvar_ratio": float(lvar / var_value) if abs(var_value) > 1e-12 else float("nan"),
    }


def regime_liquidity_profile(
    prices: np.ndarray | pd.Series,
    regime_labels: np.ndarray,
    num_states: int,
    volumes: np.ndarray | pd.Series | None = None,
) -> dict:
    """
    Compute regime-conditional liquidity profiles.

    For each regime state k, estimates:
      - spread_k: Roll spread within regime k
      - volume_k: Mean volume in regime k
      - liquidity_score_k: Composite score (lower = less liquid)

    This captures the empirical observation that liquidity dries up
    during crisis regimes — spreads widen and volumes may spike or collapse.

    Args:
        prices: Price series (Close prices).
        regime_labels: Array of regime assignments (1-based, same length as prices).
        num_states: Number of regime states.
        volumes: Optional volume series for volume-weighted analysis.

    Returns:
        Dict with per-regime liquidity metrics and overall profile.
    """
    prices = np.asarray(prices, dtype=float)
    regime_labels = np.asarray(regime_labels, dtype=int)

    if len(prices) != len(regime_labels):
        raise ValueError(
            f"prices ({len(prices)}) and regime_labels ({len(regime_labels)}) must have same length"
        )

    profiles = []
    for k in range(1, num_states + 1):
        mask = regime_labels == k
        n_obs = int(mask.sum())

        if n_obs < 10:
            profiles.append({
                "regime": k,
                "n_obs": n_obs,
                "spread_pct": None,
                "spread_abs": None,
                "mean_volume": None,
                "liquidity_score": None,
                "insufficient_data": True,
            })
            continue

        regime_prices = prices[mask]
        dp = np.diff(regime_prices)

        if len(dp) >= 3:
            cov_val = np.cov(dp[:-1], dp[1:])[0, 1]
            spread_abs = 2.0 * np.sqrt(max(0.0, -cov_val))
        else:
            spread_abs = 0.0

        mean_price = np.mean(regime_prices)
        spread_pct = (spread_abs / mean_price * 100.0) if mean_price > 0 else 0.0

        mean_vol = None
        if volumes is not None:
            vol_arr = np.asarray(volumes, dtype=float)
            if len(vol_arr) == len(regime_labels):
                regime_vols = vol_arr[mask]
                mean_vol = float(np.mean(regime_vols))

        # Liquidity score: inverse of spread (higher = more liquid)
        # Normalized to [0, 100] later
        liq_score = 1.0 / (1.0 + spread_pct) * 100.0

        profiles.append({
            "regime": k,
            "n_obs": n_obs,
            "spread_pct": float(spread_pct),
            "spread_abs": float(spread_abs),
            "mean_volume": mean_vol,
            "liquidity_score": float(liq_score),
            "insufficient_data": False,
        })

    # Compute regime-weighted average spread
    valid = [p for p in profiles if not p.get("insufficient_data")]
    if valid:
        total_obs = sum(p["n_obs"] for p in valid)
        weighted_spread = sum(
            p["spread_pct"] * p["n_obs"] / total_obs for p in valid
        )
    else:
        weighted_spread = 0.0

    return {
        "num_states": num_states,
        "profiles": profiles,
        "weighted_avg_spread_pct": float(weighted_spread),
        "n_total": len(prices),
    }


def compute_lvar_with_regime(
    var_value: float,
    regime_profiles: dict,
    current_regime_probs: np.ndarray,
    position_value: float = 1.0,
    alpha: float = 0.05,
    holding_period: int = 1,
) -> dict:
    """
    Compute regime-conditional LVaR using regime-weighted spreads.

    Combines regime-specific spread estimates with current regime
    probabilities to produce a probability-weighted LVaR.

    Args:
        var_value: Base VaR (negative, e.g. -3.2%).
        regime_profiles: Output of regime_liquidity_profile().
        current_regime_probs: Current regime probability vector.
        position_value: Position notional in USD.
        alpha: VaR confidence level.
        holding_period: Holding period in days.

    Returns:
        Dict with regime-weighted LVaR and per-regime breakdown.
    """
    probs = np.asarray(current_regime_probs, dtype=float)
    profiles = regime_profiles["profiles"]

    regime_lvars = []
    weighted_spread = 0.0
    weighted_spread_vol = 0.0

    for i, profile in enumerate(profiles):
        prob = float(probs[i]) if i < len(probs) else 0.0
        spread = profile.get("spread_pct") or 0.0
        weighted_spread += prob * spread

    # Use weighted spread for the LVaR calculation
    result = liquidity_adjusted_var(
        var_value=var_value,
        spread_pct=weighted_spread,
        spread_vol_pct=0.0,  # conservative: no vol adjustment in regime-weighted
        position_value=position_value,
        alpha=alpha,
        holding_period=holding_period,
    )

    # Per-regime breakdown
    for i, profile in enumerate(profiles):
        prob = float(probs[i]) if i < len(probs) else 0.0
        spread = profile.get("spread_pct") or 0.0
        regime_lvar = liquidity_adjusted_var(
            var_value=var_value,
            spread_pct=spread,
            position_value=position_value,
            alpha=alpha,
            holding_period=holding_period,
        )
        regime_lvars.append({
            "regime": profile["regime"],
            "probability": prob,
            "spread_pct": spread,
            "lvar": regime_lvar["lvar"],
            "liquidity_cost_pct": regime_lvar["liquidity_cost_pct"],
        })

    result["regime_breakdown"] = regime_lvars
    result["regime_weighted_spread_pct"] = float(weighted_spread)
    return result



# ── DexScreener-enhanced LVaR ──


def fetch_dexscreener_spread(pair_address: str) -> dict | None:
    """Fetch spread estimate from DexScreener liquidity data.

    Returns dict with spread_pct and spread_vol_pct compatible with
    liquidity_adjusted_var(), or None if DexScreener is unavailable.
    """
    try:
        from cortex.data.dexscreener import extract_liquidity_metrics

        metrics = extract_liquidity_metrics(pair_address)
        if metrics.get("error") or metrics.get("spread_pct") is None:
            return None

        return {
            "spread_pct": metrics["spread_pct"],
            "spread_vol_pct": metrics["spread_vol_pct"],
            "source": "dexscreener",
            "volume_24h": metrics.get("volume_24h"),
            "liquidity": metrics.get("liquidity"),
        }
    except Exception as e:
        logger.warning("DexScreener spread fetch failed: %s", e)
        return None


def liquidity_adjusted_var_with_dexscreener(
    var_value: float,
    pair_address: str | None = None,
    prices: np.ndarray | pd.Series | None = None,
    position_value: float = 1.0,
    alpha: float = 0.05,
    holding_period: int = 1,
) -> dict:
    """Compute LVaR using DexScreener liquidity data with Roll estimator fallback.

    Tries DexScreener first for real-time DEX spread data. Falls back to Roll
    estimator from price series if DexScreener is unavailable.

    Args:
        var_value: Base VaR (negative, e.g. -3.2%).
        pair_address: DEX pair address for live spread data.
        prices: Price series for Roll estimator fallback.
        position_value: Position notional in USD.
        alpha: VaR confidence level.
        holding_period: Holding period in days.
    """
    spread_data = None
    source_used = "none"

    if pair_address:
        spread_data = fetch_dexscreener_spread(pair_address)
        if spread_data:
            source_used = "dexscreener"

    if spread_data is None and prices is not None:
        p = np.asarray(prices, dtype=float)
        if len(p) >= 10:
            roll = estimate_spread(p, method="roll")
            spread_data = {
                "spread_pct": roll["spread_pct"],
                "spread_vol_pct": roll.get("spread_vol_pct", 0.0),
            }
            source_used = "roll"

    if spread_data is None:
        spread_data = {"spread_pct": 0.5, "spread_vol_pct": 0.15}
        source_used = "default"
        logger.warning("No spread data available, using conservative defaults")

    result = liquidity_adjusted_var(
        var_value=var_value,
        spread_pct=spread_data["spread_pct"],
        spread_vol_pct=spread_data["spread_vol_pct"],
        position_value=position_value,
        alpha=alpha,
        holding_period=holding_period,
    )
    result["spread_source"] = source_used
    return result


# ── On-chain enhanced LVaR (Wave 10) ──


def fetch_onchain_spread(token_address: str, limit: int = 50) -> dict | None:
    """Get realized spread from on-chain DEX swap data.

    Returns dict compatible with liquidity_adjusted_var() (spread_pct, spread_vol_pct),
    or None if on-chain data is unavailable.
    """
    try:
        from cortex.data.onchain_liquidity import (
            compute_realized_spread,
            fetch_swap_history,
        )

        swaps = fetch_swap_history(token_address, limit=limit)
        if len(swaps) < 3:
            return None

        result = compute_realized_spread(swaps)
        if result["realized_spread_pct"] <= 0:
            return None

        return {
            "spread_pct": result["realized_spread_pct"],
            "spread_vol_pct": result["realized_spread_vol_pct"],
            "n_swaps": result["n_swaps"],
            "by_dex": result["by_dex"],
        }
    except Exception as e:
        logger.warning("On-chain spread fetch failed: %s", e)
        return None


def liquidity_adjusted_var_with_onchain(
    var_value: float,
    token_address: str | None = None,
    pair_address: str | None = None,
    prices: np.ndarray | pd.Series | None = None,
    position_value: float = 1.0,
    alpha: float = 0.05,
    holding_period: int = 1,
) -> dict:
    """Compute LVaR using on-chain data with DexScreener and Roll fallbacks.

    Priority: on-chain realized spread > DexScreener > Roll estimator > defaults.

    Args:
        var_value: Base VaR (negative, e.g. -3.2%).
        token_address: Solana token mint for on-chain swap data.
        pair_address: DEX pair address for live spread data.
        prices: Price series for Roll estimator fallback.
        position_value: Position notional in USD.
        alpha: VaR confidence level.
        holding_period: Holding period in days.
    """
    spread_data = None
    source_used = "none"

    if token_address:
        spread_data = fetch_onchain_spread(token_address)
        if spread_data:
            source_used = "onchain"

    if spread_data is None and pair_address:
        spread_data = fetch_dexscreener_spread(pair_address)
        if spread_data:
            source_used = "dexscreener"

    if spread_data is None and prices is not None:
        p = np.asarray(prices, dtype=float)
        if len(p) >= 10:
            roll = estimate_spread(p, method="roll")
            spread_data = {
                "spread_pct": roll["spread_pct"],
                "spread_vol_pct": roll.get("spread_vol_pct", 0.0),
            }
            source_used = "roll"

    if spread_data is None:
        spread_data = {"spread_pct": 0.5, "spread_vol_pct": 0.15}
        source_used = "default"
        logger.warning("No spread data available, using conservative defaults")

    result = liquidity_adjusted_var(
        var_value=var_value,
        spread_pct=spread_data["spread_pct"],
        spread_vol_pct=spread_data.get("spread_vol_pct", 0.0),
        position_value=position_value,
        alpha=alpha,
        holding_period=holding_period,
    )
    result["spread_source"] = source_used
    if "by_dex" in spread_data:
        result["by_dex"] = spread_data["by_dex"]
    return result