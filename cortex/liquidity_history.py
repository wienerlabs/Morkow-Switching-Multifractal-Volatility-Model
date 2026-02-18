"""TVL/Liquidity time series storage and retrieval for LVaR.

Stores periodic liquidity snapshots (TVL, spread, depth) per pool and
provides historical spread statistics for the LVaR module.
"""
from __future__ import annotations

__all__ = [
    "take_snapshot",
    "store_snapshot",
    "get_snapshots",
    "get_historical_spread_stats",
    "get_tvl_series",
    "get_tracked_pools",
]

import logging
import time

import numpy as np

logger = logging.getLogger(__name__)


def take_snapshot(pool_address: str) -> dict | None:
    """Capture a liquidity snapshot for a single pool.

    Returns a dict with TVL, spread estimate, bid/ask depth, timestamp,
    or None if pool data is unavailable.
    """
    try:
        from cortex.data.onchain_liquidity import build_liquidity_depth_curve, get_clmm_tick_data

        tick_data = get_clmm_tick_data(pool_address)
        if not tick_data:
            return None

        depth = build_liquidity_depth_curve(pool_address)

        tvl = tick_data.get("tvl", 0.0)
        current_price = tick_data.get("current_price", 0.0)
        total_bid = depth.get("total_bid_liquidity", 0.0)
        total_ask = depth.get("total_ask_liquidity", 0.0)
        imbalance = depth.get("depth_imbalance", 0.0)

        # Estimate spread from depth curve if we have ticks
        spread_pct = 0.0
        if depth.get("bid_prices") and depth.get("ask_prices"):
            best_bid = depth["bid_prices"][0] if depth["bid_prices"] else 0.0
            best_ask = depth["ask_prices"][0] if depth["ask_prices"] else 0.0
            mid = (best_bid + best_ask) / 2.0 if (best_bid + best_ask) > 0 else 0.0
            if mid > 0:
                spread_pct = (best_ask - best_bid) / mid * 100.0

        return {
            "pool": pool_address,
            "timestamp": time.time(),
            "tvl": float(tvl),
            "current_price": float(current_price),
            "spread_pct": float(max(spread_pct, 0.0)),
            "total_bid_liquidity": float(total_bid),
            "total_ask_liquidity": float(total_ask),
            "depth_imbalance": float(imbalance),
        }
    except Exception as e:
        logger.warning("Snapshot failed for pool %s: %s", pool_address, e)
        return None


def store_snapshot(pool_address: str, snapshot: dict) -> None:
    """Append a snapshot to the pool's time series in the liquidity store."""
    from api.stores import _liquidity_store
    from cortex.config import LIQUIDITY_SNAPSHOT_RETENTION_HOURS

    existing = _liquidity_store.get(pool_address, {"snapshots": []})
    snapshots: list[dict] = existing.get("snapshots", [])
    snapshots.append(snapshot)

    # Trim old snapshots beyond retention window
    cutoff = time.time() - (LIQUIDITY_SNAPSHOT_RETENTION_HOURS * 3600)
    snapshots = [s for s in snapshots if s["timestamp"] >= cutoff]

    _liquidity_store[pool_address] = {"snapshots": snapshots}


def get_snapshots(
    pool_address: str,
    start_time: float | None = None,
    end_time: float | None = None,
) -> list[dict]:
    """Retrieve liquidity snapshots for a pool within a time range."""
    from api.stores import _liquidity_store

    data = _liquidity_store.get(pool_address)
    if not data:
        return []

    snapshots = data.get("snapshots", [])
    if start_time is not None:
        snapshots = [s for s in snapshots if s["timestamp"] >= start_time]
    if end_time is not None:
        snapshots = [s for s in snapshots if s["timestamp"] <= end_time]

    return sorted(snapshots, key=lambda s: s["timestamp"])


def get_historical_spread_stats(
    pool_address: str,
    lookback_hours: float = 24.0,
) -> dict:
    """Compute spread statistics from stored time series for LVaR input.

    Returns mean spread, spread volatility, and sample count —
    ready to feed into liquidity_adjusted_var().
    """
    cutoff = time.time() - (lookback_hours * 3600)
    snapshots = get_snapshots(pool_address, start_time=cutoff)

    if len(snapshots) < 2:
        return {
            "spread_pct": 0.0,
            "spread_vol_pct": 0.0,
            "n_samples": len(snapshots),
            "source": "liquidity_history",
            "sufficient_data": False,
        }

    spreads = np.array([s["spread_pct"] for s in snapshots])

    return {
        "spread_pct": float(np.mean(spreads)),
        "spread_vol_pct": float(np.std(spreads)),
        "n_samples": len(snapshots),
        "min_spread": float(np.min(spreads)),
        "max_spread": float(np.max(spreads)),
        "source": "liquidity_history",
        "sufficient_data": True,
    }


def get_tvl_series(
    pool_address: str,
    lookback_hours: float = 24.0,
) -> dict:
    """Get TVL time series for a pool."""
    cutoff = time.time() - (lookback_hours * 3600)
    snapshots = get_snapshots(pool_address, start_time=cutoff)

    if not snapshots:
        return {"timestamps": [], "tvl_values": [], "n_samples": 0}

    return {
        "timestamps": [s["timestamp"] for s in snapshots],
        "tvl_values": [s["tvl"] for s in snapshots],
        "n_samples": len(snapshots),
        "current_tvl": snapshots[-1]["tvl"],
        "min_tvl": min(s["tvl"] for s in snapshots),
        "max_tvl": max(s["tvl"] for s in snapshots),
    }


def get_tracked_pools() -> list[str]:
    """Return list of pool addresses currently being tracked."""
    from api.stores import _liquidity_store
    return list(_liquidity_store.keys())
