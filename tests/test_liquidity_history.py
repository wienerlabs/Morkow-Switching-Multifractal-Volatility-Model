"""Tests for cortex/liquidity_history.py — TVL/Liquidity time series collection."""

import time
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _clear_liquidity_store():
    """Reset the liquidity store between tests."""
    from api.stores import _liquidity_store
    keys = list(_liquidity_store.keys())
    for k in keys:
        _liquidity_store._data.pop(k, None)
    yield
    keys = list(_liquidity_store.keys())
    for k in keys:
        _liquidity_store._data.pop(k, None)


@pytest.fixture
def sample_snapshot():
    return {
        "pool": "pool_abc",
        "timestamp": time.time(),
        "tvl": 1_500_000.0,
        "current_price": 150.0,
        "spread_pct": 0.12,
        "total_bid_liquidity": 500_000.0,
        "total_ask_liquidity": 480_000.0,
        "depth_imbalance": 0.02,
    }


class TestStoreSnapshot:
    def test_stores_and_retrieves(self, sample_snapshot):
        from cortex.liquidity_history import get_snapshots, store_snapshot

        store_snapshot("pool_abc", sample_snapshot)
        result = get_snapshots("pool_abc")
        assert len(result) == 1
        assert result[0]["tvl"] == 1_500_000.0

    def test_multiple_snapshots_ordered(self):
        from cortex.liquidity_history import get_snapshots, store_snapshot

        now = time.time()
        for i in range(5):
            store_snapshot("pool_abc", {
                "pool": "pool_abc",
                "timestamp": now + i * 60,
                "tvl": 1_000_000.0 + i * 100_000,
                "current_price": 100.0,
                "spread_pct": 0.1 + i * 0.01,
                "total_bid_liquidity": 500_000.0,
                "total_ask_liquidity": 480_000.0,
                "depth_imbalance": 0.0,
            })

        result = get_snapshots("pool_abc")
        assert len(result) == 5
        assert result[0]["timestamp"] < result[-1]["timestamp"]

    def test_time_range_filter(self):
        from cortex.liquidity_history import get_snapshots, store_snapshot

        now = time.time()
        for i in range(10):
            store_snapshot("pool_abc", {
                "pool": "pool_abc",
                "timestamp": now + i * 3600,
                "tvl": 1_000_000.0,
                "current_price": 100.0,
                "spread_pct": 0.1,
                "total_bid_liquidity": 500_000.0,
                "total_ask_liquidity": 480_000.0,
                "depth_imbalance": 0.0,
            })

        # Get middle 4 snapshots
        result = get_snapshots(
            "pool_abc",
            start_time=now + 3 * 3600,
            end_time=now + 6 * 3600,
        )
        assert len(result) == 4

    def test_retention_trims_old(self):
        from cortex.liquidity_history import get_snapshots, store_snapshot

        old_time = time.time() - (200 * 3600)  # 200 hours ago (retention is 168h)
        store_snapshot("pool_abc", {
            "pool": "pool_abc",
            "timestamp": old_time,
            "tvl": 1_000_000.0,
            "current_price": 100.0,
            "spread_pct": 0.1,
            "total_bid_liquidity": 500_000.0,
            "total_ask_liquidity": 480_000.0,
            "depth_imbalance": 0.0,
        })

        # Now add a current snapshot — old one should be trimmed
        store_snapshot("pool_abc", {
            "pool": "pool_abc",
            "timestamp": time.time(),
            "tvl": 2_000_000.0,
            "current_price": 200.0,
            "spread_pct": 0.2,
            "total_bid_liquidity": 600_000.0,
            "total_ask_liquidity": 580_000.0,
            "depth_imbalance": 0.01,
        })

        result = get_snapshots("pool_abc")
        assert len(result) == 1
        assert result[0]["tvl"] == 2_000_000.0


class TestHistoricalSpreadStats:
    def test_insufficient_data(self):
        from cortex.liquidity_history import get_historical_spread_stats

        stats = get_historical_spread_stats("pool_xyz")
        assert stats["sufficient_data"] is False
        assert stats["n_samples"] == 0

    def test_spread_statistics(self):
        from cortex.liquidity_history import get_historical_spread_stats, store_snapshot

        now = time.time()
        spreads = [0.10, 0.12, 0.08, 0.15, 0.11, 0.09, 0.13, 0.14, 0.10, 0.12]
        for i, s in enumerate(spreads):
            store_snapshot("pool_abc", {
                "pool": "pool_abc",
                "timestamp": now - (len(spreads) - i) * 60,
                "tvl": 1_000_000.0,
                "current_price": 100.0,
                "spread_pct": s,
                "total_bid_liquidity": 500_000.0,
                "total_ask_liquidity": 480_000.0,
                "depth_imbalance": 0.0,
            })

        stats = get_historical_spread_stats("pool_abc", lookback_hours=1.0)
        assert stats["sufficient_data"] is True
        assert stats["n_samples"] == 10
        assert abs(stats["spread_pct"] - np.mean(spreads)) < 1e-10
        assert abs(stats["spread_vol_pct"] - np.std(spreads)) < 1e-10
        assert stats["min_spread"] == min(spreads)
        assert stats["max_spread"] == max(spreads)
        assert stats["source"] == "liquidity_history"


class TestTVLSeries:
    def test_tvl_series(self):
        from cortex.liquidity_history import get_tvl_series, store_snapshot

        now = time.time()
        tvls = [1_000_000.0, 1_100_000.0, 950_000.0, 1_200_000.0]
        for i, tvl in enumerate(tvls):
            store_snapshot("pool_abc", {
                "pool": "pool_abc",
                "timestamp": now - (len(tvls) - i) * 60,
                "tvl": tvl,
                "current_price": 100.0,
                "spread_pct": 0.1,
                "total_bid_liquidity": 500_000.0,
                "total_ask_liquidity": 480_000.0,
                "depth_imbalance": 0.0,
            })

        result = get_tvl_series("pool_abc", lookback_hours=1.0)
        assert result["n_samples"] == 4
        assert result["current_tvl"] == 1_200_000.0
        assert result["min_tvl"] == 950_000.0
        assert result["max_tvl"] == 1_200_000.0

    def test_empty_pool(self):
        from cortex.liquidity_history import get_tvl_series

        result = get_tvl_series("nonexistent_pool")
        assert result["n_samples"] == 0
        assert result["timestamps"] == []


class TestGetTrackedPools:
    def test_returns_stored_pools(self):
        from cortex.liquidity_history import get_tracked_pools, store_snapshot

        now = time.time()
        for pool in ["pool_a", "pool_b", "pool_c"]:
            store_snapshot(pool, {
                "pool": pool,
                "timestamp": now,
                "tvl": 1_000_000.0,
                "current_price": 100.0,
                "spread_pct": 0.1,
                "total_bid_liquidity": 500_000.0,
                "total_ask_liquidity": 480_000.0,
                "depth_imbalance": 0.0,
            })

        pools = get_tracked_pools()
        assert set(pools) == {"pool_a", "pool_b", "pool_c"}


class TestTakeSnapshot:
    @patch("cortex.data.onchain_liquidity.build_liquidity_depth_curve")
    @patch("cortex.data.onchain_liquidity.get_clmm_tick_data")
    def test_successful_snapshot(self, mock_tick, mock_depth):
        from cortex.liquidity_history import take_snapshot

        mock_tick.return_value = {
            "pool": "pool_abc",
            "current_price": 150.0,
            "tvl": 2_000_000.0,
            "ticks": [{"price": 149.5, "liquidity": 100_000}],
        }
        mock_depth.return_value = {
            "bid_prices": [149.5, 149.0],
            "ask_prices": [150.5, 151.0],
            "total_bid_liquidity": 500_000.0,
            "total_ask_liquidity": 480_000.0,
            "depth_imbalance": 0.02,
        }

        result = take_snapshot("pool_abc")
        assert result is not None
        assert result["tvl"] == 2_000_000.0
        assert result["current_price"] == 150.0
        assert result["spread_pct"] > 0

    @patch("cortex.data.onchain_liquidity.get_clmm_tick_data")
    def test_unavailable_pool(self, mock_tick):
        from cortex.liquidity_history import take_snapshot

        mock_tick.return_value = None
        result = take_snapshot("nonexistent_pool")
        assert result is None


class TestTaskScheduler:
    def test_start_stop_functions_exist(self):
        from api.tasks import (
            start_liquidity_snapshot_collector,
            stop_liquidity_snapshot_collector,
        )
        assert callable(start_liquidity_snapshot_collector)
        assert callable(stop_liquidity_snapshot_collector)

    def test_disabled_when_flag_off(self):
        with patch("cortex.config.LIQUIDITY_SNAPSHOT_ENABLED", False):
            from api import tasks
            tasks._liquidity_snapshot_task = None
            tasks.start_liquidity_snapshot_collector()
            assert tasks._liquidity_snapshot_task is None
