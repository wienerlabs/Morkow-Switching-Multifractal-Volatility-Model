"""Tests for Corwin-Schultz (2012) high-low spread estimator in cortex/liquidity.py."""

import numpy as np
import pytest

from cortex.liquidity import estimate_spread


@pytest.fixture
def ohlc_data():
    """Synthetic OHLC with known spread embedded."""
    rng = np.random.RandomState(42)
    n = 200
    mid = 100.0 + np.cumsum(rng.randn(n) * 0.3)
    spread_half = 0.15
    highs = mid + spread_half + rng.exponential(0.1, n)
    lows = mid - spread_half - rng.exponential(0.1, n)
    closes = mid + rng.randn(n) * 0.05
    return closes, highs, lows


class TestCorwinSchultzFullSample:
    def test_returns_expected_keys(self, ohlc_data):
        prices, highs, lows = ohlc_data
        r = estimate_spread(prices, method="high_low", highs=highs, lows=lows)
        assert r["method"] == "high_low"
        assert "spread_pct" in r
        assert "spread_abs" in r
        assert "n_obs" in r
        assert "n_pairs" in r
        assert "mean_alpha" in r
        assert "pct_negative_alpha" in r

    def test_spread_non_negative(self, ohlc_data):
        prices, highs, lows = ohlc_data
        r = estimate_spread(prices, method="high_low", highs=highs, lows=lows)
        assert r["spread_pct"] >= 0.0
        assert r["spread_abs"] >= 0.0

    def test_spread_reasonable_magnitude(self, ohlc_data):
        prices, highs, lows = ohlc_data
        r = estimate_spread(prices, method="high_low", highs=highs, lows=lows)
        assert r["spread_pct"] < 10.0  # should be well under 10%

    def test_n_pairs_equals_n_minus_1(self, ohlc_data):
        prices, highs, lows = ohlc_data
        r = estimate_spread(prices, method="high_low", highs=highs, lows=lows)
        assert r["n_pairs"] == len(prices) - 1

    def test_zero_spread_when_high_equals_low(self):
        """When H==L (no intraday range), spread should be 0."""
        prices = np.linspace(100, 110, 50)
        highs = prices.copy()
        lows = prices.copy()
        # log(H/L) = 0 → beta=0, gamma=0 → alpha undefined but spread_frac=0
        # This will produce nan from log(1)=0, sqrt(0)=0, etc.
        # The implementation should handle gracefully
        r = estimate_spread(prices, method="high_low", highs=highs, lows=lows)
        assert r["spread_pct"] >= 0.0  # clamped non-negative


class TestCorwinSchultzRolling:
    def test_rolling_returns_expected_keys(self, ohlc_data):
        prices, highs, lows = ohlc_data
        r = estimate_spread(prices, method="high_low", highs=highs, lows=lows, window=20)
        assert r["method"] == "high_low"
        assert "rolling_spreads" in r
        assert r["window"] == 20
        assert "spread_vol_pct" in r
        assert "spread_vol_abs" in r

    def test_rolling_spread_non_negative(self, ohlc_data):
        prices, highs, lows = ohlc_data
        r = estimate_spread(prices, method="high_low", highs=highs, lows=lows, window=20)
        assert r["spread_pct"] >= 0.0

    def test_small_window_raises(self, ohlc_data):
        prices, highs, lows = ohlc_data
        with pytest.raises(ValueError, match="Rolling window must be"):
            estimate_spread(prices, method="high_low", highs=highs, lows=lows, window=3)


class TestCorwinSchultzValidation:
    def test_missing_highs_raises(self):
        prices = np.arange(50, dtype=float) + 100
        with pytest.raises(ValueError, match="requires 'highs' and 'lows'"):
            estimate_spread(prices, method="high_low")

    def test_missing_lows_raises(self):
        prices = np.arange(50, dtype=float) + 100
        with pytest.raises(ValueError, match="requires 'highs' and 'lows'"):
            estimate_spread(prices, method="high_low", highs=prices)

    def test_length_mismatch_raises(self):
        prices = np.arange(50, dtype=float) + 100
        highs = np.arange(40, dtype=float) + 101
        lows = np.arange(50, dtype=float) + 99
        with pytest.raises(ValueError, match="must have the same length"):
            estimate_spread(prices, method="high_low", highs=highs, lows=lows)


class TestCorwinSchultzMath:
    def test_wider_spread_detected(self):
        """Wider high-low range should produce larger estimated spread."""
        rng = np.random.RandomState(7)
        n = 100
        mid = 100.0 + np.cumsum(rng.randn(n) * 0.2)

        # Narrow spread
        h_narrow = mid + 0.05
        l_narrow = mid - 0.05
        r_narrow = estimate_spread(mid, method="high_low", highs=h_narrow, lows=l_narrow)

        # Wide spread
        h_wide = mid + 0.50
        l_wide = mid - 0.50
        r_wide = estimate_spread(mid, method="high_low", highs=h_wide, lows=l_wide)

        assert r_wide["spread_pct"] > r_narrow["spread_pct"]

    def test_negative_alpha_percentage_bounded(self, ohlc_data):
        prices, highs, lows = ohlc_data
        r = estimate_spread(prices, method="high_low", highs=highs, lows=lows)
        assert 0.0 <= r["pct_negative_alpha"] <= 100.0

