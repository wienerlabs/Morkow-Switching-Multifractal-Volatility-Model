"""Tests for cortex/liquidity.py — Liquidity-Adjusted VaR module."""

import numpy as np
import pytest

from cortex.liquidity import (
    compute_lvar_with_regime,
    estimate_spread,
    liquidity_adjusted_var,
    market_impact_cost,
    regime_liquidity_profile,
)


@pytest.fixture
def price_series():
    """Synthetic price series with bid-ask bounce (negative serial covariance)."""
    rng = np.random.RandomState(99)
    n = 300
    # Random walk + bid-ask bounce
    mid = 100.0 + np.cumsum(rng.randn(n) * 0.5)
    bounce = rng.choice([-0.05, 0.05], size=n)
    return mid + bounce


@pytest.fixture
def regime_labels():
    """5-state regime labels for 300 observations."""
    rng = np.random.RandomState(99)
    return rng.choice([1, 2, 3, 4, 5], size=300, p=[0.3, 0.25, 0.2, 0.15, 0.1])


class TestEstimateSpread:
    def test_full_sample_returns_dict(self, price_series):
        result = estimate_spread(price_series)
        assert "spread_pct" in result
        assert "spread_abs" in result
        assert "method" in result
        assert result["method"] == "roll"
        assert result["n_obs"] == len(price_series)

    def test_spread_is_non_negative(self, price_series):
        result = estimate_spread(price_series)
        assert result["spread_pct"] >= 0.0
        assert result["spread_abs"] >= 0.0

    def test_rolling_window(self, price_series):
        result = estimate_spread(price_series, window=30)
        assert "rolling_spreads" in result
        assert result["window"] == 30
        assert result["spread_vol_pct"] >= 0.0

    def test_too_few_prices_raises(self):
        with pytest.raises(ValueError, match="Need ≥10"):
            estimate_spread(np.array([1.0, 2.0, 3.0]))

    def test_small_window_raises(self, price_series):
        with pytest.raises(ValueError, match="Rolling window must be ≥5"):
            estimate_spread(price_series, window=3)

    def test_unknown_method_raises(self, price_series):
        with pytest.raises(ValueError, match="Unknown method"):
            estimate_spread(price_series, method="invalid")


class TestLiquidityAdjustedVar:
    def test_lvar_worse_than_var(self):
        result = liquidity_adjusted_var(var_value=-3.0, spread_pct=0.5, spread_vol_pct=0.1)
        assert result["lvar"] < result["base_var"]  # more negative

    def test_zero_spread_equals_var(self):
        result = liquidity_adjusted_var(var_value=-3.0, spread_pct=0.0, spread_vol_pct=0.0)
        np.testing.assert_allclose(result["lvar"], result["base_var"], atol=1e-10)

    def test_holding_period_scaling(self):
        r1 = liquidity_adjusted_var(var_value=-3.0, spread_pct=0.5, holding_period=1)
        r4 = liquidity_adjusted_var(var_value=-3.0, spread_pct=0.5, holding_period=4)
        # 4-day LC should be ~2x 1-day LC (sqrt(4) = 2)
        np.testing.assert_allclose(
            r4["liquidity_cost_pct"], r1["liquidity_cost_pct"] * 2.0, atol=1e-10
        )

    def test_lvar_ratio_gt_one(self):
        result = liquidity_adjusted_var(var_value=-3.0, spread_pct=0.5)
        assert result["lvar_ratio"] > 1.0

    def test_position_value_scaling(self):
        r1 = liquidity_adjusted_var(var_value=-3.0, spread_pct=0.5, position_value=100_000)
        r2 = liquidity_adjusted_var(var_value=-3.0, spread_pct=0.5, position_value=200_000)
        np.testing.assert_allclose(r2["liquidity_cost_abs"], r1["liquidity_cost_abs"] * 2.0, atol=1e-6)


class TestMarketImpactCost:
    def test_basic_impact(self):
        result = market_impact_cost(sigma=0.03, trade_size_usd=100_000, adv_usd=1_000_000)
        assert result["impact_pct"] > 0
        assert result["impact_usd"] > 0

    def test_larger_trade_more_impact(self):
        r1 = market_impact_cost(sigma=0.03, trade_size_usd=10_000, adv_usd=1_000_000)
        r2 = market_impact_cost(sigma=0.03, trade_size_usd=100_000, adv_usd=1_000_000)
        assert r2["impact_pct"] > r1["impact_pct"]

    def test_participation_warning(self):
        result = market_impact_cost(sigma=0.03, trade_size_usd=200_000, adv_usd=1_000_000, participation_rate=0.10)
        assert result["participation_warning"] is True

    def test_zero_adv_raises(self):
        with pytest.raises(ValueError, match="ADV must be positive"):
            market_impact_cost(sigma=0.03, trade_size_usd=100_000, adv_usd=0)

    def test_negative_trade_raises(self):
        with pytest.raises(ValueError, match="Trade size must be positive"):
            market_impact_cost(sigma=0.03, trade_size_usd=-100, adv_usd=1_000_000)


class TestRegimeLiquidityProfile:
    def test_output_structure(self, price_series, regime_labels):
        result = regime_liquidity_profile(price_series, regime_labels, num_states=5)
        assert result["num_states"] == 5
        assert len(result["profiles"]) == 5
        assert result["n_total"] == len(price_series)

    def test_weighted_spread_non_negative(self, price_series, regime_labels):
        result = regime_liquidity_profile(price_series, regime_labels, num_states=5)
        assert result["weighted_avg_spread_pct"] >= 0.0

    def test_length_mismatch_raises(self, price_series):
        with pytest.raises(ValueError, match="must have same length"):
            regime_liquidity_profile(price_series, np.array([1, 2, 3]), num_states=3)

    def test_insufficient_data_flagged(self):
        prices = np.arange(20, dtype=float) + 100.0
        labels = np.array([1]*15 + [2]*5)
        result = regime_liquidity_profile(prices, labels, num_states=2)
        regime_2 = [p for p in result["profiles"] if p["regime"] == 2][0]
        assert regime_2["insufficient_data"] is True


class TestComputeLvarWithRegime:
    def test_regime_weighted_lvar(self, price_series, regime_labels):
        profiles = regime_liquidity_profile(price_series, regime_labels, num_states=5)
        probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        result = compute_lvar_with_regime(
            var_value=-3.0, regime_profiles=profiles, current_regime_probs=probs,
        )
        assert result["lvar"] <= result["base_var"]
        assert len(result["regime_breakdown"]) == 5

    def test_regime_breakdown_probabilities(self, price_series, regime_labels):
        profiles = regime_liquidity_profile(price_series, regime_labels, num_states=5)
        probs = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        result = compute_lvar_with_regime(
            var_value=-3.0, regime_profiles=profiles, current_regime_probs=probs,
        )
        for rb in result["regime_breakdown"]:
            assert "regime" in rb
            assert "probability" in rb
            assert "lvar" in rb

