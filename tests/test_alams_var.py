"""
Tests for A-LAMS-VaR model.

Covers:
- Parameter estimation (fit)
- Hamilton filter validity
- Asymmetry effect (δ)
- VaR monotonicity across confidence levels
- Liquidity adjustment
- Edge cases and error handling
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

# Load alams_var directly from file to avoid triggering
# agent/src/models/__init__.py which requires xgboost.
_module_path = (
    Path(__file__).resolve().parent.parent
    / "agent" / "src" / "models" / "risk" / "alams_var.py"
)
_spec = importlib.util.spec_from_file_location("alams_var", _module_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["alams_var"] = _mod
_spec.loader.exec_module(_mod)

ALAMSVaRModel = _mod.ALAMSVaRModel
ALAMSConfig = _mod.ALAMSConfig
LiquidityConfig = _mod.LiquidityConfig


# ============= Fixtures =============


def _generate_regime_returns(n: int = 500, seed: int = 42) -> np.ndarray:
    """Generate synthetic returns with two distinct regimes."""
    rng = np.random.RandomState(seed)
    # Low vol regime: ~60% of data
    low_vol = rng.normal(0.0005, 0.01, int(n * 0.6))
    # High vol regime: ~40% of data
    high_vol = rng.normal(-0.001, 0.04, int(n * 0.4))
    # Alternate between regimes
    returns = np.empty(n)
    returns[: len(low_vol)] = low_vol
    returns[len(low_vol) : len(low_vol) + len(high_vol)] = high_vol
    # Fill remaining
    remaining = n - len(low_vol) - len(high_vol)
    if remaining > 0:
        returns[-remaining:] = rng.normal(0, 0.02, remaining)
    return returns


@pytest.fixture
def returns():
    return _generate_regime_returns()


@pytest.fixture
def fitted_model(returns):
    model = ALAMSVaRModel(
        config=ALAMSConfig(max_iter=50, tol=1e-4)
    )
    model.fit(returns)
    return model


# ============= Estimation Tests =============


class TestFit:
    def test_fit_returns_diagnostics(self, returns):
        model = ALAMSVaRModel(config=ALAMSConfig(max_iter=50))
        diag = model.fit(returns)

        assert "log_likelihood" in diag
        assert "delta" in diag
        assert "aic" in diag
        assert "bic" in diag
        assert diag["n_obs"] == len(returns)
        assert diag["n_regimes"] == 5

    def test_fit_sets_is_fitted(self, returns):
        model = ALAMSVaRModel(config=ALAMSConfig(max_iter=50))
        assert not model.is_fitted
        model.fit(returns)
        assert model.is_fitted

    def test_sigma_monotonically_increasing(self, fitted_model):
        """Regime sigmas should be sorted: sigma_0 <= sigma_1 <= ... <= sigma_{K-1}."""
        sigmas = fitted_model.sigma
        for i in range(len(sigmas) - 1):
            assert sigmas[i] <= sigmas[i + 1] + 1e-10

    def test_delta_in_valid_range(self, fitted_model):
        """Asymmetry parameter should be in (0, 0.5]."""
        assert 0.0 < fitted_model.delta <= 0.5

    def test_min_observations_enforced(self):
        model = ALAMSVaRModel(config=ALAMSConfig(min_observations=100))
        with pytest.raises(ValueError, match="at least 100"):
            model.fit(np.random.randn(50))

    def test_fit_with_custom_config(self, returns):
        config = ALAMSConfig(n_regimes=3, max_iter=30)
        model = ALAMSVaRModel(config=config)
        diag = model.fit(returns)
        assert diag["n_regimes"] == 3
        assert len(model.mu) == 3
        assert len(model.sigma) == 3


# ============= Hamilton Filter Tests =============


class TestHamiltonFilter:
    def test_filtered_probs_shape(self, fitted_model, returns):
        probs = fitted_model.filter(returns)
        assert probs.shape == (len(returns), fitted_model.K)

    def test_filtered_probs_sum_to_one(self, fitted_model, returns):
        probs = fitted_model.filter(returns)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_filtered_probs_non_negative(self, fitted_model, returns):
        probs = fitted_model.filter(returns)
        assert np.all(probs >= 0)

    def test_filter_requires_fitted(self):
        model = ALAMSVaRModel()
        with pytest.raises(RuntimeError, match="fitted"):
            model.filter(np.random.randn(100))

    def test_regime_probabilities_returns_last(self, fitted_model, returns):
        probs = fitted_model.filter(returns)
        current = fitted_model.get_regime_probabilities()
        np.testing.assert_array_equal(current, probs[-1])


# ============= Asymmetry Tests =============


class TestAsymmetry:
    def test_negative_returns_shift_to_high_vol(self, fitted_model):
        """After negative returns, higher regimes should get more probability."""
        P_neutral = fitted_model._get_transition_matrix(0.01)  # positive return
        P_neg = fitted_model._get_transition_matrix(-0.01)      # negative return

        K = fitted_model.K
        # For each starting regime, the expected next-regime index should be
        # higher (or equal) after a negative return
        for i in range(K):
            expected_neutral = np.sum(P_neutral[i] * np.arange(K))
            expected_neg = np.sum(P_neg[i] * np.arange(K))
            assert expected_neg >= expected_neutral - 1e-10

    def test_transition_rows_sum_to_one(self, fitted_model):
        for r in [-0.05, 0.0, 0.05]:
            P = fitted_model._get_transition_matrix(r)
            row_sums = P.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_transition_non_negative(self, fitted_model):
        for r in [-0.1, 0.0, 0.1]:
            P = fitted_model._get_transition_matrix(r)
            assert np.all(P >= 0)


# ============= VaR Tests =============


class TestVaR:
    def test_var_positive(self, fitted_model):
        """VaR should be a positive number (loss magnitude)."""
        var95 = fitted_model.calculate_var(0.95)
        var99 = fitted_model.calculate_var(0.99)
        assert var95 > 0
        assert var99 > 0

    def test_var_monotonic_in_confidence(self, fitted_model):
        """Higher confidence -> larger VaR (more extreme tail)."""
        var90 = fitted_model.calculate_var(0.90)
        var95 = fitted_model.calculate_var(0.95)
        var99 = fitted_model.calculate_var(0.99)
        assert var99 >= var95
        assert var95 >= var90

    def test_var_with_inline_returns(self, fitted_model, returns):
        """calculate_var should accept returns param for inline filtering."""
        var1 = fitted_model.calculate_var(0.95)
        var2 = fitted_model.calculate_var(0.95, returns=returns)
        # Both should be valid positive values
        assert var2 > 0

    def test_var_requires_fitted(self):
        model = ALAMSVaRModel()
        with pytest.raises(RuntimeError):
            model.calculate_var(0.95)


# ============= Liquidity Adjustment Tests =============


class TestLiquidityAdjustedVaR:
    def test_zero_trade_no_slippage(self, fitted_model):
        result = fitted_model.calculate_liquidity_adjusted_var(
            confidence=0.95, trade_size_usd=0.0, pool_depth_usd=1e6
        )
        assert result["slippage_component"] == 0.0
        assert result["var_total"] == result["var_pure"]

    def test_slippage_increases_with_trade_size(self, fitted_model):
        r1 = fitted_model.calculate_liquidity_adjusted_var(
            0.95, trade_size_usd=1_000, pool_depth_usd=1_000_000
        )
        r2 = fitted_model.calculate_liquidity_adjusted_var(
            0.95, trade_size_usd=100_000, pool_depth_usd=1_000_000
        )
        assert r2["slippage_component"] > r1["slippage_component"]

    def test_var_total_gte_var_pure(self, fitted_model):
        result = fitted_model.calculate_liquidity_adjusted_var(
            0.95, trade_size_usd=10_000, pool_depth_usd=500_000
        )
        assert result["var_total"] >= result["var_pure"]

    def test_response_contains_all_fields(self, fitted_model):
        result = fitted_model.calculate_liquidity_adjusted_var(
            0.95, trade_size_usd=5_000, pool_depth_usd=1_000_000
        )
        expected_keys = {
            "var_pure", "slippage_component", "var_total",
            "confidence", "current_regime", "regime_probs",
            "delta", "regime_means", "regime_sigmas",
        }
        assert expected_keys == set(result.keys())

    def test_slippage_capped(self, fitted_model):
        """Slippage should not exceed max_slippage_bps."""
        lc = fitted_model.liquidity_config
        result = fitted_model.calculate_liquidity_adjusted_var(
            0.99, trade_size_usd=1e9, pool_depth_usd=1_000
        )
        max_slippage_decimal = lc.max_slippage_bps / 10000.0
        assert result["slippage_component"] <= max_slippage_decimal + 1e-10


# ============= Edge Cases =============


class TestEdgeCases:
    def test_constant_returns(self):
        """Model should handle near-zero variance data."""
        model = ALAMSVaRModel(config=ALAMSConfig(max_iter=30, min_observations=50))
        returns = np.full(100, 0.0001)  # Nearly constant
        # Should not raise
        model.fit(returns)
        assert model.is_fitted

    def test_large_negative_shock(self, fitted_model):
        """A large negative return should shift regime to high-vol."""
        # Feed normal returns first, then a -20% shock
        normal = np.random.RandomState(42).normal(0, 0.01, 50)
        shock = np.append(normal, [-0.20])
        fitted_model.filter(shock)
        probs = fitted_model.get_regime_probabilities()
        # Higher regimes should have meaningful probability
        high_vol_prob = probs[-2:].sum()
        assert high_vol_prob > 0.01

    def test_get_current_regime(self, fitted_model, returns):
        fitted_model.filter(returns)
        regime = fitted_model.get_current_regime()
        assert 0 <= regime < fitted_model.K

    def test_summary_before_fit(self):
        model = ALAMSVaRModel()
        s = model.summary()
        assert s["is_fitted"] is False

    def test_summary_after_fit(self, fitted_model):
        s = fitted_model.summary()
        assert s["is_fitted"] is True
        assert "var_95" in s
        assert "var_99" in s
        assert len(s["regime_means"]) == fitted_model.K

    def test_ergodic_distribution_sums_to_one(self, fitted_model):
        pi = ALAMSVaRModel._ergodic_distribution(fitted_model.P_base)
        np.testing.assert_allclose(pi.sum(), 1.0, atol=1e-6)
        assert np.all(pi > 0)
