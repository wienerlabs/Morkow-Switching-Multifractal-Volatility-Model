"""Tests for portfolio_var.py."""

import numpy as np
import pandas as pd
import pytest

from cortex import portfolio as pv


@pytest.fixture(scope="module")
def portfolio_model(multivariate_returns):
    """Calibrated multivariate model for portfolio tests."""
    return pv.calibrate_multivariate(
        multivariate_returns, num_states=5, method="empirical"
    )


class TestCalibrateMultivariate:
    def test_output_keys(self, portfolio_model):
        m = portfolio_model
        assert "assets" in m
        assert "regime_cov" in m
        assert "regime_corr" in m
        assert "current_probs" in m
        assert "P_matrix" in m

    def test_regime_cov_shape(self, portfolio_model):
        m = portfolio_model
        assert m["regime_cov"].shape == (5, 3, 3)
        assert m["regime_corr"].shape == (5, 3, 3)

    def test_current_probs_sum_to_one(self, portfolio_model):
        np.testing.assert_allclose(portfolio_model["current_probs"].sum(), 1.0, atol=1e-6)

    def test_too_few_assets_raises(self):
        df = pd.DataFrame({"A": np.random.randn(100)})
        with pytest.raises(ValueError, match="Need ≥2 assets"):
            pv.calibrate_multivariate(df)

    def test_too_few_obs_raises(self):
        df = pd.DataFrame({"A": [1.0, 2.0], "B": [3.0, 4.0]})
        with pytest.raises(ValueError, match="Need ≥30"):
            pv.calibrate_multivariate(df)


class TestPortfolioVar:
    def test_var_is_negative(self, portfolio_model):
        result = pv.portfolio_var(portfolio_model, {"A": 0.4, "B": 0.35, "C": 0.25})
        assert result["portfolio_var"] < 0
        assert result["portfolio_sigma"] > 0

    def test_regime_breakdown_length(self, portfolio_model):
        result = pv.portfolio_var(portfolio_model, {"A": 0.4, "B": 0.35, "C": 0.25})
        assert len(result["regime_breakdown"]) == 5


class TestMarginalVar:
    def test_euler_decomposition_sums(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = pv.marginal_var(portfolio_model, w)
        total_component = sum(d["component_var"] for d in result["decomposition"])
        np.testing.assert_allclose(total_component, result["portfolio_var"], atol=0.01)

    def test_decomposition_has_all_assets(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = pv.marginal_var(portfolio_model, w)
        assets_in_result = {d["asset"] for d in result["decomposition"]}
        assert assets_in_result == {"A", "B", "C"}


class TestStressVar:
    def test_stress_wider_than_normal(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = pv.stress_var(portfolio_model, w, forced_regime=5)
        assert abs(result["stressed_var"]) >= abs(result["normal_var"])
        assert result["stress_multiplier"] >= 1.0

    def test_invalid_regime_raises(self, portfolio_model):
        with pytest.raises(ValueError, match="forced_regime must be"):
            pv.stress_var(portfolio_model, {"A": 1.0}, forced_regime=0)
        with pytest.raises(ValueError, match="forced_regime must be"):
            pv.stress_var(portfolio_model, {"A": 1.0}, forced_regime=6)

    def test_regime_correlation_is_matrix(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = pv.stress_var(portfolio_model, w, forced_regime=5)
        corr = result["regime_correlation"]
        assert len(corr) == 3
        assert len(corr[0]) == 3

