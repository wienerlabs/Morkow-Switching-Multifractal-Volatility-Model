"""Tests for vine copula integration (pyvinecopulib) — Wave 11B."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cortex import copula as cpv


# ── Availability flag ──

class TestVineAvailabilityFlag:
    def test_vine_available_is_bool(self):
        assert isinstance(cpv._VINE_AVAILABLE, bool)

    def test_family_map_keys(self):
        expected = {"gaussian", "student_t", "clayton", "gumbel", "frank"}
        assert set(cpv._VINE_FAMILY_MAP.keys()) == expected


# ── Tests when pyvinecopulib is NOT installed ──

class TestVineFallbackWhenUnavailable:
    def test_fit_vine_raises_without_lib(self, multivariate_returns):
        with patch.object(cpv, "_VINE_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="pyvinecopulib not installed"):
                cpv.fit_vine_copula(multivariate_returns)


# ── Tests when pyvinecopulib IS installed (mocked) ──

class TestFitVineCopulaMocked:
    """Test fit_vine_copula with a mocked pyvinecopulib."""

    def _make_mock_vine(self, d=3):
        mock_vine = MagicMock()
        mock_vine.loglik.return_value = -150.0
        mock_vine.npars = 6

        # Mock pair copula access
        mock_pc = MagicMock()
        mock_pc.family = "gaussian"
        mock_vine.get_pair_copula.return_value = mock_pc

        # Mock simulate
        mock_vine.simulate.return_value = np.random.rand(1000, d)
        return mock_vine

    def test_fit_returns_expected_keys(self, multivariate_returns):
        mock_vine = self._make_mock_vine()
        mock_pv = MagicMock()
        mock_pv.to_pseudo_obs.return_value = np.random.rand(200, 3)
        mock_pv.Vinecop.return_value = mock_vine
        mock_pv.BicopFamily.gaussian = "gaussian"
        mock_pv.BicopFamily.student = "student"
        mock_pv.BicopFamily.clayton = "clayton"
        mock_pv.BicopFamily.gumbel = "gumbel"
        mock_pv.BicopFamily.frank = "frank"
        mock_pv.FitControlsVinecop.return_value = MagicMock()

        with patch.object(cpv, "_VINE_AVAILABLE", True), \
             patch.object(cpv, "pv", mock_pv):
            result = cpv.fit_vine_copula(multivariate_returns)

        assert result["engine"] == "pyvinecopulib"
        assert result["structure"] == "rvine"
        assert result["n_obs"] == 200
        assert result["n_assets"] == 3
        assert "log_likelihood" in result
        assert "aic" in result
        assert "bic" in result
        assert "n_params" in result
        assert "_vine_object" in result

    def test_fit_with_dataframe_input(self, multivariate_returns):
        mock_vine = self._make_mock_vine()
        mock_pv = MagicMock()
        mock_pv.to_pseudo_obs.return_value = np.random.rand(200, 3)
        mock_pv.Vinecop.return_value = mock_vine
        mock_pv.BicopFamily.gaussian = "gaussian"
        mock_pv.BicopFamily.student = "student"
        mock_pv.BicopFamily.clayton = "clayton"
        mock_pv.BicopFamily.gumbel = "gumbel"
        mock_pv.BicopFamily.frank = "frank"
        mock_pv.FitControlsVinecop.return_value = MagicMock()

        with patch.object(cpv, "_VINE_AVAILABLE", True), \
             patch.object(cpv, "pv", mock_pv):
            result = cpv.fit_vine_copula(multivariate_returns)

        # Should have converted DataFrame to ndarray for pv.to_pseudo_obs
        call_args = mock_pv.to_pseudo_obs.call_args[0][0]
        assert isinstance(call_args, np.ndarray)

    def test_fit_with_ndarray_input(self, multivariate_returns):
        mock_vine = self._make_mock_vine()
        mock_pv = MagicMock()
        mock_pv.to_pseudo_obs.return_value = np.random.rand(200, 3)
        mock_pv.Vinecop.return_value = mock_vine
        mock_pv.BicopFamily.gaussian = "gaussian"
        mock_pv.BicopFamily.student = "student"
        mock_pv.BicopFamily.clayton = "clayton"
        mock_pv.BicopFamily.gumbel = "gumbel"
        mock_pv.BicopFamily.frank = "frank"
        mock_pv.FitControlsVinecop.return_value = MagicMock()

        with patch.object(cpv, "_VINE_AVAILABLE", True), \
             patch.object(cpv, "pv", mock_pv):
            result = cpv.fit_vine_copula(multivariate_returns.values)

        assert result["n_assets"] == 3

    def test_aic_bic_formula(self, multivariate_returns):
        mock_vine = self._make_mock_vine()
        mock_pv = MagicMock()
        mock_pv.to_pseudo_obs.return_value = np.random.rand(200, 3)
        mock_pv.Vinecop.return_value = mock_vine
        mock_pv.BicopFamily.gaussian = "gaussian"
        mock_pv.BicopFamily.student = "student"
        mock_pv.BicopFamily.clayton = "clayton"
        mock_pv.BicopFamily.gumbel = "gumbel"
        mock_pv.BicopFamily.frank = "frank"
        mock_pv.FitControlsVinecop.return_value = MagicMock()

        with patch.object(cpv, "_VINE_AVAILABLE", True), \
             patch.object(cpv, "pv", mock_pv):
            result = cpv.fit_vine_copula(multivariate_returns)

        ll = result["log_likelihood"]
        k = result["n_params"]
        n = result["n_obs"]
        expected_aic = 2 * k - 2 * ll
        expected_bic = k * np.log(n) - 2 * ll
        assert abs(result["aic"] - expected_aic) < 1e-10
        assert abs(result["bic"] - expected_bic) < 1e-10

    def test_custom_family_set(self, multivariate_returns):
        mock_vine = self._make_mock_vine()
        mock_pv = MagicMock()
        mock_pv.to_pseudo_obs.return_value = np.random.rand(200, 3)
        mock_pv.Vinecop.return_value = mock_vine
        mock_pv.BicopFamily.gaussian = "gaussian"
        mock_pv.BicopFamily.student = "student"
        mock_pv.FitControlsVinecop.return_value = MagicMock()

        with patch.object(cpv, "_VINE_AVAILABLE", True), \
             patch.object(cpv, "pv", mock_pv):
            result = cpv.fit_vine_copula(
                multivariate_returns,
                family_set=["gaussian", "student_t"],
            )

        assert result["engine"] == "pyvinecopulib"


class TestVineCopulaSimulateMocked:
    def test_simulate_returns_array(self):
        mock_vine = MagicMock()
        mock_vine.simulate.return_value = np.random.rand(5000, 3)

        vine_fit = {"_vine_object": mock_vine}
        result = cpv.vine_copula_simulate(vine_fit, n_samples=5000, seed=42)

        assert isinstance(result, np.ndarray)
        assert result.shape == (5000, 3)
        mock_vine.simulate.assert_called_once_with(n=5000, seeds=[42])

    def test_simulate_different_seeds(self):
        mock_vine = MagicMock()
        mock_vine.simulate.return_value = np.random.rand(100, 2)

        vine_fit = {"_vine_object": mock_vine}
        cpv.vine_copula_simulate(vine_fit, n_samples=100, seed=99)
        mock_vine.simulate.assert_called_with(n=100, seeds=[99])


class TestVineCopulaPortfolioVarMocked:
    def _make_mock_model(self):
        return {
            "assets": ["A", "B", "C"],
            "current_probs": np.array([0.3, 0.4, 0.2, 0.1]),
            "num_states": 4,
            "per_asset": {
                "A": {"sigma_states": [0.5, 1.0, 1.5, 2.0]},
                "B": {"sigma_states": [0.6, 1.1, 1.6, 2.1]},
                "C": {"sigma_states": [0.4, 0.9, 1.4, 1.9]},
            },
        }

    def test_portfolio_var_output_keys(self):
        model = self._make_mock_model()
        weights = {"A": 0.4, "B": 0.3, "C": 0.3}

        mock_vine = MagicMock()
        mock_vine.simulate.return_value = np.random.rand(1000, 3)
        vine_fit = {
            "_vine_object": mock_vine,
            "structure": "rvine",
            "n_params": 6,
        }

        mock_pvar = MagicMock(return_value={"portfolio_var": -2.5})
        with patch("cortex.portfolio.portfolio_var", mock_pvar):
            result = cpv.vine_copula_portfolio_var(
                model, weights, vine_fit, n_simulations=1000
            )

        expected_keys = {
            "vine_var", "gaussian_var", "var_ratio",
            "engine", "structure", "n_params", "n_simulations", "alpha",
        }
        assert expected_keys == set(result.keys())

    def test_portfolio_var_engine_field(self):
        model = self._make_mock_model()
        weights = {"A": 0.5, "B": 0.3, "C": 0.2}

        mock_vine = MagicMock()
        mock_vine.simulate.return_value = np.random.rand(500, 3)
        vine_fit = {
            "_vine_object": mock_vine,
            "structure": "cvine",
            "n_params": 4,
        }

        mock_pvar = MagicMock(return_value={"portfolio_var": -1.8})
        with patch("cortex.portfolio.portfolio_var", mock_pvar):
            result = cpv.vine_copula_portfolio_var(
                model, weights, vine_fit, n_simulations=500
            )

        assert result["engine"] == "pyvinecopulib"
        assert result["structure"] == "cvine"
        assert result["n_simulations"] == 500

    def test_var_is_negative(self):
        """VaR should be negative (loss) for typical returns."""
        model = self._make_mock_model()
        weights = {"A": 0.4, "B": 0.3, "C": 0.3}

        rng = np.random.RandomState(42)
        mock_vine = MagicMock()
        mock_vine.simulate.return_value = rng.uniform(0.01, 0.99, (5000, 3))
        vine_fit = {"_vine_object": mock_vine, "structure": "rvine", "n_params": 6}

        mock_pvar = MagicMock(return_value={"portfolio_var": -3.0})
        with patch("cortex.portfolio.portfolio_var", mock_pvar):
            result = cpv.vine_copula_portfolio_var(
                model, weights, vine_fit, n_simulations=5000
            )

        assert result["vine_var"] < 0
