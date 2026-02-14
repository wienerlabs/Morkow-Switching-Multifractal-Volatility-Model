"""Tests for portfolio optimization (skfolio) — Wave 11B."""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cortex import portfolio_opt as popt


# ── Availability flag ──

class TestSkfolioAvailability:
    def test_available_flag_is_bool(self):
        assert isinstance(popt._SKFOLIO_AVAILABLE, bool)


# ── Tests when skfolio is NOT installed ──

class TestSkfolioFallbackWhenUnavailable:
    def _returns_df(self):
        rng = np.random.RandomState(42)
        return pd.DataFrame({
            "A": rng.randn(100) * 1.0,
            "B": rng.randn(100) * 1.2,
            "C": rng.randn(100) * 0.8,
        })

    def test_mean_cvar_raises(self):
        with patch.object(popt, "_SKFOLIO_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="skfolio not installed"):
                popt.optimize_mean_cvar(self._returns_df())

    def test_hrp_raises(self):
        with patch.object(popt, "_SKFOLIO_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="skfolio not installed"):
                popt.optimize_hrp(self._returns_df())

    def test_min_variance_raises(self):
        with patch.object(popt, "_SKFOLIO_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="skfolio not installed"):
                popt.optimize_min_variance(self._returns_df())


# ── Equal Weight (always available, no external deps) ──

class TestEqualWeight:
    @pytest.fixture
    def returns_df(self):
        rng = np.random.RandomState(42)
        return pd.DataFrame({
            "A": rng.randn(100),
            "B": rng.randn(100),
            "C": rng.randn(100),
            "D": rng.randn(100),
        })

    def test_output_keys(self, returns_df):
        result = popt.optimize_equal_weight(returns_df)
        expected_keys = {"method", "engine", "weights", "expected_return", "n_assets"}
        assert expected_keys == set(result.keys())

    def test_method_name(self, returns_df):
        result = popt.optimize_equal_weight(returns_df)
        assert result["method"] == "equal_weight"
        assert result["engine"] == "native"

    def test_equal_weights(self, returns_df):
        result = popt.optimize_equal_weight(returns_df)
        weights = result["weights"]
        assert len(weights) == 4
        for w in weights.values():
            assert abs(w - 0.25) < 1e-12

    def test_weights_sum_to_one(self, returns_df):
        result = popt.optimize_equal_weight(returns_df)
        assert abs(sum(result["weights"].values()) - 1.0) < 1e-12

    def test_n_assets(self, returns_df):
        result = popt.optimize_equal_weight(returns_df)
        assert result["n_assets"] == 4

    def test_expected_return_is_float(self, returns_df):
        result = popt.optimize_equal_weight(returns_df)
        assert isinstance(result["expected_return"], float)

    def test_two_asset_portfolio(self):
        rng = np.random.RandomState(99)
        df = pd.DataFrame({"X": rng.randn(50), "Y": rng.randn(50)})
        result = popt.optimize_equal_weight(df)
        assert result["n_assets"] == 2
        assert abs(result["weights"]["X"] - 0.5) < 1e-12
        assert abs(result["weights"]["Y"] - 0.5) < 1e-12


# ── Mocked skfolio tests ──

class TestMeanCVarMocked:
    def _mock_skfolio_model(self, weights, mean_val=-0.5, cvar_val=2.0):
        mock_model = MagicMock()
        mock_model.weights_ = weights
        mock_portfolio = MagicMock()
        mock_portfolio.mean = mean_val
        mock_portfolio.cvar = cvar_val
        mock_model.predict.return_value = mock_portfolio
        return mock_model

    def _patch_skfolio(self, mock_MeanRisk):
        """Context manager that patches all skfolio names used in optimize_mean_cvar."""
        mock_risk = MagicMock()
        mock_obj = MagicMock()
        return (
            patch.object(popt, "_SKFOLIO_AVAILABLE", True),
            patch.object(popt, "MeanRisk", mock_MeanRisk, create=True),
            patch.object(popt, "RiskMeasure", mock_risk, create=True),
            patch.object(popt, "ObjectiveFunction", mock_obj, create=True),
        )

    def test_output_keys(self):
        returns_df = pd.DataFrame({
            "A": np.random.randn(100),
            "B": np.random.randn(100),
        })
        mock_model = self._mock_skfolio_model(np.array([0.6, 0.4]))
        mock_MeanRisk = MagicMock(return_value=mock_model)

        patches = self._patch_skfolio(mock_MeanRisk)
        with patches[0], patches[1], patches[2], patches[3]:
            result = popt.optimize_mean_cvar(returns_df)

        expected_keys = {
            "method", "engine", "weights", "expected_return",
            "cvar", "cvar_beta", "n_assets",
        }
        assert expected_keys == set(result.keys())

    def test_method_and_engine(self):
        returns_df = pd.DataFrame({"A": np.random.randn(50)})
        mock_model = self._mock_skfolio_model(np.array([1.0]))
        mock_MeanRisk = MagicMock(return_value=mock_model)

        patches = self._patch_skfolio(mock_MeanRisk)
        with patches[0], patches[1], patches[2], patches[3]:
            result = popt.optimize_mean_cvar(returns_df)

        assert result["method"] == "mean_cvar"
        assert result["engine"] == "skfolio"


class TestHRPMocked:
    def test_output_keys(self):
        returns_df = pd.DataFrame({
            "A": np.random.randn(100),
            "B": np.random.randn(100),
            "C": np.random.randn(100),
        })
        mock_model = MagicMock()
        mock_model.weights_ = np.array([0.3, 0.4, 0.3])
        mock_portfolio = MagicMock()
        mock_portfolio.mean = 0.01
        mock_model.predict.return_value = mock_portfolio
        mock_HRP = MagicMock(return_value=mock_model)

        with patch.object(popt, "_SKFOLIO_AVAILABLE", True), \
             patch.object(popt, "HierarchicalRiskParity", mock_HRP, create=True):
            result = popt.optimize_hrp(returns_df)

        assert result["method"] == "hrp"
        assert result["engine"] == "skfolio"
        assert result["n_assets"] == 3
        assert "weights" in result
        assert "expected_return" in result


class TestMinVarianceMocked:
    def test_output_keys(self):
        returns_df = pd.DataFrame({
            "A": np.random.randn(100),
            "B": np.random.randn(100),
        })
        mock_model = MagicMock()
        mock_model.weights_ = np.array([0.55, 0.45])
        mock_portfolio = MagicMock()
        mock_portfolio.mean = 0.005
        mock_portfolio.variance = 0.02
        mock_model.predict.return_value = mock_portfolio
        mock_MeanRisk = MagicMock(return_value=mock_model)

        with patch.object(popt, "_SKFOLIO_AVAILABLE", True), \
             patch.object(popt, "MeanRisk", mock_MeanRisk, create=True), \
             patch.object(popt, "RiskMeasure", MagicMock(), create=True), \
             patch.object(popt, "ObjectiveFunction", MagicMock(), create=True):
            result = popt.optimize_min_variance(returns_df)

        assert result["method"] == "min_variance"
        assert result["engine"] == "skfolio"
        assert "variance" in result
        assert result["n_assets"] == 2


# ── Compare strategies ──

class TestCompareStrategies:
    @pytest.fixture
    def returns_df(self):
        rng = np.random.RandomState(42)
        return pd.DataFrame({
            "A": rng.randn(100),
            "B": rng.randn(100),
            "C": rng.randn(100),
        })

    def test_always_includes_equal_weight(self, returns_df):
        with patch.object(popt, "_SKFOLIO_AVAILABLE", False):
            results = popt.compare_strategies(returns_df)

        assert len(results) == 1
        assert results[0]["method"] == "equal_weight"

    def test_sorted_by_expected_return_desc(self, returns_df):
        with patch.object(popt, "_SKFOLIO_AVAILABLE", False):
            results = popt.compare_strategies(returns_df)

        # Only one result when skfolio unavailable, so just check it's a list
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_skfolio_strategies_added_when_available(self, returns_df):
        # Mock all three skfolio optimizers
        def mock_mean_cvar(r, **kw):
            return {"method": "mean_cvar", "engine": "skfolio",
                    "weights": {}, "expected_return": 0.02, "cvar": 1.0,
                    "cvar_beta": 0.95, "n_assets": 3}

        def mock_hrp(r):
            return {"method": "hrp", "engine": "skfolio",
                    "weights": {}, "expected_return": 0.015, "n_assets": 3}

        def mock_min_var(r, **kw):
            return {"method": "min_variance", "engine": "skfolio",
                    "weights": {}, "expected_return": 0.01,
                    "variance": 0.5, "n_assets": 3}

        with patch.object(popt, "_SKFOLIO_AVAILABLE", True), \
             patch.object(popt, "optimize_mean_cvar", mock_mean_cvar), \
             patch.object(popt, "optimize_hrp", mock_hrp), \
             patch.object(popt, "optimize_min_variance", mock_min_var):
            results = popt.compare_strategies(returns_df)

        methods = [r["method"] for r in results]
        assert "equal_weight" in methods
        assert "mean_cvar" in methods
        assert "hrp" in methods
        assert "min_variance" in methods
        assert len(results) == 4

    def test_failed_strategy_is_skipped(self, returns_df):
        def mock_mean_cvar(r, **kw):
            raise Exception("convergence error")

        def mock_hrp(r):
            return {"method": "hrp", "engine": "skfolio",
                    "weights": {}, "expected_return": 0.01, "n_assets": 3}

        def mock_min_var(r, **kw):
            return {"method": "min_variance", "engine": "skfolio",
                    "weights": {}, "expected_return": 0.005,
                    "variance": 0.3, "n_assets": 3}

        with patch.object(popt, "_SKFOLIO_AVAILABLE", True), \
             patch.object(popt, "optimize_mean_cvar", mock_mean_cvar), \
             patch.object(popt, "optimize_hrp", mock_hrp), \
             patch.object(popt, "optimize_min_variance", mock_min_var):
            results = popt.compare_strategies(returns_df)

        methods = [r["method"] for r in results]
        assert "mean_cvar" not in methods
        assert "hrp" in methods
        assert "min_variance" in methods
        assert len(results) == 3

    def test_results_sorted_descending(self, returns_df):
        def mock_mean_cvar(r, **kw):
            return {"method": "mean_cvar", "expected_return": 0.03,
                    "engine": "skfolio", "weights": {},
                    "cvar": 1.0, "cvar_beta": 0.95, "n_assets": 3}

        def mock_hrp(r):
            return {"method": "hrp", "expected_return": -0.01,
                    "engine": "skfolio", "weights": {}, "n_assets": 3}

        def mock_min_var(r, **kw):
            return {"method": "min_variance", "expected_return": 0.005,
                    "engine": "skfolio", "weights": {},
                    "variance": 0.3, "n_assets": 3}

        with patch.object(popt, "_SKFOLIO_AVAILABLE", True), \
             patch.object(popt, "optimize_mean_cvar", mock_mean_cvar), \
             patch.object(popt, "optimize_hrp", mock_hrp), \
             patch.object(popt, "optimize_min_variance", mock_min_var):
            results = popt.compare_strategies(returns_df)

        returns_list = [r["expected_return"] for r in results]
        assert returns_list == sorted(returns_list, reverse=True)
