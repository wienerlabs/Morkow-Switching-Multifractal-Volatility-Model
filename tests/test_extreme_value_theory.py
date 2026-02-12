"""Tests for extreme_value_theory.py — EVT/GPD tail risk module."""

import math

import numpy as np
import pandas as pd
import pytest

from cortex import evt


@pytest.fixture(scope="module")
def heavy_tail_returns():
    """Synthetic returns with heavy tails (Student-t, df=3)."""
    rng = np.random.RandomState(77)
    return pd.Series(rng.standard_t(df=3, size=500) * 2.0, name="heavy")


@pytest.fixture(scope="module")
def gpd_fit(heavy_tail_returns):
    """Pre-fitted GPD for reuse across tests."""
    losses = -heavy_tail_returns.values
    th = evt.select_threshold(heavy_tail_returns, method="variance_stability", min_exceedances=30)
    return evt.fit_gpd(losses, threshold=th["threshold"])


class TestFitGPD:
    def test_returns_required_keys(self, heavy_tail_returns):
        losses = -heavy_tail_returns.values
        th = evt.select_threshold(heavy_tail_returns, method="percentile", min_exceedances=30)
        result = evt.fit_gpd(losses, threshold=th["threshold"])
        for key in ["xi", "beta", "threshold", "n_total", "n_exceedances", "log_likelihood", "aic", "bic"]:
            assert key in result

    def test_beta_positive(self, gpd_fit):
        assert gpd_fit["beta"] > 0

    def test_n_exceedances_reasonable(self, gpd_fit):
        assert gpd_fit["n_exceedances"] >= 10
        assert gpd_fit["n_exceedances"] < gpd_fit["n_total"]

    def test_too_few_exceedances_raises(self):
        losses = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="exceedances"):
            evt.fit_gpd(losses, threshold=100.0)


class TestSelectThreshold:
    def test_percentile_method(self, heavy_tail_returns):
        result = evt.select_threshold(heavy_tail_returns, method="percentile", min_exceedances=30)
        assert "threshold" in result
        assert result["method"] == "percentile"
        assert result["n_exceedances"] >= 30

    def test_mean_excess_method(self, heavy_tail_returns):
        result = evt.select_threshold(heavy_tail_returns, method="mean_excess", min_exceedances=30)
        assert "threshold" in result
        assert result["method"] in ("mean_excess", "percentile")  # may fallback

    def test_variance_stability_method(self, heavy_tail_returns):
        result = evt.select_threshold(heavy_tail_returns, method="variance_stability", min_exceedances=30)
        assert "threshold" in result

    def test_invalid_method_raises(self, heavy_tail_returns):
        with pytest.raises(ValueError, match="Unknown"):
            evt.select_threshold(heavy_tail_returns, method="bogus")


class TestEVTVaR:
    def test_basic_computation(self, gpd_fit):
        var_val = evt.evt_var(
            xi=gpd_fit["xi"], beta=gpd_fit["beta"], threshold=gpd_fit["threshold"],
            n_total=gpd_fit["n_total"], n_exceedances=gpd_fit["n_exceedances"], alpha=0.01,
        )
        assert var_val > gpd_fit["threshold"]

    def test_more_extreme_alpha_gives_larger_var(self, gpd_fit):
        var_01 = evt.evt_var(xi=gpd_fit["xi"], beta=gpd_fit["beta"], threshold=gpd_fit["threshold"],
                             n_total=gpd_fit["n_total"], n_exceedances=gpd_fit["n_exceedances"], alpha=0.01)
        var_001 = evt.evt_var(xi=gpd_fit["xi"], beta=gpd_fit["beta"], threshold=gpd_fit["threshold"],
                              n_total=gpd_fit["n_total"], n_exceedances=gpd_fit["n_exceedances"], alpha=0.001)
        assert var_001 > var_01

    def test_invalid_beta_raises(self):
        with pytest.raises(ValueError, match="β"):
            evt.evt_var(xi=0.2, beta=-1.0, threshold=1.0, n_total=100, n_exceedances=10, alpha=0.01)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha"):
            evt.evt_var(xi=0.2, beta=1.0, threshold=1.0, n_total=100, n_exceedances=10, alpha=0.0)

    def test_exponential_limit(self):
        """When xi ≈ 0, should use exponential formula without error."""
        var_val = evt.evt_var(xi=0.0, beta=1.0, threshold=2.0, n_total=1000, n_exceedances=100, alpha=0.01)
        assert var_val > 2.0
        assert math.isfinite(var_val)


class TestEVTCVaR:
    def test_cvar_exceeds_var(self, gpd_fit):
        var_val = evt.evt_var(xi=gpd_fit["xi"], beta=gpd_fit["beta"], threshold=gpd_fit["threshold"],
                              n_total=gpd_fit["n_total"], n_exceedances=gpd_fit["n_exceedances"], alpha=0.01)
        cvar_val = evt.evt_cvar(xi=gpd_fit["xi"], beta=gpd_fit["beta"],
                                threshold=gpd_fit["threshold"], var_value=var_val, alpha=0.01)
        assert cvar_val > var_val

    def test_xi_ge_1_raises(self):
        with pytest.raises(ValueError, match="ξ ≥ 1"):
            evt.evt_cvar(xi=1.0, beta=1.0, threshold=1.0, var_value=2.0, alpha=0.01)


class TestEVTBacktest:
    def test_returns_list(self, heavy_tail_returns, gpd_fit):
        results = evt.evt_backtest(
            heavy_tail_returns, xi=gpd_fit["xi"], beta=gpd_fit["beta"],
            threshold=gpd_fit["threshold"], n_total=gpd_fit["n_total"],
            n_exceedances=gpd_fit["n_exceedances"],
        )
        assert isinstance(results, list)
        assert len(results) == 4  # default alphas

    def test_breach_rate_keys(self, heavy_tail_returns, gpd_fit):
        results = evt.evt_backtest(
            heavy_tail_returns, xi=gpd_fit["xi"], beta=gpd_fit["beta"],
            threshold=gpd_fit["threshold"], n_total=gpd_fit["n_total"],
            n_exceedances=gpd_fit["n_exceedances"],
        )
        row = results[0]
        for key in ["alpha", "confidence", "evt_var", "breach_count", "breach_rate", "kupiec_lr"]:
            assert key in row


class TestCompareVaRMethods:
    def test_three_methods_present(self, heavy_tail_returns, gpd_fit):
        results = evt.compare_var_methods(
            heavy_tail_returns, sigma_forecast=2.0,
            xi=gpd_fit["xi"], beta=gpd_fit["beta"], threshold=gpd_fit["threshold"],
            n_total=gpd_fit["n_total"], n_exceedances=gpd_fit["n_exceedances"],
            alphas=[0.01],
        )
        methods = {r["method"] for r in results}
        assert methods == {"normal", "student_t", "evt_gpd"}

    def test_evt_more_extreme_at_tail(self, heavy_tail_returns, gpd_fit):
        results = evt.compare_var_methods(
            heavy_tail_returns, sigma_forecast=2.0,
            xi=gpd_fit["xi"], beta=gpd_fit["beta"], threshold=gpd_fit["threshold"],
            n_total=gpd_fit["n_total"], n_exceedances=gpd_fit["n_exceedances"],
            alphas=[0.001],
        )
        by_method = {r["method"]: r["var_value"] for r in results}
        # EVT should give more extreme (more negative) VaR than Normal at 0.1%
        assert by_method["evt_gpd"] < by_method["normal"]

