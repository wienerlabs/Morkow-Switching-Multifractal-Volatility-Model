"""Tests for copula_portfolio_var.py."""

import numpy as np
import pandas as pd
import pytest

from cortex import copula as cpv
from cortex import portfolio as pv


@pytest.fixture(scope="module")
def portfolio_model(multivariate_returns):
    return pv.calibrate_multivariate(
        multivariate_returns, num_states=5, method="empirical"
    )


@pytest.fixture(scope="module")
def gaussian_fit(multivariate_returns):
    return cpv.fit_copula(multivariate_returns, family="gaussian")


@pytest.fixture(scope="module")
def student_t_fit(multivariate_returns):
    return cpv.fit_copula(multivariate_returns, family="student_t")


@pytest.fixture(scope="module")
def clayton_fit(multivariate_returns):
    return cpv.fit_copula(multivariate_returns, family="clayton")


@pytest.fixture(scope="module")
def gumbel_fit(multivariate_returns):
    return cpv.fit_copula(multivariate_returns, family="gumbel")


@pytest.fixture(scope="module")
def frank_fit(multivariate_returns):
    return cpv.fit_copula(multivariate_returns, family="frank")


class TestFitCopula:
    def test_gaussian_output_keys(self, gaussian_fit):
        for key in ("family", "params", "log_likelihood", "aic", "bic",
                     "n_obs", "n_assets", "n_params", "tail_dependence"):
            assert key in gaussian_fit

    def test_gaussian_has_correlation_matrix(self, gaussian_fit):
        R = np.array(gaussian_fit["params"]["R"])
        assert R.shape == (3, 3)
        np.testing.assert_allclose(np.diag(R), 1.0, atol=1e-6)

    def test_gaussian_no_tail_dependence(self, gaussian_fit):
        td = gaussian_fit["tail_dependence"]
        assert td["lambda_lower"] == 0.0
        assert td["lambda_upper"] == 0.0

    def test_student_t_has_nu(self, student_t_fit):
        assert "nu" in student_t_fit["params"]
        assert student_t_fit["params"]["nu"] > 2.0

    def test_student_t_symmetric_tail(self, student_t_fit):
        td = student_t_fit["tail_dependence"]
        assert td["lambda_lower"] == td["lambda_upper"]

    def test_clayton_lower_tail(self, clayton_fit):
        td = clayton_fit["tail_dependence"]
        assert td["lambda_lower"] > 0.0
        assert td["lambda_upper"] == 0.0

    def test_gumbel_upper_tail(self, gumbel_fit):
        td = gumbel_fit["tail_dependence"]
        assert td["lambda_lower"] == 0.0
        assert td["lambda_upper"] > 0.0

    def test_frank_no_tail_dependence(self, frank_fit):
        td = frank_fit["tail_dependence"]
        assert td["lambda_lower"] == 0.0
        assert td["lambda_upper"] == 0.0

    def test_invalid_family_raises(self, multivariate_returns):
        with pytest.raises(ValueError, match="Unknown copula family"):
            cpv.fit_copula(multivariate_returns, family="invalid")

    def test_accepts_numpy_array(self, multivariate_returns):
        fit = cpv.fit_copula(multivariate_returns.values, family="gaussian")
        assert fit["n_assets"] == 3

    def test_aic_bic_finite(self, gaussian_fit, student_t_fit, clayton_fit):
        for fit in (gaussian_fit, student_t_fit, clayton_fit):
            assert np.isfinite(fit["aic"])
            assert np.isfinite(fit["bic"])

    def test_params_json_serializable(self, student_t_fit):
        import json
        json.dumps(student_t_fit["params"])


class TestCopulaPortfolioVar:
    def test_produces_negative_var(self, portfolio_model, gaussian_fit):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.copula_portfolio_var(portfolio_model, w, gaussian_fit, n_simulations=5000)
        assert result["copula_var"] < 0

    def test_output_keys(self, portfolio_model, gaussian_fit):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.copula_portfolio_var(portfolio_model, w, gaussian_fit, n_simulations=5000)
        for key in ("copula_var", "gaussian_var", "var_ratio", "copula_family",
                     "tail_dependence", "n_simulations", "alpha"):
            assert key in result

    def test_deterministic_with_seed(self, portfolio_model, gaussian_fit):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        r1 = cpv.copula_portfolio_var(portfolio_model, w, gaussian_fit, seed=99, n_simulations=3000)
        r2 = cpv.copula_portfolio_var(portfolio_model, w, gaussian_fit, seed=99, n_simulations=3000)
        assert r1["copula_var"] == r2["copula_var"]


class TestRegimeConditionalCopulas:
    def test_returns_k_regimes(self, portfolio_model):
        results = cpv.regime_conditional_copulas(portfolio_model, family="gaussian")
        assert len(results) == 5

    def test_each_regime_has_copula(self, portfolio_model):
        results = cpv.regime_conditional_copulas(portfolio_model, family="gaussian")
        for rc in results:
            assert "regime" in rc
            assert "n_obs" in rc
            assert "copula" in rc
            assert rc["copula"]["family"] == "gaussian"


class TestCompareCopulas:
    def test_ranks_all_families(self, multivariate_returns):
        ranking = cpv.compare_copulas(multivariate_returns)
        assert len(ranking) == 5
        families = {r["family"] for r in ranking}
        assert families == set(cpv.COPULA_FAMILIES)

    def test_sorted_by_aic(self, multivariate_returns):
        ranking = cpv.compare_copulas(multivariate_returns)
        aics = [r["aic"] for r in ranking]
        assert aics == sorted(aics)

    def test_best_flag(self, multivariate_returns):
        ranking = cpv.compare_copulas(multivariate_returns)
        assert ranking[0]["best"] is True
        assert ranking[0]["rank"] == 1
        for r in ranking[1:]:
            assert r["best"] is False

    def test_subset_families(self, multivariate_returns):
        ranking = cpv.compare_copulas(multivariate_returns, families=["gaussian", "frank"])
        assert len(ranking) == 2


class TestRegimeDependentCopulaVar:
    """Tests for regime_dependent_copula_var() and _sample_regime_copula()."""

    def test_output_keys(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.regime_dependent_copula_var(portfolio_model, w, n_simulations=3000)
        expected = {
            "regime_dependent_var", "static_var", "var_difference_pct",
            "current_regime_copula", "regime_tail_dependence",
            "dominant_regime", "regime_probs", "n_simulations", "alpha",
        }
        assert expected.issubset(result.keys())

    def test_var_is_negative(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.regime_dependent_copula_var(portfolio_model, w, n_simulations=3000)
        assert result["regime_dependent_var"] < 0
        assert result["static_var"] < 0

    def test_regime_tail_dependence_per_regime(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.regime_dependent_copula_var(portfolio_model, w, n_simulations=3000)
        rtd = result["regime_tail_dependence"]
        assert len(rtd) == 5
        for item in rtd:
            assert "regime" in item
            assert "family" in item
            assert "lambda_lower" in item
            assert "lambda_upper" in item

    def test_crisis_regimes_use_student_t(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.regime_dependent_copula_var(portfolio_model, w, n_simulations=3000)
        rtd = result["regime_tail_dependence"]
        # States 4 and 5 (indices 3,4) should use student_t (crisis)
        for item in rtd:
            if item["regime"] >= 4:
                assert item["family"] == "student_t"

    def test_calm_regimes_use_gaussian(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.regime_dependent_copula_var(portfolio_model, w, n_simulations=3000)
        rtd = result["regime_tail_dependence"]
        # States 1-3 (indices 0,1,2) should use gaussian (calm)
        for item in rtd:
            if item["regime"] <= 3:
                assert item["family"] == "gaussian"

    def test_crisis_tail_dependence_ge_calm(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.regime_dependent_copula_var(portfolio_model, w, n_simulations=3000)
        rtd = result["regime_tail_dependence"]
        calm_max = max(
            (item["lambda_lower"] + item["lambda_upper"])
            for item in rtd if item["regime"] <= 3
        )
        crisis_min = min(
            (item["lambda_lower"] + item["lambda_upper"])
            for item in rtd if item["regime"] >= 4
        )
        # Student-t has non-negative tail dependence, Gaussian always zero
        assert crisis_min >= calm_max
        # Calm regimes (Gaussian) must have exactly zero tail dependence
        assert calm_max == 0.0

    def test_var_difference_pct_computed(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.regime_dependent_copula_var(portfolio_model, w, n_simulations=3000)
        assert isinstance(result["var_difference_pct"], float)
        assert np.isfinite(result["var_difference_pct"])

    def test_regime_probs_sum_to_one(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.regime_dependent_copula_var(portfolio_model, w, n_simulations=3000)
        np.testing.assert_allclose(sum(result["regime_probs"]), 1.0, atol=1e-6)

    def test_dominant_regime_valid(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.regime_dependent_copula_var(portfolio_model, w, n_simulations=3000)
        assert 1 <= result["dominant_regime"] <= 5

    def test_deterministic_with_seed(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        r1 = cpv.regime_dependent_copula_var(portfolio_model, w, seed=77, n_simulations=3000)
        r2 = cpv.regime_dependent_copula_var(portfolio_model, w, seed=77, n_simulations=3000)
        assert r1["regime_dependent_var"] == r2["regime_dependent_var"]

    def test_current_regime_copula_is_valid(self, portfolio_model):
        w = {"A": 0.4, "B": 0.35, "C": 0.25}
        result = cpv.regime_dependent_copula_var(portfolio_model, w, n_simulations=3000)
        rc = result["current_regime_copula"]
        assert "family" in rc
        assert "params" in rc
        assert "tail_dependence" in rc
        assert rc["family"] in cpv.COPULA_FAMILIES


class TestSampleRegimeCopula:
    """Tests for _sample_regime_copula() helper."""

    def test_output_shape(self, portfolio_model):
        K = 5
        d = 3
        n_samples = 500
        rng = np.random.RandomState(42)
        # Build regime copulas manually
        rc_list = cpv.regime_conditional_copulas(portfolio_model, family="gaussian")
        probs = np.ones(K) / K
        u = cpv._sample_regime_copula(rc_list, probs, n_samples, d, rng)
        assert u.shape == (n_samples, d)

    def test_values_in_unit_interval(self, portfolio_model):
        rc_list = cpv.regime_conditional_copulas(portfolio_model, family="gaussian")
        probs = np.ones(5) / 5
        rng = np.random.RandomState(42)
        u = cpv._sample_regime_copula(rc_list, probs, 500, 3, rng)
        assert np.all(u >= 0) and np.all(u <= 1)

    def test_concentrated_on_single_regime(self, portfolio_model):
        rc_list = cpv.regime_conditional_copulas(portfolio_model, family="gaussian")
        probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        rng = np.random.RandomState(42)
        u = cpv._sample_regime_copula(rc_list, probs, 500, 3, rng)
        assert u.shape == (500, 3)

