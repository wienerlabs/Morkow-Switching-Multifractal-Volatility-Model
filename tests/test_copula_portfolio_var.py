"""Tests for copula_portfolio_var.py."""

import numpy as np
import pandas as pd
import pytest

import copula_portfolio_var as cpv
import portfolio_var as pv


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

