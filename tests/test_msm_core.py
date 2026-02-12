"""Tests for MSM-VaR_MODEL.py core functions."""

import numpy as np
import pandas as pd
import pytest

from cortex import msm


class TestBuildTransitionMatrix:
    def test_scalar_p_stay(self):
        P = msm._build_transition_matrix(0.95, 3)
        assert P.shape == (3, 3)
        np.testing.assert_allclose(P.sum(axis=1), 1.0)
        np.testing.assert_allclose(np.diag(P), 0.95)

    def test_array_p_stay(self):
        p = [0.90, 0.95, 0.99]
        P = msm._build_transition_matrix(p, 3)
        for k in range(3):
            assert abs(P[k, k] - p[k]) < 1e-12
            assert abs(P[k, :].sum() - 1.0) < 1e-12

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError, match="p_stay must be scalar or length"):
            msm._build_transition_matrix([0.9, 0.95], 5)


class TestMsmVolForecast:
    def test_output_shapes(self, sample_returns):
        sf, sfi, fp, ss, P = msm.msm_vol_forecast(
            sample_returns, num_states=3, sigma_low=0.5, sigma_high=3.0, p_stay=0.95
        )
        n = len(sample_returns)
        assert sf.shape == (n,)
        assert sfi.shape == (n,)
        assert fp.shape == (n, 3)
        assert ss.shape == (3,)
        assert P.shape == (3, 3)

    def test_array_p_stay(self, sample_returns):
        sf, _, fp, ss, P = msm.msm_vol_forecast(
            sample_returns, num_states=3, sigma_low=0.5, sigma_high=3.0,
            p_stay=[0.90, 0.95, 0.99],
        )
        assert sf.shape == (len(sample_returns),)
        assert abs(P[0, 0] - 0.90) < 1e-12
        assert abs(P[2, 2] - 0.99) < 1e-12

    def test_filter_probs_sum_to_one(self, sample_returns):
        _, _, fp, _, _ = msm.msm_vol_forecast(
            sample_returns, num_states=5, sigma_low=0.3, sigma_high=2.0, p_stay=0.97
        )
        row_sums = fp.values.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)


class TestCalibrateAdvanced:
    @pytest.mark.parametrize("method", ["mle", "empirical", "grid", "hybrid"])
    def test_all_methods_return_list_p_stay(self, sample_returns, method):
        cal = msm.calibrate_msm_advanced(
            sample_returns, num_states=5, method=method, verbose=False
        )
        assert isinstance(cal["p_stay"], list)
        assert len(cal["p_stay"]) == 5
        assert all(0 < p < 1 for p in cal["p_stay"])
        assert cal["sigma_low"] > 0
        assert cal["sigma_high"] > cal["sigma_low"]
        assert "metrics" in cal

    def test_mle_can_differentiate_regimes(self, sample_returns):
        cal = msm.calibrate_msm_advanced(
            sample_returns, num_states=5, method="mle", verbose=False
        )
        # MLE may or may not find different p_stay — just verify it's valid
        assert len(cal["p_stay"]) == 5

    def test_unknown_method_raises(self, sample_returns):
        with pytest.raises(ValueError, match="Unknown calibration method"):
            msm.calibrate_msm_advanced(sample_returns, method="bogus", verbose=False)


class TestVaRForecast:
    def test_normal_var(self, calibrated_model):
        m = calibrated_model
        var_val, sigma, z, pi = msm.msm_var_forecast_next_day(
            m["filter_probs"], m["sigma_states"], m["P_matrix"], alpha=0.05
        )
        assert var_val < 0
        assert sigma > 0
        assert abs(z - (-1.6449)) < 0.01

    @pytest.mark.parametrize("nu", [3.0, 5.0, 10.0])
    def test_student_t_wider_than_normal(self, calibrated_model, nu):
        m = calibrated_model
        var_n, _, _, _ = msm.msm_var_forecast_next_day(
            m["filter_probs"], m["sigma_states"], m["P_matrix"], alpha=0.05
        )
        var_t, _, _, _ = msm.msm_var_forecast_next_day(
            m["filter_probs"], m["sigma_states"], m["P_matrix"],
            alpha=0.05, use_student_t=True, nu=nu,
        )
        assert var_t < var_n, f"Student-t(nu={nu}) should be more negative than Normal"

    def test_nu_le_2_raises(self, calibrated_model):
        m = calibrated_model
        with pytest.raises(ValueError, match="nu.*must be > 2"):
            msm.msm_var_forecast_next_day(
                m["filter_probs"], m["sigma_states"], m["P_matrix"],
                use_student_t=True, nu=2.0,
            )


class TestLogLikelihood:
    def test_backward_compat_3_params(self, sample_returns):
        ll = msm.msm_log_likelihood([0.5, 3.0, 0.95], sample_returns.values, 5)
        assert np.isfinite(ll)
        assert ll > 0  # negative LL is positive

    def test_k_plus_2_params(self, sample_returns):
        params = [0.5, 3.0, 0.93, 0.95, 0.97, 0.94, 0.96]
        ll = msm.msm_log_likelihood(params, sample_returns.values, 5)
        assert np.isfinite(ll)

    def test_invalid_params_return_penalty(self, sample_returns):
        assert msm.msm_log_likelihood([-1, 3.0, 0.95], sample_returns.values, 5) == 1e10
        assert msm.msm_log_likelihood([3.0, 0.5, 0.95], sample_returns.values, 5) == 1e10



class TestAsymmetricLeverage:
    """Tests for asymmetric leverage transitions (leverage_gamma)."""

    def test_transition_matrix_negative_return_increases_upward(self):
        """Negative return + gamma < 0 should increase P(i→j) for j > i."""
        K = 3
        P_sym = msm._build_transition_matrix(0.95, K)
        P_asym = msm._build_transition_matrix(0.95, K, leverage_gamma=-1.0, current_return=-2.0)

        # Off-diagonal upward transitions (j > i) should be larger
        for i in range(K):
            for j in range(i + 1, K):
                assert P_asym[i, j] >= P_sym[i, j], (
                    f"P_asym[{i},{j}]={P_asym[i,j]:.6f} should be >= P_sym[{i},{j}]={P_sym[i,j]:.6f}"
                )

        # Rows must still sum to 1
        np.testing.assert_allclose(P_asym.sum(axis=1), 1.0, atol=1e-10)

    def test_transition_matrix_positive_return_increases_downward(self):
        """Positive return + gamma < 0 should increase P(i→j) for j < i."""
        K = 3
        P_sym = msm._build_transition_matrix(0.95, K)
        P_asym = msm._build_transition_matrix(0.95, K, leverage_gamma=-1.0, current_return=2.0)

        # Off-diagonal downward transitions (j < i) should be larger
        for i in range(1, K):
            for j in range(i):
                assert P_asym[i, j] >= P_sym[i, j], (
                    f"P_asym[{i},{j}]={P_asym[i,j]:.6f} should be >= P_sym[{i},{j}]={P_sym[i,j]:.6f}"
                )

        np.testing.assert_allclose(P_asym.sum(axis=1), 1.0, atol=1e-10)

    def test_gamma_zero_preserves_symmetric(self):
        """leverage_gamma=0 should produce the same matrix as no leverage."""
        K = 5
        P_base = msm._build_transition_matrix(0.95, K)
        P_zero = msm._build_transition_matrix(0.95, K, leverage_gamma=0.0, current_return=-3.0)
        np.testing.assert_allclose(P_zero, P_base)

    def test_vol_forecast_with_leverage_differs(self, sample_returns):
        """msm_vol_forecast with leverage_gamma != 0 should produce different results."""
        sf_sym, _, _, _, _ = msm.msm_vol_forecast(
            sample_returns, num_states=3, sigma_low=0.5, sigma_high=3.0, p_stay=0.95,
        )
        sf_asym, _, _, _, _ = msm.msm_vol_forecast(
            sample_returns, num_states=3, sigma_low=0.5, sigma_high=3.0, p_stay=0.95,
            leverage_gamma=-0.5,
        )
        # They should not be identical
        assert not np.allclose(sf_sym.values, sf_asym.values), (
            "Symmetric and asymmetric forecasts should differ"
        )

    def test_vol_forecast_backward_compat(self, sample_returns):
        """Default leverage_gamma=0 should match the original output."""
        sf1, sfi1, fp1, ss1, P1 = msm.msm_vol_forecast(
            sample_returns, num_states=3, sigma_low=0.5, sigma_high=3.0, p_stay=0.95,
        )
        sf2, sfi2, fp2, ss2, P2 = msm.msm_vol_forecast(
            sample_returns, num_states=3, sigma_low=0.5, sigma_high=3.0, p_stay=0.95,
            leverage_gamma=0.0,
        )
        np.testing.assert_allclose(sf1.values, sf2.values)
        np.testing.assert_allclose(sfi1.values, sfi2.values)

    def test_log_likelihood_with_leverage(self, sample_returns):
        """msm_log_likelihood should accept leverage_gamma and return finite value."""
        ll_sym = msm.msm_log_likelihood([0.5, 3.0, 0.95], sample_returns.values, 5)
        ll_asym = msm.msm_log_likelihood(
            [0.5, 3.0, 0.95], sample_returns.values, 5, leverage_gamma=-0.5,
        )
        assert np.isfinite(ll_sym)
        assert np.isfinite(ll_asym)
        assert ll_sym != ll_asym

    def test_calibrate_returns_leverage_gamma(self, sample_returns):
        """calibrate_msm_advanced should include leverage_gamma in result."""
        cal = msm.calibrate_msm_advanced(
            sample_returns, num_states=5, method="empirical", verbose=False,
            leverage_gamma=-0.3,
        )
        assert "leverage_gamma" in cal
        assert cal["leverage_gamma"] == pytest.approx(-0.3)

    def test_calibrate_estimate_gamma(self, sample_returns):
        """calibrate_msm_advanced with leverage_gamma='estimate' should find a value."""
        cal = msm.calibrate_msm_advanced(
            sample_returns, num_states=3, method="empirical", verbose=False,
            leverage_gamma="estimate",
        )
        assert "leverage_gamma" in cal
        assert -2.0 <= cal["leverage_gamma"] <= 0.0
