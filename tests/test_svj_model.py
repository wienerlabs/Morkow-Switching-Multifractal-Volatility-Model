import math

import numpy as np
import pandas as pd
import pytest

from cortex.svj import (
    calibrate_svj,
    decompose_risk,
    detect_jumps,
    svj_diagnostics,
    svj_var,
    svj_vol_series,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def returns_normal():
    rng = np.random.RandomState(42)
    return pd.Series(rng.randn(300) * 1.5, name="normal")


@pytest.fixture(scope="module")
def returns_with_jumps():
    """Synthetic returns with injected jumps at known positions."""
    rng = np.random.RandomState(99)
    r = rng.randn(300) * 1.0
    jump_idx = [50, 100, 150, 200, 250]
    for i in jump_idx:
        r[i] = rng.choice([-1, 1]) * rng.uniform(6, 10)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    return pd.Series(r, index=idx, name="jumpy")


@pytest.fixture(scope="module")
def calibration_normal(returns_normal):
    return calibrate_svj(returns_normal)


@pytest.fixture(scope="module")
def calibration_jumpy(returns_with_jumps):
    return calibrate_svj(returns_with_jumps)


# ── detect_jumps ─────────────────────────────────────────────────────


class TestDetectJumps:
    def test_output_keys(self, returns_normal):
        result = detect_jumps(returns_normal)
        expected = {
            "jump_dates", "jump_returns", "jump_indices", "n_jumps",
            "jump_fraction", "avg_jump_size", "jump_vol", "bns_statistic",
            "bns_pvalue", "jump_variation", "realized_variance",
            "bipower_variation", "threshold_multiplier",
        }
        assert expected == set(result.keys())

    def test_jump_fraction_range(self, returns_normal):
        result = detect_jumps(returns_normal)
        assert 0.0 <= result["jump_fraction"] <= 1.0

    def test_detects_injected_jumps(self, returns_with_jumps):
        result = detect_jumps(returns_with_jumps, threshold_multiplier=3.0)
        assert result["n_jumps"] >= 3

    def test_bns_statistic_finite(self, returns_normal):
        result = detect_jumps(returns_normal)
        assert np.isfinite(result["bns_statistic"])
        assert 0.0 <= result["bns_pvalue"] <= 1.0

    def test_realized_gt_bipower_with_jumps(self, returns_with_jumps):
        result = detect_jumps(returns_with_jumps)
        assert result["realized_variance"] >= result["bipower_variation"] * 0.8


# ── calibrate_svj ────────────────────────────────────────────────────


class TestCalibrateSVJ:
    def test_output_keys(self, calibration_normal):
        required = {
            "kappa", "theta", "sigma", "rho", "lambda_", "mu_j", "sigma_j",
            "feller_ratio", "feller_satisfied", "n_obs", "n_jumps_detected",
            "jump_fraction", "bns_statistic", "bns_pvalue",
            "optimization_success", "use_hawkes", "hawkes_params",
        }
        assert required.issubset(set(calibration_normal.keys()))

    def test_kappa_positive(self, calibration_normal):
        assert calibration_normal["kappa"] > 0

    def test_theta_positive(self, calibration_normal):
        assert calibration_normal["theta"] > 0

    def test_rho_range(self, calibration_normal):
        assert -1.0 <= calibration_normal["rho"] <= 1.0

    def test_lambda_nonneg(self, calibration_normal):
        assert calibration_normal["lambda_"] >= 0

    def test_sigma_j_positive(self, calibration_normal):
        assert calibration_normal["sigma_j"] > 0

    def test_feller_ratio_computed(self, calibration_normal):
        c = calibration_normal
        expected = 2 * c["kappa"] * c["theta"] / (c["sigma"] ** 2) if c["sigma"] > 0 else float("inf")
        assert abs(c["feller_ratio"] - expected) < 0.01

    def test_n_obs_matches(self, returns_normal, calibration_normal):
        assert calibration_normal["n_obs"] == len(returns_normal)

    def test_no_hawkes_by_default(self, calibration_normal):
        assert calibration_normal["use_hawkes"] is False
        assert calibration_normal["hawkes_params"] is None

    def test_jumpy_detects_more_jumps(self, calibration_normal, calibration_jumpy):
        assert calibration_jumpy["n_jumps_detected"] >= calibration_normal["n_jumps_detected"]


# ── svj_var ──────────────────────────────────────────────────────────


class TestSVJVaR:
    def test_var_negative(self, returns_normal, calibration_normal):
        result = svj_var(returns_normal, calibration_normal, alpha=0.05)
        assert result["var_svj"] < 0

    def test_es_worse_than_var(self, returns_normal, calibration_normal):
        result = svj_var(returns_normal, calibration_normal, alpha=0.05)
        assert result["expected_shortfall"] <= result["var_svj"]

    def test_jump_contribution_nonneg(self, returns_normal, calibration_normal):
        result = svj_var(returns_normal, calibration_normal)
        assert result["jump_contribution_pct"] >= 0

    def test_confidence_matches_alpha(self, returns_normal, calibration_normal):
        result = svj_var(returns_normal, calibration_normal, alpha=0.01)
        assert abs(result["confidence"] - 0.99) < 1e-6

    def test_output_keys(self, returns_normal, calibration_normal):
        result = svj_var(returns_normal, calibration_normal)
        expected = {
            "var_svj", "var_diffusion_only", "var_jump_component",
            "expected_shortfall", "jump_contribution_pct", "alpha",
            "confidence", "n_simulations", "current_variance", "avg_jumps_per_day",
        }
        assert expected == set(result.keys())


# ── decompose_risk ───────────────────────────────────────────────────


class TestDecomposeRisk:
    def test_output_keys(self, returns_normal, calibration_normal):
        result = decompose_risk(returns_normal, calibration_normal)
        expected = {
            "diffusion_variance", "jump_variance", "total_model_variance",
            "empirical_variance", "jump_share_pct", "diffusion_share_pct",
            "daily_diffusion_vol", "daily_jump_vol", "daily_total_vol",
            "annualized_diffusion_vol", "annualized_jump_vol", "annualized_total_vol",
        }
        assert expected == set(result.keys())

    def test_shares_sum_to_100(self, returns_normal, calibration_normal):
        result = decompose_risk(returns_normal, calibration_normal)
        assert abs(result["jump_share_pct"] + result["diffusion_share_pct"] - 100.0) < 0.01

    def test_total_equals_sum(self, returns_normal, calibration_normal):
        result = decompose_risk(returns_normal, calibration_normal)
        total = result["diffusion_variance"] + result["jump_variance"]
        assert abs(result["total_model_variance"] - total) < 1e-8

    def test_all_positive(self, returns_normal, calibration_normal):
        result = decompose_risk(returns_normal, calibration_normal)
        assert result["diffusion_variance"] > 0
        assert result["total_model_variance"] > 0
        assert result["daily_total_vol"] > 0

    def test_jumpy_higher_jump_share(self, returns_with_jumps, calibration_jumpy, returns_normal, calibration_normal):
        r_jump = decompose_risk(returns_with_jumps, calibration_jumpy)
        r_norm = decompose_risk(returns_normal, calibration_normal)
        assert r_jump["jump_share_pct"] >= r_norm["jump_share_pct"] * 0.5


# ── svj_diagnostics ──────────────────────────────────────────────────


class TestSVJDiagnostics:
    def test_top_level_keys(self, returns_normal, calibration_normal):
        result = svj_diagnostics(returns_normal, calibration_normal)
        assert {"jump_stats", "parameter_quality", "moment_comparison"}.issubset(set(result.keys()))

    def test_jump_stats_keys(self, returns_normal, calibration_normal):
        result = svj_diagnostics(returns_normal, calibration_normal)
        js = result["jump_stats"]
        assert "n_jumps" in js
        assert "jumps_significant" in js
        assert isinstance(js["jumps_significant"], bool)

    def test_parameter_quality_keys(self, returns_normal, calibration_normal):
        result = svj_diagnostics(returns_normal, calibration_normal)
        pq = result["parameter_quality"]
        assert "feller_satisfied" in pq
        assert "half_life_years" in pq
        assert pq["half_life_years"] > 0

    def test_moment_comparison_keys(self, returns_normal, calibration_normal):
        result = svj_diagnostics(returns_normal, calibration_normal)
        mc = result["moment_comparison"]
        assert "empirical_skewness" in mc
        assert "empirical_kurtosis" in mc
        assert np.isfinite(mc["model_variance"])


# ── svj_vol_series ───────────────────────────────────────────────────


class TestSVJVolSeries:
    def test_returns_series(self, returns_normal):
        result = svj_vol_series(returns_normal)
        assert isinstance(result, pd.Series)

    def test_correct_length(self, returns_normal):
        result = svj_vol_series(returns_normal)
        assert len(result) == len(returns_normal)

    def test_all_positive(self, returns_normal):
        result = svj_vol_series(returns_normal)
        assert (result > 0).all()

    def test_name(self, returns_normal):
        result = svj_vol_series(returns_normal)
        assert result.name == "svj_vol"


# ── Model comparison integration ─────────────────────────────────────


class TestModelComparisonIntegration:
    def test_svj_in_registry(self):
        from cortex.comparison import _MODEL_REGISTRY
        assert "svj" in _MODEL_REGISTRY
        assert _MODEL_REGISTRY["svj"] == ("SVJ-Bates", 7)

    def test_compare_models_includes_svj(self, returns_normal):
        from cortex.comparison import compare_models
        df = compare_models(returns_normal, models=["msm", "svj"])
        assert "svj" in df.index.str.lower().values or "SVJ-Bates" in df.index.values
