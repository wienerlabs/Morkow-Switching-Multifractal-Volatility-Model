import numpy as np
import pandas as pd
import pytest

from cortex.rough_vol import (
    calibrate_rough_bergomi,
    calibrate_rough_heston,
    compare_rough_vs_msm,
    estimate_roughness,
    generate_fbm,
    rough_bergomi_vol_series,
    rough_vol_forecast,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def long_returns():
    """300-point synthetic returns — enough for variogram estimation."""
    rng = np.random.RandomState(42)
    return pd.Series(rng.randn(300) * 1.5, name="synthetic")


@pytest.fixture(scope="module")
def rough_fbm_returns():
    """Returns driven by rough fBm (H=0.1) — should detect roughness."""
    fbm_inc = generate_fbm(500, H=0.1, seed=77)
    vol = np.exp(np.cumsum(fbm_inc) * 0.3)
    rng = np.random.RandomState(77)
    returns = rng.randn(500) * vol
    return pd.Series(returns)


@pytest.fixture(scope="module")
def bergomi_cal(long_returns):
    return calibrate_rough_bergomi(long_returns)


@pytest.fixture(scope="module")
def heston_cal(long_returns):
    return calibrate_rough_heston(long_returns)


# ── generate_fbm tests ───────────────────────────────────────────────


class TestGenerateFBM:
    def test_correct_length(self):
        inc = generate_fbm(100, H=0.1, seed=1)
        assert len(inc) == 100

    def test_reproducibility(self):
        a = generate_fbm(50, H=0.3, seed=42)
        b = generate_fbm(50, H=0.3, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds(self):
        a = generate_fbm(50, H=0.3, seed=1)
        b = generate_fbm(50, H=0.3, seed=2)
        assert not np.allclose(a, b)

    def test_invalid_H_raises(self):
        with pytest.raises(ValueError, match="Hurst exponent"):
            generate_fbm(100, H=0.0)
        with pytest.raises(ValueError, match="Hurst exponent"):
            generate_fbm(100, H=1.0)

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError, match="n must be"):
            generate_fbm(1, H=0.5)

    def test_mean_near_zero(self):
        inc = generate_fbm(10000, H=0.5, seed=42)
        assert abs(np.mean(inc)) < 0.1

    def test_rough_H_produces_output(self):
        inc = generate_fbm(200, H=0.05, seed=10)
        assert len(inc) == 200
        assert np.all(np.isfinite(inc))

    def test_smooth_H_produces_output(self):
        inc = generate_fbm(200, H=0.9, seed=10)
        assert len(inc) == 200
        assert np.all(np.isfinite(inc))


# ── estimate_roughness tests ─────────────────────────────────────────


class TestEstimateRoughness:
    def test_returns_valid_H(self, long_returns):
        result = estimate_roughness(long_returns)
        assert 0.0 < result["H"] < 1.0

    def test_has_required_keys(self, long_returns):
        result = estimate_roughness(long_returns)
        for key in ["H", "H_se", "r_squared", "method", "lags", "variogram", "is_rough", "interpretation"]:
            assert key in result

    def test_method_is_variogram(self, long_returns):
        result = estimate_roughness(long_returns)
        assert result["method"] == "variogram"

    def test_short_series_raises(self):
        with pytest.raises(ValueError):
            estimate_roughness(np.random.randn(20))

    def test_r_squared_positive(self, long_returns):
        result = estimate_roughness(long_returns)
        assert result["r_squared"] > 0.0


# ── calibrate_rough_bergomi tests ────────────────────────────────────


class TestCalibrateRoughBergomi:
    def test_returns_model_name(self, bergomi_cal):
        assert bergomi_cal["model"] == "rough_bergomi"

    def test_has_required_keys(self, bergomi_cal):
        for key in ["model", "H", "nu", "V0", "sigma0", "metrics", "method"]:
            assert key in bergomi_cal

    def test_H_in_valid_range(self, bergomi_cal):
        assert 0.0 < bergomi_cal["H"] < 1.0

    def test_nu_positive(self, bergomi_cal):
        assert bergomi_cal["nu"] > 0

    def test_V0_positive(self, bergomi_cal):
        assert bergomi_cal["V0"] > 0

    def test_metrics_has_correlation(self, bergomi_cal):
        assert "vol_correlation" in bergomi_cal["metrics"]


# ── calibrate_rough_heston tests ─────────────────────────────────────


class TestCalibrateRoughHeston:
    def test_returns_model_name(self, heston_cal):
        assert heston_cal["model"] == "rough_heston"

    def test_has_required_keys(self, heston_cal):
        for key in ["model", "H", "lambda_", "theta", "xi", "V0", "metrics", "method"]:
            assert key in heston_cal

    def test_lambda_positive(self, heston_cal):
        assert heston_cal["lambda_"] > 0

    def test_theta_positive(self, heston_cal):
        assert heston_cal["theta"] > 0

    def test_xi_positive(self, heston_cal):
        assert heston_cal["xi"] > 0

    def test_optimization_info_in_metrics(self, heston_cal):
        assert "optimization_success" in heston_cal["metrics"]
        assert "optimization_nit" in heston_cal["metrics"]


# ── rough_vol_forecast tests ─────────────────────────────────────────


class TestRoughVolForecast:
    def test_bergomi_forecast_length(self, long_returns, bergomi_cal):
        result = rough_vol_forecast(long_returns, bergomi_cal, horizon=10, n_paths=100, seed=42)
        assert len(result["point_forecast"]) == 10

    def test_heston_forecast_length(self, long_returns, heston_cal):
        result = rough_vol_forecast(long_returns, heston_cal, horizon=5, n_paths=100, seed=42)
        assert len(result["point_forecast"]) == 5

    def test_forecast_has_confidence_bands(self, long_returns, bergomi_cal):
        result = rough_vol_forecast(long_returns, bergomi_cal, horizon=10, n_paths=200, seed=42)
        assert len(result["lower_95"]) == 10
        assert len(result["upper_95"]) == 10
        for i in range(10):
            assert result["lower_95"][i] <= result["upper_95"][i]

    def test_forecast_positive_values(self, long_returns, bergomi_cal):
        result = rough_vol_forecast(long_returns, bergomi_cal, horizon=10, n_paths=100, seed=42)
        assert all(v > 0 for v in result["point_forecast"])

    def test_forecast_model_name(self, long_returns, bergomi_cal):
        result = rough_vol_forecast(long_returns, bergomi_cal, horizon=5, n_paths=100, seed=42)
        assert result["model"] == "rough_bergomi"

    def test_unknown_model_raises(self, long_returns):
        fake_cal = {"model": "unknown", "H": 0.1}
        with pytest.raises(ValueError, match="Unknown model"):
            rough_vol_forecast(long_returns, fake_cal, horizon=5, n_paths=100)


# ── compare_rough_vs_msm tests ───────────────────────────────────────


class TestCompareRoughVsMSM:
    def test_returns_winner(self, calibrated_model):
        result = compare_rough_vs_msm(
            calibrated_model["returns"], calibrated_model["calibration"]
        )
        assert result["winner"] in ("rough_bergomi", "msm")

    def test_has_both_models(self, calibrated_model):
        result = compare_rough_vs_msm(
            calibrated_model["returns"], calibrated_model["calibration"]
        )
        assert "rough_bergomi" in result
        assert "msm" in result

    def test_metrics_are_finite(self, calibrated_model):
        result = compare_rough_vs_msm(
            calibrated_model["returns"], calibrated_model["calibration"]
        )
        assert np.isfinite(result["rough_bergomi"]["mae"])
        assert np.isfinite(result["msm"]["mae"])

    def test_comparison_metrics_present(self, calibrated_model):
        result = compare_rough_vs_msm(
            calibrated_model["returns"], calibrated_model["calibration"]
        )
        cm = result["comparison_metrics"]
        assert "mae_ratio" in cm
        assert "rmse_ratio" in cm
        assert "corr_diff" in cm


# ── rough_bergomi_vol_series tests ───────────────────────────────────


class TestRoughBergomiVolSeries:
    def test_returns_series(self, long_returns):
        result = rough_bergomi_vol_series(long_returns)
        assert isinstance(result, pd.Series)

    def test_same_length_as_input(self, long_returns):
        result = rough_bergomi_vol_series(long_returns)
        assert len(result) == len(long_returns)

    def test_all_positive(self, long_returns):
        result = rough_bergomi_vol_series(long_returns)
        assert (result > 0).all()


# ── model_comparison integration ─────────────────────────────────────


class TestModelComparisonIntegration:
    def test_rough_bergomi_in_registry(self):
        from cortex.comparison import _MODEL_REGISTRY
        assert "rough_bergomi" in _MODEL_REGISTRY

    def test_compare_includes_rough(self, sample_returns):
        from cortex.comparison import compare_models
        df = compare_models(sample_returns, models=["rough_bergomi", "ewma"])
        assert len(df) == 2
        assert "Rough-Bergomi" in df["model"].values

