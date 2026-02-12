import numpy as np
import pandas as pd
import pytest

from cortex.multifractal import (
    compare_fractal_regimes,
    hurst_dfa,
    hurst_rs,
    long_range_dependence_test,
    multifractal_spectrum,
    multifractal_width,
)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def random_walk():
    """Pure random walk — H ≈ 0.5."""
    rng = np.random.RandomState(99)
    return pd.Series(rng.randn(500))


@pytest.fixture(scope="module")
def persistent_series():
    """Cumulative sum of random walk — strongly persistent H > 0.5."""
    rng = np.random.RandomState(99)
    increments = rng.randn(500)
    return pd.Series(np.cumsum(increments))


# ── hurst_rs tests ────────────────────────────────────────────────────


class TestHurstRS:
    def test_returns_valid_H(self, random_walk):
        result = hurst_rs(random_walk)
        assert 0.0 < result["H"] < 1.5

    def test_output_keys(self, random_walk):
        result = hurst_rs(random_walk)
        expected = {"H", "H_se", "r_squared", "intercept", "p_value",
                    "window_sizes", "rs_values", "interpretation", "method"}
        assert expected == set(result.keys())

    def test_method_is_rs(self, random_walk):
        assert hurst_rs(random_walk)["method"] == "rs"

    def test_r_squared_positive(self, random_walk):
        assert hurst_rs(random_walk)["r_squared"] > 0.5

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="50"):
            hurst_rs(np.random.randn(20))

    def test_persistent_series_high_H(self, persistent_series):
        result = hurst_rs(persistent_series)
        assert result["H"] > 0.55


# ── hurst_dfa tests ───────────────────────────────────────────────────


class TestHurstDFA:
    def test_returns_valid_H(self, random_walk):
        result = hurst_dfa(random_walk)
        assert 0.0 < result["H"] < 1.5

    def test_output_keys(self, random_walk):
        result = hurst_dfa(random_walk)
        expected = {"H", "H_se", "r_squared", "intercept", "p_value",
                    "order", "window_sizes", "fluctuations", "interpretation", "method"}
        assert expected == set(result.keys())

    def test_method_is_dfa(self, random_walk):
        assert hurst_dfa(random_walk)["method"] == "dfa"

    def test_order_2(self, random_walk):
        result = hurst_dfa(random_walk, order=2)
        assert result["H"] > 0.0

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="50"):
            hurst_dfa(np.random.randn(20))


# ── multifractal_spectrum tests ───────────────────────────────────────


class TestMultifractalSpectrum:
    def test_output_keys(self, random_walk):
        spec = multifractal_spectrum(random_walk)
        expected = {"q_values", "tau_q", "H_q", "alpha", "f_alpha",
                    "width", "peak_alpha", "is_multifractal"}
        assert expected == set(spec.keys())

    def test_width_positive(self, random_walk):
        spec = multifractal_spectrum(random_walk)
        assert spec["width"] > 0.0

    def test_lists_same_length(self, random_walk):
        spec = multifractal_spectrum(random_walk)
        n = len(spec["q_values"])
        assert len(spec["tau_q"]) == n
        assert len(spec["alpha"]) == n
        assert len(spec["f_alpha"]) == n
        assert len(spec["H_q"]) == n

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="50"):
            multifractal_spectrum(np.random.randn(20))


# ── multifractal_width tests ─────────────────────────────────────────


class TestMultifractalWidth:
    def test_output_keys(self, random_walk):
        result = multifractal_width(random_walk)
        expected = {"width", "alpha_min", "alpha_max", "peak_alpha", "is_multifractal"}
        assert expected == set(result.keys())

    def test_alpha_range_consistent(self, random_walk):
        result = multifractal_width(random_walk)
        assert result["alpha_max"] > result["alpha_min"]
        assert abs(result["width"] - (result["alpha_max"] - result["alpha_min"])) < 1e-10


# ── long_range_dependence_test tests ─────────────────────────────────


class TestLongRangeDependence:
    def test_output_keys(self, random_walk):
        result = long_range_dependence_test(random_walk)
        expected = {"H_rs", "H_rs_se", "H_dfa", "H_dfa_se",
                    "is_long_range_dependent", "confidence_z", "interpretation"}
        assert expected == set(result.keys())

    def test_returns_bool(self, random_walk):
        result = long_range_dependence_test(random_walk)
        assert isinstance(result["is_long_range_dependent"], bool)

    def test_persistent_series_detected(self, persistent_series):
        result = long_range_dependence_test(persistent_series)
        assert result["H_rs"] > 0.5
        assert result["H_dfa"] > 0.5


# ── compare_fractal_regimes tests ────────────────────────────────────


class TestCompareFractalRegimes:
    def test_output_keys(self, random_walk):
        n = len(random_walk)
        fp = pd.DataFrame(
            np.random.dirichlet([1, 1, 1], size=n),
            columns=["state_1", "state_2", "state_3"],
            index=random_walk.index if hasattr(random_walk, "index") else range(n),
        )
        sigma_states = np.array([1.0, 2.0, 4.0])
        result = compare_fractal_regimes(random_walk, fp, sigma_states)
        assert "per_regime" in result
        assert "n_states" in result
        assert "summary" in result
        assert result["n_states"] == 3

    def test_regime_items_structure(self, random_walk):
        n = len(random_walk)
        fp = pd.DataFrame(
            np.random.dirichlet([1, 1, 1], size=n),
            columns=["state_1", "state_2", "state_3"],
            index=random_walk.index if hasattr(random_walk, "index") else range(n),
        )
        sigma_states = np.array([1.0, 2.0, 4.0])
        result = compare_fractal_regimes(random_walk, fp, sigma_states)
        for item in result["per_regime"]:
            assert "regime" in item
            assert "sigma" in item
            assert "n_obs" in item
            assert "interpretation" in item

    def test_with_calibrated_model(self, calibrated_model):
        result = compare_fractal_regimes(
            calibrated_model["returns"],
            calibrated_model["filter_probs"],
            calibrated_model["sigma_states"],
        )
        assert result["n_states"] == len(calibrated_model["sigma_states"])
        total_obs = sum(r["n_obs"] for r in result["per_regime"])
        expected = min(len(calibrated_model["returns"]), len(calibrated_model["filter_probs"]))
        assert total_obs == expected


# ── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_constant_series_rs_raises(self):
        with pytest.raises(ValueError):
            hurst_rs(np.ones(100))

    def test_numpy_array_input(self):
        rng = np.random.RandomState(7)
        arr = rng.randn(200)
        result = hurst_rs(arr)
        assert 0.0 < result["H"] < 1.5

    def test_rs_and_dfa_agree_roughly(self, random_walk):
        rs = hurst_rs(random_walk)
        dfa = hurst_dfa(random_walk)
        assert abs(rs["H"] - dfa["H"]) < 0.5  # same ballpark


# ── MSM integration ──────────────────────────────────────────────────


class TestMSMIntegration:
    def test_full_diagnostics_pipeline(self, calibrated_model):
        """End-to-end: calibrated MSM → fractal diagnostics."""
        ret = calibrated_model["returns"]
        rs = hurst_rs(ret)
        dfa = hurst_dfa(ret)
        spec = multifractal_spectrum(ret)
        lrd = long_range_dependence_test(ret)

        assert 0.0 < rs["H"] < 1.5
        assert 0.0 < dfa["H"] < 1.5
        assert spec["width"] > 0.0
        assert isinstance(lrd["is_long_range_dependent"], bool)

