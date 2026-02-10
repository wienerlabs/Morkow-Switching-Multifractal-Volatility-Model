"""Tests for model_comparison.py."""

import numpy as np
import pandas as pd
import pytest

import model_comparison as mc

msm = __import__("MSM-VaR_MODEL")


@pytest.fixture(scope="module")
def comparison_data():
    """Synthetic returns for comparison tests."""
    rng = np.random.RandomState(99)
    return pd.Series(rng.randn(200) * 1.5, name="test")


class TestIndividualModels:
    def test_rolling_vol(self, comparison_data):
        result = mc.rolling_vol_forecast(comparison_data, window=20)
        assert isinstance(result, pd.Series)
        assert len(result) == len(comparison_data)
        assert result.notna().sum() > 0

    def test_ewma_vol(self, comparison_data):
        result = mc.ewma_vol_forecast(comparison_data, lam=0.94)
        assert isinstance(result, pd.Series)
        assert len(result) == len(comparison_data)
        assert result.iloc[-1] > 0

    @pytest.mark.slow
    def test_garch_vol(self, comparison_data):
        result = mc.garch_vol_forecast(comparison_data, min_obs=50)
        assert isinstance(result, pd.Series)
        assert result.notna().sum() > 0


class TestCompareModels:
    def test_returns_dataframe(self, comparison_data):
        cal = msm.calibrate_msm_advanced(
            comparison_data, num_states=5, method="empirical", verbose=False
        )
        sf, _, _, _, _ = msm.msm_vol_forecast(
            comparison_data,
            num_states=cal["num_states"],
            sigma_low=cal["sigma_low"],
            sigma_high=cal["sigma_high"],
            p_stay=cal["p_stay"],
        )
        df = mc.compare_models(
            comparison_data, sf, cal,
            models=["rolling_20", "ewma"],
            alpha=0.05,
        )
        assert isinstance(df, pd.DataFrame)
        assert "MSM" in df.index
        assert "breach_rate" in df.columns

    def test_all_models_have_metrics(self, comparison_data):
        cal = msm.calibrate_msm_advanced(
            comparison_data, num_states=5, method="empirical", verbose=False
        )
        sf, _, _, _, _ = msm.msm_vol_forecast(
            comparison_data,
            num_states=cal["num_states"],
            sigma_low=cal["sigma_low"],
            sigma_high=cal["sigma_high"],
            p_stay=cal["p_stay"],
        )
        df = mc.compare_models(
            comparison_data, sf, cal,
            models=["rolling_20", "ewma"],
            alpha=0.05,
        )
        for col in ["breach_rate", "kupiec_pvalue", "aic"]:
            assert col in df.columns


class TestGenerateReport:
    def test_report_structure(self, comparison_data):
        cal = msm.calibrate_msm_advanced(
            comparison_data, num_states=5, method="empirical", verbose=False
        )
        sf, _, _, _, _ = msm.msm_vol_forecast(
            comparison_data,
            num_states=cal["num_states"],
            sigma_low=cal["sigma_low"],
            sigma_high=cal["sigma_high"],
            p_stay=cal["p_stay"],
        )
        df = mc.compare_models(
            comparison_data, sf, cal,
            models=["rolling_20", "ewma"],
            alpha=0.05,
        )
        report = mc.generate_comparison_report(df)
        assert "summary_table" in report
        assert "winners" in report
        assert "ranking" in report
        assert isinstance(report["ranking"], list)

