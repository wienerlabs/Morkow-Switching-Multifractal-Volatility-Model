"""Integration tests: MSM → EVT → Hawkes → SVJ → Guardian full pipeline.

Uses real model functions (no mocks) with synthetic data from conftest fixtures.
"""
import numpy as np
import pandas as pd
import pytest

from cortex import msm, evt, hawkes, svj
from cortex.guardian import assess_trade, _cache


@pytest.fixture(autouse=True)
def clear_guardian_cache():
    _cache.clear()
    yield
    _cache.clear()


class TestFullPipeline:
    """End-to-end: synthetic data → MSM → EVT → Hawkes → SVJ → Guardian."""

    def test_msm_to_evt_pipeline(self, sample_returns, calibrated_model):
        """MSM calibration feeds into EVT tail risk estimation."""
        m = calibrated_model
        assert "calibration" in m
        assert "filter_probs" in m

        # EVT: select threshold → fit GPD → compute VaR
        losses = np.abs(sample_returns.values)
        thresh_result = evt.select_threshold(sample_returns, method="percentile")
        assert "threshold" in thresh_result
        assert thresh_result["n_exceedances"] >= 10

        gpd = evt.fit_gpd(losses, thresh_result["threshold"])
        assert gpd["xi"] is not None
        assert gpd["beta"] > 0
        assert gpd["n_exceedances"] >= 10

        var_99 = evt.evt_var(
            xi=gpd["xi"], beta=gpd["beta"],
            threshold=gpd["threshold"],
            n_total=gpd["n_total"],
            n_exceedances=gpd["n_exceedances"],
            alpha=0.01,
        )
        assert var_99 > 0  # positive loss magnitude

    def test_hawkes_pipeline(self, sample_returns):
        """Extract events → fit Hawkes → detect flash crash risk."""
        events = hawkes.extract_events(sample_returns, threshold_percentile=5.0)
        assert events["n_events"] >= 5, "Need enough extreme events"

        fit = hawkes.fit_hawkes(events["event_times"], events["T"])
        assert fit["mu"] > 0
        assert fit["alpha"] >= 0
        assert fit["beta"] > 0
        assert 0 <= fit["branching_ratio"] < 2.0

        risk = hawkes.detect_flash_crash_risk(events["event_times"], fit)
        assert risk["risk_level"] in ("low", "moderate", "elevated", "critical")
        assert 0 <= risk["contagion_risk_score"] <= 1.0

    def test_svj_pipeline(self, sample_returns):
        """Calibrate SVJ → decompose risk into diffusion + jump."""
        # Add synthetic jumps to make SVJ calibration meaningful
        jumpy = sample_returns.copy()
        rng = np.random.RandomState(99)
        jump_idx = rng.choice(len(jumpy), size=6, replace=False)
        jumpy.iloc[jump_idx] += rng.choice([-1, 1], size=6) * rng.uniform(4, 7, size=6)

        cal = svj.calibrate_svj(jumpy)
        assert "lambda_" in cal
        assert "mu_j" in cal
        assert "sigma_j" in cal
        assert "theta" in cal

        risk = svj.decompose_risk(jumpy, cal)
        assert "jump_share_pct" in risk
        assert 0 <= risk["jump_share_pct"] <= 100
        assert risk["daily_jump_vol"] >= 0

    def test_full_guardian_pipeline(self, sample_returns, calibrated_model):
        """Full chain: MSM + EVT + Hawkes + SVJ → Guardian assess_trade."""
        m = calibrated_model

        # 1. EVT
        losses = np.abs(sample_returns.values)
        thresh = evt.select_threshold(sample_returns, method="percentile")
        gpd = evt.fit_gpd(losses, thresh["threshold"])

        # 2. SVJ (with synthetic jumps)
        jumpy = sample_returns.copy()
        rng = np.random.RandomState(77)
        jump_idx = rng.choice(len(jumpy), size=5, replace=False)
        jumpy.iloc[jump_idx] += rng.choice([-1, 1], size=5) * rng.uniform(4, 7, size=5)
        svj_cal = svj.calibrate_svj(jumpy)

        # 3. Hawkes
        events = hawkes.extract_events(sample_returns, threshold_percentile=5.0)
        assert events["n_events"] >= 5
        hawkes_fit = hawkes.fit_hawkes(events["event_times"], events["T"])

        # 4. Guardian assess_trade with all components
        result = assess_trade(
            token="INTEGRATION_TEST",
            trade_size_usd=10000,
            direction="long",
            model_data=m,
            evt_data=gpd,
            svj_data={"returns": jumpy, "calibration": svj_cal},
            hawkes_data={
                "event_times": events["event_times"],
                "mu": hawkes_fit["mu"],
                "alpha": hawkes_fit["alpha"],
                "beta": hawkes_fit["beta"],
            },
        )

        assert "approved" in result
        assert "risk_score" in result
        assert "recommended_size" in result
        assert "component_scores" in result
        assert "confidence" in result
        assert isinstance(result["approved"], bool)
        assert 0 <= result["risk_score"] <= 100
        assert result["recommended_size"] > 0
        assert result["confidence"] > 0
        assert result["from_cache"] is False
        # Should have at least 4 component scores (evt, svj, hawkes, regime)
        assert len(result["component_scores"]) >= 4

    def test_msm_var_forecast(self, calibrated_model):
        """MSM VaR forecast produces valid risk estimate."""
        m = calibrated_model
        var_t1, sigma_t1, z_alpha, pi_t1 = msm.msm_var_forecast_next_day(
            m["filter_probs"], m["sigma_states"], m["P_matrix"], alpha=0.05,
        )
        assert var_t1 < 0  # VaR is negative (loss)
        assert sigma_t1 > 0
        assert z_alpha < 0  # quantile for alpha=0.05 is negative
        assert len(pi_t1) == m["calibration"]["num_states"]
        assert abs(sum(pi_t1) - 1.0) < 1e-6

    def test_guardian_partial_data(self, calibrated_model):
        """Guardian works with only MSM data (no EVT/SVJ/Hawkes)."""
        result = assess_trade(
            token="PARTIAL_INT",
            trade_size_usd=5000,
            direction="short",
            model_data=calibrated_model,
            evt_data=None,
            svj_data=None,
            hawkes_data=None,
        )
        assert result["approved"] is True or result["approved"] is False
        assert result["confidence"] < 1.0  # partial data → lower confidence
        assert len(result["component_scores"]) >= 1

