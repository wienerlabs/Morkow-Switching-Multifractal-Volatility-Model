"""Tests for Guardian launch_tracker component integration."""
import os
import time
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("TESTING", "1")

from cortex.guardian import _cache, assess_trade


@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()


def _minimal_model_data():
    """Minimal model_data for assess_trade."""
    probs = pd.DataFrame(np.array([[0.7, 0.2, 0.05, 0.03, 0.02]]))
    return {
        "filter_probs": probs,
        "calibration": {"num_states": 5, "sigma_states": [0.01, 0.02, 0.03, 0.04, 0.05]},
    }


def _minimal_evt_data():
    return {"xi": 0.1, "beta": 0.5, "threshold": 1.0, "n_total": 300, "n_exceedances": 30}


# ── _score_launch_tracker integration ──

@patch("cortex.evt.evt_var", return_value=3.0)
@patch("cortex.guardian.LAUNCH_TRACKER_ENABLED", True)
def test_launch_tracker_included_when_enabled(mock_var):
    """When LAUNCH_TRACKER_ENABLED and launch_data provided, component appears in scores."""
    launch_data = {
        "score": 45,
        "cex_funded": True,
        "bundle_detected": False,
        "deployer_age_days": 2.0,
        "top10_concentration_pct": 60.0,
        "deploy_to_first_trade_sec": 120.0,
        "risk_factors": ["deployer_cex_funded:Binance"],
    }
    result = assess_trade(
        token="TEST",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        launch_data=launch_data,
    )
    components = [s["component"] for s in result["component_scores"]]
    assert "launch_tracker" in components
    lt = next(s for s in result["component_scores"] if s["component"] == "launch_tracker")
    assert lt["score"] == 45.0
    assert lt["details"]["cex_funded"] is True


@patch("cortex.evt.evt_var", return_value=3.0)
def test_launch_tracker_not_included_when_disabled(mock_var):
    """When LAUNCH_TRACKER_ENABLED is False (default), component is excluded."""
    launch_data = {"score": 80, "cex_funded": True, "bundle_detected": True}
    result = assess_trade(
        token="TEST2",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        launch_data=launch_data,
    )
    components = [s["component"] for s in result["component_scores"]]
    assert "launch_tracker" not in components


@patch("cortex.evt.evt_var", return_value=3.0)
@patch("cortex.guardian.LAUNCH_TRACKER_ENABLED", True)
@patch("cortex.guardian.LAUNCH_TRACKER_VETO_THRESHOLD", 90)
def test_launch_tracker_hard_veto(mock_var):
    """Score >= 90 + cex_funded + bundle_detected triggers hard veto."""
    launch_data = {
        "score": 95,
        "cex_funded": True,
        "bundle_detected": True,
        "deployer_age_days": 0.5,
        "top10_concentration_pct": 92.0,
        "deploy_to_first_trade_sec": 5.0,
        "risk_factors": ["deployer_cex_funded:Binance", "bundle_detected:4_wallets"],
    }
    result = assess_trade(
        token="SCAM",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        launch_data=launch_data,
    )
    assert "launch_tracker_suspicious_launch" in result["veto_reasons"]
    assert result["approved"] is False


@patch("cortex.evt.evt_var", return_value=3.0)
@patch("cortex.guardian.LAUNCH_TRACKER_ENABLED", True)
@patch("cortex.guardian.LAUNCH_TRACKER_VETO_THRESHOLD", 90)
def test_launch_tracker_high_score_no_veto_without_both_flags(mock_var):
    """Score >= 90 but missing cex_funded or bundle_detected → no hard veto."""
    launch_data = {
        "score": 95,
        "cex_funded": True,
        "bundle_detected": False,  # missing bundle
        "deployer_age_days": 0.5,
        "top10_concentration_pct": 92.0,
        "deploy_to_first_trade_sec": 5.0,
        "risk_factors": ["deployer_cex_funded:Binance"],
    }
    result = assess_trade(
        token="MAYBE",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        launch_data=launch_data,
    )
    assert "launch_tracker_suspicious_launch" not in result["veto_reasons"]


@patch("cortex.evt.evt_var", return_value=3.0)
def test_launch_data_none_no_effect(mock_var):
    """When launch_data is None, no launch_tracker component added."""
    result = assess_trade(
        token="SAFE",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        launch_data=None,
    )
    components = [s["component"] for s in result["component_scores"]]
    assert "launch_tracker" not in components


@patch("cortex.evt.evt_var", return_value=3.0)
@patch("cortex.guardian.LAUNCH_TRACKER_ENABLED", True)
def test_launch_tracker_score_clamped(mock_var):
    """Score values outside 0-100 are clamped."""
    launch_data = {"score": 150, "cex_funded": False, "bundle_detected": False}
    result = assess_trade(
        token="CLAMP",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        launch_data=launch_data,
    )
    lt = next(s for s in result["component_scores"] if s["component"] == "launch_tracker")
    assert lt["score"] == 100.0


@patch("cortex.evt.evt_var", return_value=3.0)
@patch("cortex.guardian.LAUNCH_TRACKER_ENABLED", True)
def test_launch_tracker_zero_score(mock_var):
    """Score 0 (old token) contributes zero risk."""
    launch_data = {
        "score": 0,
        "cex_funded": False,
        "bundle_detected": False,
        "deployer_age_days": 365.0,
        "top10_concentration_pct": 15.0,
        "deploy_to_first_trade_sec": 0.0,
        "risk_factors": [],
    }
    result = assess_trade(
        token="OLD",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        launch_data=launch_data,
    )
    lt = next(s for s in result["component_scores"] if s["component"] == "launch_tracker")
    assert lt["score"] == 0.0
