"""Tests for Guardian ghost_watcher component integration."""
import os
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


# ── _score_ghost_watcher integration ──


@patch("cortex.evt.evt_var", return_value=3.0)
@patch("cortex.guardian.GHOST_WATCHER_ENABLED", True)
def test_ghost_watcher_included_when_enabled(mock_var):
    """When GHOST_WATCHER_ENABLED and ghost_watcher_data provided, component appears in scores."""
    gw_data = {
        "risk_score": 55,
        "dormant_whales_detected": 3,
        "wallets_reactivating": 2,
        "aggregate_dormant_balance_pct": 8.5,
        "cluster_detected": False,
        "risk_factors": ["dormant_reactivation:2_wallets", "reactivation_cluster:3_wallets"],
    }
    result = assess_trade(
        token="GW_TEST",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        ghost_watcher_data=gw_data,
    )
    components = [s["component"] for s in result["component_scores"]]
    assert "ghost_watcher" in components
    gw = next(s for s in result["component_scores"] if s["component"] == "ghost_watcher")
    assert gw["score"] == 55.0
    assert gw["details"]["dormant_whales_detected"] == 3


@patch("cortex.evt.evt_var", return_value=3.0)
def test_ghost_watcher_not_included_when_disabled(mock_var):
    """When GHOST_WATCHER_ENABLED is False (default), component is excluded."""
    gw_data = {"risk_score": 80, "cluster_detected": True}
    result = assess_trade(
        token="GW_OFF",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        ghost_watcher_data=gw_data,
    )
    components = [s["component"] for s in result["component_scores"]]
    assert "ghost_watcher" not in components


@patch("cortex.evt.evt_var", return_value=3.0)
@patch("cortex.guardian.GHOST_WATCHER_ENABLED", True)
@patch("cortex.guardian.GHOST_WATCHER_VETO_THRESHOLD", 80)
def test_ghost_watcher_hard_veto(mock_var):
    """Score >= 80 + cluster_detected + high concentration triggers hard veto."""
    gw_data = {
        "risk_score": 85,
        "dormant_whales_detected": 5,
        "wallets_reactivating": 4,
        "aggregate_dormant_balance_pct": 15.0,
        "cluster_detected": True,
        "risk_factors": [
            "dormant_reactivation:4_wallets",
            "reactivation_cluster:4_wallets",
            "high_concentration:15.0%_supply",
        ],
    }
    result = assess_trade(
        token="RUG",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        ghost_watcher_data=gw_data,
    )
    assert "ghost_watcher_coordinated_dump" in result["veto_reasons"]
    assert result["approved"] is False


@patch("cortex.evt.evt_var", return_value=3.0)
@patch("cortex.guardian.GHOST_WATCHER_ENABLED", True)
@patch("cortex.guardian.GHOST_WATCHER_VETO_THRESHOLD", 80)
def test_ghost_watcher_high_score_no_veto_without_cluster_and_concentration(mock_var):
    """Score >= 80 but missing cluster or high concentration → no hard veto."""
    gw_data = {
        "risk_score": 85,
        "dormant_whales_detected": 2,
        "wallets_reactivating": 1,
        "aggregate_dormant_balance_pct": 5.0,  # below 10% threshold
        "cluster_detected": False,
        "risk_factors": ["dormant_reactivation:1_wallets"],
    }
    result = assess_trade(
        token="MAYBE_GW",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        ghost_watcher_data=gw_data,
    )
    assert "ghost_watcher_coordinated_dump" not in result["veto_reasons"]


@patch("cortex.evt.evt_var", return_value=3.0)
def test_ghost_watcher_data_none_no_effect(mock_var):
    """When ghost_watcher_data is None, no ghost_watcher component added."""
    result = assess_trade(
        token="SAFE_GW",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        ghost_watcher_data=None,
    )
    components = [s["component"] for s in result["component_scores"]]
    assert "ghost_watcher" not in components


@patch("cortex.evt.evt_var", return_value=3.0)
@patch("cortex.guardian.GHOST_WATCHER_ENABLED", True)
def test_ghost_watcher_score_clamped(mock_var):
    """Score values outside 0-100 are clamped."""
    gw_data = {"risk_score": 150, "cluster_detected": False}
    result = assess_trade(
        token="CLAMP_GW",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        ghost_watcher_data=gw_data,
    )
    gw = next(s for s in result["component_scores"] if s["component"] == "ghost_watcher")
    assert gw["score"] == 100.0


@patch("cortex.evt.evt_var", return_value=3.0)
@patch("cortex.guardian.GHOST_WATCHER_ENABLED", True)
def test_ghost_watcher_zero_score(mock_var):
    """Score 0 (no dormant whales) contributes zero risk."""
    gw_data = {
        "risk_score": 0,
        "dormant_whales_detected": 0,
        "wallets_reactivating": 0,
        "aggregate_dormant_balance_pct": 0.0,
        "cluster_detected": False,
        "risk_factors": [],
    }
    result = assess_trade(
        token="CLEAN_GW",
        trade_size_usd=100,
        direction="long",
        model_data=_minimal_model_data(),
        evt_data=_minimal_evt_data(),
        svj_data=None,
        hawkes_data=None,
        ghost_watcher_data=gw_data,
    )
    gw = next(s for s in result["component_scores"] if s["component"] == "ghost_watcher")
    assert gw["score"] == 0.0
