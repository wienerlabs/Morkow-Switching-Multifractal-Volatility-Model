"""Tests for cortex/guardian.py — Guardian risk veto layer."""
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from cortex import guardian
from cortex.guardian import (
    _cache,
    _recommend_size,
    _score_evt,
    _score_hawkes,
    _score_news,
    _score_regime,
    _score_svj,
    assess_trade,
    CIRCUIT_BREAKER_THRESHOLD,
    CACHE_TTL_SECONDS,
)


@pytest.fixture(autouse=True)
def clear_cache():
    _cache.clear()
    yield
    _cache.clear()


# ── _score_evt ──

@patch("cortex.evt.evt_var", return_value=3.0)
def test_score_evt_low_risk(mock_var):
    evt_data = {"xi": 0.1, "beta": 0.5, "threshold": 1.0, "n_total": 300, "n_exceedances": 30}
    result = _score_evt(evt_data)
    assert result["component"] == "evt"
    assert 0 <= result["score"] <= 30


@patch("cortex.evt.evt_var", return_value=14.0)
def test_score_evt_high_risk(mock_var):
    evt_data = {"xi": 0.5, "beta": 2.0, "threshold": 3.0, "n_total": 300, "n_exceedances": 30}
    result = _score_evt(evt_data)
    assert result["score"] > 70


# ── _score_svj ──

@patch("cortex.svj.decompose_risk")
def test_score_svj_low_jump(mock_decompose):
    mock_decompose.return_value = {"jump_share_pct": 10.0, "daily_jump_vol": 0.5}
    svj_data = {"returns": pd.Series([1.0] * 100), "calibration": {"lambda_": 5.0}}
    result = _score_svj(svj_data)
    assert result["component"] == "svj"
    assert result["score"] < 30


@patch("cortex.svj.decompose_risk")
def test_score_svj_high_jump(mock_decompose):
    mock_decompose.return_value = {"jump_share_pct": 80.0, "daily_jump_vol": 3.0}
    svj_data = {"returns": pd.Series([1.0] * 100), "calibration": {"lambda_": 90.0}}
    result = _score_svj(svj_data)
    assert result["score"] > 60


# ── _score_hawkes ──

@patch("cortex.hawkes.detect_flash_crash_risk")
def test_score_hawkes_low_contagion(mock_risk):
    mock_risk.return_value = {
        "contagion_risk_score": 0.1, "risk_level": "low",
        "intensity_ratio": 1.2, "recent_event_count": 2,
    }
    hawkes_data = {"event_times": np.array([1.0, 5.0]), "mu": 0.05, "alpha": 0.1, "beta": 1.0}
    result = _score_hawkes(hawkes_data)
    assert result["component"] == "hawkes"
    assert result["score"] < 25


@patch("cortex.hawkes.detect_flash_crash_risk")
def test_score_hawkes_high_contagion(mock_risk):
    mock_risk.return_value = {
        "contagion_risk_score": 0.85, "risk_level": "critical",
        "intensity_ratio": 4.0, "recent_event_count": 10,
    }
    hawkes_data = {"event_times": np.array([1.0, 2.0, 3.0]), "mu": 0.05, "alpha": 0.8, "beta": 1.0}
    result = _score_hawkes(hawkes_data)
    assert result["score"] > 75


# ── _score_regime ──

def test_score_regime_low_vol():
    probs = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
    fp = pd.DataFrame([probs])
    model_data = {"filter_probs": fp, "calibration": {"num_states": 5}}
    result = _score_regime(model_data)
    assert result["component"] == "regime"
    assert result["score"] < 20


def test_score_regime_crisis():
    probs = np.array([0.02, 0.03, 0.05, 0.1, 0.8])
    fp = pd.DataFrame([probs])
    model_data = {"filter_probs": fp, "calibration": {"num_states": 5}}
    result = _score_regime(model_data)
    assert result["score"] > 70


# ── _score_news ──

def test_score_news_no_items():
    result = _score_news({"n_items": 0}, "long")
    assert result["score"] == 50.0


def test_score_news_bullish_long():
    signal = {"sentiment_ewma": 0.8, "strength": 0.7, "confidence": 0.9,
              "entropy": 0.5, "direction": "LONG", "n_items": 5}
    result = _score_news(signal, "long")
    assert result["score"] < 40


def test_score_news_bearish_long():
    signal = {"sentiment_ewma": -0.8, "strength": 0.7, "confidence": 0.9,
              "entropy": 0.5, "direction": "SHORT", "n_items": 5}
    result = _score_news(signal, "long")
    assert result["score"] > 60


# ── _recommend_size ──

def test_recommend_size_low_risk():
    size = _recommend_size(10000, 20.0, 1, 5)
    assert size == pytest.approx(8000.0, abs=1)


def test_recommend_size_crisis_regime():
    size = _recommend_size(10000, 20.0, 5, 5)
    assert size < 5000


# ── assess_trade ──

@patch("cortex.guardian._score_evt")
@patch("cortex.guardian._score_svj")
@patch("cortex.guardian._score_hawkes")
def test_assess_trade_approved(mock_hawkes, mock_svj, mock_evt):
    mock_evt.return_value = {"component": "evt", "score": 15.0, "details": {}}
    mock_svj.return_value = {"component": "svj", "score": 10.0, "details": {"jump_share_pct": 15}}
    mock_hawkes.return_value = {"component": "hawkes", "score": 8.0,
                                "details": {"contagion_risk_score": 0.08}}
    probs = np.array([0.7, 0.15, 0.1, 0.03, 0.02])
    fp = pd.DataFrame([probs])
    model_data = {"filter_probs": fp, "calibration": {"num_states": 5}}
    result = assess_trade("SOL", 5000, "long", model_data,
                          {"xi": 0.1, "beta": 0.5, "threshold": 1.0, "n_total": 300, "n_exceedances": 30},
                          {"returns": pd.Series([1.0]*100), "calibration": {"lambda_": 5}},
                          {"event_times": np.array([1.0]), "mu": 0.05, "alpha": 0.1, "beta": 1.0})
    assert result["approved"] is True
    assert result["risk_score"] < 75
    assert result["from_cache"] is False


@patch("cortex.guardian._score_evt")
@patch("cortex.guardian._score_svj")
@patch("cortex.guardian._score_hawkes")
def test_assess_trade_vetoed(mock_hawkes, mock_svj, mock_evt):
    mock_evt.return_value = {"component": "evt", "score": 95.0, "details": {}}
    mock_svj.return_value = {"component": "svj", "score": 85.0, "details": {"jump_share_pct": 70}}
    mock_hawkes.return_value = {"component": "hawkes", "score": 92.0,
                                "details": {"contagion_risk_score": 0.92}}
    probs = np.array([0.02, 0.03, 0.05, 0.1, 0.8])
    fp = pd.DataFrame([probs])
    model_data = {"filter_probs": fp, "calibration": {"num_states": 5}}
    result = assess_trade("SOL", 5000, "long", model_data,
                          {"xi": 0.5, "beta": 2.0, "threshold": 3.0, "n_total": 300, "n_exceedances": 30},
                          {"returns": pd.Series([1.0]*100), "calibration": {"lambda_": 90}},
                          {"event_times": np.array([1.0, 2.0, 3.0]), "mu": 0.05, "alpha": 0.8, "beta": 1.0})
    assert result["approved"] is False
    assert len(result["veto_reasons"]) > 0


@patch("cortex.guardian._score_evt")
def test_circuit_breaker_trigger(mock_evt):
    mock_evt.return_value = {"component": "evt", "score": 95.0, "details": {}}
    result = assess_trade("SOL", 5000, "long", None,
                          {"xi": 0.5, "beta": 2.0, "threshold": 3.0, "n_total": 300, "n_exceedances": 30},
                          None, None)
    assert "evt_extreme_tail" in result["veto_reasons"]
    assert result["approved"] is False


def test_assess_trade_cache():
    """Second call within TTL returns cached result."""
    probs = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
    fp = pd.DataFrame([probs])
    model_data = {"filter_probs": fp, "calibration": {"num_states": 5}}
    with patch("cortex.guardian._score_regime") as mock_regime:
        mock_regime.return_value = {"component": "regime", "score": 10.0,
                                    "details": {"current_regime": 1, "num_states": 5}}
        r1 = assess_trade("CACHE_TEST", 1000, "long", model_data, None, None, None)
        r2 = assess_trade("CACHE_TEST", 1000, "long", model_data, None, None, None)
    assert r2["from_cache"] is True
    assert mock_regime.call_count == 1


def test_assess_trade_partial_models():
    """Works with only regime data (no EVT/SVJ/Hawkes)."""
    probs = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
    fp = pd.DataFrame([probs])
    model_data = {"filter_probs": fp, "calibration": {"num_states": 5}}
    result = assess_trade("PARTIAL", 5000, "short", model_data, None, None, None)
    assert "risk_score" in result
    assert result["confidence"] < 1.0
