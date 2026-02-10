"""Tests for regime_analytics.py."""

import numpy as np
import pandas as pd
import pytest

import regime_analytics as ra


class TestComputeExpectedDurations:
    def test_scalar_p_stay(self):
        d = ra.compute_expected_durations(0.97, 5)
        assert len(d) == 5
        assert all(v == d[1] for v in d.values())
        assert abs(d[1] - 33.33) < 0.01

    def test_array_p_stay(self):
        d = ra.compute_expected_durations([0.90, 0.95, 0.99], 3)
        assert len(d) == 3
        assert abs(d[1] - 10.0) < 0.01
        assert abs(d[2] - 20.0) < 0.01
        assert abs(d[3] - 100.0) < 0.01

    def test_invalid_p_stay_raises(self):
        with pytest.raises(ValueError):
            ra.compute_expected_durations(1.0, 5)
        with pytest.raises(ValueError):
            ra.compute_expected_durations(0.0, 5)

    def test_invalid_num_states_raises(self):
        with pytest.raises(ValueError):
            ra.compute_expected_durations(0.95, 1)


class TestExtractRegimeHistory:
    def test_returns_list_of_dicts(self, calibrated_model):
        m = calibrated_model
        history = ra.extract_regime_history(
            m["filter_probs"], m["sigma_states"], m["returns"]
        )
        assert isinstance(history, list)
        assert len(history) > 0
        first = history[0]
        assert "regime" in first
        assert "start" in first
        assert "end" in first
        assert "duration_days" in first

    def test_regimes_cover_full_series(self, calibrated_model):
        m = calibrated_model
        history = ra.extract_regime_history(
            m["filter_probs"], m["sigma_states"], m["returns"]
        )
        total_days = sum(h["duration_days"] for h in history)
        assert total_days == len(m["returns"])


class TestDetectRegimeTransition:
    def test_returns_dict(self, calibrated_model):
        m = calibrated_model
        result = ra.detect_regime_transition(
            m["filter_probs"], m["sigma_states"], lookback=5
        )
        assert "current_regime" in result
        assert "transition_detected" in result
        assert isinstance(result["current_regime"], int)
        assert isinstance(result["transition_detected"], bool)


class TestComputeRegimeStatistics:
    def test_returns_list(self, calibrated_model):
        m = calibrated_model
        stats = ra.compute_regime_statistics(
            m["filter_probs"], m["sigma_states"], m["returns"]
        )
        assert isinstance(stats, list)
        assert len(stats) == len(m["sigma_states"])
        first = stats[0]
        assert "regime" in first
        assert "mean_return" in first
        assert "volatility" in first
        assert "time_fraction" in first

