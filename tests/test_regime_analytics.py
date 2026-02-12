"""Tests for regime_analytics.py."""

import numpy as np
import pandas as pd
import pytest

from cortex import regime as ra


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
    def test_returns_dataframe(self, calibrated_model):
        m = calibrated_model
        df = ra.extract_regime_history(
            m["filter_probs"], m["returns"], m["sigma_states"]
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "regime" in df.columns
        assert "start" in df.columns
        assert "end" in df.columns
        assert "duration" in df.columns

    def test_regimes_cover_full_series(self, calibrated_model):
        m = calibrated_model
        df = ra.extract_regime_history(
            m["filter_probs"], m["returns"], m["sigma_states"]
        )
        total_days = int(df["duration"].sum())
        assert total_days == len(m["returns"])


class TestDetectRegimeTransition:
    def test_returns_dict(self, calibrated_model):
        m = calibrated_model
        result = ra.detect_regime_transition(
            m["filter_probs"], threshold=0.3
        )
        assert "current_regime" in result
        assert "alert" in result
        assert isinstance(result["current_regime"], int)
        assert isinstance(result["alert"], bool)


class TestComputeRegimeStatistics:
    def test_returns_dataframe(self, calibrated_model):
        m = calibrated_model
        df = ra.compute_regime_statistics(
            m["returns"], m["filter_probs"], m["sigma_states"]
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(m["sigma_states"])
        assert "regime" in df.columns
        assert "mean_return" in df.columns
        assert "volatility" in df.columns
        assert "frequency" in df.columns

