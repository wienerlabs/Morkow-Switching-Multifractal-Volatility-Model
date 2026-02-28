"""Tests for HMMRegimeDetector."""

import numpy as np
import pytest
from cortex.hmm_regime import HMMRegimeDetector


def _hmmlearn_available():
    try:
        import hmmlearn
        return True
    except ImportError:
        return False


class TestHMMRegimeDetector:
    def _make_data(self, n=200):
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, n)
        volumes = np.random.uniform(1000, 5000, n)
        return returns, volumes

    def test_init_defaults(self):
        det = HMMRegimeDetector()
        assert det._n_states == 3
        assert det._min_bars == 100
        assert not det._fitted

    def test_build_features_too_short(self):
        det = HMMRegimeDetector(min_bars=100)
        result = det._build_features(np.zeros(50), np.zeros(50))
        assert result is None

    def test_build_features_length_mismatch(self):
        det = HMMRegimeDetector(min_bars=10)
        r = np.zeros(100)
        v = np.zeros(90)
        result = det._build_features(r, v)
        assert result is not None
        assert result.shape[0] == 90  # uses min_len

    def test_fit_without_hmmlearn(self):
        det = HMMRegimeDetector(min_bars=10)
        r, v = self._make_data(200)
        # This may pass or fail depending on whether hmmlearn is installed
        # Either way it shouldn't raise
        det.fit(r, v)

    def test_predict_before_fit_returns_none(self):
        det = HMMRegimeDetector()
        r, v = self._make_data()
        assert det.predict(r, v) is None

    def test_get_regime_multipliers_before_fit(self):
        det = HMMRegimeDetector()
        assert det.get_regime_multipliers() is None

    def test_needs_retrain(self):
        det = HMMRegimeDetector(retrain_interval=5)
        assert not det.needs_retrain()
        det._samples_since_fit = 5
        assert det.needs_retrain()

    @pytest.mark.skipif(
        not _hmmlearn_available(),
        reason="hmmlearn not installed"
    )
    def test_full_pipeline(self):
        det = HMMRegimeDetector(min_bars=50, n_states=3)
        r, v = self._make_data(300)
        assert det.fit(r, v)
        assert det._fitted

        state = det.predict(r[-100:], v[-100:])
        assert state is not None
        assert 0 <= state < 3

        mults = det.get_regime_multipliers()
        assert mults is not None
        assert "regime_multiplier" in mults
        assert 0.5 <= mults["regime_multiplier"] <= 1.5
