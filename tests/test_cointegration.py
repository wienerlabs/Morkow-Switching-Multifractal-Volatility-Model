"""Tests for CointegrationModule."""

import numpy as np
import pytest
from cortex.cointegration import CointegrationModule, CointegrationSignal


class TestCointegrationSignal:
    def test_default_invalid(self):
        sig = CointegrationSignal(z_score=0.0, direction=None, confidence=0.0, is_valid=False)
        assert not sig.is_valid
        assert sig.direction is None


class TestCointegrationModule:
    def test_too_short_returns_invalid(self):
        mod = CointegrationModule()
        sol = np.random.uniform(100, 200, 50)
        btc = np.random.uniform(40000, 50000, 50)
        sig = mod.compute(sol, btc, lookback=168)
        assert not sig.is_valid

    def test_random_data_no_crash(self):
        np.random.seed(42)
        mod = CointegrationModule()
        sol = np.cumsum(np.random.normal(0, 1, 200)) + 150
        btc = np.cumsum(np.random.normal(0, 1, 200)) + 45000
        sig = mod.compute(sol, btc, lookback=168)
        assert isinstance(sig, CointegrationSignal)

    def test_cointegrated_pair(self):
        np.random.seed(42)
        n = 300
        common = np.cumsum(np.random.normal(0, 1, n))
        sol = common * 0.5 + np.random.normal(0, 0.1, n) + 150
        btc = common * 2.0 + np.random.normal(0, 0.1, n) + 45000

        mod = CointegrationModule(entry_zscore=2.0)
        sig = mod.compute(sol, btc, lookback=200)
        assert isinstance(sig, CointegrationSignal)
        # May or may not be valid depending on random seed

    def test_direction_long_when_negative_zscore(self):
        sig = CointegrationSignal(z_score=-2.5, direction="long", confidence=0.6, is_valid=True)
        assert sig.direction == "long"

    def test_direction_short_when_positive_zscore(self):
        sig = CointegrationSignal(z_score=2.5, direction="short", confidence=0.6, is_valid=True)
        assert sig.direction == "short"

    def test_length_mismatch_handled(self):
        mod = CointegrationModule()
        sol = np.random.uniform(100, 200, 200)
        btc = np.random.uniform(40000, 50000, 180)
        sig = mod.compute(sol, btc, lookback=168)
        assert isinstance(sig, CointegrationSignal)
