"""Tests for SignalPnLTracker — direction-correctness Sharpe weight optimizer."""

import pytest
from cortex.signal_pnl_tracker import SignalPnLTracker, SignalOutcome, compute_reward


class TestComputeReward:
    def test_correct_long(self):
        assert compute_reward("long", True) == 1.0

    def test_wrong_long(self):
        assert compute_reward("long", False) == -1.0

    def test_correct_short(self):
        assert compute_reward("short", True) == 1.0

    def test_no_direction(self):
        assert compute_reward(None, True) == 0.0

    def test_no_direction_wrong(self):
        assert compute_reward(None, False) == 0.0


class TestSignalPnLTracker:
    def _make_outcome(self, agent: str, correct: bool, direction: str = "long") -> SignalOutcome:
        reward = compute_reward(direction, correct)
        return SignalOutcome(agent_name=agent, direction=direction, confidence=0.7, was_correct=correct, reward=reward)

    def test_empty_returns_none(self):
        tracker = SignalPnLTracker()
        assert tracker.compute_sharpe_weights() is None

    def test_below_min_samples_returns_none(self):
        tracker = SignalPnLTracker(min_samples=5)
        for i in range(4):
            tracker.record([self._make_outcome("ta", True)])
        assert tracker.compute_sharpe_weights() is None

    def test_at_min_samples_returns_weights(self):
        tracker = SignalPnLTracker(min_samples=5)
        for i in range(5):
            tracker.record([
                self._make_outcome("ta", i % 2 == 0),
                self._make_outcome("macro", i % 3 == 0),
                self._make_outcome("risk", True),
            ])
        weights = tracker.compute_sharpe_weights()
        assert weights is not None
        assert set(weights.keys()) == {"ta", "macro", "risk"}
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_weights_bounded(self):
        tracker = SignalPnLTracker(min_samples=5, w_min=0.05, w_max=0.40)
        for i in range(10):
            tracker.record([
                self._make_outcome("ta", True),
                self._make_outcome("macro", False),
                self._make_outcome("risk", i % 2 == 0),
            ])
        weights = tracker.compute_sharpe_weights()
        assert weights is not None
        for w in weights.values():
            assert 0.05 <= w <= 0.40 + 1e-6

    def test_better_agent_gets_higher_weight(self):
        tracker = SignalPnLTracker(min_samples=5, w_min=0.01, w_max=0.99)
        for i in range(20):
            # good_agent: mostly correct (18/20), bad_agent: mostly wrong (2/20)
            # Mixing in some variance so std > 0 for meaningful Sharpe
            tracker.record([
                self._make_outcome("good_agent", i % 10 != 0),
                self._make_outcome("bad_agent", i % 10 == 0),
            ])
        weights = tracker.compute_sharpe_weights()
        assert weights is not None
        assert weights["good_agent"] > weights["bad_agent"]

    def test_sample_counts(self):
        tracker = SignalPnLTracker()
        tracker.record([self._make_outcome("ta", True)])
        tracker.record([self._make_outcome("ta", False), self._make_outcome("macro", True)])
        assert tracker.sample_counts == {"ta": 2, "macro": 1}

    def test_default_min_samples_is_5(self):
        tracker = SignalPnLTracker()
        assert tracker._min_samples == 5
