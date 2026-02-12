"""Tests for cortex/circuit_breaker.py — CircuitBreaker state machine."""

import time
from unittest.mock import patch

import pytest

from cortex.circuit_breaker import (
    CBState,
    CircuitBreaker,
    _breakers,
    get_all_states,
    is_blocked,
    record_score,
    reset_all,
    reset_breaker,
)


@pytest.fixture(autouse=True)
def clean_breakers():
    """Reset module-level breakers before each test."""
    _breakers.clear()
    yield
    _breakers.clear()


class TestCircuitBreakerUnit:
    def test_initial_state_closed(self):
        cb = CircuitBreaker("test", threshold=90, consecutive=3, cooldown=300)
        assert cb.state == CBState.CLOSED
        assert cb.fail_count == 0

    def test_low_scores_stay_closed(self):
        cb = CircuitBreaker("test", threshold=90, consecutive=3, cooldown=300)
        for _ in range(10):
            cb.record_score(50.0)
        assert cb.state == CBState.CLOSED

    def test_consecutive_high_scores_trip(self):
        cb = CircuitBreaker("test", threshold=90, consecutive=3, cooldown=300)
        cb.record_score(95.0)
        cb.record_score(95.0)
        assert cb.state == CBState.CLOSED
        cb.record_score(95.0)
        assert cb.state == CBState.OPEN

    def test_open_blocks(self):
        cb = CircuitBreaker("test", threshold=90, consecutive=2, cooldown=300)
        cb.record_score(95.0)
        cb.record_score(95.0)
        assert cb.is_blocked() is True

    def test_cooldown_transitions_to_half_open(self):
        cb = CircuitBreaker("test", threshold=90, consecutive=2, cooldown=1)
        cb.record_score(95.0)
        cb.record_score(95.0)
        assert cb.state == CBState.OPEN
        time.sleep(1.1)
        assert cb.is_blocked() is False
        assert cb.state == CBState.HALF_OPEN

    def test_half_open_recovers_on_low_score(self):
        cb = CircuitBreaker("test", threshold=90, consecutive=2, cooldown=0.1)
        cb.record_score(95.0)
        cb.record_score(95.0)
        time.sleep(0.15)
        # First call while OPEN: cooldown expired → transitions to HALF_OPEN and returns
        cb.record_score(30.0)
        assert cb.state == CBState.HALF_OPEN
        # Second call while HALF_OPEN with low score → CLOSED
        cb.record_score(30.0)
        assert cb.state == CBState.CLOSED

    def test_half_open_trips_again_on_high_score(self):
        cb = CircuitBreaker("test", threshold=90, consecutive=1, cooldown=0.1)
        cb.record_score(95.0)
        assert cb.state == CBState.OPEN
        time.sleep(0.15)
        # First call: cooldown expired → HALF_OPEN and returns
        cb.record_score(95.0)
        assert cb.state == CBState.HALF_OPEN
        # Second call while HALF_OPEN with high score → trips to OPEN
        cb.record_score(95.0)
        assert cb.state == CBState.OPEN

    def test_reset(self):
        cb = CircuitBreaker("test", threshold=90, consecutive=2, cooldown=300)
        cb.record_score(95.0)
        cb.record_score(95.0)
        assert cb.state == CBState.OPEN
        cb.reset()
        assert cb.state == CBState.CLOSED
        assert cb.fail_count == 0

    def test_status_dict(self):
        cb = CircuitBreaker("test", threshold=90, consecutive=3, cooldown=300)
        s = cb.status()
        assert s["name"] == "test"
        assert s["state"] == "closed"
        assert s["threshold"] == 90
        assert s["cooldown_remaining"] is None

    def test_history_capped_at_100(self):
        cb = CircuitBreaker("test", threshold=90, consecutive=999, cooldown=300)
        for i in range(150):
            cb.record_score(float(i))
        assert len(cb._history) == 100

    def test_fail_count_decrements_on_low_score(self):
        cb = CircuitBreaker("test", threshold=90, consecutive=5, cooldown=300)
        cb.record_score(95.0)
        cb.record_score(95.0)
        assert cb.fail_count == 2
        cb.record_score(10.0)
        assert cb.fail_count == 1


class TestModuleLevelFunctions:
    def test_record_score_initializes_breakers(self):
        result = record_score(50.0)
        assert "global" in result
        assert "lp" in result

    def test_is_blocked_default_false(self):
        blocked, blockers = is_blocked()
        assert blocked is False
        assert blockers == []

    def test_strategy_specific_blocking(self):
        for _ in range(3):
            record_score(95.0, strategy="lp")
        blocked, blockers = is_blocked(strategy="lp")
        assert blocked is True
        assert "lp" in blockers

    def test_get_all_states_returns_list(self):
        record_score(50.0)
        states = get_all_states()
        assert isinstance(states, list)
        assert len(states) >= 4  # global + lp + arb + perp

    def test_reset_breaker(self):
        for _ in range(3):
            record_score(95.0)
        assert reset_breaker("global") is True
        blocked, _ = is_blocked()
        assert blocked is False

    def test_reset_unknown_breaker(self):
        record_score(50.0)
        assert reset_breaker("nonexistent") is False

    def test_reset_all(self):
        for _ in range(3):
            record_score(95.0, strategy="arb")
        reset_all()
        blocked, _ = is_blocked(strategy="arb")
        assert blocked is False

