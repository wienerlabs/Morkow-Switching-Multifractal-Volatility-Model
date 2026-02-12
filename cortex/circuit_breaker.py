"""Circuit Breaker State Machine — per-strategy risk isolation.

States: CLOSED (normal) → OPEN (blocked) → HALF_OPEN (probing) → CLOSED/OPEN.
Each strategy (lp, arb, perp) + a global breaker are tracked independently.
"""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Any

from cortex.config import CB_CONSECUTIVE_CHECKS, CB_COOLDOWN_SECONDS, CB_STRATEGIES, CB_THRESHOLD

logger = logging.getLogger(__name__)


class CBState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, name: str, threshold: float = CB_THRESHOLD,
                 consecutive: int = CB_CONSECUTIVE_CHECKS,
                 cooldown: float = CB_COOLDOWN_SECONDS):
        self.name = name
        self.threshold = threshold
        self.consecutive = consecutive
        self.cooldown = cooldown
        self.state = CBState.CLOSED
        self.fail_count = 0
        self.opened_at: float | None = None
        self._history: list[float] = []

    def record_score(self, score: float) -> None:
        self._history.append(score)
        if len(self._history) > 100:
            self._history = self._history[-100:]

        if self.state == CBState.OPEN:
            if self.opened_at and (time.time() - self.opened_at) >= self.cooldown:
                self.state = CBState.HALF_OPEN
                self.fail_count = 0
                logger.info("circuit_breaker %s → HALF_OPEN (cooldown expired)", self.name)
            return

        if score >= self.threshold:
            self.fail_count += 1
            if self.fail_count >= self.consecutive:
                self._trip()
        else:
            if self.state == CBState.HALF_OPEN:
                self.state = CBState.CLOSED
                self.fail_count = 0
                logger.info("circuit_breaker %s → CLOSED (recovered)", self.name)
            else:
                self.fail_count = max(0, self.fail_count - 1)

    def _trip(self) -> None:
        self.state = CBState.OPEN
        self.opened_at = time.time()
        logger.warning("circuit_breaker %s → OPEN (score >= %.0f × %d consecutive)",
                       self.name, self.threshold, self.consecutive)

    def is_blocked(self) -> bool:
        if self.state == CBState.OPEN:
            if self.opened_at and (time.time() - self.opened_at) >= self.cooldown:
                self.state = CBState.HALF_OPEN
                self.fail_count = 0
                return False
            return True
        return False

    def reset(self) -> None:
        self.state = CBState.CLOSED
        self.fail_count = 0
        self.opened_at = None

    def status(self) -> dict[str, Any]:
        remaining = None
        if self.state == CBState.OPEN and self.opened_at:
            remaining = max(0.0, self.cooldown - (time.time() - self.opened_at))
        return {
            "name": self.name,
            "state": self.state.value,
            "fail_count": self.fail_count,
            "threshold": self.threshold,
            "consecutive_required": self.consecutive,
            "cooldown_seconds": self.cooldown,
            "opened_at": self.opened_at,
            "cooldown_remaining": remaining,
            "history_len": len(self._history),
        }


_breakers: dict[str, CircuitBreaker] = {}


def _ensure_breakers() -> None:
    if _breakers:
        return
    _breakers["global"] = CircuitBreaker("global")
    for s in CB_STRATEGIES:
        _breakers[s] = CircuitBreaker(s)


def record_score(score: float, strategy: str | None = None) -> dict[str, Any]:
    _ensure_breakers()
    _breakers["global"].record_score(score)
    if strategy and strategy in _breakers:
        _breakers[strategy].record_score(score)
    return {name: cb.status() for name, cb in _breakers.items()}


def is_blocked(strategy: str | None = None) -> tuple[bool, list[str]]:
    _ensure_breakers()
    blockers: list[str] = []
    if _breakers["global"].is_blocked():
        blockers.append("global")
    if strategy and strategy in _breakers and _breakers[strategy].is_blocked():
        blockers.append(strategy)
    return (len(blockers) > 0, blockers)


def get_all_states() -> list[dict[str, Any]]:
    _ensure_breakers()
    return [cb.status() for cb in _breakers.values()]


def reset_breaker(name: str) -> bool:
    _ensure_breakers()
    if name not in _breakers:
        return False
    _breakers[name].reset()
    return True


def reset_all() -> None:
    _ensure_breakers()
    for cb in _breakers.values():
        cb.reset()

