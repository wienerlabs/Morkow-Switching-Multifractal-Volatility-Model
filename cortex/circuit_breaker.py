"""Circuit Breaker State Machine — per-strategy risk isolation with outcome tracking.

States: CLOSED (normal) → OPEN (blocked) → HALF_OPEN (probing) → CLOSED/OPEN.
Each strategy (lp, arb, perp) + a global breaker are tracked independently.

Strategy-specific failure rules:
  - LP:   3 consecutive impermanent losses → pause LP strategy
  - Arb:  5 consecutive failed executions → pause arb strategy
  - Perp: 2 consecutive stop-losses → pause perps strategy
  - Spot: 4 consecutive market losses → pause spot strategy
  - Lending: 3 consecutive rate losses → pause lending strategy

Outcome tracking records trade results per strategy and triggers circuit
breakers based on consecutive failures (separate from risk-score-based trips).
"""

__all__ = [
    "CBState",
    "CircuitBreaker",
    "OutcomeCircuitBreaker",
    "record_score",
    "record_trade_outcome",
    "is_blocked",
    "get_all_states",
    "get_outcome_states",
    "reset_breaker",
    "reset_all",
    "persist_cb_state",
    "restore_cb_state",
]
from __future__ import annotations

import logging
import time
from collections import deque
from enum import Enum
from typing import Any

from cortex.config import CB_CONSECUTIVE_CHECKS, CB_COOLDOWN_SECONDS, CB_STRATEGIES, CB_THRESHOLD

logger = logging.getLogger(__name__)

# Strategy-specific failure rules
STRATEGY_FAILURE_RULES: dict[str, dict[str, Any]] = {
    "lp": {
        "consecutive_loss_limit": 3,
        "loss_type": "impermanent_loss",
        "cooldown_seconds": 600,        # 10 minutes
        "description": "3 consecutive IL events → pause LP strategy",
    },
    "arb": {
        "consecutive_loss_limit": 5,
        "loss_type": "failed_execution",
        "cooldown_seconds": 300,        # 5 minutes
        "description": "5 consecutive failed arb executions → pause arb",
    },
    "perp": {
        "consecutive_loss_limit": 2,
        "loss_type": "stop_loss",
        "cooldown_seconds": 900,        # 15 minutes (most aggressive cooldown)
        "description": "2 consecutive stop-losses → pause perps",
    },
    "spot": {
        "consecutive_loss_limit": 4,
        "loss_type": "market_loss",
        "cooldown_seconds": 300,
        "description": "4 consecutive market losses → pause spot",
    },
    "lending": {
        "consecutive_loss_limit": 3,
        "loss_type": "rate_loss",
        "cooldown_seconds": 600,
        "description": "3 consecutive rate losses → pause lending",
    },
}


class CBState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Risk-score-based circuit breaker with configurable threshold."""

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


class OutcomeCircuitBreaker:
    """Trade-outcome-based circuit breaker with strategy-specific failure rules.

    Tracks consecutive losses per strategy and trips when the strategy's
    consecutive loss limit is reached.  Separate from the risk-score-based
    CircuitBreaker.
    """

    def __init__(self, strategy: str):
        rules = STRATEGY_FAILURE_RULES.get(strategy, {})
        self.strategy = strategy
        self.loss_limit = rules.get("consecutive_loss_limit", 3)
        self.loss_type = rules.get("loss_type", "unknown")
        self.cooldown = rules.get("cooldown_seconds", 300.0)
        self.description = rules.get("description", f"{strategy} outcome breaker")

        self.state = CBState.CLOSED
        self.consecutive_losses = 0
        self.opened_at: float | None = None
        self.total_trades = 0
        self.total_losses = 0
        self.total_wins = 0
        self._outcomes: deque[dict[str, Any]] = deque(maxlen=100)

    def record_outcome(self, success: bool, pnl: float = 0.0,
                       loss_type: str = "", details: str = "") -> None:
        """Record a trade outcome and check if the breaker should trip.

        Args:
            success: Whether the trade was successful
            pnl: Profit/loss amount
            loss_type: Type of loss (e.g., "impermanent_loss", "stop_loss")
            details: Optional description
        """
        self.total_trades += 1
        self._outcomes.append({
            "success": success,
            "pnl": pnl,
            "loss_type": loss_type,
            "details": details,
            "ts": time.time(),
        })

        if self.state == CBState.OPEN:
            if self.opened_at and (time.time() - self.opened_at) >= self.cooldown:
                self.state = CBState.HALF_OPEN
                self.consecutive_losses = 0
                logger.info("outcome_breaker %s → HALF_OPEN (cooldown expired)", self.strategy)
            return

        if success:
            self.total_wins += 1
            if self.state == CBState.HALF_OPEN:
                # Successful probe → recover
                self.state = CBState.CLOSED
                self.consecutive_losses = 0
                logger.info("outcome_breaker %s → CLOSED (probe succeeded)", self.strategy)
            else:
                self.consecutive_losses = 0
        else:
            self.total_losses += 1
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.loss_limit:
                self._trip()
            elif self.state == CBState.HALF_OPEN:
                # Failed probe → reopen
                self._trip()

    def _trip(self) -> None:
        self.state = CBState.OPEN
        self.opened_at = time.time()
        logger.warning(
            "outcome_breaker %s → OPEN (%d consecutive %s losses, limit %d)",
            self.strategy, self.consecutive_losses, self.loss_type, self.loss_limit,
        )

    def is_blocked(self) -> bool:
        if self.state == CBState.OPEN:
            if self.opened_at and (time.time() - self.opened_at) >= self.cooldown:
                self.state = CBState.HALF_OPEN
                self.consecutive_losses = 0
                return False
            return True
        return False

    def reset(self) -> None:
        self.state = CBState.CLOSED
        self.consecutive_losses = 0
        self.opened_at = None

    def status(self) -> dict[str, Any]:
        remaining = None
        if self.state == CBState.OPEN and self.opened_at:
            remaining = max(0.0, self.cooldown - (time.time() - self.opened_at))

        win_rate = self.total_wins / self.total_trades if self.total_trades > 0 else 0.0
        return {
            "name": f"{self.strategy}_outcome",
            "strategy": self.strategy,
            "state": self.state.value,
            "consecutive_losses": self.consecutive_losses,
            "loss_limit": self.loss_limit,
            "loss_type": self.loss_type,
            "cooldown_seconds": self.cooldown,
            "opened_at": self.opened_at,
            "cooldown_remaining": remaining,
            "total_trades": self.total_trades,
            "total_wins": self.total_wins,
            "total_losses": self.total_losses,
            "win_rate": round(win_rate, 4),
            "description": self.description,
        }


# ── Module-level singleton state ──────────────────────────────────────

_breakers: dict[str, CircuitBreaker] = {}
_outcome_breakers: dict[str, OutcomeCircuitBreaker] = {}


def _ensure_breakers() -> None:
    if not _breakers:
        _breakers["global"] = CircuitBreaker("global")
        for s in CB_STRATEGIES:
            _breakers[s] = CircuitBreaker(s)

    if not _outcome_breakers:
        for s in STRATEGY_FAILURE_RULES:
            _outcome_breakers[s] = OutcomeCircuitBreaker(s)


def record_score(score: float, strategy: str | None = None) -> dict[str, Any]:
    """Record a risk score for the global and strategy-specific breakers."""
    _ensure_breakers()
    _breakers["global"].record_score(score)
    if strategy and strategy in _breakers:
        _breakers[strategy].record_score(score)
    return {name: cb.status() for name, cb in _breakers.items()}


def record_trade_outcome(
    strategy: str,
    success: bool,
    pnl: float = 0.0,
    loss_type: str = "",
    details: str = "",
) -> dict[str, Any]:
    """Record a trade outcome for strategy-specific outcome breakers.

    Args:
        strategy: Strategy name (lp, arb, perp, spot, lending)
        success: Whether the trade was successful
        pnl: Profit/loss amount
        loss_type: Type of loss (e.g., "impermanent_loss", "stop_loss", "failed_execution")
        details: Optional description

    Returns:
        Updated status of the outcome breaker for this strategy.
    """
    _ensure_breakers()
    if strategy not in _outcome_breakers:
        # Unknown strategy, create a default breaker
        _outcome_breakers[strategy] = OutcomeCircuitBreaker(strategy)

    _outcome_breakers[strategy].record_outcome(
        success=success, pnl=pnl, loss_type=loss_type, details=details,
    )
    return _outcome_breakers[strategy].status()


def is_blocked(strategy: str | None = None) -> tuple[bool, list[str]]:
    """Check if a strategy (or global) is blocked by any breaker.

    Checks both risk-score-based and outcome-based breakers.
    """
    _ensure_breakers()
    blockers: list[str] = []

    # Risk-score breakers
    if _breakers["global"].is_blocked():
        blockers.append("global")
    if strategy and strategy in _breakers and _breakers[strategy].is_blocked():
        blockers.append(strategy)

    # Outcome breakers
    if strategy and strategy in _outcome_breakers and _outcome_breakers[strategy].is_blocked():
        blockers.append(f"{strategy}_outcome")

    return (len(blockers) > 0, blockers)


def get_all_states() -> list[dict[str, Any]]:
    """Get status of all circuit breakers (risk-score + outcome)."""
    _ensure_breakers()
    states: list[dict[str, Any]] = [cb.status() for cb in _breakers.values()]
    states.extend(ob.status() for ob in _outcome_breakers.values())
    return states


def get_outcome_states() -> list[dict[str, Any]]:
    """Get status of outcome-based circuit breakers only."""
    _ensure_breakers()
    return [ob.status() for ob in _outcome_breakers.values()]


def reset_breaker(name: str) -> bool:
    """Reset a specific breaker by name."""
    _ensure_breakers()
    if name in _breakers:
        _breakers[name].reset()
        return True
    # Check outcome breakers (by strategy name)
    for strategy, ob in _outcome_breakers.items():
        if name in (strategy, f"{strategy}_outcome"):
            ob.reset()
            return True
    return False


def reset_all() -> None:
    """Reset all circuit breakers (risk-score + outcome)."""
    _ensure_breakers()
    for cb in _breakers.values():
        cb.reset()
    for ob in _outcome_breakers.values():
        ob.reset()


# ── Persistence ──────────────────────────────────────────────────────

_cb_store: Any = None


def _get_cb_store():
    """Lazy-init PersistentStore for circuit breaker snapshots."""
    global _cb_store
    if _cb_store is None:
        from cortex.persistence import PersistentStore
        _cb_store = PersistentStore("circuit_breaker")
    return _cb_store


def persist_cb_state() -> None:
    """Snapshot all breaker states to Redis via PersistentStore."""
    _ensure_breakers()
    store = _get_cb_store()

    score_snapshot: dict[str, dict] = {}
    for name, cb in _breakers.items():
        score_snapshot[name] = {
            "state": cb.state.value,
            "fail_count": cb.fail_count,
            "opened_at": cb.opened_at,
            "_history": cb._history[-100:],
        }
    store["score_breakers"] = score_snapshot

    outcome_snapshot: dict[str, dict] = {}
    for strategy, ob in _outcome_breakers.items():
        outcome_snapshot[strategy] = {
            "state": ob.state.value,
            "consecutive_losses": ob.consecutive_losses,
            "opened_at": ob.opened_at,
            "total_trades": ob.total_trades,
            "total_losses": ob.total_losses,
            "total_wins": ob.total_wins,
            "_outcomes": list(ob._outcomes),
        }
    store["outcome_breakers"] = outcome_snapshot
    logger.info("Circuit breaker states persisted to Redis")


def restore_cb_state() -> None:
    """Reload breaker states from Redis snapshot into module-level singletons."""
    _ensure_breakers()
    store = _get_cb_store()

    score_data = store.get("score_breakers")
    if score_data:
        for name, snap in score_data.items():
            if name in _breakers:
                cb = _breakers[name]
                cb.state = CBState(snap["state"])
                cb.fail_count = snap["fail_count"]
                cb.opened_at = snap.get("opened_at")
                cb._history = snap.get("_history", [])
        logger.info("Restored %d score-based circuit breaker(s) from Redis", len(score_data))

    outcome_data = store.get("outcome_breakers")
    if outcome_data:
        for strategy, snap in outcome_data.items():
            if strategy in _outcome_breakers:
                ob = _outcome_breakers[strategy]
                ob.state = CBState(snap["state"])
                ob.consecutive_losses = snap["consecutive_losses"]
                ob.opened_at = snap.get("opened_at")
                ob.total_trades = snap["total_trades"]
                ob.total_losses = snap["total_losses"]
                ob.total_wins = snap["total_wins"]
                ob._outcomes = deque(snap.get("_outcomes", []), maxlen=100)
        logger.info("Restored %d outcome-based circuit breaker(s) from Redis", len(outcome_data))
