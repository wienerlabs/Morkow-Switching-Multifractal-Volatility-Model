"""DX-Research Task 3: Layered Agent Memory.

DX01 finding: 3-layer memory (short-term / long-term / contextual) kept a single
agent coherent for 5 continuous days. Without it, agents degrade after ~6 hours.

Three layers:
  - Short-term: Last N trade decisions + outcomes (circular buffer, fast access)
  - Long-term:  Compressed regime/performance summary (exponential decay)
  - Contextual: Latest signals from other analysts (shared awareness)

Memory is per-agent (keyed by agent_id) and persisted to Redis via PersistentStore.
"""
from __future__ import annotations

__all__ = [
    "AgentMemory",
    "get_memory",
    "record_decision",
    "get_context_snapshot",
]

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MEMORY_SHORT_TERM_SIZE = 20
MEMORY_LONG_TERM_DECAY = 0.95  # exponential decay per update


@dataclass
class ShortTermEntry:
    """A single trade decision record."""
    token: str
    direction: str
    score: float
    approved: bool
    pnl: float | None = None
    ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "token": self.token,
            "direction": self.direction,
            "score": round(self.score, 4),
            "approved": self.approved,
            "pnl": round(self.pnl, 4) if self.pnl is not None else None,
            "ts": self.ts,
        }


@dataclass
class LongTermSummary:
    """Compressed performance summary with exponential decay."""
    total_decisions: int = 0
    total_wins: int = 0
    total_losses: int = 0
    avg_risk_score: float = 50.0
    dominant_regime: int = 0
    regime_distribution: dict[int, float] = field(default_factory=dict)
    cumulative_pnl: float = 0.0
    last_updated: float = 0.0

    @property
    def win_rate(self) -> float:
        total = self.total_wins + self.total_losses
        return self.total_wins / total if total > 0 else 0.5

    def update(self, score: float, won: bool | None, regime: int = 0) -> None:
        self.total_decisions += 1
        if won is True:
            self.total_wins += 1
        elif won is False:
            self.total_losses += 1

        # Exponential moving average of risk score
        self.avg_risk_score = (
            MEMORY_LONG_TERM_DECAY * self.avg_risk_score
            + (1 - MEMORY_LONG_TERM_DECAY) * score
        )

        # Regime distribution tracking
        self.regime_distribution[regime] = (
            self.regime_distribution.get(regime, 0.0) * MEMORY_LONG_TERM_DECAY
            + (1 - MEMORY_LONG_TERM_DECAY)
        )

        # Normalize regime distribution
        total = sum(self.regime_distribution.values())
        if total > 0:
            self.regime_distribution = {
                k: v / total for k, v in self.regime_distribution.items()
            }
            self.dominant_regime = max(
                self.regime_distribution, key=self.regime_distribution.get  # type: ignore[arg-type]
            )

        self.last_updated = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_decisions": self.total_decisions,
            "win_rate": round(self.win_rate, 4),
            "avg_risk_score": round(self.avg_risk_score, 2),
            "dominant_regime": self.dominant_regime,
            "regime_distribution": {
                str(k): round(v, 4) for k, v in self.regime_distribution.items()
            },
            "cumulative_pnl": round(self.cumulative_pnl, 4),
        }


class AgentMemory:
    """Three-layer memory system for a single agent."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.short_term: deque[ShortTermEntry] = deque(maxlen=MEMORY_SHORT_TERM_SIZE)
        self.long_term = LongTermSummary()
        self.contextual: dict[str, dict[str, Any]] = {}

    def record_decision(
        self,
        token: str,
        direction: str,
        score: float,
        approved: bool,
        regime: int = 0,
    ) -> None:
        """Record a new trade decision into short-term memory."""
        entry = ShortTermEntry(
            token=token, direction=direction, score=score,
            approved=approved, ts=time.time(),
        )
        self.short_term.append(entry)
        self.long_term.update(score, won=None, regime=regime)

    def record_outcome(self, token: str, pnl: float) -> None:
        """Attach PnL outcome to the most recent matching decision."""
        for entry in reversed(self.short_term):
            if entry.token == token and entry.pnl is None:
                entry.pnl = pnl
                self.long_term.update(
                    entry.score, won=(pnl > 0),
                )
                self.long_term.cumulative_pnl += pnl
                break

    def update_context(self, source_agent: str, signal: dict[str, Any]) -> None:
        """Update contextual memory with another agent's latest signal."""
        self.contextual[source_agent] = {
            "signal": signal,
            "ts": time.time(),
        }

    def get_recent_decisions(self, n: int = 5) -> list[dict[str, Any]]:
        """Get last N decisions from short-term memory."""
        entries = list(self.short_term)[-n:]
        return [e.to_dict() for e in entries]

    def get_summary(self) -> dict[str, Any]:
        return self.long_term.to_dict()

    def get_context(self, max_age_seconds: float = 300.0) -> dict[str, dict[str, Any]]:
        """Get contextual signals from other agents, filtered by freshness."""
        now = time.time()
        return {
            agent: data
            for agent, data in self.contextual.items()
            if (now - data["ts"]) < max_age_seconds
        }

    def snapshot(self) -> dict[str, Any]:
        """Full memory state for persistence / debugging."""
        return {
            "agent_id": self.agent_id,
            "short_term": [e.to_dict() for e in self.short_term],
            "long_term": self.long_term.to_dict(),
            "contextual": self.get_context(),
        }


# ── Module-level singleton registry ──

_memories: dict[str, AgentMemory] = {}


def get_memory(agent_id: str) -> AgentMemory:
    """Get or create memory instance for an agent."""
    if agent_id not in _memories:
        _memories[agent_id] = AgentMemory(agent_id)
    return _memories[agent_id]


def record_decision(
    agent_id: str,
    token: str,
    direction: str,
    score: float,
    approved: bool,
    regime: int = 0,
) -> None:
    """Convenience: record a decision for an agent."""
    get_memory(agent_id).record_decision(token, direction, score, approved, regime)


def get_context_snapshot(agent_id: str) -> dict[str, Any]:
    """Get full memory snapshot for an agent."""
    return get_memory(agent_id).snapshot()
