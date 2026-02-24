"""Base classes for the multi-agent framework.

Every specialist agent inherits from BaseAgent and returns an AgentSignal.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class AgentSignal:
    """Standardized output from any specialist agent."""

    agent_name: str
    score: float           # 0-100, lower = safer / more bullish
    confidence: float = 0.5  # 0.0-1.0
    direction: str | None = None  # "long", "short", or None
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(100.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.direction is not None and self.direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long', 'short', or None — got {self.direction!r}")


class BaseAgent(ABC):
    """Abstract base for all specialist agents."""

    name: str = "base"
    weight: float = 1.0

    @abstractmethod
    def analyze(self, token: str, data: pd.DataFrame, context: dict[str, Any]) -> AgentSignal:
        """Run analysis and return a signal."""

    def analyze_backtest(
        self, token: str, data: pd.DataFrame, bar_idx: int, context: dict[str, Any]
    ) -> AgentSignal:
        """Backtest mode: only uses data up to bar_idx (inclusive)."""
        return self.analyze(token, data.iloc[: bar_idx + 1], context)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, weight={self.weight})"


class AgentRegistry:
    """Registry of active specialist agents."""

    _agents: dict[str, BaseAgent] = {}

    @classmethod
    def register(cls, agent: BaseAgent) -> None:
        cls._agents[agent.name] = agent

    @classmethod
    def unregister(cls, name: str) -> None:
        cls._agents.pop(name, None)

    @classmethod
    def get(cls, name: str) -> BaseAgent | None:
        return cls._agents.get(name)

    @classmethod
    def get_all(cls) -> list[BaseAgent]:
        return list(cls._agents.values())

    @classmethod
    def clear(cls) -> None:
        cls._agents.clear()
