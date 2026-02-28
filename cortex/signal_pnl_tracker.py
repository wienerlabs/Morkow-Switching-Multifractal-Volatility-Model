"""Direction-correctness Sharpe tracker for agent weight optimization."""

from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict

import numpy as np


def compute_reward(direction: str | None, was_correct: bool) -> float:
    if direction is None:
        return 0.0
    return 1.0 if was_correct else -1.0


@dataclass
class SignalOutcome:
    agent_name: str
    direction: str | None
    confidence: float
    was_correct: bool
    reward: float


class SignalPnLTracker:
    """Track per-agent direction-correctness rewards, compute Sharpe-based MVO weights."""

    def __init__(
        self,
        min_samples: int = 5,
        w_min: float = 0.05,
        w_max: float = 0.40,
    ) -> None:
        self._min_samples = min_samples
        self._w_min = w_min
        self._w_max = w_max
        self._history: dict[str, list[float]] = defaultdict(list)

    def record(self, outcomes: list[SignalOutcome]) -> None:
        for o in outcomes:
            self._history[o.agent_name].append(o.reward)

    def compute_sharpe_weights(self) -> dict[str, float] | None:
        agents = sorted(self._history.keys())
        if not agents:
            return None

        # Need min_samples for ALL agents
        for a in agents:
            if len(self._history[a]) < self._min_samples:
                return None

        n = len(agents)
        sharpes = {}
        for a in agents:
            rewards = np.array(self._history[a], dtype=float)
            mu = rewards.mean()
            std = rewards.std(ddof=1)
            sharpes[a] = mu / std if std > 1e-9 else 0.0

        # Simple MVO: weight proportional to positive Sharpe
        raw = {}
        for a in agents:
            raw[a] = max(sharpes[a], 0.01)  # floor at 0.01 to avoid zero weights

        total = sum(raw.values())
        weights = {a: raw[a] / total for a in agents}

        # Clip to [w_min, w_max]
        for a in agents:
            weights[a] = max(self._w_min, min(self._w_max, weights[a]))

        # Re-normalize
        total = sum(weights.values())
        weights = {a: w / total for a, w in weights.items()}

        return weights

    @property
    def sample_counts(self) -> dict[str, int]:
        return {a: len(v) for a, v in self._history.items()}
