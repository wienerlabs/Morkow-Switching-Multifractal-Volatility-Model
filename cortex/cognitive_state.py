"""Dynamic Cognitive State Machine — agent emotion derived from market signals.

Combines Fear/Greed index + regime state + Hawkes contagion + news sentiment
into a 5-level emotion that dynamically adjusts Guardian threshold, Kelly
fraction, and position size. Replaces static Fear/Greed thresholds.

Each state transition is logged with reason and timestamp for audit trail.
"""
from __future__ import annotations

__all__ = [
    "CognitiveStateMachine",
    "EmotionLevel",
    "get_cognitive_state",
    "CognitiveTransition",
]

import enum
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from cortex.config import (
    APPROVAL_THRESHOLD,
    GUARDIAN_KELLY_FRACTION,
)

logger = logging.getLogger(__name__)

# Feature flag
COGNITIVE_STATE_ENABLED_KEY = "COGNITIVE_STATE_ENABLED"


class EmotionLevel(str, enum.Enum):
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


# Numeric mapping: -2 (extreme fear) to +2 (extreme greed)
_LEVEL_VALUE = {
    EmotionLevel.EXTREME_FEAR: -2,
    EmotionLevel.FEAR: -1,
    EmotionLevel.NEUTRAL: 0,
    EmotionLevel.GREED: 1,
    EmotionLevel.EXTREME_GREED: 2,
}

# Threshold adjustment per emotion level (added to APPROVAL_THRESHOLD)
_THRESHOLD_DELTA = {
    EmotionLevel.EXTREME_FEAR: -8.0,   # contrarian opportunity: lower bar
    EmotionLevel.FEAR: -3.0,
    EmotionLevel.NEUTRAL: 0.0,
    EmotionLevel.GREED: 4.0,
    EmotionLevel.EXTREME_GREED: 10.0,  # overheated: raise bar significantly
}

# Kelly fraction multiplier per emotion level
_KELLY_MULTIPLIER = {
    EmotionLevel.EXTREME_FEAR: 0.6,    # reduce size in extreme fear (high uncertainty)
    EmotionLevel.FEAR: 0.8,
    EmotionLevel.NEUTRAL: 1.0,
    EmotionLevel.GREED: 0.85,
    EmotionLevel.EXTREME_GREED: 0.5,   # cut size significantly in extreme greed
}

# Position size haircut (multiplier on recommended_size)
_SIZE_MULTIPLIER = {
    EmotionLevel.EXTREME_FEAR: 0.7,
    EmotionLevel.FEAR: 0.9,
    EmotionLevel.NEUTRAL: 1.0,
    EmotionLevel.GREED: 0.85,
    EmotionLevel.EXTREME_GREED: 0.6,
}


@dataclass
class CognitiveTransition:
    from_level: EmotionLevel
    to_level: EmotionLevel
    reason: str
    raw_score: float
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.from_level.value,
            "to": self.to_level.value,
            "reason": self.reason,
            "raw_score": round(self.raw_score, 4),
            "timestamp": self.timestamp,
        }


def _classify_fear_greed(value: int) -> tuple[float, str]:
    """Map Fear/Greed index (0-100) to a -1..+1 signal."""
    normalized = (value - 50) / 50.0
    if value <= 20:
        return normalized, "extreme_fear_zone"
    elif value <= 40:
        return normalized, "fear_zone"
    elif value >= 80:
        return normalized, "extreme_greed_zone"
    elif value >= 60:
        return normalized, "greed_zone"
    return normalized, "neutral_zone"


def _classify_regime(regime: int, num_states: int, crisis_prob: float) -> tuple[float, str]:
    """Map regime state to a -1..+1 signal. Crisis → negative (fear), calm → positive (greed)."""
    if num_states <= 1:
        return 0.0, "single_regime"
    ratio = (regime - 1) / (num_states - 1)
    # Invert: high regime (crisis) = fear = negative
    signal = -(ratio * 2.0 - 1.0)
    if crisis_prob > 0.5:
        signal = max(-1.0, signal - 0.3)
    if regime >= num_states:
        return max(-1.0, signal), "crisis_regime"
    elif regime >= num_states - 1:
        return signal, "near_crisis"
    elif regime <= 1:
        return signal, "calm_regime"
    return signal, "mid_regime"


def _classify_hawkes(contagion_score: float) -> tuple[float, str]:
    """Map Hawkes contagion (0-1) to a -1..+1 signal. High contagion → negative (fear)."""
    signal = -(contagion_score * 2.0 - 1.0)
    if contagion_score > 0.75:
        return signal, "flash_crash_risk"
    elif contagion_score > 0.5:
        return signal, "elevated_contagion"
    return signal, "low_contagion"


def _classify_news_sentiment(ewma: float) -> tuple[float, str]:
    """Map news sentiment EWMA (-1..+1) to greed/fear signal."""
    if ewma > 0.5:
        return ewma, "bullish_news"
    elif ewma < -0.5:
        return ewma, "bearish_news"
    return ewma, "mixed_news"


class CognitiveStateMachine:
    """Living emotion state that adapts to market conditions.

    Inputs (all optional, partial data is fine):
        - Fear/Greed index (0-100)
        - Regime state (current regime, num_states, crisis_prob)
        - Hawkes contagion score (0-1)
        - News sentiment EWMA (-1..+1)

    Output:
        - EmotionLevel (5 states)
        - Threshold adjustment for Guardian
        - Kelly fraction multiplier
        - Position size multiplier
    """

    WEIGHTS = {
        "fear_greed": 0.35,
        "regime": 0.30,
        "hawkes": 0.20,
        "news": 0.15,
    }

    def __init__(self, smoothing: float = 0.3, max_history: int = 100) -> None:
        self._current_level = EmotionLevel.NEUTRAL
        self._raw_score: float = 0.0
        self._smoothed_score: float = 0.0
        self._smoothing = smoothing  # EMA alpha: higher = more reactive
        self._transitions: list[CognitiveTransition] = []
        self._max_history = max_history
        self._last_inputs: dict[str, Any] = {}
        self._last_update: float = 0.0

    @property
    def level(self) -> EmotionLevel:
        return self._current_level

    @property
    def raw_score(self) -> float:
        return self._raw_score

    @property
    def smoothed_score(self) -> float:
        return self._smoothed_score

    def update(
        self,
        fear_greed_value: int | None = None,
        regime: int | None = None,
        num_states: int = 5,
        crisis_prob: float = 0.0,
        hawkes_contagion: float | None = None,
        news_ewma: float | None = None,
    ) -> EmotionLevel:
        """Ingest latest signals and compute new emotion level."""
        signals: dict[str, tuple[float, str]] = {}
        reasons: list[str] = []

        if fear_greed_value is not None:
            sig, reason = _classify_fear_greed(fear_greed_value)
            signals["fear_greed"] = (sig, reason)
            reasons.append(f"fg={fear_greed_value}({reason})")

        if regime is not None:
            sig, reason = _classify_regime(regime, num_states, crisis_prob)
            signals["regime"] = (sig, reason)
            reasons.append(f"regime={regime}/{num_states}({reason})")

        if hawkes_contagion is not None:
            sig, reason = _classify_hawkes(hawkes_contagion)
            signals["hawkes"] = (sig, reason)
            reasons.append(f"hawkes={hawkes_contagion:.2f}({reason})")

        if news_ewma is not None:
            sig, reason = _classify_news_sentiment(news_ewma)
            signals["news"] = (sig, reason)
            reasons.append(f"news={news_ewma:.2f}({reason})")

        if not signals:
            return self._current_level

        # Weighted average of available signals (renormalize weights)
        total_weight = sum(self.WEIGHTS[k] for k in signals)
        if total_weight > 0:
            self._raw_score = sum(
                signals[k][0] * self.WEIGHTS[k] / total_weight
                for k in signals
            )
        else:
            self._raw_score = 0.0

        # EMA smoothing
        if self._last_update == 0:
            self._smoothed_score = self._raw_score
        else:
            self._smoothed_score = (
                self._smoothing * self._raw_score
                + (1 - self._smoothing) * self._smoothed_score
            )

        # Map smoothed score to emotion level
        new_level = self._score_to_level(self._smoothed_score)

        self._last_inputs = {
            "fear_greed_value": fear_greed_value,
            "regime": regime,
            "num_states": num_states,
            "crisis_prob": crisis_prob,
            "hawkes_contagion": hawkes_contagion,
            "news_ewma": news_ewma,
            "signals": {k: {"value": v[0], "reason": v[1]} for k, v in signals.items()},
        }
        self._last_update = time.time()

        if new_level != self._current_level:
            transition = CognitiveTransition(
                from_level=self._current_level,
                to_level=new_level,
                reason="; ".join(reasons),
                raw_score=self._raw_score,
                timestamp=self._last_update,
            )
            self._transitions.append(transition)
            if len(self._transitions) > self._max_history:
                self._transitions = self._transitions[-self._max_history:]

            logger.info(
                "cognitive_state TRANSITION %s → %s score=%.3f reason='%s'",
                self._current_level.value, new_level.value,
                self._smoothed_score, transition.reason,
            )
            self._current_level = new_level

        return self._current_level

    def get_threshold_adjustment(self) -> float:
        """Return the threshold delta to add to APPROVAL_THRESHOLD."""
        return _THRESHOLD_DELTA[self._current_level]

    def get_effective_threshold(self) -> float:
        """Return adjusted approval threshold."""
        base = APPROVAL_THRESHOLD
        delta = self.get_threshold_adjustment()
        return max(50.0, min(95.0, base + delta))

    def get_kelly_multiplier(self) -> float:
        """Return Kelly fraction multiplier for current emotion."""
        return _KELLY_MULTIPLIER[self._current_level]

    def get_size_multiplier(self) -> float:
        """Return position size multiplier for current emotion."""
        return _SIZE_MULTIPLIER[self._current_level]

    def get_adjustments(self) -> dict[str, Any]:
        """Return all adjustments as a dict for Guardian integration."""
        return {
            "emotion_level": self._current_level.value,
            "raw_score": round(self._raw_score, 4),
            "smoothed_score": round(self._smoothed_score, 4),
            "threshold_delta": self.get_threshold_adjustment(),
            "effective_threshold": self.get_effective_threshold(),
            "kelly_multiplier": self.get_kelly_multiplier(),
            "size_multiplier": self.get_size_multiplier(),
            "last_inputs": self._last_inputs,
            "last_update": self._last_update,
        }

    def get_transitions(self, limit: int = 20) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self._transitions[-limit:]]

    def snapshot(self) -> dict[str, Any]:
        """Serialize state for persistence."""
        return {
            "level": self._current_level.value,
            "raw_score": self._raw_score,
            "smoothed_score": self._smoothed_score,
            "last_inputs": self._last_inputs,
            "last_update": self._last_update,
            "transitions": [t.to_dict() for t in self._transitions],
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore state from a snapshot."""
        self._current_level = EmotionLevel(data.get("level", "neutral"))
        self._raw_score = data.get("raw_score", 0.0)
        self._smoothed_score = data.get("smoothed_score", 0.0)
        self._last_inputs = data.get("last_inputs", {})
        self._last_update = data.get("last_update", 0.0)
        transitions = data.get("transitions", [])
        self._transitions = [
            CognitiveTransition(
                from_level=EmotionLevel(t["from"]),
                to_level=EmotionLevel(t["to"]),
                reason=t["reason"],
                raw_score=t["raw_score"],
                timestamp=t["timestamp"],
            )
            for t in transitions
        ]

    @staticmethod
    def _score_to_level(score: float) -> EmotionLevel:
        """Map smoothed score (-1..+1) to emotion level with hysteresis bands."""
        if score <= -0.6:
            return EmotionLevel.EXTREME_FEAR
        elif score <= -0.2:
            return EmotionLevel.FEAR
        elif score >= 0.6:
            return EmotionLevel.EXTREME_GREED
        elif score >= 0.2:
            return EmotionLevel.GREED
        return EmotionLevel.NEUTRAL


_csm: CognitiveStateMachine | None = None


def get_cognitive_state() -> CognitiveStateMachine:
    global _csm
    if _csm is None:
        _csm = CognitiveStateMachine()
    return _csm
