"""Adaptive Guardian Weights — Exponential Gradient online learning.

Updates component weights after each trade outcome using multiplicative
weight update (Exponential Gradient):

    w_{k,t+1} = w_k * exp(eta * r_k) / Z_t

where r_k is the component's prediction accuracy for the trade outcome
and Z_t is a normalization factor ensuring weights sum to 1.

This lets the system learn which risk components (EVT, SVJ, Hawkes,
Regime, News, A-LAMS) are most predictive over time.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from typing import Any

from cortex.config import (
    ADAPTIVE_WEIGHTS_ENABLED,
    ADAPTIVE_WEIGHTS_LR,
    ADAPTIVE_WEIGHTS_MIN_SAMPLES,
    GUARDIAN_WEIGHTS,
)

logger = logging.getLogger(__name__)

_current_weights: dict[str, float] = dict(GUARDIAN_WEIGHTS)
_outcome_history: deque[dict] = deque(maxlen=500)
_update_count: int = 0


def get_weights() -> dict[str, float]:
    """Return current adaptive weights (or static defaults if disabled)."""
    if not ADAPTIVE_WEIGHTS_ENABLED:
        return dict(GUARDIAN_WEIGHTS)
    if _update_count < ADAPTIVE_WEIGHTS_MIN_SAMPLES:
        return dict(GUARDIAN_WEIGHTS)
    return dict(_current_weights)


def record_outcome(
    component_scores: list[dict],
    risk_score: float,
    trade_pnl: float,
) -> dict[str, Any]:
    """Record trade outcome and update weights via Exponential Gradient.

    Args:
        component_scores: List of {"component": str, "score": float} from Guardian.
        risk_score: Composite risk score at trade time.
        trade_pnl: Realized PnL of the trade (positive = win, negative = loss).

    Returns:
        Dict with updated weights and update metadata.
    """
    global _update_count

    if not ADAPTIVE_WEIGHTS_ENABLED:
        return {"enabled": False, "weights": dict(GUARDIAN_WEIGHTS)}

    _outcome_history.append({
        "component_scores": {s["component"]: s["score"] for s in component_scores},
        "risk_score": risk_score,
        "pnl": trade_pnl,
    })
    _update_count += 1

    if _update_count < ADAPTIVE_WEIGHTS_MIN_SAMPLES:
        return {
            "enabled": True,
            "active": False,
            "reason": f"need {ADAPTIVE_WEIGHTS_MIN_SAMPLES} samples, have {_update_count}",
            "weights": dict(GUARDIAN_WEIGHTS),
        }

    # Compute per-component reward: positive if component correctly predicted outcome
    # High score + loss = component was right (warned of risk) → positive reward
    # Low score + win = component was right (allowed trade) → positive reward
    # High score + win = component was wrong (false alarm) → negative reward
    # Low score + loss = component was wrong (missed risk) → negative reward
    outcome_positive = trade_pnl > 0
    score_map = {s["component"]: s["score"] for s in component_scores}

    eta = ADAPTIVE_WEIGHTS_LR
    new_weights: dict[str, float] = {}

    for comp, w in _current_weights.items():
        score = score_map.get(comp, 50.0)
        predicted_risky = score > 50.0

        if (predicted_risky and not outcome_positive) or (not predicted_risky and outcome_positive):
            reward = 1.0  # correct prediction
        else:
            reward = -1.0  # incorrect prediction

        new_weights[comp] = w * math.exp(eta * reward)

    # Normalize
    total = sum(new_weights.values())
    if total > 0:
        for comp in new_weights:
            new_weights[comp] /= total

    _current_weights.update(new_weights)

    logger.debug(
        "adaptive_weights_update n=%d weights=%s",
        _update_count,
        {k: round(v, 4) for k, v in _current_weights.items()},
    )

    return {
        "enabled": True,
        "active": True,
        "n_updates": _update_count,
        "weights": {k: round(v, 6) for k, v in _current_weights.items()},
    }


def get_stats() -> dict[str, Any]:
    """Return adaptive weight statistics."""
    return {
        "enabled": ADAPTIVE_WEIGHTS_ENABLED,
        "active": _update_count >= ADAPTIVE_WEIGHTS_MIN_SAMPLES,
        "n_updates": _update_count,
        "current_weights": {k: round(v, 6) for k, v in _current_weights.items()},
        "default_weights": dict(GUARDIAN_WEIGHTS),
        "learning_rate": ADAPTIVE_WEIGHTS_LR,
        "min_samples": ADAPTIVE_WEIGHTS_MIN_SAMPLES,
    }


def reset() -> None:
    """Reset to default weights (for testing)."""
    global _current_weights, _update_count
    _current_weights = dict(GUARDIAN_WEIGHTS)
    _outcome_history.clear()
    _update_count = 0
