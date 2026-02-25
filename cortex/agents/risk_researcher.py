"""Risk Researcher agent — EVT, SVJ, Hawkes, MSM model scoring."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from cortex.agents.base import AgentSignal, BaseAgent

logger = logging.getLogger(__name__)


class RiskResearcherAgent(BaseAgent):
    """Scores risk using pre-calibrated EVT, SVJ, Hawkes, and MSM models."""

    name = "risk_researcher"
    weight = 0.50  # highest weight — this is the core risk engine

    # Sub-weights for internal model composition (normalized)
    MODEL_WEIGHTS = {"evt": 0.30, "svj": 0.21, "hawkes": 0.25, "regime": 0.24}

    def analyze(self, token: str, data: pd.DataFrame, context: dict[str, Any]) -> AgentSignal:
        """Score risk using calibrated model params from context."""
        evt_data = context.get("evt_data")
        svj_data = context.get("svj_data")
        hawkes_data = context.get("hawkes_data")
        model_data = context.get("model_data")

        scores = {}
        details = {}

        # EVT scoring
        if evt_data:
            try:
                from cortex.guardian import _score_evt
                result = _score_evt(evt_data)
                scores["evt"] = result.get("score", 50.0)
                details["evt"] = result
            except Exception:
                scores["evt"] = 50.0
                details["evt"] = {"score": 50.0, "error": True}
        else:
            scores["evt"] = 50.0

        # SVJ scoring
        if svj_data:
            try:
                from cortex.guardian import _score_svj
                result = _score_svj(svj_data)
                scores["svj"] = result.get("score", 50.0)
                details["svj"] = result
            except Exception:
                scores["svj"] = 50.0
                details["svj"] = {"score": 50.0, "error": True}
        else:
            scores["svj"] = 50.0

        # Hawkes scoring
        if hawkes_data:
            try:
                from cortex.guardian import _score_hawkes
                result = _score_hawkes(hawkes_data)
                scores["hawkes"] = result.get("score", 50.0)
                details["hawkes"] = result
            except Exception:
                scores["hawkes"] = 50.0
                details["hawkes"] = {"score": 50.0, "error": True}
        else:
            scores["hawkes"] = 50.0

        # Regime scoring
        if model_data:
            try:
                from cortex.guardian import _score_regime
                result = _score_regime(model_data)
                scores["regime"] = result.get("score", 50.0)
                details["regime"] = result
            except Exception:
                scores["regime"] = 50.0
                details["regime"] = {"score": 50.0, "error": True}
        else:
            scores["regime"] = 50.0

        # Weighted composite
        composite = sum(
            scores[k] * self.MODEL_WEIGHTS[k] for k in self.MODEL_WEIGHTS
        )

        # Direction based on regime (only if we have regime data)
        regime_state = 0
        direction = None
        if model_data:
            regime_state = model_data.get("current_regime", 0)
            if regime_state <= 1:
                direction = "long"
            elif regime_state >= 3:
                direction = "short"

        # Confidence: higher when models agree (low variance)
        score_values = list(scores.values())
        if len(score_values) > 1:
            import numpy as np
            variance = np.var(score_values)
            max_var = 2500  # max possible variance (0 and 100)
            agreement = 1.0 - min(variance / max_var, 1.0)
            confidence = 0.4 + agreement * 0.4  # 0.4-0.8
        else:
            confidence = 0.5

        parts = [f"{k}={v:.0f}" for k, v in scores.items()]
        reasoning = f"Risk: {', '.join(parts)} → composite={composite:.1f} (regime={regime_state})"

        return AgentSignal(
            agent_name=self.name,
            score=composite,
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            metadata={
                "evt_score": round(scores["evt"], 2),
                "svj_score": round(scores["svj"], 2),
                "hawkes_score": round(scores["hawkes"], 2),
                "regime_score": round(scores["regime"], 2),
                "regime_state": regime_state,
                "model_details": details,
            },
        )
