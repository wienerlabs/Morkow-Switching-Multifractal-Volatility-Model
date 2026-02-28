"""Agent Coordinator — dispatches to specialist agents and synthesizes signals.

Collects AgentSignals from all registered agents, applies confidence-weighted
aggregation, and produces a Guardian-compatible approval decision with
component-level transparency.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from cortex.agents.base import AgentSignal, BaseAgent
from cortex.config import SHARPE_WEIGHTS_ENABLED, HMM_REGIME_ENABLED

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorDecision:
    """Synthesized output from the multi-agent coordinator."""

    approved: bool
    risk_score: float
    direction: str | None
    confidence: float
    signals: list[AgentSignal]
    veto_reasons: list[str] = field(default_factory=list)
    recommended_size: float = 0.0
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AgentCoordinator:
    """Dispatches analysis to specialist agents and aggregates their signals."""

    def __init__(
        self,
        agents: list[BaseAgent],
        approval_threshold: float = 60.0,
        veto_score: float = 85.0,
        pnl_tracker=None,
        hmm_detector=None,
    ) -> None:
        self.agents = agents
        self.approval_threshold = approval_threshold
        self.veto_score = veto_score
        self._pnl_tracker = pnl_tracker
        self._hmm_detector = hmm_detector

    def evaluate(
        self,
        token: str,
        data: pd.DataFrame,
        context: dict[str, Any],
        trade_size_usd: float = 0.0,
    ) -> CoordinatorDecision:
        """Run all agents in live mode and synthesize."""
        signals = []
        for agent in self.agents:
            try:
                sig = agent.analyze(token, data, context)
                signals.append(sig)
            except Exception:
                logger.warning("agent_%s_failed", agent.name, exc_info=True)

        return self._synthesize(signals, trade_size_usd)

    def evaluate_backtest(
        self,
        token: str,
        data: pd.DataFrame,
        bar_idx: int,
        context: dict[str, Any],
        trade_size_usd: float = 0.0,
    ) -> CoordinatorDecision:
        """Run all agents in backtest mode and synthesize."""
        signals = []
        for agent in self.agents:
            try:
                sig = agent.analyze_backtest(token, data, bar_idx, context)
                signals.append(sig)
            except Exception:
                logger.warning("agent_%s_backtest_failed", agent.name, exc_info=True)

        return self._synthesize(signals, trade_size_usd)

    def _synthesize(
        self, signals: list[AgentSignal], trade_size_usd: float
    ) -> CoordinatorDecision:
        """Aggregate agent signals into a single decision."""
        if not signals:
            return CoordinatorDecision(
                approved=False,
                risk_score=50.0,
                direction=None,
                confidence=0.0,
                signals=[],
                veto_reasons=["no_agent_signals"],
                reasoning="No agents produced signals",
            )

        # Build weight map — dynamic Sharpe weights if available, else static
        agent_weights = {a.name: a.weight for a in self.agents}

        if SHARPE_WEIGHTS_ENABLED and self._pnl_tracker is not None:
            sharpe_weights = self._pnl_tracker.compute_sharpe_weights()
            if sharpe_weights is not None:
                agent_weights = sharpe_weights

        # Apply regime multiplier if available
        regime_multiplier = 1.0
        if HMM_REGIME_ENABLED and self._hmm_detector is not None:
            regime_mults = self._hmm_detector.get_regime_multipliers()
            if regime_mults is not None:
                regime_multiplier = regime_mults.get("regime_multiplier", 1.0)

        # Confidence-weighted risk score
        total_weight = 0.0
        weighted_score = 0.0
        for sig in signals:
            w = agent_weights.get(sig.agent_name, 1.0) * regime_multiplier * sig.confidence
            weighted_score += sig.score * w
            total_weight += w

        risk_score = weighted_score / total_weight if total_weight > 0 else 50.0
        risk_score = max(0.0, min(100.0, risk_score))

        # Direction consensus: weighted vote
        direction_votes: dict[str, float] = {"long": 0.0, "short": 0.0}
        for sig in signals:
            if sig.direction is not None:
                w = agent_weights.get(sig.agent_name, 1.0) * regime_multiplier * sig.confidence
                direction_votes[sig.direction] += w

        if direction_votes["long"] > direction_votes["short"] and direction_votes["long"] > 0:
            direction = "long"
        elif direction_votes["short"] > direction_votes["long"] and direction_votes["short"] > 0:
            direction = "short"
        else:
            direction = None

        # Aggregate confidence
        confidences = [s.confidence for s in signals]
        avg_confidence = sum(confidences) / len(confidences)

        # Veto logic
        veto_reasons = []
        for sig in signals:
            if sig.score >= self.veto_score:
                veto_reasons.append(f"{sig.agent_name}_extreme_risk")

        # Approval: below threshold AND no vetoes AND has direction
        approved = (
            risk_score < self.approval_threshold
            and len(veto_reasons) == 0
            and direction is not None
        )

        # Size scaling: linear from threshold
        if approved and trade_size_usd > 0:
            scale = max(0.0, 1.0 - risk_score / 100.0)
            recommended_size = round(trade_size_usd * scale, 2)
        else:
            recommended_size = 0.0

        # Reasoning summary
        parts = [f"{s.agent_name}={s.score:.0f}(c={s.confidence:.2f})" for s in signals]
        reasoning = (
            f"Coordinator: {', '.join(parts)} → "
            f"composite={risk_score:.1f}, dir={direction}, "
            f"{'APPROVED' if approved else 'REJECTED'}"
        )

        return CoordinatorDecision(
            approved=approved,
            risk_score=round(risk_score, 2),
            direction=direction,
            confidence=round(avg_confidence, 4),
            signals=signals,
            veto_reasons=veto_reasons,
            recommended_size=recommended_size,
            reasoning=reasoning,
            metadata={
                "agent_scores": {s.agent_name: round(s.score, 2) for s in signals},
                "agent_directions": {s.agent_name: s.direction for s in signals},
                "agent_confidences": {s.agent_name: round(s.confidence, 4) for s in signals},
                "direction_votes": {k: round(v, 4) for k, v in direction_votes.items()},
                "threshold": self.approval_threshold,
                "veto_threshold": self.veto_score,
            },
        )

    def to_guardian_format(self, decision: CoordinatorDecision) -> dict[str, Any]:
        """Convert CoordinatorDecision to Guardian assess_trade() compatible dict."""
        component_scores = []
        for sig in decision.signals:
            component_scores.append({
                "component": sig.agent_name,
                "score": round(sig.score, 2),
                "details": sig.metadata,
            })

        return {
            "approved": decision.approved,
            "risk_score": decision.risk_score,
            "veto_reasons": decision.veto_reasons,
            "recommended_size": decision.recommended_size,
            "regime_state": self._extract_regime(decision),
            "confidence": decision.confidence,
            "calibrated_confidence": None,
            "effective_threshold": self.approval_threshold,
            "hawkes_deferred": False,
            "copula_gate_triggered": False,
            "expires_at": None,
            "component_scores": component_scores,
            "circuit_breaker": None,
            "portfolio_limits": None,
            "debate": None,
            "human_override": None,
            "cognitive_state": None,
            "from_cache": False,
            "request_id": None,
            "agent_coordinator": {
                "reasoning": decision.reasoning,
                "direction_votes": decision.metadata.get("direction_votes", {}),
            },
        }

    def _extract_regime(self, decision: CoordinatorDecision) -> int:
        """Extract regime state from risk_researcher signal if available."""
        for sig in decision.signals:
            if sig.agent_name == "risk_researcher":
                return sig.metadata.get("regime_state", 2)
        return 2
