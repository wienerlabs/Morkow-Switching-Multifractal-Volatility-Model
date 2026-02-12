"""Adversarial Debate System — multi-round challenge/defense between trading agents.

Three agent roles:
  - Trader: proposes trade with reasoning (bullish bias)
  - Risk Manager: challenges with risk concerns (bearish bias)
  - Portfolio Manager: arbitrates final decision (neutral)

Rule-based implementation. Pluggable LLM interface for future upgrade.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from cortex.config import DEBATE_MAX_ROUNDS

logger = logging.getLogger(__name__)


def _trader_argue(risk_score: float, component_scores: list[dict],
                  direction: str, trade_size_usd: float, round_num: int) -> dict[str, Any]:
    low_components = [s for s in component_scores if s["score"] < 50]
    confidence = max(0.1, 1.0 - risk_score / 100.0)
    confidence *= max(0.5, 1.0 - 0.1 * round_num)

    arguments: list[str] = []
    if low_components:
        names = ", ".join(s["component"] for s in low_components)
        arguments.append(f"Favorable signals from: {names}")
    if risk_score < 50:
        arguments.append(f"Overall risk score {risk_score:.0f} is within acceptable range")
    if risk_score >= 50:
        arguments.append(f"Risk score {risk_score:.0f} is elevated but manageable with reduced size")

    return {"role": "trader", "position": "approve", "confidence": round(confidence, 4),
            "arguments": arguments, "suggested_action": direction}


def _risk_manager_argue(risk_score: float, component_scores: list[dict],
                        veto_reasons: list[str], round_num: int) -> dict[str, Any]:
    high_components = [s for s in component_scores if s["score"] >= 60]
    confidence = min(0.95, risk_score / 100.0)
    confidence *= min(1.0, 0.8 + 0.1 * round_num)

    arguments: list[str] = []
    if veto_reasons:
        arguments.append(f"Active veto triggers: {', '.join(veto_reasons)}")
    if high_components:
        for s in high_components:
            arguments.append(f"{s['component']} risk at {s['score']:.0f}/100")
    if risk_score >= 75:
        arguments.append("Composite risk exceeds approval threshold")
    elif risk_score >= 50:
        arguments.append("Elevated risk warrants significant size reduction")

    position = "reject" if risk_score >= 75 or veto_reasons else "reduce"
    return {"role": "risk_manager", "position": position, "confidence": round(confidence, 4),
            "arguments": arguments, "suggested_action": "block" if position == "reject" else "reduce_size"}


def _portfolio_manager_arbitrate(trader: dict[str, Any], risk_mgr: dict[str, Any],
                                 risk_score: float, original_approved: bool) -> dict[str, Any]:
    t_conf = trader["confidence"]
    r_conf = risk_mgr["confidence"]
    trader_weight = 0.4 * t_conf
    risk_weight = 0.6 * r_conf
    total = trader_weight + risk_weight
    risk_lean = risk_weight / total if total > 0 else 0.5
    approve = risk_lean < 0.55 and risk_score < 75

    reasoning: list[str] = []
    if approve:
        reasoning.append(f"Trader confidence ({t_conf:.2f}) sufficient against risk ({r_conf:.2f})")
        if risk_score > 50:
            reasoning.append("Recommending reduced position size")
    else:
        reasoning.append(f"Risk manager concerns outweigh (risk_lean={risk_lean:.2f})")
        if risk_mgr["position"] == "reject":
            reasoning.append("Hard veto conditions present")

    return {"role": "portfolio_manager", "decision": "approve" if approve else "reject",
            "confidence": round(abs(0.5 - risk_lean) * 2, 4), "reasoning": reasoning,
            "trader_weight": round(trader_weight, 4), "risk_weight": round(risk_weight, 4)}


def run_debate(risk_score: float, component_scores: list[dict], veto_reasons: list[str],
               direction: str, trade_size_usd: float, original_approved: bool) -> dict[str, Any]:
    start = time.time()
    rounds: list[dict[str, Any]] = []

    for r in range(DEBATE_MAX_ROUNDS):
        trader = _trader_argue(risk_score, component_scores, direction, trade_size_usd, r)
        risk_mgr = _risk_manager_argue(risk_score, component_scores, veto_reasons, r)
        pm = _portfolio_manager_arbitrate(trader, risk_mgr, risk_score, original_approved)
        rounds.append({"round": r + 1, "trader": trader, "risk_manager": risk_mgr, "arbitrator": pm})
        if pm["confidence"] > 0.7:
            break

    final = rounds[-1]["arbitrator"]
    elapsed_ms = (time.time() - start) * 1000

    return {
        "final_decision": final["decision"],
        "final_confidence": final["confidence"],
        "rounds": rounds,
        "num_rounds": len(rounds),
        "elapsed_ms": round(elapsed_ms, 1),
        "original_approved": original_approved,
        "decision_changed": (final["decision"] == "approve") != original_approved,
    }

