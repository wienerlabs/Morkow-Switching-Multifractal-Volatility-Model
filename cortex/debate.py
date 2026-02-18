"""Adversarial Debate System — evidence-based multi-round challenge/defense.

Four agent roles:
  - Trader:         proposes trade with quantitative evidence (bullish bias)
  - Risk Manager:   challenges with risk models + VaR evidence (bearish bias)
  - Devil's Advocate: challenges whichever side is winning (contrarian)
  - Portfolio Manager: arbitrates with Bayesian confidence update (neutral)

Strategy-aware: LP, arbitrage, perps, spot each trigger different evidence
patterns and risk thresholds.

Evidence is sourced from:
  - Guardian component scores (EVT, SVJ, Hawkes, Regime, News, A-LAMS)
  - Kelly Criterion statistics
  - Portfolio drawdown + correlated exposure
  - Circuit breaker states
  - A-LAMS VaR (regime-conditional)
"""
from __future__ import annotations

__all__ = [
    "DebateEvidence",
    "AgentArgument",
    "DebateContext",
    "enrich_context",
    "run_debate",
    "record_debate_outcome",
    "persist_debate_state",
    "restore_debate_state",
]

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

from cortex.config import (
    APPROVAL_THRESHOLD,
    DEBATE_MAX_ROUNDS,
    DEBATE_EMPIRICAL_PRIOR_ENABLED,
    DEBATE_PRIOR_MAX,
    DEBATE_PRIOR_MIN,
    MAX_CORRELATED_EXPOSURE,
    MAX_DAILY_DRAWDOWN,
)

logger = logging.getLogger(__name__)

# ── Strategy risk profiles ──────────────────────────────────────────────

STRATEGY_PROFILES: dict[str, dict[str, Any]] = {
    "lp": {
        "risk_tolerance": 0.45,
        "key_risks": ["impermanent_loss", "pool_imbalance", "smart_contract"],
        "size_cap_pct": 0.25,
        "min_confidence": 0.55,
        "consecutive_loss_limit": 3,
        "loss_type": "impermanent_loss",
    },
    "arb": {
        "risk_tolerance": 0.35,
        "key_risks": ["execution_risk", "latency", "price_movement"],
        "size_cap_pct": 0.15,
        "min_confidence": 0.65,
        "consecutive_loss_limit": 5,
        "loss_type": "failed_execution",
    },
    "perp": {
        "risk_tolerance": 0.55,
        "key_risks": ["liquidation", "funding_rate", "leverage_decay"],
        "size_cap_pct": 0.20,
        "min_confidence": 0.60,
        "consecutive_loss_limit": 2,
        "loss_type": "stop_loss",
    },
    "spot": {
        "risk_tolerance": 0.40,
        "key_risks": ["market_risk", "slippage", "volatility"],
        "size_cap_pct": 0.30,
        "min_confidence": 0.50,
        "consecutive_loss_limit": 4,
        "loss_type": "market_loss",
    },
    "lending": {
        "risk_tolerance": 0.30,
        "key_risks": ["utilization_rate", "liquidation_risk", "rate_change"],
        "size_cap_pct": 0.35,
        "min_confidence": 0.45,
        "consecutive_loss_limit": 3,
        "loss_type": "rate_loss",
    },
}

# ── Task 7: Empirical Bayes Debate Prior ─────────────────────────────

_debate_outcomes: list[dict] = []


def record_debate_outcome(strategy: str, approved: bool, pnl: float) -> None:
    """Record a debate outcome for empirical prior computation."""
    _debate_outcomes.append({
        "strategy": strategy, "approved": approved,
        "win": pnl > 0, "ts": time.time(),
    })
    if len(_debate_outcomes) > 1000:
        _debate_outcomes[:] = _debate_outcomes[-500:]


def _compute_empirical_prior(strategy: str) -> float:
    """Compute Laplace-smoothed empirical prior from past debate outcomes.

    prior = (wins + 1) / (total + 2)   (Laplace / add-one smoothing)
    Clamped to [DEBATE_PRIOR_MIN, DEBATE_PRIOR_MAX] for stability.
    """
    if not DEBATE_EMPIRICAL_PRIOR_ENABLED:
        return 0.5

    relevant = [o for o in _debate_outcomes if o["strategy"] == strategy]
    if len(relevant) < 5:
        return 0.5

    wins = sum(1 for o in relevant if o["win"])
    total = len(relevant)
    prior = (wins + 1) / (total + 2)
    return max(DEBATE_PRIOR_MIN, min(DEBATE_PRIOR_MAX, prior))


@dataclass
class DebateEvidence:
    """Structured evidence item for debate arguments."""
    source: str
    claim: str
    value: float
    threshold: float | None = None
    severity: str = "medium"  # low, medium, high, critical
    quantitative: bool = True

    def supports_approval(self) -> bool:
        if self.threshold is None:
            return self.value < 50.0
        return self.value < self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "claim": self.claim,
            "value": round(self.value, 4),
            "threshold": round(self.threshold, 4) if self.threshold else None,
            "severity": self.severity,
            "supports_approval": self.supports_approval(),
        }


@dataclass
class AgentArgument:
    """Structured argument from a debate agent."""
    role: str
    position: str  # approve, reject, reduce
    confidence: float
    arguments: list[str]
    evidence: list[DebateEvidence]
    suggested_action: str
    suggested_size_pct: float | None = None
    bayesian_prior: float = 0.5
    bayesian_posterior: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "position": self.position,
            "confidence": round(self.confidence, 4),
            "arguments": self.arguments,
            "evidence": [e.to_dict() for e in self.evidence],
            "suggested_action": self.suggested_action,
            "suggested_size_pct": round(self.suggested_size_pct, 4) if self.suggested_size_pct else None,
            "bayesian_posterior": round(self.bayesian_posterior, 4),
        }


@dataclass
class DebateContext:
    """Full context available to debate agents."""
    risk_score: float
    component_scores: list[dict]
    veto_reasons: list[str]
    direction: str
    trade_size_usd: float
    original_approved: bool
    strategy: str = "spot"
    # Optional enrichment data
    kelly_stats: dict[str, Any] = field(default_factory=dict)
    drawdown: dict[str, Any] = field(default_factory=dict)
    correlation: dict[str, Any] = field(default_factory=dict)
    alams_data: dict[str, Any] = field(default_factory=dict)
    circuit_breaker_states: list[dict] = field(default_factory=list)
    portfolio_value: float = 100_000.0


def _collect_evidence(ctx: DebateContext) -> dict[str, list[DebateEvidence]]:
    """Collect all available evidence, categorized by source."""
    bullish: list[DebateEvidence] = []
    bearish: list[DebateEvidence] = []

    # ── Component scores ──
    for cs in ctx.component_scores:
        score = cs["score"]
        component = cs["component"]
        ev = DebateEvidence(
            source=f"guardian:{component}",
            claim=f"{component} risk score {score:.0f}/100",
            value=score,
            threshold=60.0,
            severity="high" if score > 80 else "medium" if score > 60 else "low",
        )
        if ev.supports_approval():
            bullish.append(ev)
        else:
            bearish.append(ev)

    # ── Composite risk score ──
    bearish.append(DebateEvidence(
        source="guardian:composite",
        claim=f"Composite risk score {ctx.risk_score:.1f}/100",
        value=ctx.risk_score,
        threshold=APPROVAL_THRESHOLD,
        severity="critical" if ctx.risk_score > 85 else "high" if ctx.risk_score > 70 else "medium",
    ))

    # ── Veto reasons ──
    for reason in ctx.veto_reasons:
        bearish.append(DebateEvidence(
            source="guardian:veto",
            claim=f"Veto trigger: {reason}",
            value=100.0,
            threshold=0.0,
            severity="critical",
            quantitative=False,
        ))

    # ── Kelly Criterion ──
    if ctx.kelly_stats.get("active"):
        kelly_f = ctx.kelly_stats.get("kelly_fraction", 0.0)
        win_rate = ctx.kelly_stats.get("win_rate", 0.5)
        size_pct = ctx.trade_size_usd / max(ctx.portfolio_value, 1.0)

        if size_pct <= kelly_f * 1.5:
            bullish.append(DebateEvidence(
                source="kelly",
                claim=f"Position size {size_pct:.1%} within Kelly bound ({kelly_f:.1%})",
                value=size_pct * 100,
                threshold=kelly_f * 150,
                severity="low",
            ))
        else:
            bearish.append(DebateEvidence(
                source="kelly",
                claim=f"Position size {size_pct:.1%} exceeds Kelly ({kelly_f:.1%}) by {(size_pct / kelly_f - 1):.0%}",
                value=size_pct * 100,
                threshold=kelly_f * 150,
                severity="high",
            ))

        if win_rate > 0.55:
            bullish.append(DebateEvidence(
                source="kelly",
                claim=f"Win rate {win_rate:.1%} shows positive edge",
                value=win_rate * 100,
                threshold=50.0,
                severity="low",
            ))
        elif win_rate < 0.45:
            bearish.append(DebateEvidence(
                source="kelly",
                claim=f"Win rate {win_rate:.1%} shows negative edge",
                value=(1 - win_rate) * 100,
                threshold=50.0,
                severity="high",
            ))

    # ── Drawdown ──
    if ctx.drawdown:
        daily_dd = ctx.drawdown.get("daily_drawdown_pct", 0.0)
        weekly_dd = ctx.drawdown.get("weekly_drawdown_pct", 0.0)

        if daily_dd >= MAX_DAILY_DRAWDOWN * 0.8:
            bearish.append(DebateEvidence(
                source="drawdown",
                claim=f"Daily drawdown {daily_dd:.2%} approaching limit ({MAX_DAILY_DRAWDOWN:.0%})",
                value=daily_dd * 100,
                threshold=MAX_DAILY_DRAWDOWN * 100,
                severity="critical" if daily_dd >= MAX_DAILY_DRAWDOWN else "high",
            ))
        else:
            bullish.append(DebateEvidence(
                source="drawdown",
                claim=f"Daily drawdown {daily_dd:.2%} well within limits",
                value=daily_dd * 100,
                threshold=MAX_DAILY_DRAWDOWN * 100,
                severity="low",
            ))

        if weekly_dd >= MAX_DAILY_DRAWDOWN * 0.8:
            bearish.append(DebateEvidence(
                source="drawdown",
                claim=f"Weekly drawdown {weekly_dd:.2%} elevated",
                value=weekly_dd * 100,
                threshold=MAX_DAILY_DRAWDOWN * 100,
                severity="high",
            ))

    # ── Correlation ──
    if ctx.correlation and ctx.correlation.get("group"):
        exp_pct = ctx.correlation.get("exposure_pct", 0.0)
        if exp_pct >= MAX_CORRELATED_EXPOSURE * 0.8:
            bearish.append(DebateEvidence(
                source="correlation",
                claim=f"Correlated exposure {exp_pct:.2%} in {ctx.correlation['group']}",
                value=exp_pct * 100,
                threshold=MAX_CORRELATED_EXPOSURE * 100,
                severity="high" if exp_pct >= MAX_CORRELATED_EXPOSURE else "medium",
            ))

    # ── A-LAMS VaR ──
    if ctx.alams_data:
        var_total = ctx.alams_data.get("var_total", 0.0)
        regime = ctx.alams_data.get("current_regime", 0)
        delta = ctx.alams_data.get("delta", 0.0)

        if var_total > 0.05:
            bearish.append(DebateEvidence(
                source="alams",
                claim=f"A-LAMS VaR(95%) at {var_total:.2%} — extreme tail risk",
                value=var_total * 100,
                threshold=5.0,
                severity="critical" if var_total > 0.08 else "high",
            ))
        elif var_total > 0.02:
            bearish.append(DebateEvidence(
                source="alams",
                claim=f"A-LAMS VaR(95%) at {var_total:.2%} — elevated",
                value=var_total * 100,
                threshold=5.0,
                severity="medium",
            ))
        else:
            bullish.append(DebateEvidence(
                source="alams",
                claim=f"A-LAMS VaR(95%) at {var_total:.2%} — low risk",
                value=var_total * 100,
                threshold=5.0,
                severity="low",
            ))

        if regime >= 4:
            bearish.append(DebateEvidence(
                source="alams:regime",
                claim=f"Crisis regime (state {regime}) detected",
                value=float(regime),
                threshold=3.0,
                severity="critical",
            ))
        elif regime >= 3:
            bearish.append(DebateEvidence(
                source="alams:regime",
                claim=f"High-volatility regime (state {regime}) detected",
                value=float(regime),
                threshold=3.0,
                severity="high",
            ))

        if delta > 0.3:
            bearish.append(DebateEvidence(
                source="alams:asymmetry",
                claim=f"High asymmetry δ={delta:.3f} — negative returns amplify volatility transitions",
                value=delta,
                threshold=0.3,
                severity="medium",
            ))

    # ── Circuit breakers ──
    for cb in ctx.circuit_breaker_states:
        if cb.get("state") == "open":
            bearish.append(DebateEvidence(
                source=f"circuit_breaker:{cb['name']}",
                claim=f"Circuit breaker '{cb['name']}' is OPEN",
                value=100.0,
                threshold=0.0,
                severity="critical",
                quantitative=False,
            ))
        elif cb.get("state") == "half_open":
            bearish.append(DebateEvidence(
                source=f"circuit_breaker:{cb['name']}",
                claim=f"Circuit breaker '{cb['name']}' is HALF_OPEN (probing)",
                value=50.0,
                threshold=50.0,
                severity="high",
                quantitative=False,
            ))

    return {"bullish": bullish, "bearish": bearish}


def _bayesian_update(prior: float, evidence: list[DebateEvidence], direction: str) -> float:
    """Update prior probability using log-likelihood from evidence items.

    Each evidence item's severity maps to a likelihood ratio:
    - critical: 5x (strongly shifts belief)
    - high: 3x
    - medium: 1.5x
    - low: 1.2x
    """
    severity_lr = {"critical": 5.0, "high": 3.0, "medium": 1.5, "low": 1.2}
    log_odds = math.log(prior / (1.0 - prior)) if 0 < prior < 1 else 0.0

    for ev in evidence:
        lr = severity_lr.get(ev.severity, 1.5)
        supports = ev.supports_approval()
        if direction == "approve" and supports:
            log_odds += math.log(lr)
        elif direction == "approve" and not supports:
            log_odds -= math.log(lr)
        elif direction == "reject" and not supports:
            log_odds += math.log(lr)
        elif direction == "reject" and supports:
            log_odds -= math.log(lr)

    log_odds = max(-10.0, min(10.0, log_odds))
    return 1.0 / (1.0 + math.exp(-log_odds))


def _trader_argue(ctx: DebateContext, evidence: dict[str, list[DebateEvidence]],
                  round_num: int, prev_risk_args: AgentArgument | None) -> AgentArgument:
    """Trader builds quantitative case for approval."""
    profile = STRATEGY_PROFILES.get(ctx.strategy, STRATEGY_PROFILES["spot"])
    bullish = evidence["bullish"]
    bearish = evidence["bearish"]

    # Base confidence from risk score inversion
    base_conf = max(0.1, 1.0 - ctx.risk_score / 100.0)
    # Decay confidence slightly each round (concessions)
    base_conf *= max(0.5, 1.0 - 0.08 * round_num)

    arguments: list[str] = []
    trader_evidence: list[DebateEvidence] = []

    # Lead with favorable component scores
    low_components = [cs for cs in ctx.component_scores if cs["score"] < 50]
    if low_components:
        names = ", ".join(cs["component"] for cs in low_components)
        arguments.append(f"Favorable risk signals from: {names}")

    # Present bullish evidence
    for ev in sorted(bullish, key=lambda e: -{"low": 1, "medium": 2, "high": 3, "critical": 4}[e.severity]):
        arguments.append(ev.claim)
        trader_evidence.append(ev)

    # Strategy-specific arguments
    if ctx.strategy == "arb":
        arguments.append("Arbitrage opportunities are time-sensitive and structurally risk-bounded")
    elif ctx.strategy == "lp":
        arguments.append("LP positions generate yield that compounds over holding period")
    elif ctx.strategy == "lending":
        arguments.append("Lending positions have bounded downside via collateral")

    # Counter risk manager's previous arguments
    if prev_risk_args and round_num > 0:
        critical_risks = [e for e in prev_risk_args.evidence if e.severity == "critical"]
        if not critical_risks:
            arguments.append("Risk manager raised no critical-severity concerns")
        else:
            # Acknowledge but propose mitigation
            arguments.append(f"Acknowledge {len(critical_risks)} critical concern(s) — propose {profile['size_cap_pct']:.0%} max allocation cap")

    # Propose position size
    risk_scale = max(0.0, 1.0 - ctx.risk_score / 100.0)
    suggested_size = min(profile["size_cap_pct"], risk_scale * profile["size_cap_pct"])

    # Task 7: Empirical Bayes prior
    prior = _compute_empirical_prior(ctx.strategy)

    # Bayesian posterior
    posterior = _bayesian_update(prior, bullish, "approve")

    return AgentArgument(
        role="trader",
        position="approve",
        confidence=round(min(0.95, base_conf * (1 + len(bullish) * 0.05)), 4),
        arguments=arguments[:8],
        evidence=trader_evidence[:6],
        suggested_action=ctx.direction,
        suggested_size_pct=round(suggested_size, 4),
        bayesian_prior=round(prior, 4),
        bayesian_posterior=round(posterior, 4),
    )


def _risk_manager_argue(ctx: DebateContext, evidence: dict[str, list[DebateEvidence]],
                        round_num: int, prev_trader_args: AgentArgument | None) -> AgentArgument:
    """Risk Manager builds quantitative case against or for size reduction."""
    profile = STRATEGY_PROFILES.get(ctx.strategy, STRATEGY_PROFILES["spot"])
    bearish = evidence["bearish"]

    base_conf = min(0.95, ctx.risk_score / 100.0)
    base_conf *= min(1.0, 0.8 + 0.08 * round_num)

    arguments: list[str] = []
    rm_evidence: list[DebateEvidence] = []

    # Lead with veto reasons
    if ctx.veto_reasons:
        arguments.append(f"Active veto triggers ({len(ctx.veto_reasons)}): {', '.join(ctx.veto_reasons[:3])}")

    # Present bearish evidence sorted by severity
    for ev in sorted(bearish, key=lambda e: -{"low": 1, "medium": 2, "high": 3, "critical": 4}[e.severity]):
        arguments.append(ev.claim)
        rm_evidence.append(ev)

    # Strategy-specific risk warnings
    strat_risks = profile.get("key_risks", [])
    for risk_type in strat_risks:
        if risk_type == "impermanent_loss" and ctx.alams_data.get("current_regime", 0) >= 3:
            arguments.append(f"IL risk amplified in high-vol regime (regime {ctx.alams_data.get('current_regime', 0)})")
        elif risk_type == "liquidation" and ctx.risk_score > 60:
            arguments.append("Liquidation cascade risk elevated in current market conditions")
        elif risk_type == "execution_risk" and any("hawkes" in v for v in ctx.veto_reasons):
            arguments.append("Execution risk compounded by flash crash contagion (Hawkes)")

    # Counter trader's previous arguments
    if prev_trader_args and round_num > 0:
        if prev_trader_args.suggested_size_pct and prev_trader_args.suggested_size_pct > profile["size_cap_pct"] * 0.5:
            arguments.append(
                f"Proposed size {prev_trader_args.suggested_size_pct:.1%} exceeds "
                f"half of strategy cap ({profile['size_cap_pct'] * 0.5:.1%}) — too aggressive"
            )

    # Determine position
    critical_count = sum(1 for e in bearish if e.severity == "critical")
    position = "reject" if critical_count >= 2 or ctx.risk_score >= 80 or len(ctx.veto_reasons) > 0 else "reduce"

    # Task 7: Empirical Bayes prior
    prior = _compute_empirical_prior(ctx.strategy)

    # Bayesian posterior (risk perspective)
    posterior = _bayesian_update(prior, bearish, "reject")

    return AgentArgument(
        role="risk_manager",
        position=position,
        confidence=round(min(0.95, base_conf * (1 + len(bearish) * 0.03)), 4),
        arguments=arguments[:8],
        evidence=rm_evidence[:6],
        suggested_action="block" if position == "reject" else "reduce_size",
        suggested_size_pct=round(profile["size_cap_pct"] * 0.3, 4) if position == "reduce" else 0.0,
        bayesian_prior=round(prior, 4),
        bayesian_posterior=round(posterior, 4),
    )


def _devils_advocate_argue(ctx: DebateContext, trader: AgentArgument,
                           risk_mgr: AgentArgument, round_num: int) -> AgentArgument:
    """Devil's Advocate challenges the majority position to stress-test reasoning."""
    # Determine which side is winning
    trader_strength = trader.confidence * trader.bayesian_posterior
    risk_strength = risk_mgr.confidence * risk_mgr.bayesian_posterior

    # Challenge the stronger side
    target = "trader" if trader_strength > risk_strength else "risk_manager"
    arguments: list[str] = []
    da_evidence: list[DebateEvidence] = []

    if target == "trader":
        # Challenge the bullish case
        arguments.append(f"Challenging trader (confidence {trader.confidence:.2f}, posterior {trader.bayesian_posterior:.2f})")

        # Look for weak bullish evidence
        weak_bull = [e for e in trader.evidence if e.severity in ("low", "medium")]
        if weak_bull:
            arguments.append(f"{len(weak_bull)} of {len(trader.evidence)} bullish evidence items are low/medium severity")

        # Question suggested size
        if trader.suggested_size_pct and trader.suggested_size_pct > 0.10:
            arguments.append(f"Proposed position {trader.suggested_size_pct:.1%} is aggressive given {ctx.risk_score:.0f} risk score")

        # Survivorship bias check
        if ctx.kelly_stats.get("active") and ctx.kelly_stats.get("n_trades", 0) < 100:
            arguments.append(f"Kelly based on only {ctx.kelly_stats.get('n_trades', 0)} trades — insufficient for reliable edge estimation")

        position = "reduce"
        confidence = risk_strength / (trader_strength + risk_strength) if (trader_strength + risk_strength) > 0 else 0.5
    else:
        # Challenge the bearish case
        arguments.append(f"Challenging risk manager (confidence {risk_mgr.confidence:.2f}, posterior {risk_mgr.bayesian_posterior:.2f})")

        # Check if risk is overweighted
        non_critical = [e for e in risk_mgr.evidence if e.severity != "critical"]
        critical = [e for e in risk_mgr.evidence if e.severity == "critical"]
        if len(non_critical) > len(critical) * 2:
            arguments.append(f"Only {len(critical)} critical risks vs {len(non_critical)} non-critical — risk may be overstated")

        # Check if similar risk was recently profitable
        if ctx.kelly_stats.get("active") and ctx.kelly_stats.get("win_rate", 0) > 0.55:
            arguments.append(f"Historical win rate {ctx.kelly_stats['win_rate']:.1%} suggests risk aversion costs alpha")

        position = "approve"
        confidence = trader_strength / (trader_strength + risk_strength) if (trader_strength + risk_strength) > 0 else 0.5

    return AgentArgument(
        role="devils_advocate",
        position=position,
        confidence=round(min(0.85, confidence), 4),
        arguments=arguments[:5],
        evidence=da_evidence,
        suggested_action="challenge_" + target,
        bayesian_prior=0.5,
        bayesian_posterior=round(confidence, 4),
    )


def _portfolio_manager_arbitrate(
    trader: AgentArgument,
    risk_mgr: AgentArgument,
    devils_advocate: AgentArgument,
    ctx: DebateContext,
    round_num: int,
) -> dict[str, Any]:
    """Portfolio Manager arbitrates using weighted Bayesian synthesis."""
    profile = STRATEGY_PROFILES.get(ctx.strategy, STRATEGY_PROFILES["spot"])

    # Weight agents: risk manager gets 50%, trader 30%, devil's advocate 20%
    t_weight = 0.30 * trader.confidence * trader.bayesian_posterior
    r_weight = 0.50 * risk_mgr.confidence * risk_mgr.bayesian_posterior
    d_weight = 0.20 * devils_advocate.confidence * devils_advocate.bayesian_posterior

    total_w = t_weight + r_weight + d_weight
    if total_w == 0:
        total_w = 1.0

    # Compute approval score: trader + DA(if approve) vs risk + DA(if reject)
    approval_score = t_weight / total_w
    if devils_advocate.position == "approve":
        approval_score += d_weight / total_w
    rejection_score = r_weight / total_w
    if devils_advocate.position in ("reject", "reduce"):
        rejection_score += d_weight / total_w

    # Normalize
    total_score = approval_score + rejection_score
    if total_score > 0:
        approval_pct = approval_score / total_score
    else:
        approval_pct = 0.5

    # Decision thresholds (strategy-dependent)
    approve_threshold = 1.0 - profile["risk_tolerance"]
    approve = approval_pct >= approve_threshold and ctx.risk_score < APPROVAL_THRESHOLD

    # Force reject on hard veto conditions
    if ctx.veto_reasons:
        approve = False

    reasoning: list[str] = []
    if approve:
        reasoning.append(f"Approval score {approval_pct:.2f} >= threshold {approve_threshold:.2f}")
        if ctx.risk_score > 50:
            reasoning.append("Recommending reduced position size due to elevated risk")
    else:
        reasoning.append(f"Approval score {approval_pct:.2f} < threshold {approve_threshold:.2f}")
        if risk_mgr.position == "reject":
            reasoning.append("Risk manager recommends hard rejection")
        if ctx.veto_reasons:
            reasoning.append(f"Hard veto conditions present ({len(ctx.veto_reasons)})")

    # Determine recommended size
    if approve:
        # Blend trader and risk manager suggestions
        t_size = trader.suggested_size_pct or profile["size_cap_pct"]
        r_size = risk_mgr.suggested_size_pct or 0.0
        recommended_size = t_size * approval_pct + r_size * (1 - approval_pct)
        recommended_size = min(profile["size_cap_pct"], recommended_size)
    else:
        recommended_size = 0.0

    pm_confidence = abs(approval_pct - 0.5) * 2  # 0 at 50/50, 1 at unanimous

    return {
        "role": "portfolio_manager",
        "decision": "approve" if approve else "reject",
        "confidence": round(pm_confidence, 4),
        "reasoning": reasoning,
        "approval_score": round(approval_pct, 4),
        "rejection_score": round(1 - approval_pct, 4),
        "recommended_size_pct": round(recommended_size, 4),
        "trader_weight": round(t_weight, 4),
        "risk_weight": round(r_weight, 4),
        "da_weight": round(d_weight, 4),
        "strategy_profile": ctx.strategy,
    }


def enrich_context(ctx: DebateContext) -> DebateContext:
    """Enrich debate context with live data from risk models (best-effort)."""
    # Kelly stats
    try:
        from cortex.guardian import get_kelly_stats
        ctx.kelly_stats = get_kelly_stats()
    except Exception:
        pass

    # Portfolio drawdown + correlation
    try:
        from cortex.portfolio_risk import get_drawdown, get_correlated_exposure, get_portfolio_value
        ctx.drawdown = get_drawdown()
        ctx.portfolio_value = get_portfolio_value()
        # token not available here, so skip correlation
    except Exception:
        pass

    # Circuit breaker states
    try:
        from cortex.circuit_breaker import get_all_states
        ctx.circuit_breaker_states = get_all_states()
    except Exception:
        pass

    return ctx


def run_debate(
    risk_score: float,
    component_scores: list[dict],
    veto_reasons: list[str],
    direction: str,
    trade_size_usd: float,
    original_approved: bool,
    strategy: str = "spot",
    alams_data: dict[str, Any] | None = None,
    kelly_stats: dict[str, Any] | None = None,
    drawdown: dict[str, Any] | None = None,
    correlation: dict[str, Any] | None = None,
    circuit_breaker_states: list[dict] | None = None,
    enrich: bool = True,
) -> dict[str, Any]:
    """Run multi-round adversarial debate with evidence marshaling.

    Returns structured debate result with per-round transcripts, evidence chains,
    Bayesian posteriors, and final arbitrated decision.
    """
    start = time.time()

    ctx = DebateContext(
        risk_score=risk_score,
        component_scores=component_scores,
        veto_reasons=veto_reasons,
        direction=direction,
        trade_size_usd=trade_size_usd,
        original_approved=original_approved,
        strategy=strategy,
        alams_data=alams_data or {},
        kelly_stats=kelly_stats or {},
        drawdown=drawdown or {},
        correlation=correlation or {},
        circuit_breaker_states=circuit_breaker_states or [],
    )

    if enrich:
        ctx = enrich_context(ctx)

    # Collect evidence once (shared across rounds)
    evidence = _collect_evidence(ctx)
    total_evidence = len(evidence["bullish"]) + len(evidence["bearish"])

    rounds: list[dict[str, Any]] = []
    prev_trader: AgentArgument | None = None
    prev_risk: AgentArgument | None = None

    for r in range(DEBATE_MAX_ROUNDS):
        trader = _trader_argue(ctx, evidence, r, prev_risk)
        risk_mgr = _risk_manager_argue(ctx, evidence, r, prev_trader)
        da = _devils_advocate_argue(ctx, trader, risk_mgr, r)
        pm = _portfolio_manager_arbitrate(trader, risk_mgr, da, ctx, r)

        rounds.append({
            "round": r + 1,
            "trader": trader.to_dict(),
            "risk_manager": risk_mgr.to_dict(),
            "devils_advocate": da.to_dict(),
            "arbitrator": pm,
        })

        prev_trader = trader
        prev_risk = risk_mgr

        # Early termination: PM has high confidence
        if pm["confidence"] > 0.7:
            break

    final = rounds[-1]["arbitrator"]
    elapsed_ms = (time.time() - start) * 1000

    result = {
        "final_decision": final["decision"],
        "final_confidence": final["confidence"],
        "recommended_size_pct": final["recommended_size_pct"],
        "approval_score": final["approval_score"],
        "rounds": rounds,
        "num_rounds": len(rounds),
        "elapsed_ms": round(elapsed_ms, 1),
        "original_approved": original_approved,
        "decision_changed": (final["decision"] == "approve") != original_approved,
        "strategy": strategy,
        "evidence_summary": {
            "total": total_evidence,
            "bullish": len(evidence["bullish"]),
            "bearish": len(evidence["bearish"]),
            "bullish_items": [e.to_dict() for e in evidence["bullish"][:5]],
            "bearish_items": [e.to_dict() for e in evidence["bearish"][:5]],
        },
    }

    # Persist transcript to tiered storage (non-blocking, best-effort)
    try:
        from cortex.debate_store import get_debate_store
        store = get_debate_store()
        store.store(
            result,
            token=ctx.direction,  # token not in context, use direction as fallback
            trade_size_usd=trade_size_usd,
            direction=direction,
        )
    except Exception:
        logger.debug("Debate transcript persistence failed (non-critical)", exc_info=True)

    return result


# ── Debate Outcome Persistence ───────────────────────────────────────

_debate_store: Any = None


def _get_debate_outcome_store():
    """Lazy-init PersistentStore for debate outcome priors."""
    global _debate_store
    if _debate_store is None:
        from cortex.persistence import PersistentStore
        _debate_store = PersistentStore("debate_outcomes")
    return _debate_store


def persist_debate_state() -> None:
    """Snapshot debate outcomes to Redis."""
    store = _get_debate_outcome_store()
    store["outcomes"] = _debate_outcomes[-1000:]
    logger.info("Persisted %d debate outcome entries to Redis", len(_debate_outcomes))


def restore_debate_state() -> None:
    """Restore debate outcomes from Redis."""
    store = _get_debate_outcome_store()
    data = store.get("outcomes")
    if data:
        _debate_outcomes.clear()
        _debate_outcomes.extend(data)
        logger.info("Restored %d debate outcome entries from Redis", len(_debate_outcomes))
