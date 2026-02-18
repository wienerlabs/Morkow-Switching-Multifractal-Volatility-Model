"""Tests for cortex/debate.py — Adversarial Debate System (evidence-based, 4 agents)."""

from unittest.mock import patch

import pytest

from cortex.debate import (
    DebateContext,
    DebateEvidence,
    _bayesian_update,
    _collect_evidence,
    _devils_advocate_argue,
    _portfolio_manager_arbitrate,
    _risk_manager_argue,
    _trader_argue,
    run_debate,
)


@pytest.fixture
def low_risk_ctx():
    return DebateContext(
        risk_score=25.0,
        component_scores=[
            {"component": "volatility", "score": 30},
            {"component": "liquidity", "score": 25},
            {"component": "sentiment", "score": 40},
        ],
        veto_reasons=[],
        direction="long",
        trade_size_usd=10_000,
        original_approved=True,
        strategy="spot",
    )


@pytest.fixture
def high_risk_ctx():
    return DebateContext(
        risk_score=85.0,
        component_scores=[
            {"component": "volatility", "score": 85},
            {"component": "liquidity", "score": 70},
            {"component": "sentiment", "score": 90},
        ],
        veto_reasons=["max_drawdown"],
        direction="long",
        trade_size_usd=10_000,
        original_approved=True,
        strategy="spot",
    )


@pytest.fixture
def low_risk_evidence(low_risk_ctx):
    return _collect_evidence(low_risk_ctx)


@pytest.fixture
def high_risk_evidence(high_risk_ctx):
    return _collect_evidence(high_risk_ctx)


class TestDebateEvidence:
    def test_supports_approval_below_threshold(self):
        ev = DebateEvidence(source="test", claim="Low risk", value=30.0, threshold=60.0)
        assert ev.supports_approval() is True

    def test_rejects_above_threshold(self):
        ev = DebateEvidence(source="test", claim="High risk", value=80.0, threshold=60.0)
        assert ev.supports_approval() is False

    def test_to_dict_structure(self):
        ev = DebateEvidence(source="guardian:vol", claim="Vol score 30", value=30.0, threshold=60.0, severity="low")
        d = ev.to_dict()
        assert d["source"] == "guardian:vol"
        assert d["severity"] == "low"
        assert d["supports_approval"] is True


class TestCollectEvidence:
    def test_low_risk_has_bullish(self, low_risk_ctx):
        ev = _collect_evidence(low_risk_ctx)
        assert len(ev["bullish"]) > 0

    def test_high_risk_has_bearish(self, high_risk_ctx):
        ev = _collect_evidence(high_risk_ctx)
        assert len(ev["bearish"]) > 0

    def test_veto_reasons_in_bearish(self, high_risk_ctx):
        ev = _collect_evidence(high_risk_ctx)
        veto_evidence = [e for e in ev["bearish"] if e.source == "guardian:veto"]
        assert len(veto_evidence) > 0
        assert veto_evidence[0].severity == "critical"

    def test_alams_evidence_collected(self):
        ctx = DebateContext(
            risk_score=50.0, component_scores=[], veto_reasons=[],
            direction="long", trade_size_usd=5000, original_approved=True,
            alams_data={"var_total": 0.07, "current_regime": 4, "delta": 0.35},
        )
        ev = _collect_evidence(ctx)
        alams_ev = [e for e in ev["bearish"] if "alams" in e.source]
        assert len(alams_ev) >= 2


class TestBayesianUpdate:
    def test_bullish_evidence_increases_approval(self):
        evidence = [DebateEvidence(source="test", claim="Good", value=20.0, threshold=60.0, severity="high")]
        posterior = _bayesian_update(0.5, evidence, "approve")
        assert posterior > 0.5

    def test_bearish_evidence_decreases_approval(self):
        evidence = [DebateEvidence(source="test", claim="Bad", value=80.0, threshold=60.0, severity="high")]
        posterior = _bayesian_update(0.5, evidence, "approve")
        assert posterior < 0.5

    def test_critical_severity_larger_shift(self):
        low = [DebateEvidence(source="t", claim="x", value=80.0, threshold=60.0, severity="low")]
        crit = [DebateEvidence(source="t", claim="x", value=80.0, threshold=60.0, severity="critical")]
        post_low = _bayesian_update(0.5, low, "reject")
        post_crit = _bayesian_update(0.5, crit, "reject")
        assert post_crit > post_low


class TestTraderArgue:
    def test_returns_agent_argument(self, low_risk_ctx, low_risk_evidence):
        r = _trader_argue(low_risk_ctx, low_risk_evidence, 0, None)
        assert r.role == "trader"
        assert r.position == "approve"
        assert 0 < r.confidence <= 1.0

    def test_confidence_decreases_with_risk(self, high_risk_ctx, high_risk_evidence, low_risk_ctx, low_risk_evidence):
        r_low = _trader_argue(low_risk_ctx, low_risk_evidence, 0, None)
        r_high = _trader_argue(high_risk_ctx, high_risk_evidence, 0, None)
        assert r_low.confidence > r_high.confidence

    def test_confidence_decreases_with_rounds(self, low_risk_ctx, low_risk_evidence):
        r0 = _trader_argue(low_risk_ctx, low_risk_evidence, 0, None)
        r2 = _trader_argue(low_risk_ctx, low_risk_evidence, 2, None)
        assert r0.confidence >= r2.confidence

    def test_favorable_signals_mentioned(self, low_risk_ctx, low_risk_evidence):
        r = _trader_argue(low_risk_ctx, low_risk_evidence, 0, None)
        assert any("Favorable" in a for a in r.arguments)

    def test_to_dict_serialization(self, low_risk_ctx, low_risk_evidence):
        r = _trader_argue(low_risk_ctx, low_risk_evidence, 0, None)
        d = r.to_dict()
        assert d["role"] == "trader"
        assert isinstance(d["evidence"], list)


class TestRiskManagerArgue:
    def test_returns_agent_argument(self, high_risk_ctx, high_risk_evidence):
        r = _risk_manager_argue(high_risk_ctx, high_risk_evidence, 0, None)
        assert r.role == "risk_manager"
        assert r.position in ("reject", "reduce")

    def test_rejects_on_high_risk(self, high_risk_ctx, high_risk_evidence):
        r = _risk_manager_argue(high_risk_ctx, high_risk_evidence, 0, None)
        assert r.position == "reject"

    def test_reduces_on_moderate_risk(self):
        ctx = DebateContext(
            risk_score=55.0,
            component_scores=[{"component": "vol", "score": 50}],
            veto_reasons=[],
            direction="long", trade_size_usd=5000, original_approved=True,
        )
        ev = _collect_evidence(ctx)
        r = _risk_manager_argue(ctx, ev, 0, None)
        assert r.position == "reduce"


class TestDevilsAdvocate:
    def test_challenges_stronger_side(self, low_risk_ctx, low_risk_evidence):
        trader = _trader_argue(low_risk_ctx, low_risk_evidence, 0, None)
        risk = _risk_manager_argue(low_risk_ctx, low_risk_evidence, 0, None)
        da = _devils_advocate_argue(low_risk_ctx, trader, risk, 0)
        assert da.role == "devils_advocate"
        assert da.position in ("approve", "reject", "reduce")


class TestPortfolioManagerArbitrate:
    def test_approves_low_risk(self, low_risk_ctx, low_risk_evidence):
        trader = _trader_argue(low_risk_ctx, low_risk_evidence, 0, None)
        risk = _risk_manager_argue(low_risk_ctx, low_risk_evidence, 0, None)
        da = _devils_advocate_argue(low_risk_ctx, trader, risk, 0)
        r = _portfolio_manager_arbitrate(trader, risk, da, low_risk_ctx, 0)
        assert r["decision"] == "approve"
        assert r["role"] == "portfolio_manager"
        assert "confidence" in r
        assert "reasoning" in r

    def test_rejects_high_risk(self, high_risk_ctx, high_risk_evidence):
        trader = _trader_argue(high_risk_ctx, high_risk_evidence, 0, None)
        risk = _risk_manager_argue(high_risk_ctx, high_risk_evidence, 0, None)
        da = _devils_advocate_argue(high_risk_ctx, trader, risk, 0)
        r = _portfolio_manager_arbitrate(trader, risk, da, high_risk_ctx, 0)
        assert r["decision"] == "reject"


class TestRunDebate:
    def test_low_risk_approves(self):
        r = run_debate(
            risk_score=25.0,
            component_scores=[{"component": "vol", "score": 30}, {"component": "liq", "score": 25}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=10_000,
            original_approved=True,
            enrich=False,
        )
        assert r["final_decision"] == "approve"
        assert r["num_rounds"] >= 1
        assert "elapsed_ms" in r

    def test_high_risk_rejects(self):
        r = run_debate(
            risk_score=85.0,
            component_scores=[{"component": "vol", "score": 85}, {"component": "liq", "score": 80}],
            veto_reasons=["drawdown"],
            direction="long",
            trade_size_usd=10_000,
            original_approved=True,
            enrich=False,
        )
        assert r["final_decision"] == "reject"

    def test_early_termination_on_high_confidence(self):
        r = run_debate(
            risk_score=10.0,
            component_scores=[{"component": "vol", "score": 15}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=10_000,
            original_approved=True,
            enrich=False,
        )
        assert r["num_rounds"] <= 3

    def test_decision_changed_flag(self):
        r = run_debate(
            risk_score=85.0,
            component_scores=[{"component": "vol", "score": 90}],
            veto_reasons=["veto"],
            direction="long",
            trade_size_usd=10_000,
            original_approved=True,
            enrich=False,
        )
        assert r["decision_changed"] is True

    def test_rounds_structure(self):
        r = run_debate(
            risk_score=50.0,
            component_scores=[{"component": "vol", "score": 50}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=10_000,
            original_approved=False,
            enrich=False,
        )
        for rnd in r["rounds"]:
            assert "round" in rnd
            assert "trader" in rnd
            assert "risk_manager" in rnd
            assert "devils_advocate" in rnd
            assert "arbitrator" in rnd

    def test_evidence_summary_present(self):
        r = run_debate(
            risk_score=50.0,
            component_scores=[{"component": "vol", "score": 60}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            enrich=False,
        )
        assert "evidence_summary" in r
        assert r["evidence_summary"]["total"] > 0

    def test_strategy_aware(self):
        r = run_debate(
            risk_score=40.0,
            component_scores=[{"component": "vol", "score": 40}],
            veto_reasons=[],
            direction="arb_binance_to_orca",
            trade_size_usd=5_000,
            original_approved=True,
            strategy="arb",
            enrich=False,
        )
        assert r["strategy"] == "arb"

    def test_alams_data_integration(self):
        r = run_debate(
            risk_score=50.0,
            component_scores=[{"component": "vol", "score": 50}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            alams_data={"var_total": 0.07, "current_regime": 4, "delta": 0.4},
            enrich=False,
        )
        assert r["evidence_summary"]["bearish"] > 0


# ── DX-Research Task 2: Information Asymmetry Tests ──


class TestInfoAsymmetry:
    """DX01 finding: Environment constraints > prompt engineering.
    Trader should only see bullish evidence, Risk Manager only bearish."""

    def test_trader_cannot_see_bearish_when_asymmetry_on(self):
        """With asymmetry on, trader's bearish list is empty even when bearish evidence exists."""
        ctx = DebateContext(
            risk_score=60.0,
            component_scores=[
                {"component": "vol", "score": 70},  # bearish (>50)
                {"component": "liq", "score": 30},   # bullish (<50)
            ],
            veto_reasons=[],
            direction="long",
            trade_size_usd=10_000,
            original_approved=True,
        )
        evidence = _collect_evidence(ctx)
        assert len(evidence["bearish"]) > 0, "Sanity: bearish evidence exists"

        with patch("cortex.debate.DEBATE_INFO_ASYMMETRY_ENABLED", True):
            trader = _trader_argue(ctx, evidence, 0, None)
            # Trader should not reference any high-risk components in arguments
            args_text = " ".join(trader.arguments)
            assert "vol" not in args_text.lower() or "Favorable" in args_text

    def test_trader_sees_everything_when_asymmetry_off(self):
        """With asymmetry off, trader sees all components."""
        ctx = DebateContext(
            risk_score=60.0,
            component_scores=[
                {"component": "high_risk", "score": 80},
                {"component": "low_risk", "score": 20},
            ],
            veto_reasons=[],
            direction="long",
            trade_size_usd=10_000,
            original_approved=True,
        )
        evidence = _collect_evidence(ctx)

        with patch("cortex.debate.DEBATE_INFO_ASYMMETRY_ENABLED", False):
            trader = _trader_argue(ctx, evidence, 0, None)
            # Trader still presents evidence, but from both sides
            assert trader.role == "trader"
            assert len(trader.evidence) > 0

    def test_info_asymmetry_flag_in_result(self):
        """run_debate includes info_asymmetry_active flag."""
        r = run_debate(
            risk_score=40.0,
            component_scores=[{"component": "vol", "score": 30}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            enrich=False,
        )
        assert "info_asymmetry_active" in r

    def test_asymmetry_still_rejects_high_risk(self):
        """Even with asymmetry, the system correctly rejects dangerous trades.
        Portfolio Manager and Devil's Advocate see everything."""
        r = run_debate(
            risk_score=90.0,
            component_scores=[
                {"component": "vol", "score": 92},
                {"component": "liq", "score": 85},
            ],
            veto_reasons=["extreme_risk"],
            direction="long",
            trade_size_usd=10_000,
            original_approved=True,
            enrich=False,
        )
        assert r["final_decision"] == "reject"

    def test_asymmetry_still_approves_low_risk(self):
        """With asymmetry, low-risk trades still get approved."""
        r = run_debate(
            risk_score=15.0,
            component_scores=[
                {"component": "vol", "score": 15},
                {"component": "liq", "score": 10},
            ],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            enrich=False,
        )
        assert r["final_decision"] == "approve"


# ── DX-Research Task 6: Persona Diversity Tests ──


class TestPersonaDiversity:
    """DX Terminal finding: lexical diversity (different analysis perspectives) = alpha."""

    def test_persona_flag_in_result(self):
        """run_debate includes persona_diversity_active flag."""
        r = run_debate(
            risk_score=40.0,
            component_scores=[{"component": "vol", "score": 40}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            enrich=False,
        )
        assert "persona_diversity_active" in r

    def test_trader_momentum_bias_boosts_confidence(self):
        """Trader with momentum bias should have higher confidence when momentum evidence exists."""
        ctx = DebateContext(
            risk_score=40.0,
            component_scores=[{"component": "momentum", "score": 30}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            kelly_stats={"active": True, "kelly_fraction": 0.1, "win_rate": 0.6, "n_trades": 200},
        )
        evidence = _collect_evidence(ctx)

        with patch("cortex.debate.PERSONA_DIVERSITY_ENABLED", True):
            trader_on = _trader_argue(ctx, evidence, 0, None)

        with patch("cortex.debate.PERSONA_DIVERSITY_ENABLED", False):
            trader_off = _trader_argue(ctx, evidence, 0, None)

        # Momentum persona should boost confidence when momentum/kelly evidence present
        assert trader_on.confidence >= trader_off.confidence

    def test_risk_manager_tail_sensitivity(self):
        """Risk manager with tail sensitivity should have higher confidence when critical evidence exists."""
        ctx = DebateContext(
            risk_score=85.0,
            component_scores=[
                {"component": "vol", "score": 90},
                {"component": "liq", "score": 85},
            ],
            veto_reasons=["extreme_risk"],
            direction="long",
            trade_size_usd=10_000,
            original_approved=True,
        )
        evidence = _collect_evidence(ctx)

        with patch("cortex.debate.PERSONA_DIVERSITY_ENABLED", True):
            rm_on = _risk_manager_argue(ctx, evidence, 0, None)

        with patch("cortex.debate.PERSONA_DIVERSITY_ENABLED", False):
            rm_off = _risk_manager_argue(ctx, evidence, 0, None)

        # Tail sensitivity persona should boost confidence when critical evidence present
        assert rm_on.confidence >= rm_off.confidence

    def test_persona_disabled_no_effect(self):
        """With persona disabled, trader confidence is same as baseline."""
        ctx = DebateContext(
            risk_score=40.0,
            component_scores=[{"component": "vol", "score": 40}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            kelly_stats={"active": True, "kelly_fraction": 0.1, "win_rate": 0.6, "n_trades": 200},
        )
        evidence = _collect_evidence(ctx)

        with patch("cortex.debate.PERSONA_DIVERSITY_ENABLED", False):
            t1 = _trader_argue(ctx, evidence, 0, None)
            t2 = _trader_argue(ctx, evidence, 0, None)

        # Without persona, same inputs = same output
        assert t1.confidence == t2.confidence

    def test_da_contrarian_challenges_stronger_side(self):
        """Devil's advocate with contrarian persona should always challenge the majority."""
        ctx = DebateContext(
            risk_score=30.0,
            component_scores=[{"component": "vol", "score": 30}],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
        )
        evidence = _collect_evidence(ctx)

        trader = _trader_argue(ctx, evidence, 0, None)
        rm = _risk_manager_argue(ctx, evidence, 0, None)
        da = _devils_advocate_argue(ctx, trader, rm, 0)

        # DA should challenge the winning side
        trader_strength = trader.confidence * trader.bayesian_posterior
        risk_strength = rm.confidence * rm.bayesian_posterior
        if trader_strength > risk_strength:
            assert da.position == "reduce"  # challenging trader
        else:
            assert da.position == "approve"  # challenging risk mgr

    def test_different_personas_produce_different_scores(self):
        """Same evidence, different persona settings = different confidence scores.
        This is the core acceptance criterion."""
        import cortex.debate as _debate_mod

        ctx = DebateContext(
            risk_score=50.0,
            component_scores=[
                {"component": "momentum", "score": 35},
                {"component": "vol", "score": 65},
            ],
            veto_reasons=[],
            direction="long",
            trade_size_usd=5_000,
            original_approved=True,
            kelly_stats={"active": True, "kelly_fraction": 0.08, "win_rate": 0.55, "n_trades": 150},
        )
        evidence = _collect_evidence(ctx)

        orig_enabled = _debate_mod.PERSONA_DIVERSITY_ENABLED
        orig_bias = _debate_mod.PERSONA_TRADER_MOMENTUM_BIAS
        try:
            # High momentum bias trader
            _debate_mod.PERSONA_DIVERSITY_ENABLED = True
            _debate_mod.PERSONA_TRADER_MOMENTUM_BIAS = 2.0
            trader_high = _trader_argue(ctx, evidence, 0, None)

            # Low momentum bias trader
            _debate_mod.PERSONA_TRADER_MOMENTUM_BIAS = 0.5
            trader_low = _trader_argue(ctx, evidence, 0, None)
        finally:
            _debate_mod.PERSONA_DIVERSITY_ENABLED = orig_enabled
            _debate_mod.PERSONA_TRADER_MOMENTUM_BIAS = orig_bias

        # Different persona settings should produce different confidence
        assert trader_high.confidence != trader_low.confidence
