"""Tests for cortex/debate.py — Adversarial Debate System (evidence-based, 4 agents)."""

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
