"""Tests for cortex/debate.py — Adversarial Debate System."""

import pytest

from cortex.debate import (
    _portfolio_manager_arbitrate,
    _risk_manager_argue,
    _trader_argue,
    run_debate,
)


@pytest.fixture
def low_risk_components():
    return [
        {"component": "volatility", "score": 30},
        {"component": "liquidity", "score": 25},
        {"component": "sentiment", "score": 40},
    ]


@pytest.fixture
def high_risk_components():
    return [
        {"component": "volatility", "score": 85},
        {"component": "liquidity", "score": 70},
        {"component": "sentiment", "score": 90},
    ]


class TestTraderArgue:
    def test_returns_expected_keys(self, low_risk_components):
        r = _trader_argue(30.0, low_risk_components, "long", 10_000, 0)
        assert r["role"] == "trader"
        assert r["position"] == "approve"
        assert "confidence" in r
        assert "arguments" in r

    def test_confidence_decreases_with_risk(self, high_risk_components):
        r_low = _trader_argue(20.0, [], "long", 10_000, 0)
        r_high = _trader_argue(80.0, high_risk_components, "long", 10_000, 0)
        assert r_low["confidence"] > r_high["confidence"]

    def test_confidence_decreases_with_rounds(self, low_risk_components):
        r0 = _trader_argue(30.0, low_risk_components, "long", 10_000, 0)
        r2 = _trader_argue(30.0, low_risk_components, "long", 10_000, 2)
        assert r0["confidence"] >= r2["confidence"]

    def test_favorable_signals_mentioned(self, low_risk_components):
        r = _trader_argue(30.0, low_risk_components, "long", 10_000, 0)
        assert any("Favorable" in a for a in r["arguments"])


class TestRiskManagerArgue:
    def test_returns_expected_keys(self, high_risk_components):
        r = _risk_manager_argue(80.0, high_risk_components, [], 0)
        assert r["role"] == "risk_manager"
        assert "position" in r
        assert "confidence" in r

    def test_rejects_on_high_risk(self, high_risk_components):
        r = _risk_manager_argue(80.0, high_risk_components, ["drawdown_limit"], 0)
        assert r["position"] == "reject"

    def test_reduces_on_moderate_risk(self, low_risk_components):
        r = _risk_manager_argue(55.0, low_risk_components, [], 0)
        assert r["position"] == "reduce"

    def test_veto_reasons_included(self, high_risk_components):
        r = _risk_manager_argue(80.0, high_risk_components, ["max_drawdown"], 0)
        assert any("veto" in a.lower() for a in r["arguments"])


class TestPortfolioManagerArbitrate:
    def test_approves_when_trader_strong(self):
        trader = {"confidence": 0.9, "position": "approve"}
        risk = {"confidence": 0.3, "position": "reduce"}
        r = _portfolio_manager_arbitrate(trader, risk, 40.0, True)
        assert r["decision"] == "approve"

    def test_rejects_when_risk_dominant(self):
        trader = {"confidence": 0.2, "position": "approve"}
        risk = {"confidence": 0.9, "position": "reject"}
        r = _portfolio_manager_arbitrate(trader, risk, 80.0, True)
        assert r["decision"] == "reject"

    def test_returns_expected_keys(self):
        trader = {"confidence": 0.5, "position": "approve"}
        risk = {"confidence": 0.5, "position": "reduce"}
        r = _portfolio_manager_arbitrate(trader, risk, 50.0, True)
        assert r["role"] == "portfolio_manager"
        assert "decision" in r
        assert "confidence" in r
        assert "reasoning" in r


class TestRunDebate:
    def test_low_risk_approves(self, low_risk_components):
        r = run_debate(25.0, low_risk_components, [], "long", 10_000, True)
        assert r["final_decision"] == "approve"
        assert r["num_rounds"] >= 1
        assert "elapsed_ms" in r

    def test_high_risk_rejects(self, high_risk_components):
        r = run_debate(85.0, high_risk_components, ["drawdown"], "long", 10_000, True)
        assert r["final_decision"] == "reject"

    def test_early_termination_on_high_confidence(self, low_risk_components):
        r = run_debate(10.0, low_risk_components, [], "long", 10_000, True)
        # Very low risk → high PM confidence → should terminate early
        assert r["num_rounds"] <= 3

    def test_decision_changed_flag(self, high_risk_components):
        r = run_debate(85.0, high_risk_components, ["veto"], "long", 10_000, True)
        # original_approved=True but debate rejects → decision_changed=True
        assert r["decision_changed"] is True

    def test_rounds_structure(self, low_risk_components):
        r = run_debate(50.0, low_risk_components, [], "long", 10_000, False)
        for rnd in r["rounds"]:
            assert "round" in rnd
            assert "trader" in rnd
            assert "risk_manager" in rnd
            assert "arbitrator" in rnd

