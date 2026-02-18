"""Tests for cortex/narrator.py — LLM-powered narrative engine.

Mocks OpenAI API calls to test context assembly, error handling,
and the disabled/enabled feature flag behavior without real LLM calls.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import cortex.narrator as narrator_mod
from cortex.narrator import (
    explain_decision,
    interpret_news,
    market_briefing,
    answer_question,
    get_narrator_status,
    _collect_guardian_context,
    _collect_debate_context,
    _collect_news_context,
)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def mock_assessment():
    return {
        "approved": True,
        "risk_score": 42.5,
        "veto_reasons": [],
        "recommended_size": 5000.0,
        "regime_state": 2,
        "confidence": 0.85,
        "effective_threshold": 75.0,
        "component_scores": [
            {"component": "evt", "score": 35.0, "details": {}},
            {"component": "svj", "score": 28.0, "details": {}},
            {"component": "hawkes", "score": 15.0, "details": {}},
            {"component": "regime", "score": 40.0, "details": {}},
            {"component": "news", "score": 55.0, "details": {}},
            {"component": "alams", "score": 50.0, "details": {}},
        ],
        "circuit_breaker": None,
        "portfolio_limits": None,
        "debate": {
            "final_decision": "approve",
            "final_confidence": 0.72,
            "num_rounds": 2,
            "decision_changed": False,
            "recommended_size_pct": 0.15,
            "evidence_summary": {
                "bullish": 4,
                "bearish": 3,
                "bullish_items": [{"claim": "Low EVT tail risk"}],
                "bearish_items": [{"claim": "Elevated news risk"}],
            },
            "rounds": [{
                "round": 1,
                "arbitrator": {
                    "approval_score": 0.65,
                    "reasoning": ["Approval score 0.65 >= threshold 0.60"],
                },
            }],
        },
    }


@pytest.fixture
def mock_news_items():
    return [
        {
            "title": "SOL surges 15% on ETF news",
            "source": "CoinDesk",
            "sentiment": {"label": "Bullish", "score": 0.8},
            "impact": 8.5,
        },
        {
            "title": "SEC warns about DeFi protocols",
            "source": "TheBlock",
            "sentiment": {"label": "Bearish", "score": -0.6},
            "impact": 7.2,
        },
    ]


@pytest.fixture
def mock_news_signal():
    return {
        "sentiment_ewma": 0.25,
        "direction": "LONG",
        "strength": 0.45,
        "confidence": 0.68,
        "entropy": 1.2,
        "n_items": 15,
        "n_sources": 3,
        "bull_pct": 0.5,
        "bear_pct": 0.3,
        "neutral_pct": 0.2,
    }


def _mock_llm_response(content: str = "Test LLM response"):
    """Create a mock OpenAI ChatCompletion response."""
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


# ── Context Collector Tests ────────────────────────────────────────────


class TestContextCollectors:
    def test_guardian_context_basic(self, mock_assessment):
        ctx = _collect_guardian_context(mock_assessment)
        assert "42.5/100" in ctx
        assert "Approved: True" in ctx
        assert "Regime State: 2" in ctx
        assert "evt: 35.0/100" in ctx
        assert "hawkes: 15.0/100" in ctx

    def test_guardian_context_with_veto(self, mock_assessment):
        mock_assessment["veto_reasons"] = ["evt_extreme_tail", "hawkes_flash_crash_risk"]
        ctx = _collect_guardian_context(mock_assessment)
        assert "evt_extreme_tail" in ctx
        assert "hawkes_flash_crash_risk" in ctx

    def test_guardian_context_with_portfolio_block(self, mock_assessment):
        mock_assessment["portfolio_limits"] = {"blocked": True, "blockers": ["daily_drawdown"]}
        ctx = _collect_guardian_context(mock_assessment)
        assert "BLOCKED" in ctx
        assert "daily_drawdown" in ctx

    def test_debate_context_basic(self, mock_assessment):
        ctx = _collect_debate_context(mock_assessment["debate"])
        assert "approve" in ctx
        assert "0.72" in ctx or "72" in ctx
        assert "Rounds: 2" in ctx

    def test_debate_context_empty(self):
        ctx = _collect_debate_context({})
        assert "No debate data" in ctx

    def test_debate_context_none(self):
        ctx = _collect_debate_context(None)
        assert "No debate data" in ctx

    def test_news_context_basic(self, mock_news_signal, mock_news_items):
        ctx = _collect_news_context(mock_news_signal, mock_news_items)
        assert "LONG" in ctx
        assert "0.25" in ctx
        assert "SOL surges" in ctx
        assert "SEC warns" in ctx

    def test_news_context_none(self):
        ctx = _collect_news_context(None)
        assert "No news data" in ctx

    def test_news_context_without_items(self, mock_news_signal):
        ctx = _collect_news_context(mock_news_signal)
        assert "LONG" in ctx
        assert "Top Headlines" not in ctx


# ── Disabled State Tests ───────────────────────────────────────────────


class TestNarratorDisabled:
    """When NARRATOR_ENABLED=false, all functions return {'enabled': False} immediately."""

    @pytest.mark.asyncio
    async def test_explain_disabled(self, mock_assessment):
        with patch.object(narrator_mod, "NARRATOR_ENABLED", False):
            result = await explain_decision(mock_assessment, "SOL", "long", 10000)
        assert result["enabled"] is False
        assert result["narrative"] is None

    @pytest.mark.asyncio
    async def test_interpret_news_disabled(self, mock_news_items, mock_news_signal):
        with patch.object(narrator_mod, "NARRATOR_ENABLED", False):
            result = await interpret_news(mock_news_items, mock_news_signal)
        assert result["enabled"] is False
        assert result["interpretation"] is None

    @pytest.mark.asyncio
    async def test_briefing_disabled(self):
        with patch.object(narrator_mod, "NARRATOR_ENABLED", False):
            result = await market_briefing()
        assert result["enabled"] is False
        assert result["briefing"] is None

    @pytest.mark.asyncio
    async def test_answer_disabled(self):
        with patch.object(narrator_mod, "NARRATOR_ENABLED", False):
            result = await answer_question("What is the current risk?")
        assert result["enabled"] is False
        assert result["answer"] is None


# ── LLM Call Tests (mocked) ────────────────────────────────────────────


class TestNarratorLLMCalls:
    """With NARRATOR_ENABLED=true, functions call the LLM and return results."""

    @pytest.mark.asyncio
    async def test_explain_success(self, mock_assessment):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("APPROVE with moderate confidence. Risk score 42.5 is well below threshold.")
        )

        with patch.object(narrator_mod, "NARRATOR_ENABLED", True), \
             patch.object(narrator_mod, "_client", mock_client):
            result = await explain_decision(mock_assessment, "SOL", "long", 10000)

        assert result["enabled"] is True
        assert "APPROVE" in result["narrative"]
        assert result["model"] is not None
        assert result["latency_ms"] >= 0
        assert result["token"] == "SOL"

    @pytest.mark.asyncio
    async def test_interpret_news_success(self, mock_news_items, mock_news_signal):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("The SOL ETF news is the primary market mover.")
        )

        with patch.object(narrator_mod, "NARRATOR_ENABLED", True), \
             patch.object(narrator_mod, "_client", mock_client):
            result = await interpret_news(mock_news_items, mock_news_signal)

        assert result["enabled"] is True
        assert "ETF" in result["interpretation"]
        assert result["n_items"] == 2

    @pytest.mark.asyncio
    async def test_briefing_success(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("Market is in a low-volatility regime.")
        )

        with patch.object(narrator_mod, "NARRATOR_ENABLED", True), \
             patch.object(narrator_mod, "_client", mock_client):
            result = await market_briefing()

        assert result["enabled"] is True
        assert "low-volatility" in result["briefing"]

    @pytest.mark.asyncio
    async def test_answer_success(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_mock_llm_response("The current regime state is 2 (Low Vol).")
        )

        with patch.object(narrator_mod, "NARRATOR_ENABLED", True), \
             patch.object(narrator_mod, "_client", mock_client):
            result = await answer_question("What is the current regime?")

        assert result["enabled"] is True
        assert "regime" in result["answer"].lower()
        assert result["question"] == "What is the current regime?"


# ── Error Handling Tests ───────────────────────────────────────────────


class TestNarratorErrorHandling:
    @pytest.mark.asyncio
    async def test_explain_llm_error(self, mock_assessment):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=Exception("API timeout"))

        with patch.object(narrator_mod, "NARRATOR_ENABLED", True), \
             patch.object(narrator_mod, "_client", mock_client):
            result = await explain_decision(mock_assessment, "SOL", "long", 10000)

        assert result["enabled"] is True
        assert result["narrative"] is None
        assert "API timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_interpret_no_items(self):
        with patch.object(narrator_mod, "NARRATOR_ENABLED", True):
            result = await interpret_news([], None)

        assert result["enabled"] is True
        assert result["interpretation"] is None
        assert "No news items" in result["error"]

    @pytest.mark.asyncio
    async def test_briefing_llm_error(self):
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Rate limited"))

        with patch.object(narrator_mod, "NARRATOR_ENABLED", True), \
             patch.object(narrator_mod, "_client", mock_client):
            result = await market_briefing()

        assert result["enabled"] is True
        assert result["briefing"] is None
        assert "Rate limited" in result["error"]


# ── Status Tests ───────────────────────────────────────────────────────


class TestNarratorStatus:
    def test_status_returns_dict(self):
        status = get_narrator_status()
        assert "enabled" in status
        assert "model" in status
        assert "api_key_set" in status
        assert "call_count" in status
        assert "error_count" in status
        assert "avg_latency_ms" in status


# ── System Prompt Tests ────────────────────────────────────────────────


class TestSystemPrompts:
    """Verify system prompts contain critical instructions."""

    def test_explain_system_prompt(self):
        from cortex.narrator import EXPLAIN_SYSTEM
        assert "Guardian" in EXPLAIN_SYSTEM or "guardian" in EXPLAIN_SYSTEM
        assert "APPROVE" in EXPLAIN_SYSTEM or "approve" in EXPLAIN_SYSTEM.lower()
        assert "debate" in EXPLAIN_SYSTEM.lower()

    def test_news_system_prompt(self):
        from cortex.narrator import INTERPRET_NEWS_SYSTEM
        assert "crypto" in INTERPRET_NEWS_SYSTEM.lower()
        assert "lexicon" in INTERPRET_NEWS_SYSTEM.lower()
        assert "sentiment" in INTERPRET_NEWS_SYSTEM.lower()

    def test_briefing_system_prompt(self):
        from cortex.narrator import BRIEFING_SYSTEM
        assert "briefing" in BRIEFING_SYSTEM.lower()
        assert "regime" in BRIEFING_SYSTEM.lower()

    def test_question_system_prompt(self):
        from cortex.narrator import QUESTION_SYSTEM
        assert "question" in QUESTION_SYSTEM.lower() or "operator" in QUESTION_SYSTEM.lower()
