"""Tests for api/routes/narrator.py — Narrator API endpoint tests.

Tests the FastAPI route layer using TestClient, with mocked narrator core.
"""

import pytest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

HEADERS = {"X-API-Key": "test"}


# ── Disabled State (503) ─────────────────────────────────────────────


class TestNarratorDisabledEndpoints:
    """When NARRATOR_ENABLED=false, all narrator endpoints return 503."""

    def test_explain_disabled(self):
        with patch("api.routes.narrator.NARRATOR_ENABLED", False):
            resp = client.post(
                "/api/v1/narrator/explain",
                json={"token": "SOL", "direction": "long", "trade_size_usd": 10000},
                headers=HEADERS,
            )
        assert resp.status_code == 503
        assert "disabled" in resp.json()["detail"].lower()

    def test_news_disabled(self):
        with patch("api.routes.narrator.NARRATOR_ENABLED", False):
            resp = client.post(
                "/api/v1/narrator/news",
                json={},
                headers=HEADERS,
            )
        assert resp.status_code == 503

    def test_briefing_disabled(self):
        with patch("api.routes.narrator.NARRATOR_ENABLED", False):
            resp = client.get("/api/v1/narrator/briefing", headers=HEADERS)
        assert resp.status_code == 503

    def test_ask_disabled(self):
        with patch("api.routes.narrator.NARRATOR_ENABLED", False):
            resp = client.post(
                "/api/v1/narrator/ask",
                json={"question": "What is the current regime?"},
                headers=HEADERS,
            )
        assert resp.status_code == 503


# ── Status endpoint (always works) ───────────────────────────────────


class TestNarratorStatusEndpoint:
    def test_status_returns_200(self):
        resp = client.get("/api/v1/narrator/status", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "enabled" in data
        assert "model" in data
        assert "api_key_set" in data
        assert "call_count" in data


# ── Enabled State (mocked LLM) ──────────────────────────────────────


class TestNarratorEnabledEndpoints:
    """When NARRATOR_ENABLED=true with mocked LLM, endpoints return 200."""

    def test_explain_with_assessment(self):
        mock_result = {
            "enabled": True,
            "narrative": "APPROVE — risk is manageable.",
            "model": "gpt-4o-mini",
            "latency_ms": 150.0,
            "token": "SOL",
            "direction": "long",
        }

        with patch("api.routes.narrator.NARRATOR_ENABLED", True), \
             patch("cortex.narrator.explain_decision", new_callable=AsyncMock, return_value=mock_result):
            resp = client.post(
                "/api/v1/narrator/explain",
                json={
                    "token": "SOL",
                    "direction": "long",
                    "trade_size_usd": 10000,
                    "assessment": {
                        "approved": True,
                        "risk_score": 42.5,
                        "veto_reasons": [],
                        "recommended_size": 5000,
                        "regime_state": 2,
                        "confidence": 0.85,
                        "component_scores": [],
                    },
                },
                headers=HEADERS,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["narrative"] == "APPROVE — risk is manageable."
        assert data["model"] == "gpt-4o-mini"

    def test_news_interpretation(self):
        mock_result = {
            "enabled": True,
            "interpretation": "ETF news is the primary catalyst.",
            "model": "gpt-4o-mini",
            "latency_ms": 200.0,
            "n_items": 2,
        }

        with patch("api.routes.narrator.NARRATOR_ENABLED", True), \
             patch("cortex.narrator.interpret_news", new_callable=AsyncMock, return_value=mock_result):
            resp = client.post(
                "/api/v1/narrator/news",
                json={
                    "news_items": [{"title": "SOL ETF approved"}],
                    "news_signal": {"direction": "LONG", "strength": 0.8},
                },
                headers=HEADERS,
            )

        assert resp.status_code == 200
        assert "ETF" in resp.json()["interpretation"]

    def test_market_briefing(self):
        mock_result = {
            "enabled": True,
            "briefing": "Market is in low-vol regime with bullish sentiment.",
            "model": "gpt-4o-mini",
            "latency_ms": 300.0,
        }

        with patch("api.routes.narrator.NARRATOR_ENABLED", True), \
             patch("cortex.narrator.market_briefing", new_callable=AsyncMock, return_value=mock_result):
            resp = client.get("/api/v1/narrator/briefing", headers=HEADERS)

        assert resp.status_code == 200
        assert "low-vol" in resp.json()["briefing"]

    def test_ask_question(self):
        mock_result = {
            "enabled": True,
            "answer": "Current regime is state 2 (Low Vol).",
            "model": "gpt-4o-mini",
            "latency_ms": 250.0,
            "question": "What regime are we in?",
        }

        with patch("api.routes.narrator.NARRATOR_ENABLED", True), \
             patch("cortex.narrator.answer_question", new_callable=AsyncMock, return_value=mock_result):
            resp = client.post(
                "/api/v1/narrator/ask",
                json={"question": "What regime are we in?"},
                headers=HEADERS,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "regime" in data["answer"].lower()
        assert data["question"] == "What regime are we in?"

    def test_ask_requires_question(self):
        """Ensure question field is required and non-empty."""
        with patch("api.routes.narrator.NARRATOR_ENABLED", True):
            resp = client.post(
                "/api/v1/narrator/ask",
                json={},
                headers=HEADERS,
            )
        assert resp.status_code == 422  # Pydantic validation error
