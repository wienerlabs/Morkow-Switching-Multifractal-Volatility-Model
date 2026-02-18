"""Tests for api/routes/dx.py — DX-Research API endpoints."""
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test"}


# ── DX Status ────────────────────────────────────────────────────────────


class TestDXStatus:
    def test_status_returns_flags(self):
        resp = client.get("/api/v1/dx/status", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "stigmergy_enabled" in data
        assert "ising_cascade_enabled" in data
        assert "vault_delta_enabled" in data
        assert "human_override_enabled" in data
        assert "persona_diversity_enabled" in data


# ── Stigmergy Endpoints ──────────────────────────────────────────────────


class TestStigmeryEndpoints:
    def test_board_snapshot_empty(self):
        resp = client.get("/api/v1/dx/stigmergy", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["total_tokens"] == 0

    def test_board_disabled(self):
        with patch("cortex.config.STIGMERGY_ENABLED", False):
            resp = client.get("/api/v1/dx/stigmergy", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_token_consensus(self):
        resp = client.get("/api/v1/dx/stigmergy/SOL", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["token"] == "SOL"
        assert "direction" in data
        assert "conviction" in data

    def test_token_disabled(self):
        with patch("cortex.config.STIGMERGY_ENABLED", False):
            resp = client.get("/api/v1/dx/stigmergy/SOL", headers=HEADERS)
        assert resp.status_code == 503

    def test_board_with_signals(self):
        from cortex.stigmergy import deposit_signal, get_board
        import cortex.stigmergy as stig_module
        old_board = stig_module._board
        stig_module._board = None
        try:
            deposit_signal("test_src", "BTC", "bullish", 0.8)
            resp = client.get("/api/v1/dx/stigmergy", headers=HEADERS)
            assert resp.status_code == 200
            data = resp.json()
            assert data["total_tokens"] >= 1
            assert "BTC" in data["tokens"]
            assert data["tokens"]["BTC"]["direction"] == "bullish"
        finally:
            stig_module._board = old_board


# ── Ising Cascade Endpoint ───────────────────────────────────────────────


class TestCascadeEndpoint:
    def test_cascade_for_token(self):
        resp = client.get("/api/v1/dx/cascade/SOL", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["token"] == "SOL"
        assert "cascade_risk" in data
        assert "cascade_score" in data
        assert "magnetization" in data

    def test_cascade_disabled(self):
        with patch("cortex.config.ISING_CASCADE_ENABLED", False):
            resp = client.get("/api/v1/dx/cascade/SOL", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False


# ── Vault Delta Endpoint ─────────────────────────────────────────────────


class TestVaultDeltaEndpoint:
    def test_vault_empty(self):
        resp = client.get("/api/v1/dx/vault/vault-1", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["component"] == "vault_delta"

    def test_vault_disabled(self):
        with patch("cortex.config.VAULT_DELTA_ENABLED", False):
            resp = client.get("/api/v1/dx/vault/vault-1", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_vault_with_data(self):
        from cortex.vault_delta import ingest_snapshot
        import cortex.vault_delta as vd_module
        old_tracker = vd_module._tracker
        vd_module._tracker = None
        try:
            now = time.time()
            ingest_snapshot("test-vault", 1_000_000, 1.0, ts=now - 86400)
            ingest_snapshot("test-vault", 950_000, 0.98, ts=now)
            resp = client.get("/api/v1/dx/vault/test-vault", headers=HEADERS)
            assert resp.status_code == 200
            data = resp.json()
            assert data["score"] > 0
        finally:
            vd_module._tracker = old_tracker


# ── Override Endpoints ───────────────────────────────────────────────────


class TestOverrideEndpoints:
    @pytest.fixture(autouse=True)
    def reset_registry(self):
        import cortex.human_override as ovr_module
        old = ovr_module._registry
        ovr_module._registry = None
        yield
        ovr_module._registry = old

    def test_list_empty(self):
        resp = client.get("/api/v1/dx/overrides", headers=HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True
        assert data["overrides"] == []

    def test_list_disabled(self):
        with patch("cortex.config.HUMAN_OVERRIDE_ENABLED", False):
            resp = client.get("/api/v1/dx/overrides", headers=HEADERS)
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    def test_create_override(self):
        resp = client.post(
            "/api/v1/dx/overrides",
            json={
                "action": "force_reject",
                "token": "SOL",
                "reason": "test override",
                "created_by": "test",
            },
            headers=HEADERS,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "force_reject"
        assert data["token"] == "SOL"
        assert "id" in data

    def test_create_disabled(self):
        with patch("cortex.config.HUMAN_OVERRIDE_ENABLED", False):
            resp = client.post(
                "/api/v1/dx/overrides",
                json={"action": "force_reject", "token": "SOL", "reason": "test"},
                headers=HEADERS,
            )
        assert resp.status_code == 503

    def test_create_and_revoke(self):
        create_resp = client.post(
            "/api/v1/dx/overrides",
            json={"action": "cooldown", "token": "ETH", "reason": "test"},
            headers=HEADERS,
        )
        assert create_resp.status_code == 200
        override_id = create_resp.json()["id"]

        revoke_resp = client.delete(
            f"/api/v1/dx/overrides/{override_id}?revoked_by=test",
            headers=HEADERS,
        )
        assert revoke_resp.status_code == 200
        assert revoke_resp.json()["revoked"] is True

    def test_revoke_nonexistent(self):
        resp = client.delete(
            "/api/v1/dx/overrides/fake-id?revoked_by=test",
            headers=HEADERS,
        )
        assert resp.status_code == 404

    def test_create_and_list(self):
        client.post(
            "/api/v1/dx/overrides",
            json={"action": "force_approve", "token": "BTC", "reason": "bullish"},
            headers=HEADERS,
        )
        resp = client.get("/api/v1/dx/overrides", headers=HEADERS)
        data = resp.json()
        assert len(data["overrides"]) == 1
        assert data["overrides"][0]["token"] == "BTC"
        assert len(data["audit_log"]) >= 1


# ── Narrator DX Context Collectors ───────────────────────────────────────


class TestNarratorDXCollectors:
    def test_stigmergy_collector_no_data(self):
        from cortex.narrator import _collect_stigmergy_context
        result = _collect_stigmergy_context("SOL")
        assert "SOL" in result
        assert "direction=" in result

    def test_stigmergy_collector_board(self):
        from cortex.narrator import _collect_stigmergy_context
        result = _collect_stigmergy_context()
        assert "Stigmergy" in result

    def test_cascade_collector(self):
        from cortex.narrator import _collect_cascade_context
        result = _collect_cascade_context("SOL")
        assert "risk=" in result
        assert "score=" in result

    def test_cascade_collector_no_token(self):
        from cortex.narrator import _collect_cascade_context
        result = _collect_cascade_context()
        assert "no token" in result.lower()

    def test_memory_collector_empty(self):
        from cortex.narrator import _collect_memory_context
        result = _collect_memory_context("test-agent")
        assert "no decisions" in result.lower()

    def test_vault_collector_empty(self):
        from cortex.narrator import _collect_vault_context
        result = _collect_vault_context()
        assert "Vault" in result

    def test_override_collector_empty(self):
        from cortex.narrator import _collect_override_context
        result = _collect_override_context()
        assert "no active" in result.lower()

    def test_stigmergy_disabled(self):
        from cortex.narrator import _collect_stigmergy_context
        with patch("cortex.config.STIGMERGY_ENABLED", False):
            result = _collect_stigmergy_context()
        assert "disabled" in result.lower()

    def test_cascade_disabled(self):
        from cortex.narrator import _collect_cascade_context
        with patch("cortex.config.ISING_CASCADE_ENABLED", False):
            result = _collect_cascade_context("SOL")
        assert "disabled" in result.lower()

    def test_vault_disabled(self):
        from cortex.narrator import _collect_vault_context
        with patch("cortex.config.VAULT_DELTA_ENABLED", False):
            result = _collect_vault_context()
        assert "disabled" in result.lower()

    def test_override_disabled(self):
        from cortex.narrator import _collect_override_context
        with patch("cortex.config.HUMAN_OVERRIDE_ENABLED", False):
            result = _collect_override_context()
        assert "disabled" in result.lower()
