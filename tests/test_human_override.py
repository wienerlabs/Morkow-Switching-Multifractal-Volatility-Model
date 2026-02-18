"""Tests for cortex/human_override.py — DX-Research Task 8: Human Override."""
import time
from unittest.mock import patch

import pytest

from cortex.human_override import (
    OverrideAction,
    OverrideEntry,
    OverrideRegistry,
    OverrideResult,
    check_override,
    create_override,
    get_registry,
    list_active_overrides,
    revoke_override,
)
import cortex.human_override as override_module


@pytest.fixture(autouse=True)
def reset_registry():
    old = override_module._registry
    override_module._registry = None
    yield
    override_module._registry = old


class TestOverrideEntry:
    def test_creation(self):
        entry = OverrideEntry(
            id="abc123", action=OverrideAction.FORCE_REJECT,
            token="SOL", reason="whale dump incoming",
            created_by="trader_joe", created_at=time.time(),
            expires_at=time.time() + 3600,
        )
        assert entry.is_active
        assert entry.ttl_remaining > 3599
        assert entry.action == OverrideAction.FORCE_REJECT

    def test_expired_entry(self):
        entry = OverrideEntry(
            id="abc123", action=OverrideAction.FORCE_REJECT,
            token="SOL", reason="test",
            created_by="op", created_at=time.time() - 7200,
            expires_at=time.time() - 3600,
        )
        assert not entry.is_active
        assert entry.ttl_remaining == 0.0

    def test_revoked_entry(self):
        entry = OverrideEntry(
            id="abc123", action=OverrideAction.FORCE_APPROVE,
            token="BTC", reason="test",
            created_by="op", created_at=time.time(),
            expires_at=time.time() + 3600,
            revoked=True,
        )
        assert not entry.is_active

    def test_to_dict(self):
        entry = OverrideEntry(
            id="abc123", action=OverrideAction.SIZE_CAP,
            token="ETH", reason="limit exposure",
            created_by="risk_team", created_at=time.time(),
            expires_at=time.time() + 1800,
            size_cap_usd=10_000.0,
        )
        d = entry.to_dict()
        assert d["id"] == "abc123"
        assert d["action"] == "size_cap"
        assert d["size_cap_usd"] == 10_000.0
        assert d["is_active"] is True


class TestOverrideRegistry:
    def test_create_override(self):
        reg = OverrideRegistry()
        entry = reg.create(
            action=OverrideAction.FORCE_REJECT,
            token="SOL",
            reason="flash crash risk",
            created_by="ops",
            ttl=1800,
        )
        assert entry.is_active
        assert entry.token == "SOL"
        assert len(entry.id) == 12

    def test_create_global_override(self):
        reg = OverrideRegistry()
        entry = reg.create(
            action=OverrideAction.COOLDOWN,
            token="*",
            reason="system maintenance",
            created_by="admin",
        )
        assert entry.token == "*"

    def test_create_size_cap_requires_usd(self):
        reg = OverrideRegistry()
        with pytest.raises(ValueError, match="size_cap_usd required"):
            reg.create(action=OverrideAction.SIZE_CAP, reason="test")

    def test_create_size_cap(self):
        reg = OverrideRegistry()
        entry = reg.create(
            action=OverrideAction.SIZE_CAP,
            token="SOL",
            reason="reduce exposure",
            size_cap_usd=5_000.0,
        )
        assert entry.size_cap_usd == 5_000.0

    def test_ttl_clamped_to_max(self):
        reg = OverrideRegistry()
        entry = reg.create(
            action=OverrideAction.FORCE_REJECT,
            reason="test",
            ttl=999_999,  # way over max
        )
        # Should be clamped to HUMAN_OVERRIDE_MAX_TTL (86400)
        assert entry.expires_at - entry.created_at <= 86400 + 1

    def test_check_no_overrides(self):
        reg = OverrideRegistry()
        result = reg.check("SOL")
        assert not result.has_override

    def test_check_matching_token(self):
        reg = OverrideRegistry()
        reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="test")
        result = reg.check("SOL")
        assert result.has_override
        assert result.action == OverrideAction.FORCE_REJECT

    def test_check_global_override(self):
        reg = OverrideRegistry()
        reg.create(action=OverrideAction.COOLDOWN, token="*", reason="maintenance")
        result = reg.check("BTC")
        assert result.has_override
        assert result.action == OverrideAction.COOLDOWN

    def test_check_token_specific_over_global(self):
        reg = OverrideRegistry()
        reg.create(action=OverrideAction.FORCE_APPROVE, token="*", reason="global allow")
        reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="SOL blocked")
        result = reg.check("SOL")
        assert result.action == OverrideAction.FORCE_REJECT

    def test_check_priority_reject_over_approve(self):
        reg = OverrideRegistry()
        reg.create(action=OverrideAction.FORCE_APPROVE, token="SOL", reason="allow")
        reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="block")
        result = reg.check("SOL")
        assert result.action == OverrideAction.FORCE_REJECT

    def test_check_case_insensitive(self):
        reg = OverrideRegistry()
        reg.create(action=OverrideAction.FORCE_REJECT, token="sol", reason="test")
        result = reg.check("SOL")
        assert result.has_override

    def test_check_increments_applied_count(self):
        reg = OverrideRegistry()
        entry = reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="test")
        assert entry.applied_count == 0
        reg.check("SOL")
        assert entry.applied_count == 1
        reg.check("SOL")
        assert entry.applied_count == 2

    def test_check_expired_ignored(self):
        reg = OverrideRegistry()
        entry = reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="test", ttl=1)
        # Manually expire
        entry.expires_at = time.time() - 1
        result = reg.check("SOL")
        assert not result.has_override

    def test_check_revoked_ignored(self):
        reg = OverrideRegistry()
        entry = reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="test")
        reg.revoke(entry.id, "ops")
        result = reg.check("SOL")
        assert not result.has_override

    def test_revoke(self):
        reg = OverrideRegistry()
        entry = reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="test")
        assert reg.revoke(entry.id, "admin")
        assert entry.revoked
        assert entry.revoked_by == "admin"
        assert entry.revoked_at is not None

    def test_revoke_nonexistent(self):
        reg = OverrideRegistry()
        assert not reg.revoke("nonexistent_id")

    def test_revoke_already_revoked(self):
        reg = OverrideRegistry()
        entry = reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="test")
        reg.revoke(entry.id)
        assert not reg.revoke(entry.id)

    def test_list_active(self):
        reg = OverrideRegistry()
        reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="a")
        reg.create(action=OverrideAction.FORCE_APPROVE, token="BTC", reason="b")
        e3 = reg.create(action=OverrideAction.COOLDOWN, token="ETH", reason="c")
        reg.revoke(e3.id)
        active = reg.list_active()
        assert len(active) == 2
        tokens = {o["token"] for o in active}
        assert tokens == {"SOL", "BTC"}

    def test_audit_log_records_events(self):
        reg = OverrideRegistry()
        entry = reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="test")
        reg.check("SOL")
        reg.revoke(entry.id)
        log = reg.get_audit_log()
        events = [e["event"] for e in log]
        assert "CREATED" in events
        assert "APPLIED" in events
        assert "REVOKED" in events

    def test_audit_log_capped(self):
        reg = OverrideRegistry()
        for i in range(1100):
            reg.create(action=OverrideAction.FORCE_REJECT, reason=f"test_{i}", ttl=3600)
        assert len(reg._audit_log) <= 1000

    def test_prune_expired(self):
        reg = OverrideRegistry()
        e1 = reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="test", ttl=1)
        e1.expires_at = time.time() - 1  # force expire
        reg.create(action=OverrideAction.FORCE_APPROVE, token="BTC", reason="test", ttl=3600)
        pruned = reg._prune_expired()
        assert pruned == 1
        active = reg.list_active()
        assert len(active) == 1
        assert active[0]["token"] == "BTC"

    def test_clear(self):
        reg = OverrideRegistry()
        reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="test")
        reg.clear()
        assert len(reg.list_active()) == 0
        assert len(reg.get_audit_log()) == 0

    def test_non_matching_token_not_affected(self):
        reg = OverrideRegistry()
        reg.create(action=OverrideAction.FORCE_REJECT, token="SOL", reason="block SOL")
        result = reg.check("BTC")
        assert not result.has_override


class TestModuleLevelAPI:
    def test_create_and_check(self):
        result = create_override(
            action="force_reject",
            token="SOL",
            reason="testing",
            created_by="test",
        )
        assert result["action"] == "force_reject"
        assert result["is_active"] is True

        ovr = check_override("SOL")
        assert ovr.has_override
        assert ovr.action == OverrideAction.FORCE_REJECT

    def test_create_disabled(self):
        with patch("cortex.human_override.HUMAN_OVERRIDE_ENABLED", False):
            result = create_override(action="force_reject", reason="test")
        assert "error" in result

    def test_check_disabled(self):
        create_override(action="force_reject", token="SOL", reason="test")
        with patch("cortex.human_override.HUMAN_OVERRIDE_ENABLED", False):
            ovr = check_override("SOL")
        assert not ovr.has_override

    def test_list_active_disabled(self):
        with patch("cortex.human_override.HUMAN_OVERRIDE_ENABLED", False):
            assert list_active_overrides() == []

    def test_revoke_via_module(self):
        result = create_override(
            action="force_approve", token="BTC", reason="test"
        )
        assert revoke_override(result["id"])
        ovr = check_override("BTC")
        assert not ovr.has_override

    def test_singleton_registry(self):
        r1 = get_registry()
        r2 = get_registry()
        assert r1 is r2

    def test_create_with_string_action(self):
        result = create_override(action="cooldown", token="ETH", reason="test")
        assert result["action"] == "cooldown"

    def test_create_size_cap_via_module(self):
        result = create_override(
            action="size_cap", token="SOL", reason="limit",
            size_cap_usd=5_000.0,
        )
        assert result["size_cap_usd"] == 5_000.0


class TestGuardianIntegration:
    """Test that human override affects Guardian decisions.

    Uses proper mocking to match existing guardian test patterns.
    """

    @staticmethod
    def _make_model_data():
        import numpy as np
        import pandas as pd
        probs = np.array([0.7, 0.15, 0.1, 0.03, 0.02])
        fp = pd.DataFrame([probs])
        return {"filter_probs": fp, "calibration": {"num_states": 5}}

    @staticmethod
    def _make_evt_data():
        return {"xi": 0.1, "beta": 0.5, "threshold": 1.0, "n_total": 300, "n_exceedances": 30}

    @staticmethod
    def _make_svj_data():
        import pandas as pd
        return {"returns": pd.Series([1.0] * 100), "calibration": {"lambda_": 5}}

    @staticmethod
    def _make_hawkes_data():
        import numpy as np
        return {"event_times": np.array([1.0]), "mu": 0.05, "alpha": 0.1, "beta": 1.0}

    def _assess(self, token="SOL", direction="long", trade_size_usd=5_000):
        from cortex.guardian import assess_trade, _cache
        _cache.clear()
        return assess_trade(
            token, trade_size_usd, direction,
            self._make_model_data(), self._make_evt_data(),
            self._make_svj_data(), self._make_hawkes_data(),
        )

    @patch("cortex.guardian._score_evt", return_value={"component": "evt", "score": 15.0, "details": {}})
    @patch("cortex.guardian._score_svj", return_value={"component": "svj", "score": 10.0, "details": {"jump_share_pct": 15}})
    @patch("cortex.guardian._score_hawkes", return_value={"component": "hawkes", "score": 8.0, "details": {"contagion_risk_score": 0.08}})
    def test_force_reject_overrides_approval(self, *_mocks):
        """A force_reject override should block an otherwise approved trade."""
        create_override(
            action="force_reject", token="SOL",
            reason="whale dump detected", created_by="risk_team",
        )
        result = self._assess("SOL")
        assert result["approved"] is False
        assert result["human_override"] is not None
        assert result["human_override"]["action"] == "force_reject"

    @patch("cortex.guardian._score_evt", return_value={"component": "evt", "score": 15.0, "details": {}})
    @patch("cortex.guardian._score_svj", return_value={"component": "svj", "score": 10.0, "details": {"jump_share_pct": 15}})
    @patch("cortex.guardian._score_hawkes", return_value={"component": "hawkes", "score": 8.0, "details": {"contagion_risk_score": 0.08}})
    def test_force_approve_overrides(self, *_mocks):
        """A force_approve override should set approved=True."""
        create_override(
            action="force_approve", token="BTC",
            reason="confirmed opportunity", created_by="ops",
        )
        result = self._assess("BTC")
        assert result["approved"] is True
        assert result["human_override"] is not None

    @patch("cortex.guardian._score_evt", return_value={"component": "evt", "score": 15.0, "details": {}})
    @patch("cortex.guardian._score_svj", return_value={"component": "svj", "score": 10.0, "details": {"jump_share_pct": 15}})
    @patch("cortex.guardian._score_hawkes", return_value={"component": "hawkes", "score": 8.0, "details": {"contagion_risk_score": 0.08}})
    def test_size_cap_reduces_size(self, *_mocks):
        """SIZE_CAP should limit recommended_size."""
        create_override(
            action="size_cap", token="SOL",
            reason="reduce exposure", size_cap_usd=500.0,
        )
        result = self._assess("SOL", trade_size_usd=10_000)
        assert result["recommended_size"] <= 500.0
        assert result["human_override"] is not None

    @patch("cortex.guardian._score_evt", return_value={"component": "evt", "score": 15.0, "details": {}})
    @patch("cortex.guardian._score_svj", return_value={"component": "svj", "score": 10.0, "details": {"jump_share_pct": 15}})
    @patch("cortex.guardian._score_hawkes", return_value={"component": "hawkes", "score": 8.0, "details": {"contagion_risk_score": 0.08}})
    def test_cooldown_blocks_trading(self, *_mocks):
        """COOLDOWN should prevent any trade on the token."""
        create_override(
            action="cooldown", token="ETH",
            reason="post-exploit caution", created_by="security",
        )
        result = self._assess("ETH")
        assert result["approved"] is False
        assert result["human_override"] is not None
        assert result["human_override"]["action"] == "cooldown"

    @patch("cortex.guardian._score_evt", return_value={"component": "evt", "score": 15.0, "details": {}})
    @patch("cortex.guardian._score_svj", return_value={"component": "svj", "score": 10.0, "details": {"jump_share_pct": 15}})
    @patch("cortex.guardian._score_hawkes", return_value={"component": "hawkes", "score": 8.0, "details": {"contagion_risk_score": 0.08}})
    def test_no_override_no_field(self, *_mocks):
        """Without active overrides, human_override should be None."""
        result = self._assess("NOOVR")
        assert result["human_override"] is None

    @patch("cortex.guardian._score_evt", return_value={"component": "evt", "score": 15.0, "details": {}})
    @patch("cortex.guardian._score_svj", return_value={"component": "svj", "score": 10.0, "details": {"jump_share_pct": 15}})
    @patch("cortex.guardian._score_hawkes", return_value={"component": "hawkes", "score": 8.0, "details": {"contagion_risk_score": 0.08}})
    def test_expired_override_not_applied(self, *_mocks):
        """Expired override should not affect decisions."""
        entry_dict = create_override(
            action="force_reject", token="SOL", reason="test", ttl=1,
        )
        reg = get_registry()
        entry = reg._overrides[entry_dict["id"]]
        entry.expires_at = time.time() - 1

        result = self._assess("SOL")
        assert result["human_override"] is None
