"""DX-Research Task 8: Human-in-the-Loop Override.

DX Terminal finding: 87% of top-performing agents were human-assisted.
This module provides a registry for human overrides that can force-approve,
force-reject, cap trade size, or cooldown a token for a configurable TTL.

Architecture:
  - OverrideRegistry: In-memory store of active overrides with TTL expiry
  - OverrideEntry: Single override with type, target, reason, TTL, audit trail
  - Audit Log: Every create/expire/apply event is logged for compliance
  - Guardian Integration: assess_trade checks registry before final decision
"""
from __future__ import annotations

__all__ = [
    "OverrideAction",
    "OverrideEntry",
    "OverrideRegistry",
    "get_registry",
    "create_override",
    "check_override",
    "list_active_overrides",
    "revoke_override",
]

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from cortex.config import (
    HUMAN_OVERRIDE_DEFAULT_TTL,
    HUMAN_OVERRIDE_ENABLED,
    HUMAN_OVERRIDE_MAX_TTL,
)

logger = logging.getLogger(__name__)


class OverrideAction(str, Enum):
    FORCE_APPROVE = "force_approve"
    FORCE_REJECT = "force_reject"
    SIZE_CAP = "size_cap"
    COOLDOWN = "cooldown"


@dataclass
class OverrideEntry:
    """A single human override directive."""
    id: str
    action: OverrideAction
    token: str  # "*" = global (all tokens)
    reason: str
    created_by: str  # operator identifier
    created_at: float
    expires_at: float
    size_cap_usd: float | None = None  # only for SIZE_CAP action
    metadata: dict[str, Any] = field(default_factory=dict)
    applied_count: int = 0
    revoked: bool = False
    revoked_at: float | None = None
    revoked_by: str | None = None

    @property
    def is_active(self) -> bool:
        return not self.revoked and time.time() < self.expires_at

    @property
    def ttl_remaining(self) -> float:
        return max(0.0, self.expires_at - time.time())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action.value,
            "token": self.token,
            "reason": self.reason,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "ttl_remaining": round(self.ttl_remaining, 1),
            "size_cap_usd": self.size_cap_usd,
            "is_active": self.is_active,
            "applied_count": self.applied_count,
            "revoked": self.revoked,
            "revoked_at": self.revoked_at,
            "revoked_by": self.revoked_by,
            "metadata": self.metadata,
        }


@dataclass
class OverrideResult:
    """Result of checking overrides for a trade."""
    has_override: bool
    action: OverrideAction | None = None
    override_id: str | None = None
    reason: str = ""
    size_cap_usd: float | None = None
    created_by: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_override": self.has_override,
            "action": self.action.value if self.action else None,
            "override_id": self.override_id,
            "reason": self.reason,
            "size_cap_usd": self.size_cap_usd,
            "created_by": self.created_by,
        }


class OverrideRegistry:
    """In-memory registry of active human overrides with audit logging."""

    def __init__(self) -> None:
        self._overrides: dict[str, OverrideEntry] = {}
        self._audit_log: list[dict[str, Any]] = []

    def _log_audit(self, event: str, entry: OverrideEntry, extra: dict[str, Any] | None = None) -> None:
        record = {
            "event": event,
            "override_id": entry.id,
            "action": entry.action.value,
            "token": entry.token,
            "reason": entry.reason,
            "created_by": entry.created_by,
            "ts": time.time(),
        }
        if extra:
            record.update(extra)
        self._audit_log.append(record)
        # Keep last 1000 audit entries
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]
        logger.info("OVERRIDE AUDIT [%s]: %s %s on %s by %s — %s",
                     event, entry.action.value, entry.id[:8], entry.token,
                     entry.created_by, entry.reason)

    def create(
        self,
        action: OverrideAction,
        token: str = "*",
        reason: str = "",
        created_by: str = "operator",
        ttl: float | None = None,
        size_cap_usd: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OverrideEntry:
        """Create a new override. Returns the entry."""
        if ttl is None:
            ttl = HUMAN_OVERRIDE_DEFAULT_TTL
        ttl = min(ttl, HUMAN_OVERRIDE_MAX_TTL)
        ttl = max(1.0, ttl)

        if action == OverrideAction.SIZE_CAP and size_cap_usd is None:
            raise ValueError("size_cap_usd required for SIZE_CAP action")

        now = time.time()
        entry = OverrideEntry(
            id=uuid.uuid4().hex[:12],
            action=action,
            token=token.upper() if token != "*" else "*",
            reason=reason,
            created_by=created_by,
            created_at=now,
            expires_at=now + ttl,
            size_cap_usd=size_cap_usd,
            metadata=metadata or {},
        )
        self._overrides[entry.id] = entry
        self._log_audit("CREATED", entry, {"ttl": ttl})
        return entry

    def check(self, token: str) -> OverrideResult:
        """Check if any active override applies to this token.

        Priority: FORCE_REJECT > COOLDOWN > SIZE_CAP > FORCE_APPROVE.
        Token-specific overrides take precedence over global (*).
        """
        self._prune_expired()

        candidates: list[OverrideEntry] = []
        for entry in self._overrides.values():
            if not entry.is_active:
                continue
            if entry.token == "*" or entry.token == token.upper():
                candidates.append(entry)

        if not candidates:
            return OverrideResult(has_override=False)

        # Priority ordering
        priority = {
            OverrideAction.FORCE_REJECT: 0,
            OverrideAction.COOLDOWN: 1,
            OverrideAction.SIZE_CAP: 2,
            OverrideAction.FORCE_APPROVE: 3,
        }

        # Token-specific first, then by priority
        candidates.sort(key=lambda e: (
            0 if e.token != "*" else 1,
            priority.get(e.action, 99),
        ))

        winner = candidates[0]
        winner.applied_count += 1
        self._log_audit("APPLIED", winner, {"target_token": token})

        return OverrideResult(
            has_override=True,
            action=winner.action,
            override_id=winner.id,
            reason=winner.reason,
            size_cap_usd=winner.size_cap_usd,
            created_by=winner.created_by,
        )

    def revoke(self, override_id: str, revoked_by: str = "operator") -> bool:
        """Revoke an override before its TTL expires."""
        entry = self._overrides.get(override_id)
        if entry is None or entry.revoked:
            return False
        entry.revoked = True
        entry.revoked_at = time.time()
        entry.revoked_by = revoked_by
        self._log_audit("REVOKED", entry, {"revoked_by": revoked_by})
        return True

    def list_active(self) -> list[dict[str, Any]]:
        """List all currently active overrides."""
        self._prune_expired()
        return [e.to_dict() for e in self._overrides.values() if e.is_active]

    def get_audit_log(self, n: int = 50) -> list[dict[str, Any]]:
        """Get recent audit log entries."""
        return self._audit_log[-n:]

    def _prune_expired(self) -> int:
        """Remove expired overrides and log their expiry."""
        now = time.time()
        expired_ids = [
            oid for oid, e in self._overrides.items()
            if not e.revoked and now >= e.expires_at
        ]
        for oid in expired_ids:
            entry = self._overrides[oid]
            entry.revoked = True
            entry.revoked_at = now
            entry.revoked_by = "system:ttl_expired"
            self._log_audit("EXPIRED", entry, {"applied_count": entry.applied_count})
        return len(expired_ids)

    def clear(self) -> None:
        """Clear all overrides (for testing)."""
        self._overrides.clear()
        self._audit_log.clear()


# ── Module-level singleton ──

_registry: OverrideRegistry | None = None


def get_registry() -> OverrideRegistry:
    global _registry
    if _registry is None:
        _registry = OverrideRegistry()
    return _registry


def create_override(
    action: str | OverrideAction,
    token: str = "*",
    reason: str = "",
    created_by: str = "operator",
    ttl: float | None = None,
    size_cap_usd: float | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience: create an override and return its dict representation."""
    if not HUMAN_OVERRIDE_ENABLED:
        return {"error": "Human override system is disabled"}
    if isinstance(action, str):
        action = OverrideAction(action)
    entry = get_registry().create(
        action=action, token=token, reason=reason,
        created_by=created_by, ttl=ttl, size_cap_usd=size_cap_usd,
        metadata=metadata,
    )
    return entry.to_dict()


def check_override(token: str) -> OverrideResult:
    """Convenience: check overrides for a token."""
    if not HUMAN_OVERRIDE_ENABLED:
        return OverrideResult(has_override=False)
    return get_registry().check(token)


def revoke_override(override_id: str, revoked_by: str = "operator") -> bool:
    """Convenience: revoke an override."""
    return get_registry().revoke(override_id, revoked_by)


def list_active_overrides() -> list[dict[str, Any]]:
    """Convenience: list active overrides."""
    if not HUMAN_OVERRIDE_ENABLED:
        return []
    return get_registry().list_active()
