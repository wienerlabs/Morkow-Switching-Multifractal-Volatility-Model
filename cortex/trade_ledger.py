"""Git-like trade audit chain with SHA-256 hash-linked entries.

Every trade execution flows through stage() → commit() → push() with
cryptographic linking. Provides full audit trail for regulatory compliance,
debugging, and ML training data extraction.

Persistence: Redis-backed via PersistentStore pattern (in-memory fallback).
"""
from __future__ import annotations

__all__ = ["TradeLedger", "get_trade_ledger", "LedgerEntry", "LedgerState"]

import enum
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any

from cortex.config import PERSISTENCE_ENABLED

logger = logging.getLogger(__name__)


class LedgerState(str, enum.Enum):
    STAGED = "staged"
    COMMITTED = "committed"
    PUSHED = "pushed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class LedgerEntry:
    hash: str
    parent_hash: str
    state: LedgerState
    token: str
    direction: str
    strategy: str
    trade_size_usd: float
    intent: dict[str, Any]
    message: str
    result: dict[str, Any] | None
    guardian_score: float | None
    staged_at: float
    committed_at: float | None
    pushed_at: float | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LedgerEntry:
        data["state"] = LedgerState(data["state"])
        return cls(**data)


def _compute_hash(parent_hash: str, payload: dict[str, Any]) -> str:
    raw = json.dumps({"parent": parent_hash, **payload}, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class TradeLedger:
    """Hash-chained trade audit trail with stage/commit/push semantics."""

    def __init__(self, max_entries: int = 1000) -> None:
        self._entries: dict[str, LedgerEntry] = {}
        self._token_index: dict[str, list[str]] = {}
        self._pending: dict[str, str] = {}
        self._max_entries = max_entries
        self._head: str = "genesis"

    def stage(
        self,
        token: str,
        direction: str,
        trade_size_usd: float,
        strategy: str = "spot",
        intent: dict[str, Any] | None = None,
        guardian_score: float | None = None,
    ) -> str:
        """Create a pending ledger entry. Returns the hash."""
        payload = {
            "token": token,
            "direction": direction,
            "trade_size_usd": trade_size_usd,
            "strategy": strategy,
            "intent": intent or {},
            "ts": time.time(),
        }
        entry_hash = _compute_hash(self._head, payload)

        entry = LedgerEntry(
            hash=entry_hash,
            parent_hash=self._head,
            state=LedgerState.STAGED,
            token=token,
            direction=direction,
            strategy=strategy,
            trade_size_usd=trade_size_usd,
            intent=intent or {},
            message="",
            result=None,
            guardian_score=guardian_score,
            staged_at=time.time(),
            committed_at=None,
            pushed_at=None,
            error=None,
        )

        self._entries[entry_hash] = entry
        self._token_index.setdefault(token, []).append(entry_hash)
        self._pending[token] = entry_hash
        self._prune_if_needed()

        logger.info("trade_ledger STAGED hash=%s token=%s dir=%s size=%.2f",
                     entry_hash, token, direction, trade_size_usd)
        return entry_hash

    def commit(self, entry_hash: str, message: str) -> LedgerEntry:
        """Commit a staged entry with a message. Returns the entry."""
        entry = self._entries.get(entry_hash)
        if entry is None:
            raise KeyError(f"Entry {entry_hash} not found")
        if entry.state != LedgerState.STAGED:
            raise ValueError(f"Entry {entry_hash} is {entry.state.value}, expected staged")

        entry.state = LedgerState.COMMITTED
        entry.message = message
        entry.committed_at = time.time()
        self._head = entry_hash

        logger.info("trade_ledger COMMITTED hash=%s msg='%s'", entry_hash, message)
        return entry

    def push(
        self,
        entry_hash: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> LedgerEntry:
        """Execute (push) a committed entry. Records result or error."""
        entry = self._entries.get(entry_hash)
        if entry is None:
            raise KeyError(f"Entry {entry_hash} not found")
        if entry.state != LedgerState.COMMITTED:
            raise ValueError(f"Entry {entry_hash} is {entry.state.value}, expected committed")

        if error:
            entry.state = LedgerState.FAILED
            entry.error = error
        else:
            entry.state = LedgerState.PUSHED
            entry.result = result or {}

        entry.pushed_at = time.time()
        self._pending.pop(entry.token, None)

        logger.info("trade_ledger %s hash=%s token=%s",
                     entry.state.value.upper(), entry_hash, entry.token)
        return entry

    def rollback(self, entry_hash: str, reason: str = "") -> LedgerEntry:
        """Rollback a staged or committed entry."""
        entry = self._entries.get(entry_hash)
        if entry is None:
            raise KeyError(f"Entry {entry_hash} not found")
        if entry.state == LedgerState.PUSHED:
            raise ValueError("Cannot rollback a pushed entry")

        entry.state = LedgerState.ROLLED_BACK
        entry.error = reason or "manual rollback"
        self._pending.pop(entry.token, None)

        logger.info("trade_ledger ROLLED_BACK hash=%s reason='%s'", entry_hash, reason)
        return entry

    def show(self, entry_hash: str) -> LedgerEntry | None:
        return self._entries.get(entry_hash)

    def log(self, token: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent entries, optionally filtered by token."""
        if token:
            hashes = self._token_index.get(token, [])[-limit:]
        else:
            hashes = list(self._entries.keys())[-limit:]

        return [self._entries[h].to_dict() for h in reversed(hashes) if h in self._entries]

    def diff(self, hash1: str, hash2: str) -> dict[str, Any]:
        """Compare two ledger entries."""
        e1 = self._entries.get(hash1)
        e2 = self._entries.get(hash2)
        if not e1 or not e2:
            raise KeyError("One or both entries not found")

        d1, d2 = e1.to_dict(), e2.to_dict()
        changes = {}
        all_keys = set(d1.keys()) | set(d2.keys())
        for k in all_keys:
            v1, v2 = d1.get(k), d2.get(k)
            if v1 != v2:
                changes[k] = {"old": v1, "new": v2}

        return {"hash1": hash1, "hash2": hash2, "changes": changes}

    def get_pending(self, token: str) -> LedgerEntry | None:
        """Get the pending (staged/committed) entry for a token."""
        h = self._pending.get(token)
        if h:
            return self._entries.get(h)
        return None

    @property
    def head(self) -> str:
        return self._head

    @property
    def size(self) -> int:
        return len(self._entries)

    def stats(self) -> dict[str, Any]:
        counts = {}
        for e in self._entries.values():
            counts[e.state.value] = counts.get(e.state.value, 0) + 1
        return {
            "total": len(self._entries),
            "head": self._head,
            "pending_tokens": list(self._pending.keys()),
            "by_state": counts,
        }

    def _prune_if_needed(self) -> None:
        if len(self._entries) <= self._max_entries:
            return
        to_remove = len(self._entries) - self._max_entries
        oldest = list(self._entries.keys())[:to_remove]
        for h in oldest:
            entry = self._entries.pop(h, None)
            if entry:
                token_list = self._token_index.get(entry.token, [])
                if h in token_list:
                    token_list.remove(h)

    def snapshot(self) -> list[dict[str, Any]]:
        """Serialize all entries for persistence."""
        return [e.to_dict() for e in self._entries.values()]

    def restore(self, data: list[dict[str, Any]]) -> int:
        """Restore entries from a snapshot. Returns count restored."""
        count = 0
        for item in data:
            try:
                entry = LedgerEntry.from_dict(item)
                self._entries[entry.hash] = entry
                self._token_index.setdefault(entry.token, []).append(entry.hash)
                if entry.state in (LedgerState.STAGED, LedgerState.COMMITTED):
                    self._pending[entry.token] = entry.hash
                count += 1
            except Exception:
                logger.warning("Failed to restore ledger entry", exc_info=True)

        if self._entries:
            last = list(self._entries.values())[-1]
            self._head = last.hash

        logger.info("Restored %d trade ledger entries", count)
        return count


_ledger: TradeLedger | None = None


def get_trade_ledger() -> TradeLedger:
    global _ledger
    if _ledger is None:
        _ledger = TradeLedger()
    return _ledger
