"""Tests for cortex/trade_ledger.py — Git-like trade audit chain."""
import pytest

from cortex.trade_ledger import (
    LedgerEntry,
    LedgerState,
    TradeLedger,
    _compute_hash,
)


@pytest.fixture
def ledger():
    return TradeLedger(max_entries=100)


def test_stage_creates_entry(ledger):
    h = ledger.stage("SOL", "long", 5000.0, strategy="arb")
    assert len(h) == 16
    entry = ledger.show(h)
    assert entry is not None
    assert entry.state == LedgerState.STAGED
    assert entry.token == "SOL"
    assert entry.direction == "long"
    assert entry.trade_size_usd == 5000.0
    assert entry.parent_hash == "genesis"


def test_commit_transitions_state(ledger):
    h = ledger.stage("ETH", "short", 3000.0)
    entry = ledger.commit(h, "risk score 42, regime 2")
    assert entry.state == LedgerState.COMMITTED
    assert entry.message == "risk score 42, regime 2"
    assert entry.committed_at is not None
    assert ledger.head == h


def test_push_success(ledger):
    h = ledger.stage("BTC", "long", 10000.0)
    ledger.commit(h, "all clear")
    entry = ledger.push(h, result={"tx_hash": "0xabc", "fill_price": 62000.0})
    assert entry.state == LedgerState.PUSHED
    assert entry.result["tx_hash"] == "0xabc"
    assert entry.pushed_at is not None


def test_push_failure(ledger):
    h = ledger.stage("SOL", "long", 5000.0)
    ledger.commit(h, "execute")
    entry = ledger.push(h, error="slippage exceeded 2%")
    assert entry.state == LedgerState.FAILED
    assert entry.error == "slippage exceeded 2%"


def test_hash_chain_integrity(ledger):
    h1 = ledger.stage("SOL", "long", 1000.0)
    ledger.commit(h1, "first trade")
    ledger.push(h1, result={"ok": True})

    h2 = ledger.stage("SOL", "long", 2000.0)
    entry2 = ledger.show(h2)
    assert entry2.parent_hash == h1

    ledger.commit(h2, "second trade")
    ledger.push(h2, result={"ok": True})

    h3 = ledger.stage("SOL", "short", 3000.0)
    entry3 = ledger.show(h3)
    assert entry3.parent_hash == h2


def test_commit_requires_staged(ledger):
    h = ledger.stage("SOL", "long", 1000.0)
    ledger.commit(h, "msg")
    with pytest.raises(ValueError, match="committed"):
        ledger.commit(h, "again")


def test_push_requires_committed(ledger):
    h = ledger.stage("SOL", "long", 1000.0)
    with pytest.raises(ValueError, match="staged"):
        ledger.push(h, result={})


def test_rollback_staged(ledger):
    h = ledger.stage("SOL", "long", 1000.0)
    entry = ledger.rollback(h, "changed mind")
    assert entry.state == LedgerState.ROLLED_BACK
    assert entry.error == "changed mind"


def test_rollback_committed(ledger):
    h = ledger.stage("SOL", "long", 1000.0)
    ledger.commit(h, "hmm")
    entry = ledger.rollback(h, "abort")
    assert entry.state == LedgerState.ROLLED_BACK


def test_rollback_pushed_raises(ledger):
    h = ledger.stage("SOL", "long", 1000.0)
    ledger.commit(h, "go")
    ledger.push(h, result={})
    with pytest.raises(ValueError, match="pushed"):
        ledger.rollback(h)


def test_log_all(ledger):
    for i in range(5):
        h = ledger.stage("SOL", "long", float(i * 1000))
        ledger.commit(h, f"trade {i}")
        ledger.push(h, result={"i": i})

    entries = ledger.log(limit=10)
    assert len(entries) == 5
    assert entries[0]["trade_size_usd"] == 4000.0


def test_log_by_token(ledger):
    for token in ["SOL", "ETH", "SOL", "BTC", "SOL"]:
        h = ledger.stage(token, "long", 1000.0)
        ledger.commit(h, f"trade {token}")
        ledger.push(h, result={})

    sol_entries = ledger.log(token="SOL")
    assert len(sol_entries) == 3
    for e in sol_entries:
        assert e["token"] == "SOL"


def test_diff(ledger):
    h1 = ledger.stage("SOL", "long", 1000.0)
    ledger.commit(h1, "first")
    ledger.push(h1, result={"price": 100.0})

    h2 = ledger.stage("SOL", "long", 2000.0)
    ledger.commit(h2, "second")
    ledger.push(h2, result={"price": 110.0})

    diff = ledger.diff(h1, h2)
    assert "trade_size_usd" in diff["changes"]
    assert diff["changes"]["trade_size_usd"]["old"] == 1000.0
    assert diff["changes"]["trade_size_usd"]["new"] == 2000.0


def test_diff_missing_hash(ledger):
    h = ledger.stage("SOL", "long", 1000.0)
    with pytest.raises(KeyError):
        ledger.diff(h, "nonexistent")


def test_show_nonexistent(ledger):
    assert ledger.show("nope") is None


def test_get_pending(ledger):
    assert ledger.get_pending("SOL") is None
    h = ledger.stage("SOL", "long", 1000.0)
    pending = ledger.get_pending("SOL")
    assert pending is not None
    assert pending.hash == h

    ledger.commit(h, "go")
    ledger.push(h, result={})
    assert ledger.get_pending("SOL") is None


def test_stats(ledger):
    h1 = ledger.stage("SOL", "long", 1000.0)
    h2 = ledger.stage("ETH", "short", 2000.0)
    ledger.commit(h1, "go")
    ledger.push(h1, result={})

    stats = ledger.stats()
    assert stats["total"] == 2
    assert stats["by_state"]["pushed"] == 1
    assert stats["by_state"]["staged"] == 1
    assert "ETH" in stats["pending_tokens"]


def test_pruning():
    ledger = TradeLedger(max_entries=5)
    hashes = []
    for i in range(10):
        h = ledger.stage("SOL", "long", float(i))
        ledger.commit(h, f"t{i}")
        ledger.push(h, result={})
        hashes.append(h)

    assert ledger.size <= 5
    assert ledger.show(hashes[0]) is None
    assert ledger.show(hashes[-1]) is not None


def test_snapshot_and_restore(ledger):
    for i in range(3):
        h = ledger.stage("SOL", "long", float(i * 100))
        ledger.commit(h, f"trade {i}")
        ledger.push(h, result={"i": i})

    snapshot = ledger.snapshot()
    assert len(snapshot) == 3

    new_ledger = TradeLedger()
    restored = new_ledger.restore(snapshot)
    assert restored == 3
    assert new_ledger.size == 3

    entries = new_ledger.log(token="SOL")
    assert len(entries) == 3


def test_guardian_score_stored(ledger):
    h = ledger.stage("SOL", "long", 5000.0, guardian_score=42.5)
    entry = ledger.show(h)
    assert entry.guardian_score == 42.5


def test_entry_serialization():
    entry = LedgerEntry(
        hash="abc123",
        parent_hash="genesis",
        state=LedgerState.PUSHED,
        token="SOL",
        direction="long",
        strategy="arb",
        trade_size_usd=5000.0,
        intent={"reason": "spread detected"},
        message="execute arb",
        result={"tx": "0x123"},
        guardian_score=35.0,
        staged_at=1000.0,
        committed_at=1001.0,
        pushed_at=1002.0,
        error=None,
    )
    d = entry.to_dict()
    assert d["state"] == "pushed"

    restored = LedgerEntry.from_dict(d)
    assert restored.state == LedgerState.PUSHED
    assert restored.hash == "abc123"


def test_compute_hash_deterministic():
    h1 = _compute_hash("parent", {"a": 1, "b": 2})
    h2 = _compute_hash("parent", {"b": 2, "a": 1})
    assert h1 == h2


def test_compute_hash_different_parents():
    h1 = _compute_hash("parent_a", {"data": 1})
    h2 = _compute_hash("parent_b", {"data": 1})
    assert h1 != h2


def test_intent_data_preserved(ledger):
    intent = {"reason": "arb opportunity", "spread_bps": 150, "pools": ["orca", "raydium"]}
    h = ledger.stage("SOL", "long", 5000.0, intent=intent)
    entry = ledger.show(h)
    assert entry.intent == intent
