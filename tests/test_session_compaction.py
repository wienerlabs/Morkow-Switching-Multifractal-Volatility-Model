"""Tests for cortex/session_compaction.py — Session Compaction."""
import time

import pytest

from cortex.session_compaction import (
    SessionCompactor,
    CompactionResult,
    _estimate_tokens,
    _message_importance,
    IMPORTANCE_WEIGHTS,
)


@pytest.fixture
def compactor():
    return SessionCompactor(max_tokens=1000, compact_threshold=0.8)


def _msg(content: str, category: str = "conversation", **kwargs) -> dict:
    return {
        "role": "assistant",
        "content": content,
        "category": category,
        "timestamp": time.time(),
        **kwargs,
    }


def test_add_message(compactor):
    compactor.add_message(_msg("hello"))
    assert compactor.message_count == 1


def test_add_message_auto_timestamp(compactor):
    msg = {"role": "user", "content": "test"}
    compactor.add_message(msg)
    assert "timestamp" in msg
    assert "category" in msg


def test_current_tokens(compactor):
    compactor.add_message(_msg("a" * 400))  # ~100 tokens
    assert compactor.current_tokens > 0


def test_should_compact_below_threshold(compactor):
    compactor.add_message(_msg("short"))
    assert not compactor.should_compact()


def test_should_compact_above_threshold(compactor):
    # 1000 * 0.8 = 800 token threshold, each 'a' * 4 = 1 token
    for _ in range(20):
        compactor.add_message(_msg("a" * 200))  # ~50 tokens each, 20*50=1000
    assert compactor.should_compact()


def test_compact_empty(compactor):
    result = compactor.compact()
    assert result.original_count == 0
    assert result.compacted_count == 0
    assert result.compaction_hash == "empty"


def test_compact_preserves_high_importance():
    compactor = SessionCompactor(max_tokens=500, compact_threshold=0.5)

    # Add high importance messages
    compactor.add_message(_msg("SOL trade executed tx=0xabc", "trade_execution"))
    compactor.add_message(_msg("Guardian rejected: risk=92", "guardian_decision"))

    # Add low importance messages
    for i in range(10):
        compactor.add_message(_msg(f"status check {i}", "conversation"))

    result = compactor.compact()
    assert result.compacted_count >= 2  # high importance kept
    assert result.discarded_count > 0   # low importance discarded


def test_compact_creates_summary():
    compactor = SessionCompactor(max_tokens=500, compact_threshold=0.5)

    # Add medium importance messages
    for i in range(10):
        compactor.add_message(_msg(f"Analysis result {i}: SOL looks good", "analysis"))

    # Add low importance
    for i in range(5):
        compactor.add_message(_msg(f"hello {i}", "conversation"))

    result = compactor.compact()
    assert result.summary_count >= 1
    assert compactor.summary_count >= 1


def test_compact_reduces_tokens():
    compactor = SessionCompactor(max_tokens=2000, compact_threshold=0.5)

    for i in range(30):
        compactor.add_message(_msg(f"conversation message {i} " * 10, "conversation"))

    tokens_before = compactor.current_tokens
    result = compactor.compact()
    assert result.tokens_after < result.tokens_before
    assert result.compression_ratio < 1.0


def test_compaction_result_serialization():
    result = CompactionResult(
        original_count=50,
        compacted_count=10,
        summary_count=2,
        tokens_before=5000,
        tokens_after=1200,
        discarded_count=30,
        compaction_hash="abc123",
    )
    d = result.to_dict()
    assert d["compression_ratio"] == pytest.approx(0.24)
    assert d["original_count"] == 50


def test_get_context_includes_summaries_and_messages(compactor):
    compactor.add_message(_msg("trade executed", "trade_execution"))
    # Manually add a summary
    compactor._summaries.append({
        "role": "system",
        "content": "SESSION SUMMARY: previous trades...",
        "is_summary": True,
    })
    ctx = compactor.get_context()
    assert len(ctx) == 2
    assert ctx[0]["is_summary"] is True
    assert ctx[1]["category"] == "trade_execution"


def test_get_status(compactor):
    compactor.add_message(_msg("test"))
    status = compactor.get_status()
    assert status["messages"] == 1
    assert status["summaries"] == 0
    assert "current_tokens" in status
    assert "usage_pct" in status
    assert "should_compact" in status


def test_clear(compactor):
    compactor.add_message(_msg("test"))
    compactor._summaries.append({"content": "summary"})
    compactor.clear()
    assert compactor.message_count == 0
    assert compactor.summary_count == 0


def test_estimate_tokens():
    assert _estimate_tokens("") == 1
    assert _estimate_tokens("a" * 100) == 25
    assert _estimate_tokens("hello world") >= 1


def test_message_importance_categories():
    """Higher category = higher importance."""
    trade = _message_importance({"category": "trade_execution"})
    convo = _message_importance({"category": "conversation"})
    assert trade > convo


def test_message_importance_trade_hash_boost():
    """Messages with trade_hash get boosted."""
    base = _message_importance({"category": "conversation"})
    boosted = _message_importance({"category": "conversation", "trade_hash": "abc123"})
    assert boosted > base


def test_message_importance_error_boost():
    """Messages with error get boosted."""
    base = _message_importance({"category": "system"})
    errored = _message_importance({"category": "system", "error": "timeout"})
    assert errored > base


def test_multiple_compactions():
    """Multiple rounds of compaction work correctly."""
    compactor = SessionCompactor(max_tokens=400, compact_threshold=0.5)
    old_ts = time.time() - 7200  # old so no recency boost

    # First batch — "analysis" category (0.5 importance) → summarize bucket
    for i in range(20):
        compactor.add_message({
            "role": "assistant",
            "content": f"analysis batch 1 #{i}",
            "category": "analysis",
            "timestamp": old_ts + i,
        })
    compactor.compact()
    assert compactor._total_compactions == 1

    # Second batch
    for i in range(20):
        compactor.add_message({
            "role": "assistant",
            "content": f"analysis batch 2 #{i}",
            "category": "analysis",
            "timestamp": old_ts + 100 + i,
        })
    compactor.compact()
    assert compactor._total_compactions == 2
    assert compactor.summary_count >= 2  # one per compaction


def test_importance_weights_coverage():
    """All defined categories have a weight."""
    for cat in IMPORTANCE_WEIGHTS:
        score = _message_importance({"category": cat, "timestamp": time.time()})
        assert 0 <= score <= 1.0


def test_summary_content_structure():
    compactor = SessionCompactor(max_tokens=500, compact_threshold=0.5)
    old_ts = time.time() - 7200  # 2 hours ago, no recency boost
    for i in range(10):
        compactor.add_message({
            "role": "assistant",
            "content": f"Market signal #{i}: SOL +2%",
            "category": "market_signal",
            "timestamp": old_ts + i,
        })

    compactor.compact()
    assert compactor.summary_count >= 1
    summary = compactor._summaries[-1]
    assert "SESSION SUMMARY" in summary["content"]
    assert summary.get("is_summary") is True
    assert summary.get("source_count", 0) > 0
