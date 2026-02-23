"""Session Compaction — prevents agent context overflow via rolling summarization.

Inspired by OpenAlice's session management: rather than letting context grow
unbounded until truncation, this module implements a rolling compaction strategy
that preserves key decisions, trade outcomes, and risk state while discarding
low-value conversational filler.

Works with any LLM-backed agent (Eliza, custom, etc.) by operating on a list
of structured message dicts.
"""
from __future__ import annotations

__all__ = ["SessionCompactor", "CompactionResult"]

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)

# Default token budget: compaction triggers at 80% of max context
DEFAULT_MAX_TOKENS = 8000
DEFAULT_COMPACT_THRESHOLD = 0.8  # compact when usage >= 80%

# Message categories by importance (higher = more important, kept longer)
IMPORTANCE_WEIGHTS: dict[str, float] = {
    "trade_execution": 1.0,   # actual trade outcomes — never discard
    "risk_alert": 0.95,       # circuit breaker trips, drawdown warnings
    "guardian_decision": 0.9,  # approval/rejection with scores
    "position_update": 0.85,  # P&L, position state changes
    "strategy_change": 0.8,   # mode changes, allocation shifts
    "market_signal": 0.6,     # price alerts, regime changes
    "analysis": 0.5,          # pool/arb analysis results
    "system": 0.3,            # heartbeat, status checks
    "conversation": 0.1,      # general chat, filler
}


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


def _message_importance(msg: dict[str, Any]) -> float:
    """Score a message's importance for retention."""
    category = msg.get("category", "conversation")
    base = IMPORTANCE_WEIGHTS.get(category, 0.1)

    # Boost recent messages
    age_seconds = time.time() - msg.get("timestamp", time.time())
    recency_boost = max(0.0, 1.0 - age_seconds / 3600.0) * 0.2  # up to +0.2 for <1h old

    # Boost messages with trade hashes (audit trail)
    if msg.get("trade_hash") or msg.get("entry_hash"):
        base = max(base, 0.9)

    # Boost messages with errors (learning signal)
    if msg.get("error"):
        base = max(base, 0.7)

    return min(1.0, base + recency_boost)


@dataclass
class CompactionResult:
    """Result of a compaction operation."""
    original_count: int
    compacted_count: int
    summary_count: int
    tokens_before: int
    tokens_after: int
    discarded_count: int
    compaction_hash: str
    timestamp: float = field(default_factory=time.time)

    @property
    def compression_ratio(self) -> float:
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["compression_ratio"] = self.compression_ratio
        return d


class SessionCompactor:
    """Rolling session compaction for agent context management.

    Usage::

        compactor = SessionCompactor(max_tokens=8000)

        # Add messages as they arrive
        compactor.add_message({
            "role": "assistant",
            "content": "Approved SOL long at $150",
            "category": "guardian_decision",
            "timestamp": time.time(),
        })

        # Check if compaction needed, compact if so
        if compactor.should_compact():
            result = compactor.compact()
    """

    def __init__(
        self,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        compact_threshold: float = DEFAULT_COMPACT_THRESHOLD,
    ) -> None:
        self._max_tokens = max_tokens
        self._compact_threshold = compact_threshold
        self._messages: list[dict[str, Any]] = []
        self._summaries: list[dict[str, Any]] = []
        self._total_compactions: int = 0
        self._total_discarded: int = 0

    @property
    def message_count(self) -> int:
        return len(self._messages)

    @property
    def summary_count(self) -> int:
        return len(self._summaries)

    @property
    def current_tokens(self) -> int:
        total = 0
        for msg in self._messages:
            total += _estimate_tokens(msg.get("content", ""))
        for s in self._summaries:
            total += _estimate_tokens(s.get("content", ""))
        return total

    def add_message(self, message: dict[str, Any]) -> None:
        """Add a message to the session."""
        if "timestamp" not in message:
            message["timestamp"] = time.time()
        if "category" not in message:
            message["category"] = "conversation"
        self._messages.append(message)

    def should_compact(self) -> bool:
        """Check if the session needs compaction."""
        return self.current_tokens >= self._max_tokens * self._compact_threshold

    def compact(self) -> CompactionResult:
        """Perform rolling compaction.

        Strategy:
        1. Score all messages by importance
        2. Keep high-importance messages verbatim
        3. Summarize medium-importance messages into a digest
        4. Discard low-importance messages entirely
        """
        tokens_before = self.current_tokens
        original_count = len(self._messages)

        if not self._messages:
            return CompactionResult(
                original_count=0,
                compacted_count=0,
                summary_count=len(self._summaries),
                tokens_before=0,
                tokens_after=0,
                discarded_count=0,
                compaction_hash="empty",
            )

        # Score and sort
        scored = [(msg, _message_importance(msg)) for msg in self._messages]

        # Partition into tiers
        keep: list[dict[str, Any]] = []      # importance >= 0.7
        summarize: list[dict[str, Any]] = []  # 0.3 <= importance < 0.7
        discard: list[dict[str, Any]] = []    # importance < 0.3

        for msg, score in scored:
            if score >= 0.7:
                keep.append(msg)
            elif score >= 0.3:
                summarize.append(msg)
            else:
                discard.append(msg)

        # Build summary from medium-tier messages
        if summarize:
            summary = self._build_summary(summarize)
            self._summaries.append(summary)

        # Replace messages with only the kept ones
        self._messages = keep
        self._total_compactions += 1
        self._total_discarded += len(discard)

        tokens_after = self.current_tokens

        # Compute a hash for this compaction event
        raw = json.dumps({
            "ts": time.time(),
            "kept": len(keep),
            "summarized": len(summarize),
            "discarded": len(discard),
        }, sort_keys=True)
        compaction_hash = hashlib.sha256(raw.encode()).hexdigest()[:12]

        result = CompactionResult(
            original_count=original_count,
            compacted_count=len(keep),
            summary_count=len(self._summaries),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            discarded_count=len(discard),
            compaction_hash=compaction_hash,
        )

        logger.info(
            "session_compaction: %d → %d msgs, %d → %d tokens (%.0f%% compression), %d discarded",
            original_count, len(keep), tokens_before, tokens_after,
            (1 - result.compression_ratio) * 100, len(discard),
        )

        return result

    def _build_summary(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Build a compact summary from a list of messages."""
        categories: dict[str, list[str]] = {}
        for msg in messages:
            cat = msg.get("category", "other")
            content = msg.get("content", "")
            # Truncate individual message content for summary
            if len(content) > 200:
                content = content[:200] + "..."
            categories.setdefault(cat, []).append(content)

        summary_parts = []
        for cat, items in categories.items():
            summary_parts.append(f"[{cat}] {len(items)} messages")
            # Include first and last item as representative samples
            if items:
                summary_parts.append(f"  first: {items[0][:100]}")
                if len(items) > 1:
                    summary_parts.append(f"  last: {items[-1][:100]}")

        return {
            "role": "system",
            "content": "SESSION SUMMARY:\n" + "\n".join(summary_parts),
            "category": "system",
            "timestamp": time.time(),
            "is_summary": True,
            "source_count": len(messages),
        }

    def get_context(self) -> list[dict[str, Any]]:
        """Get the current context: summaries + active messages."""
        return self._summaries + self._messages

    def get_status(self) -> dict[str, Any]:
        return {
            "messages": len(self._messages),
            "summaries": len(self._summaries),
            "current_tokens": self.current_tokens,
            "max_tokens": self._max_tokens,
            "usage_pct": (self.current_tokens / self._max_tokens * 100) if self._max_tokens > 0 else 0,
            "total_compactions": self._total_compactions,
            "total_discarded": self._total_discarded,
            "should_compact": self.should_compact(),
        }

    def clear(self) -> None:
        """Clear all messages and summaries."""
        self._messages.clear()
        self._summaries.clear()
