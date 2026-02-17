"""Tests for the background news buffer and collector."""

import asyncio
import time
from unittest.mock import patch, MagicMock

import pytest

from cortex.news import NewsBuffer, news_buffer


# ── NewsBuffer unit tests ──

def test_buffer_empty_initially():
    buf = NewsBuffer(max_items=10)
    assert buf.get_signal() is None
    assert buf.get_full() is None
    assert buf.age_seconds == float("inf")
    assert buf.stats["has_data"] is False


def test_buffer_update_and_read():
    buf = NewsBuffer(max_items=10)
    result = {
        "items": [{"id": f"test_{i}"} for i in range(5)],
        "signal": {"direction": "LONG", "strength": 0.7},
        "source_counts": {"cryptocompare": 3, "newsdata": 1, "cryptopanic": 1},
        "meta": {"total": 5},
    }
    buf.update(result)

    assert buf.get_signal() == {"direction": "LONG", "strength": 0.7}
    full = buf.get_full()
    assert full is not None
    assert len(full["items"]) == 5
    assert buf.stats["has_data"] is True
    assert buf.stats["fetch_count"] == 1


def test_buffer_max_items_truncation():
    buf = NewsBuffer(max_items=3)
    result = {
        "items": [{"id": f"test_{i}"} for i in range(10)],
        "signal": {"direction": "NEUTRAL"},
        "source_counts": {},
        "meta": {},
    }
    buf.update(result)

    full = buf.get_full()
    assert len(full["items"]) == 3


def test_buffer_get_full_with_max_items():
    buf = NewsBuffer(max_items=100)
    result = {
        "items": [{"id": f"test_{i}"} for i in range(20)],
        "signal": {"direction": "SHORT"},
        "source_counts": {},
        "meta": {},
    }
    buf.update(result)

    subset = buf.get_full(max_items=5)
    assert len(subset["items"]) == 5

    full = buf.get_full()
    assert len(full["items"]) == 20


def test_buffer_age():
    buf = NewsBuffer(max_items=10)
    result = {"items": [], "signal": {}, "source_counts": {}, "meta": {}}
    buf.update(result)
    assert buf.age_seconds < 1.0


def test_buffer_record_error():
    buf = NewsBuffer(max_items=10)
    buf.record_error("connection timeout")
    assert buf.stats["error_count"] == 1
    assert buf.stats["last_error"] == "connection timeout"


def test_buffer_overwrite():
    buf = NewsBuffer(max_items=10)
    buf.update({"items": [{"id": "old"}], "signal": {"v": 1}, "source_counts": {}, "meta": {}})
    buf.update({"items": [{"id": "new"}], "signal": {"v": 2}, "source_counts": {}, "meta": {}})
    assert buf.get_signal() == {"v": 2}
    assert buf.stats["fetch_count"] == 2


# ── Global singleton ──

def test_global_news_buffer_exists():
    from cortex.news import news_buffer
    assert isinstance(news_buffer, NewsBuffer)


# ── Collector start/stop ──

@pytest.mark.asyncio
async def test_news_collector_start_stop():
    from api import tasks

    async def _dummy_loop():
        while True:
            await asyncio.sleep(3600)

    from api.tasks import start_news_collector, stop_news_collector

    tasks._news_collector_task = None
    with patch.object(tasks, "_news_collector_loop", _dummy_loop):
        with patch("cortex.config.NEWS_COLLECTOR_ENABLED", True):
            start_news_collector()
            assert tasks._news_collector_task is not None
            stop_news_collector()
            assert tasks._news_collector_task is None


@pytest.mark.asyncio
async def test_news_collector_disabled():
    with patch("cortex.config.NEWS_COLLECTOR_ENABLED", False):
        from api.tasks import start_news_collector, stop_news_collector
        from api import tasks
        tasks._news_collector_task = None
        start_news_collector()
        assert tasks._news_collector_task is None
