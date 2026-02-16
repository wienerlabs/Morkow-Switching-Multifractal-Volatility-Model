"""Pyth Network oracle — fully dynamic integration with all Hermes v2 endpoints.

Supports:
  1. Feed discovery — list all 585+ crypto feeds
  2. Feed search   — search by token name
  3. Live prices   — fetch any token(s) by feed ID
  4. Historical    — price at any past unix timestamp
  5. SSE streaming — real-time price stream (async)
  6. Price buffers — per-feed ring buffer for downstream volatility

No hardcoded feed IDs. The entire Pyth crypto market is accessible.
"""

import logging
import time
from collections import deque
from typing import Any

import httpx

from cortex.config import (
    PYTH_BUFFER_DEPTH,
    PYTH_FEED_CACHE_TTL,
    PYTH_HERMES_URL,
    PYTH_SSE_TIMEOUT,
)
from cortex.data.rpc_failover import get_resilient_pool

logger = logging.getLogger(__name__)

_pool = get_resilient_pool()

# ── Feed registry cache ──
_feed_cache: list[dict[str, Any]] = []
_feed_cache_ts: float = 0.0
_feed_by_id: dict[str, dict[str, Any]] = {}
_feed_by_symbol: dict[str, dict[str, Any]] = {}

# ── Price state ──
_price_buffers: dict[str, deque[dict[str, Any]]] = {}
_last_prices: dict[str, dict[str, Any]] = {}


def _ensure_buffer(feed_id: str) -> deque:
    clean = feed_id.lower().replace("0x", "")
    if clean not in _price_buffers:
        _price_buffers[clean] = deque(maxlen=PYTH_BUFFER_DEPTH)
    return _price_buffers[clean]


def _parse_pyth_price(parsed: dict) -> dict[str, Any]:
    price_data = parsed.get("price", {})
    raw_price = int(price_data.get("price", "0"))
    expo = int(price_data.get("expo", 0))
    conf = int(price_data.get("conf", "0"))
    publish_time = int(price_data.get("publish_time", 0))

    ema = parsed.get("ema_price", {})
    ema_raw = int(ema.get("price", "0"))

    feed_id = parsed.get("id", "")
    meta = _feed_by_id.get(feed_id.lower().replace("0x", ""), {})
    attrs = meta.get("attributes", {})

    return {
        "price": raw_price * (10 ** expo),
        "confidence": conf * (10 ** expo),
        "ema_price": ema_raw * (10 ** expo),
        "expo": expo,
        "publish_time": publish_time,
        "feed_id": feed_id,
        "symbol": attrs.get("base", ""),
        "description": attrs.get("description", ""),
        "timestamp": time.time(),
    }


# ── 1. Feed discovery ──

def list_feeds(
    asset_type: str = "crypto",
    query: str | None = None,
) -> list[dict[str, Any]]:
    """List all Pyth price feeds, optionally filtered by search query.

    Caches the full feed list for PYTH_FEED_CACHE_TTL seconds.
    """
    global _feed_cache, _feed_cache_ts, _feed_by_id, _feed_by_symbol

    now = time.time()
    need_refresh = not _feed_cache or (now - _feed_cache_ts) > PYTH_FEED_CACHE_TTL

    if need_refresh:
        params: dict[str, str] = {"asset_type": asset_type}
        try:
            resp = _pool.get(f"{PYTH_HERMES_URL}/v2/price_feeds", params=params)
            resp.raise_for_status()
            _feed_cache = resp.json()
            _feed_cache_ts = now
            _feed_by_id = {
                f["id"].lower().replace("0x", ""): f for f in _feed_cache
            }
            _feed_by_symbol = {}
            for f in _feed_cache:
                base = f.get("attributes", {}).get("base", "").upper()
                quote = f.get("attributes", {}).get("quote_currency", "USD").upper()
                if base and quote == "USD":
                    _feed_by_symbol[base] = f
        except Exception:
            logger.warning("Failed to fetch Pyth feed list")
            if not _feed_cache:
                return []

    if query:
        q = query.lower()
        return [
            f for f in _feed_cache
            if q in f.get("attributes", {}).get("base", "").lower()
            or q in f.get("attributes", {}).get("description", "").lower()
            or q in f.get("attributes", {}).get("symbol", "").lower()
        ]

    return _feed_cache


def search_feeds(query: str) -> list[dict[str, Any]]:
    """Search Pyth feeds by token name via Hermes API (server-side filtering)."""
    params: dict[str, str] = {"asset_type": "crypto", "query": query}
    try:
        resp = _pool.get(f"{PYTH_HERMES_URL}/v2/price_feeds", params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.warning("Pyth feed search failed for query=%s", query)
        return list_feeds(query=query)


def resolve_feed_ids(symbols: list[str]) -> list[str]:
    """Resolve token symbols (e.g. ['SOL', 'BTC']) to Pyth feed IDs.

    Triggers a feed cache refresh if needed.
    """
    list_feeds()
    ids: list[str] = []
    for sym in symbols:
        upper = sym.upper()
        feed = _feed_by_symbol.get(upper)
        if feed:
            ids.append(feed["id"])
        elif len(sym) > 20:
            ids.append(sym.replace("0x", ""))
        else:
            logger.debug("No Pyth feed found for symbol: %s", sym)
    return ids


# ── 2. Live prices ──

def fetch_prices(feed_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch latest prices for arbitrary feed IDs from Pyth Hermes."""
    if not feed_ids:
        return []

    params = [("ids[]", fid) for fid in feed_ids]
    url = f"{PYTH_HERMES_URL}/v2/updates/price/latest"

    try:
        resp = _pool.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("Pyth price fetch failed for %d feeds", len(feed_ids))
        return []

    results: list[dict[str, Any]] = []
    for parsed in data.get("parsed", []):
        entry = _parse_pyth_price(parsed)
        results.append(entry)
        fid = parsed.get("id", "").lower().replace("0x", "")
        buf = _ensure_buffer(fid)
        buf.append(entry)
        _last_prices[fid] = entry

    return results


def fetch_prices_by_symbols(symbols: list[str]) -> list[dict[str, Any]]:
    """Convenience: resolve symbols then fetch prices."""
    ids = resolve_feed_ids(symbols)
    return fetch_prices(ids)


# ── 3. Historical prices ──

def fetch_historical(
    feed_ids: list[str],
    timestamp: int,
) -> list[dict[str, Any]]:
    """Fetch prices at a specific past unix timestamp."""
    if not feed_ids:
        return []

    params = [("ids[]", fid) for fid in feed_ids]
    url = f"{PYTH_HERMES_URL}/v2/updates/price/{timestamp}"

    try:
        resp = _pool.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("Pyth historical fetch failed for ts=%d", timestamp)
        return []

    return [_parse_pyth_price(p) for p in data.get("parsed", [])]


# ── 4. SSE streaming helpers ──

_stream_status: dict[str, Any] = {
    "active": False,
    "feed_ids": [],
    "events_received": 0,
    "started_at": None,
}


async def run_price_stream(feed_ids: list[str]) -> None:
    """Connect to Pyth SSE stream and buffer incoming prices.

    This is an async generator meant to be run as a background task.
    The SSE endpoint auto-closes after ~24h; we reconnect automatically.
    """
    if not feed_ids:
        return

    params = "&".join(f"ids[]={fid}" for fid in feed_ids)
    url = f"{PYTH_HERMES_URL}/v2/updates/price/stream?{params}"

    _stream_status["active"] = True
    _stream_status["feed_ids"] = feed_ids
    _stream_status["started_at"] = time.time()

    import asyncio

    while _stream_status["active"]:
        try:
            async with httpx.AsyncClient(timeout=PYTH_SSE_TIMEOUT) as client:
                async with client.stream("GET", url) as resp:
                    buffer = ""
                    async for chunk in resp.aiter_text():
                        buffer += chunk
                        while "\n\n" in buffer:
                            event_str, buffer = buffer.split("\n\n", 1)
                            _process_sse_event(event_str)
        except Exception:
            logger.warning("Pyth SSE stream disconnected, reconnecting in 3s")
            await asyncio.sleep(3)


def _process_sse_event(raw: str) -> None:
    """Parse a single SSE event and update price buffers."""
    import json as _json
    data_line = ""
    for line in raw.strip().split("\n"):
        if line.startswith("data:"):
            data_line = line[5:].strip()

    if not data_line:
        return

    try:
        payload = _json.loads(data_line)
    except Exception:
        return

    for parsed in payload.get("parsed", []):
        entry = _parse_pyth_price(parsed)
        fid = parsed.get("id", "").lower().replace("0x", "")
        buf = _ensure_buffer(fid)
        buf.append(entry)
        _last_prices[fid] = entry
        _stream_status["events_received"] += 1


def stop_stream() -> None:
    _stream_status["active"] = False


# ── 5. Query helpers ──

def get_latest_prices(symbols: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Return latest cached prices, keyed by symbol.

    If symbols is None, returns all cached prices.
    If cache is empty for requested symbols, does a live fetch.
    """
    if symbols is None:
        result: dict[str, dict[str, Any]] = {}
        for fid, entry in _last_prices.items():
            key = entry.get("symbol") or fid
            result[key] = entry
        return result

    ids = resolve_feed_ids(symbols)
    cached = {}
    missing_ids = []
    for fid in ids:
        clean = fid.lower().replace("0x", "")
        if clean in _last_prices:
            entry = _last_prices[clean]
            key = entry.get("symbol") or clean
            cached[key] = entry
        else:
            missing_ids.append(fid)

    if missing_ids:
        fresh = fetch_prices(missing_ids)
        for entry in fresh:
            key = entry.get("symbol") or entry.get("feed_id", "")
            cached[key] = entry

    return cached


def get_price_buffer(feed_id: str) -> list[dict[str, Any]]:
    """Return the ring-buffer history for a specific feed."""
    clean = feed_id.lower().replace("0x", "")
    buf = _price_buffers.get(clean)
    return list(buf) if buf else []


def get_oracle_status() -> dict[str, Any]:
    return {
        "hermes_url": PYTH_HERMES_URL,
        "total_feeds_known": len(_feed_cache),
        "feed_cache_age_s": round(time.time() - _feed_cache_ts, 1) if _feed_cache_ts else None,
        "prices_cached": len(_last_prices),
        "buffers_active": len(_price_buffers),
        "buffer_depth": PYTH_BUFFER_DEPTH,
        "stream": dict(_stream_status),
        "timestamp": time.time(),
    }
