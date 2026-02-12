"""Pyth Network oracle integration — real-time price feeds via SSE + REST fallback.

Connects to Pyth Hermes API for sub-second price updates.  Falls back to
Birdeye REST when Pyth is unavailable.  Maintains a per-token ring buffer
of recent prices for downstream volatility calculations.
"""

import asyncio
import atexit
import json
import logging
import time
from collections import deque
from typing import Any

import httpx

from cortex.config import (
    BIRDEYE_BASE,
    PYTH_BUFFER_DEPTH,
    PYTH_HERMES_URL,
    PYTH_PRICE_FEEDS,
    PYTH_SSE_TIMEOUT,
    SOLANA_HTTP_TIMEOUT,
)

logger = logging.getLogger(__name__)

_pool = httpx.Client(timeout=SOLANA_HTTP_TIMEOUT)
atexit.register(_pool.close)

_price_buffers: dict[str, deque[dict[str, Any]]] = {
    token: deque(maxlen=PYTH_BUFFER_DEPTH) for token in PYTH_PRICE_FEEDS
}
_last_prices: dict[str, dict[str, Any]] = {}


def _parse_pyth_price(parsed: dict) -> dict[str, Any]:
    """Convert Pyth parsed price object to a normalised dict."""
    price_data = parsed.get("price", {})
    raw_price = int(price_data.get("price", "0"))
    expo = int(price_data.get("expo", 0))
    conf = int(price_data.get("conf", "0"))
    publish_time = int(price_data.get("publish_time", 0))

    ema = parsed.get("ema_price", {})
    ema_raw = int(ema.get("price", "0"))

    return {
        "price": raw_price * (10 ** expo),
        "confidence": conf * (10 ** expo),
        "ema_price": ema_raw * (10 ** expo),
        "publish_time": publish_time,
        "feed_id": parsed.get("id", ""),
        "timestamp": time.time(),
    }


def _feed_id_to_token(feed_id: str) -> str | None:
    clean = feed_id.lower().replace("0x", "")
    for token, fid in PYTH_PRICE_FEEDS.items():
        if fid.lower().replace("0x", "") == clean:
            return token
    return None


def fetch_pyth_prices_rest() -> dict[str, dict[str, Any]]:
    """Fetch latest prices from Pyth Hermes REST endpoint."""
    if not PYTH_PRICE_FEEDS:
        return {}

    ids_param = "&".join(
        f"ids[]={fid}" for fid in PYTH_PRICE_FEEDS.values()
    )
    url = f"{PYTH_HERMES_URL}/v2/updates/price/latest?{ids_param}"

    try:
        resp = _pool.get(url)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        logger.warning("Pyth REST fetch failed, trying Birdeye fallback")
        return _fetch_birdeye_fallback()

    result: dict[str, dict[str, Any]] = {}
    for parsed in data.get("parsed", []):
        token = _feed_id_to_token(parsed.get("id", ""))
        if token:
            entry = _parse_pyth_price(parsed)
            result[token] = entry
            _price_buffers[token].append(entry)
            _last_prices[token] = entry

    return result


def _fetch_birdeye_fallback() -> dict[str, dict[str, Any]]:
    """Fallback: fetch prices from Birdeye REST API."""
    from cortex.data.solana import TOKEN_REGISTRY

    result: dict[str, dict[str, Any]] = {}
    for token in PYTH_PRICE_FEEDS:
        mint = TOKEN_REGISTRY.get(token, {}).get("mint")
        if not mint:
            continue
        try:
            resp = _pool.get(
                f"{BIRDEYE_BASE}/defi/price",
                params={"address": mint},
                headers={"X-API-KEY": ""},
            )
            if resp.status_code == 200:
                body = resp.json()
                price = body.get("data", {}).get("value", 0.0)
                entry = {
                    "price": price,
                    "confidence": 0.0,
                    "ema_price": price,
                    "publish_time": int(time.time()),
                    "feed_id": "birdeye_fallback",
                    "timestamp": time.time(),
                    "source": "birdeye",
                }
                result[token] = entry
                _price_buffers[token].append(entry)
                _last_prices[token] = entry
        except Exception:
            logger.warning("Birdeye fallback failed for %s", token)
    return result



def get_latest_prices() -> dict[str, dict[str, Any]]:
    """Return the most recent price for each tracked token.

    Tries cached values first; falls back to a fresh REST fetch.
    """
    if _last_prices:
        return dict(_last_prices)
    return fetch_pyth_prices_rest()


def get_price_history(token: str) -> list[dict[str, Any]]:
    """Return the ring-buffer history for *token*."""
    token = token.upper()
    buf = _price_buffers.get(token)
    if buf is None:
        return []
    return list(buf)


def get_oracle_status() -> dict[str, Any]:
    """Health / status summary for the oracle subsystem."""
    return {
        "tracked_tokens": list(PYTH_PRICE_FEEDS.keys()),
        "buffer_depth": PYTH_BUFFER_DEPTH,
        "prices_cached": len(_last_prices),
        "buffer_sizes": {t: len(b) for t, b in _price_buffers.items()},
        "hermes_url": PYTH_HERMES_URL,
        "timestamp": time.time(),
    }
