"""Macro market indicators — BTC dominance, fear/greed index, total market cap.

Fetches from CoinGecko and Alternative.me with TTL-based caching to respect
rate limits.  All functions are synchronous and safe for FastAPI sync routes.
"""

import logging
import time
from typing import Any

from cortex.config import COINGECKO_BASE, FEAR_GREED_URL, MACRO_CACHE_TTL
from cortex.data.rpc_failover import get_resilient_pool

logger = logging.getLogger(__name__)

_pool = get_resilient_pool()

_macro_cache: dict[str, Any] = {}
_cache_ts: float = 0.0


def get_fear_greed() -> dict[str, Any]:
    """Fetch the crypto Fear & Greed Index from Alternative.me."""
    try:
        resp = _pool.get(FEAR_GREED_URL)
        resp.raise_for_status()
        data = resp.json()
        entry = data.get("data", [{}])[0]
        return {
            "value": int(entry.get("value", 50)),
            "classification": entry.get("value_classification", "Neutral"),
            "timestamp": int(entry.get("timestamp", time.time())),
        }
    except Exception:
        logger.warning("Fear/Greed API request failed")
        return {"value": 50, "classification": "Neutral", "timestamp": int(time.time())}


def get_btc_dominance() -> dict[str, Any]:
    """Fetch BTC dominance and total market cap from CoinGecko."""
    try:
        resp = _pool.get(f"{COINGECKO_BASE}/global")
        resp.raise_for_status()
        data = resp.json().get("data", {})
        market_cap_pct = data.get("market_cap_percentage", {})
        return {
            "btc_dominance": round(market_cap_pct.get("btc", 0.0), 2),
            "eth_dominance": round(market_cap_pct.get("eth", 0.0), 2),
            "total_market_cap_usd": data.get("total_market_cap", {}).get("usd", 0),
            "total_volume_24h_usd": data.get("total_volume", {}).get("usd", 0),
            "active_cryptocurrencies": data.get("active_cryptocurrencies", 0),
        }
    except Exception:
        logger.warning("CoinGecko global API request failed")
        return {
            "btc_dominance": 0.0,
            "eth_dominance": 0.0,
            "total_market_cap_usd": 0,
            "total_volume_24h_usd": 0,
            "active_cryptocurrencies": 0,
        }


def get_macro_indicators() -> dict[str, Any]:
    """Return all macro indicators with TTL caching."""
    global _macro_cache, _cache_ts

    now = time.time()
    if _macro_cache and (now - _cache_ts) < MACRO_CACHE_TTL:
        return _macro_cache

    fear_greed = get_fear_greed()
    btc_dom = get_btc_dominance()

    risk_level = "low"
    fg_val = fear_greed["value"]
    if fg_val <= 25:
        risk_level = "extreme_fear"
    elif fg_val <= 40:
        risk_level = "fear"
    elif fg_val >= 75:
        risk_level = "extreme_greed"
    elif fg_val >= 60:
        risk_level = "greed"
    else:
        risk_level = "neutral"

    result = {
        "fear_greed": fear_greed,
        "btc_dominance": btc_dom,
        "risk_level": risk_level,
        "timestamp": now,
    }

    _macro_cache = result
    _cache_ts = now
    return result

