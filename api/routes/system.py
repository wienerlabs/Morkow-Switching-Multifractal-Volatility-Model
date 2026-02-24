"""System health endpoint — data source availability and latency."""

import logging
import time

from fastapi import APIRouter

router = APIRouter(tags=["system"])

logger = logging.getLogger(__name__)

_SOURCES = {
    "pyth": {"label": "Pyth Network", "config_key": "PYTH_HERMES_URL"},
    "birdeye": {"label": "Birdeye", "config_key": "BIRDEYE_BASE"},
    "jupiter": {"label": "Jupiter", "config_key": "JUPITER_API_URL"},
    "helius": {"label": "Helius RPC", "config_key": "HELIUS_RPC_URL"},
    "coingecko": {"label": "CoinGecko", "config_key": "COINGECKO_BASE"},
    "dexscreener": {"label": "DexScreener", "config_key": "DEXSCREENER_BASE_URL"},
}


@router.get("/system/health", summary="Data source health status")
def get_system_health():
    """Return health status and latency for all external data sources.

    Leverages the ResilientClient's per-host health tracking to report
    status without making additional network calls.
    """
    from cortex.config import (
        BIRDEYE_BASE,
        COINGECKO_BASE,
        DEXSCREENER_BASE_URL,
        HELIUS_RPC_URL,
        JUPITER_API_URL,
        PYTH_HERMES_URL,
    )
    from cortex.data.rpc_failover import get_resilient_pool

    pool = get_resilient_pool()
    all_health = pool.get_all_health()
    endpoint_map = {e["host"]: e for e in all_health.get("endpoints", [])}

    url_map = {
        "pyth": PYTH_HERMES_URL,
        "birdeye": BIRDEYE_BASE,
        "jupiter": JUPITER_API_URL,
        "helius": HELIUS_RPC_URL,
        "coingecko": COINGECKO_BASE,
        "dexscreener": DEXSCREENER_BASE_URL,
    }

    sources = {}
    for key, url in url_map.items():
        meta = _SOURCES[key]
        if not url:
            sources[key] = {
                "label": meta["label"],
                "status": "unconfigured",
                "url": "",
                "latency_ms": None,
                "success_rate": None,
            }
            continue

        try:
            from httpx import URL
            host = str(URL(url).host)
        except Exception:
            host = url[:40]

        health = endpoint_map.get(host)
        if health:
            sources[key] = {
                "label": meta["label"],
                "status": health["status"],
                "url": url,
                "latency_ms": health["avg_latency_ms"],
                "success_rate": health["success_rate"],
                "total_requests": health["total_requests"],
                "in_cooldown": health["in_cooldown"],
            }
        else:
            sources[key] = {
                "label": meta["label"],
                "status": "unknown",
                "url": url,
                "latency_ms": None,
                "success_rate": None,
            }

    any_down = any(s["status"] == "down" for s in sources.values())
    any_degraded = any(s["status"] == "degraded" for s in sources.values())
    all_unconfigured = all(s["status"] == "unconfigured" for s in sources.values())

    if all_unconfigured:
        overall = "unconfigured"
    elif any_down:
        overall = "degraded"
    elif any_degraded:
        overall = "warning"
    else:
        overall = "healthy"

    return {
        "status": overall,
        "sources": sources,
        "timestamp": time.time(),
    }

