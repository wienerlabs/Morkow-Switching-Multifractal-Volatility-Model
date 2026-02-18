"""DexScreener DEX data adapter for the Cortex Risk Engine.

Uses the official, free,
keyless DexScreener REST API (https://docs.dexscreener.com/api/reference).

Provides:
  1. Token price feeds (complementary to Birdeye/Pyth)
  2. Pair stats & liquidity depth for LVaR spread estimation
  3. New token pair discovery
  4. In-memory price cache with TTL
"""

__all__ = [
    "is_available",
    "get_token_price",
    "get_pair_liquidity",
    "extract_liquidity_metrics",
    "get_new_tokens",
    "get_cached_prices",
]

import logging
import time
from typing import Any

import httpx

from cortex.config import (
    DEXSCREENER_BASE_URL,
    DEXSCREENER_TIMEOUT,
    DEXSCREENER_MAX_RETRIES,
    DEXSCREENER_CACHE_TTL,
    DEXSCREENER_MIN_LIQUIDITY_USD,
)

logger = logging.getLogger(__name__)

_BASE = DEXSCREENER_BASE_URL.rstrip("/")
_price_cache: dict[str, dict[str, Any]] = {}


def _request(method: str, path: str, **kwargs) -> Any:
    """HTTP request with exponential backoff retry."""
    url = f"{_BASE}{path}"
    last_err: Exception | None = None
    for attempt in range(1, DEXSCREENER_MAX_RETRIES + 1):
        try:
            resp = httpx.request(method, url, timeout=DEXSCREENER_TIMEOUT, **kwargs)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            logger.warning("DexScreener %s %s attempt %d/%d failed: %s", method, path, attempt, DEXSCREENER_MAX_RETRIES, e)
            if attempt < DEXSCREENER_MAX_RETRIES:
                time.sleep(1.0 * attempt)
    raise last_err  # type: ignore[misc]


def is_available() -> bool:
    """DexScreener is always available (no auth required)."""
    return True


# ── 1. Token Price ──


def get_token_price(token_address: str) -> dict[str, Any]:
    """Fetch token price via DexScreener token lookup.

    Returns the highest-liquidity pair's price data.
    """
    pairs = _request("GET", f"/tokens/v1/solana/{token_address}")
    if not pairs:
        return {
            "source": "dexscreener",
            "token_address": token_address,
            "price_usd": None,
            "error": "no pairs found",
            "timestamp": time.time(),
        }

    # Pick the pair with the highest USD liquidity
    best = max(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0))

    result = {
        "source": "dexscreener",
        "token_address": token_address,
        "price_usd": float(best["priceUsd"]) if best.get("priceUsd") else None,
        "price_native": float(best["priceNative"]) if best.get("priceNative") else None,
        "pair_address": best.get("pairAddress"),
        "dex_id": best.get("dexId"),
        "liquidity_usd": float(best.get("liquidity", {}).get("usd", 0) or 0),
        "volume_24h": float(best.get("volume", {}).get("h24", 0) or 0),
        "raw": best,
        "timestamp": time.time(),
    }
    _price_cache[token_address] = result
    return result


# ── 2. Pair Stats & Liquidity Depth ──


def get_pair_liquidity(pair_address: str) -> dict[str, Any]:
    """Fetch pair info for LVaR liquidity depth estimation."""
    data = _request("GET", f"/latest/dex/pairs/solana/{pair_address}")
    pair = (data.get("pairs") or [None])[0] if isinstance(data, dict) else None

    if not pair:
        return {
            "source": "dexscreener",
            "pair_address": pair_address,
            "error": "pair not found",
            "timestamp": time.time(),
        }

    return {
        "source": "dexscreener",
        "pair_address": pair_address,
        "price_usd": float(pair["priceUsd"]) if pair.get("priceUsd") else None,
        "liquidity_usd": float(pair.get("liquidity", {}).get("usd", 0) or 0),
        "liquidity_base": float(pair.get("liquidity", {}).get("base", 0) or 0),
        "liquidity_quote": float(pair.get("liquidity", {}).get("quote", 0) or 0),
        "volume_24h": float(pair.get("volume", {}).get("h24", 0) or 0),
        "volume_6h": float(pair.get("volume", {}).get("h6", 0) or 0),
        "volume_1h": float(pair.get("volume", {}).get("h1", 0) or 0),
        "txns_24h": pair.get("txns", {}).get("h24", {}),
        "price_change_24h": float(pair.get("priceChange", {}).get("h24", 0) or 0),
        "dex_id": pair.get("dexId"),
        "raw": pair,
        "timestamp": time.time(),
    }


def extract_liquidity_metrics(pair_address: str) -> dict[str, Any]:
    """Extract LVaR-compatible spread and depth metrics from DexScreener pair data.

    Returns spread_pct, spread_vol_pct suitable for liquidity_adjusted_var().
    Uses Amihud illiquidity proxy: |return| / volume as spread estimate.
    """
    try:
        data = get_pair_liquidity(pair_address)
        if data.get("error"):
            return {"source": "dexscreener", "error": data["error"], "spread_pct": None, "spread_vol_pct": None}

        volume_24h = data.get("volume_24h", 0) or 0
        liquidity = data.get("liquidity_usd", 0) or 0

        if liquidity > 0 and volume_24h > 0:
            turnover = volume_24h / liquidity
            spread_pct = max(0.01, (1.0 / (1.0 + turnover)) * 2.0)
        elif volume_24h > 0:
            spread_pct = max(0.05, 1.0 / (volume_24h / 1000.0))
        else:
            spread_pct = 1.0  # conservative fallback

        spread_vol_pct = spread_pct * 0.3

        return {
            "source": "dexscreener",
            "pair_address": pair_address,
            "spread_pct": spread_pct,
            "spread_vol_pct": spread_vol_pct,
            "volume_24h": volume_24h,
            "liquidity": liquidity,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.warning("DexScreener liquidity metrics failed for %s: %s", pair_address, e)
        return {"source": "dexscreener", "error": str(e), "spread_pct": None, "spread_vol_pct": None}


# ── 3. New Token Discovery ──


def get_new_tokens(limit: int = 20, min_liquidity: bool = True) -> list[dict[str, Any]]:
    """Fetch recently created token pairs from DexScreener.

    Uses the token-boosts endpoint as a proxy for newly active tokens,
    then enriches with pair data filtered by minimum liquidity.
    """
    try:
        boosts = _request("GET", "/token-boosts/latest/v1")
        if not boosts:
            return []

        # Filter to Solana tokens only
        solana_tokens = [t for t in boosts if t.get("chainId") == "solana"][:limit * 2]

        results: list[dict[str, Any]] = []
        for token in solana_tokens:
            addr = token.get("tokenAddress", "")
            if not addr:
                continue
            try:
                pairs = _request("GET", f"/tokens/v1/solana/{addr}")
                if not pairs:
                    continue
                best = max(pairs, key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0))
                liq_usd = float(best.get("liquidity", {}).get("usd", 0) or 0)

                if min_liquidity and liq_usd < DEXSCREENER_MIN_LIQUIDITY_USD:
                    continue

                results.append({
                    "token_address": addr,
                    "pair_address": best.get("pairAddress"),
                    "dex_id": best.get("dexId"),
                    "price_usd": float(best["priceUsd"]) if best.get("priceUsd") else None,
                    "liquidity_usd": liq_usd,
                    "volume_24h": float(best.get("volume", {}).get("h24", 0) or 0),
                    "pair_created_at": best.get("pairCreatedAt"),
                    "base_token": best.get("baseToken", {}),
                    "timestamp": time.time(),
                    "meets_min_liquidity": liq_usd >= DEXSCREENER_MIN_LIQUIDITY_USD,
                })
            except Exception as e:
                logger.debug("Skipping token %s: %s", addr[:12], e)
                continue

            if len(results) >= limit:
                break

        return results
    except Exception as e:
        logger.warning("DexScreener new tokens fetch failed: %s", e)
        return []


# ── 4. Cache Accessors ──


def get_cached_prices() -> dict[str, dict[str, Any]]:
    """Return all cached token prices, evicting stale entries."""
    now = time.time()
    stale = [k for k, v in _price_cache.items() if now - v.get("timestamp", 0) > DEXSCREENER_CACHE_TTL]
    for k in stale:
        del _price_cache[k]
    return dict(_price_cache)
