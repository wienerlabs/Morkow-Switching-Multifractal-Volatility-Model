"""Webacy API client — token safety, wallet sanctions, and wallet risk scoring.

Wraps the Webacy REST API (https://api.webacy.com) for the Guardian pipeline.
On any error (timeout, HTTP failure, missing API key), functions return safe
defaults so trading is never blocked by a Webacy outage.
"""

__all__ = [
    "is_webacy_enabled",
    "check_token_safety",
    "check_wallet_sanctions",
    "check_wallet_risk",
]

import time
from typing import Any

import httpx
import structlog

from cortex.config import (
    WEBACY_API_KEY,
    WEBACY_BASE_URL,
    WEBACY_CACHE_TTL,
    WEBACY_ENABLED,
    WEBACY_HARD_VETO_SCORE,
    WEBACY_TIMEOUT,
)

log = structlog.get_logger()

_BASE = WEBACY_BASE_URL.rstrip("/")

# Simple TTL cache: key → (timestamp, result)
_token_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _headers() -> dict[str, str]:
    return {"x-api-key": WEBACY_API_KEY}


def _cache_get(key: str) -> dict[str, Any] | None:
    entry = _token_cache.get(key)
    if entry is None:
        return None
    ts, result = entry
    if time.monotonic() - ts > WEBACY_CACHE_TTL:
        del _token_cache[key]
        return None
    return result


def _cache_set(key: str, result: dict[str, Any]) -> None:
    _token_cache[key] = (time.monotonic(), result)


def is_webacy_enabled() -> bool:
    return WEBACY_ENABLED and bool(WEBACY_API_KEY)


def _safe_token_default(mint_address: str) -> dict[str, Any]:
    return {
        "safe": True,
        "risk_score": 0,
        "mintable": False,
        "freezable": False,
        "sniper_pct": 0.0,
        "bundler_pct": 0.0,
        "holder_count": 0,
        "flags": [],
        "source": "webacy",
        "mint_address": mint_address,
        "fallback": True,
    }


def _safe_sanctions_default() -> dict[str, Any]:
    return {"sanctioned": False, "source": None}


def _safe_wallet_default() -> dict[str, Any]:
    return {"risk_score": 0, "threat_categories": [], "is_high_risk": False}


async def check_token_safety(mint_address: str) -> dict[str, Any]:
    """Check token safety via Webacy trading-lite endpoint.

    Returns a normalised dict with risk_score 0-100, safety flags, and a
    boolean ``safe`` field (True when risk_score < WEBACY_HARD_VETO_SCORE).
    """
    if not is_webacy_enabled():
        return _safe_token_default(mint_address)

    cache_key = f"token:{mint_address}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    try:
        async with httpx.AsyncClient(timeout=WEBACY_TIMEOUT) as client:
            resp = await client.get(
                f"{_BASE}/trading-lite/{mint_address}",
                params={"chain": "sol"},
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        log.warning("webacy.check_token_safety failed", mint_address=mint_address, exc_info=True)
        return _safe_token_default(mint_address)

    mintable = bool(data.get("mintable", False))
    freezable = bool(data.get("freezable", False))
    sniper_pct = float(data.get("sniper_percentage", 0))
    bundler_pct = float(data.get("bundler_percentage", 0))
    holder_count = int(data.get("holder_count", 0))

    risk_score = 0
    flags: list[str] = []

    if mintable:
        risk_score += 30
        flags.append("mintable")
    if freezable:
        risk_score += 25
        flags.append("freezable")
    if sniper_pct > 20:
        risk_score += 20
        flags.append("high_sniper_pct")
    if bundler_pct > 15:
        risk_score += 15
        flags.append("high_bundler_pct")
    if holder_count < 50:
        risk_score += 10
        flags.append("low_holder_count")

    risk_score = min(risk_score, 100)

    result: dict[str, Any] = {
        "safe": risk_score < WEBACY_HARD_VETO_SCORE,
        "risk_score": risk_score,
        "mintable": mintable,
        "freezable": freezable,
        "sniper_pct": sniper_pct,
        "bundler_pct": bundler_pct,
        "holder_count": holder_count,
        "flags": flags,
        "source": "webacy",
        "mint_address": mint_address,
        "fallback": False,
    }

    _cache_set(cache_key, result)
    return result


async def check_wallet_sanctions(address: str) -> dict[str, Any]:
    """Check if a wallet address is sanctioned (OFAC/SDN).

    Returns ``{"sanctioned": bool, "source": str | None}``.
    """
    if not is_webacy_enabled():
        return _safe_sanctions_default()

    try:
        async with httpx.AsyncClient(timeout=WEBACY_TIMEOUT) as client:
            resp = await client.get(
                f"{_BASE}/addresses/sanctioned/{address}",
                params={"chain": "sol"},
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        log.warning("webacy.check_wallet_sanctions failed", address=address, exc_info=True)
        return _safe_sanctions_default()

    return {
        "sanctioned": bool(data.get("sanctioned", False)),
        "source": data.get("source"),
    }


async def check_wallet_risk(address: str) -> dict[str, Any]:
    """Assess wallet-level risk score (0-100) with threat categorisation.

    Returns ``{"risk_score": int, "threat_categories": list[str], "is_high_risk": bool}``.
    """
    if not is_webacy_enabled():
        return _safe_wallet_default()

    try:
        async with httpx.AsyncClient(timeout=WEBACY_TIMEOUT) as client:
            resp = await client.get(
                f"{_BASE}/addresses/{address}",
                params={"chain": "sol"},
                headers=_headers(),
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        log.warning("webacy.check_wallet_risk failed", address=address, exc_info=True)
        return _safe_wallet_default()

    risk_score = int(data.get("risk_score", 0))
    threat_categories = list(data.get("threat_categories", []))

    return {
        "risk_score": risk_score,
        "threat_categories": threat_categories,
        "is_high_risk": risk_score >= WEBACY_HARD_VETO_SCORE,
    }
