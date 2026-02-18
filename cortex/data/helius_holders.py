"""Helius DAS holder distribution adapter — token holder concentration risk.

Uses Helius getTokenAccounts (DAS API) to fetch all holders of a token mint,
then computes concentration metrics for rug-pull risk assessment.

API docs: https://www.helius.dev/docs/api-reference/das/gettokenaccounts
"""

__all__ = ["is_available", "get_holder_data", "get_concentration_risk"]

import logging
import time
from typing import Any

import httpx

from cortex.config import (
    HELIUS_API_KEY,
    HELIUS_RPC_URL,
    ONCHAIN_CACHE_TTL,
    ONCHAIN_HTTP_TIMEOUT,
)

logger = logging.getLogger(__name__)

_PAGE_LIMIT = 1000
_MAX_PAGES = 50  # safety cap: 50k holders max per query
_MAX_RETRIES = 3

_cache: dict[str, dict[str, Any]] = {}
_cache_ts: dict[str, float] = {}


def is_available() -> bool:
    return bool(HELIUS_API_KEY)


def _fetch_all_accounts(token_mint: str) -> list[dict[str, Any]]:
    """Paginate through all token accounts for a mint via Helius DAS."""
    if not HELIUS_RPC_URL:
        raise RuntimeError("HELIUS_RPC_URL not configured")

    accounts: list[dict[str, Any]] = []
    page = 1

    while page <= _MAX_PAGES:
        payload = {
            "jsonrpc": "2.0",
            "id": "helius-holders",
            "method": "getTokenAccounts",
            "params": {
                "page": page,
                "limit": _PAGE_LIMIT,
                "displayOptions": {},
                "mint": token_mint,
            },
        }

        resp = _post_with_retry(payload)
        batch = resp.get("result", {}).get("token_accounts", [])

        if not batch:
            break

        accounts.extend(batch)
        page += 1

    return accounts


def _post_with_retry(payload: dict) -> dict:
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = httpx.post(
                HELIUS_RPC_URL,
                json=payload,
                timeout=ONCHAIN_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            logger.warning(
                "Helius DAS attempt %d/%d failed: %s", attempt, _MAX_RETRIES, e
            )
            if attempt < _MAX_RETRIES:
                time.sleep(1 * attempt)
    raise last_err  # type: ignore[misc]


def _compute_concentration(holders: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute concentration metrics from sorted holder list.

    Returns top10_pct, top50_pct, hhi (Herfindahl-Hirschman Index),
    and a qualitative concentration_risk label.
    """
    if not holders:
        return {
            "top10_pct": 0.0,
            "top50_pct": 0.0,
            "hhi": 0.0,
            "concentration_risk": "unknown",
        }

    total = sum(h["amount"] for h in holders)
    if total == 0:
        return {
            "top10_pct": 0.0,
            "top50_pct": 0.0,
            "hhi": 0.0,
            "concentration_risk": "unknown",
        }

    # Sort descending by amount
    sorted_holders = sorted(holders, key=lambda h: h["amount"], reverse=True)

    # Top-N percentage
    top10_sum = sum(h["amount"] for h in sorted_holders[:10])
    top50_sum = sum(h["amount"] for h in sorted_holders[:50])
    top10_pct = round(top10_sum / total * 100, 2)
    top50_pct = round(top50_sum / total * 100, 2)

    # HHI: sum of squared market shares (0-10000 scale)
    shares = [h["amount"] / total * 100 for h in sorted_holders]
    hhi = round(sum(s * s for s in shares), 2)

    # Risk label
    if top10_pct > 80:
        risk = "critical"
    elif top10_pct > 50:
        risk = "high"
    elif top10_pct > 30:
        risk = "medium"
    else:
        risk = "low"

    return {
        "top10_pct": top10_pct,
        "top50_pct": top50_pct,
        "hhi": hhi,
        "concentration_risk": risk,
    }


def get_holder_data(token_mint: str) -> dict[str, Any]:
    """Fetch holder distribution with concentration risk metrics.

    Returns:
        {
            "source": "helius_das",
            "token_mint": "...",
            "holders": [{"owner": str, "amount": int, "pct": float}, ...],
            "total_holders": int,
            "top10_pct": float,
            "top50_pct": float,
            "hhi": float,
            "concentration_risk": "low"|"medium"|"high"|"critical",
            "timestamp": float,
        }
    """
    if not is_available():
        return {
            "source": "helius_das",
            "token_mint": token_mint,
            "error": "HELIUS_API_KEY not configured",
            "holders": [],
            "total_holders": 0,
            "top10_pct": 0.0,
            "top50_pct": 0.0,
            "hhi": 0.0,
            "concentration_risk": "unknown",
            "timestamp": time.time(),
        }

    # Check cache
    now = time.time()
    if token_mint in _cache and (now - _cache_ts.get(token_mint, 0)) < ONCHAIN_CACHE_TTL:
        return _cache[token_mint]

    try:
        raw_accounts = _fetch_all_accounts(token_mint)
    except Exception as e:
        logger.warning("Helius holder fetch failed for %s: %s", token_mint, e)
        return {
            "source": "helius_das",
            "token_mint": token_mint,
            "error": str(e),
            "holders": [],
            "total_holders": 0,
            "top10_pct": 0.0,
            "top50_pct": 0.0,
            "hhi": 0.0,
            "concentration_risk": "unknown",
            "timestamp": time.time(),
        }

    # Aggregate by owner (a wallet can have multiple token accounts)
    owner_totals: dict[str, int] = {}
    for acct in raw_accounts:
        owner = acct.get("owner", "unknown")
        amount = int(acct.get("amount", 0))
        owner_totals[owner] = owner_totals.get(owner, 0) + amount

    # Build sorted holder list
    holder_list = [
        {"owner": owner, "amount": amount}
        for owner, amount in owner_totals.items()
    ]
    holder_list.sort(key=lambda h: h["amount"], reverse=True)

    total_supply = sum(h["amount"] for h in holder_list)

    # Add percentage to each holder
    for h in holder_list:
        h["pct"] = round(h["amount"] / total_supply * 100, 4) if total_supply > 0 else 0.0

    concentration = _compute_concentration(holder_list)

    result: dict[str, Any] = {
        "source": "helius_das",
        "token_mint": token_mint,
        "holders": holder_list,
        "total_holders": len(holder_list),
        **concentration,
        "timestamp": time.time(),
    }

    _cache[token_mint] = result
    _cache_ts[token_mint] = now
    return result
