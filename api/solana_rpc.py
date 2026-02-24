"""Shared Solana RPC utilities for vault/staking/vesting route files.

Provides reusable helpers for RPC calls, PDA derivation, Anchor account
deserialization, and in-memory TTL caching. All functions are synchronous
to match the existing FastAPI sync route handler pattern.
"""

__all__ = [
    "get_rpc_url",
    "fetch_account_info",
    "fetch_multiple_accounts",
    "derive_pda",
    "deserialize_anchor_account",
    "cached",
]

import base64
import functools
import logging
import time
from typing import Any, Callable

import httpx
from fastapi import HTTPException
from solders.pubkey import Pubkey  # type: ignore[import-untyped]

from cortex.config import HELIUS_RPC_URL, ONCHAIN_CACHE_TTL, ONCHAIN_HTTP_TIMEOUT

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# In-memory TTL cache
# ---------------------------------------------------------------------------

_cache: dict[str, tuple[float, Any]] = {}


def cached(ttl_seconds: int = 30):
    """Decorator that caches function results by args with a TTL."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            key = f"{fn.__module__}.{fn.__qualname__}:{args}:{sorted(kwargs.items())}"
            now = time.monotonic()
            hit = _cache.get(key)
            if hit is not None:
                expires_at, value = hit
                if now < expires_at:
                    return value
            result = fn(*args, **kwargs)
            _cache[key] = (now + ttl_seconds, result)
            return result

        return wrapper

    return decorator


def _evict_expired() -> None:
    """Remove expired entries to prevent unbounded growth."""
    now = time.monotonic()
    expired = [k for k, (exp, _) in _cache.items() if now >= exp]
    for k in expired:
        del _cache[k]


# ---------------------------------------------------------------------------
# RPC helpers
# ---------------------------------------------------------------------------


def get_rpc_url() -> str:
    """Return configured Helius RPC URL or raise 503."""
    if not HELIUS_RPC_URL:
        raise HTTPException(
            status_code=503,
            detail="Solana RPC not configured (HELIUS_RPC_URL is empty)",
        )
    return HELIUS_RPC_URL


def _rpc_post(payload: dict) -> dict:
    """POST a JSON-RPC request with retries, matching helius_holders.py pattern."""
    url = get_rpc_url()
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = httpx.post(url, json=payload, timeout=ONCHAIN_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.warning("RPC error: %s", data["error"])
            return data
        except Exception as e:
            last_err = e
            logger.warning(
                "Solana RPC attempt %d/%d failed: %s", attempt, _MAX_RETRIES, e
            )
            if attempt < _MAX_RETRIES:
                time.sleep(1 * attempt)
    raise last_err  # type: ignore[misc]


@cached(ttl_seconds=30)
def fetch_account_info(address: str) -> dict | None:
    """Fetch a single account via getAccountInfo. Returns decoded data or None."""
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [address, {"encoding": "base64"}],
    }
    result = _rpc_post(payload).get("result")
    if result is None or result.get("value") is None:
        return None

    value = result["value"]
    raw_data = value.get("data")
    decoded: bytes | None = None
    if isinstance(raw_data, list) and len(raw_data) >= 1:
        try:
            decoded = base64.b64decode(raw_data[0])
        except Exception:
            logger.warning("Failed to base64-decode account data for %s", address)

    return {
        "data": decoded,
        "owner": value.get("owner"),
        "lamports": value.get("lamports"),
        "executable": value.get("executable", False),
    }


def fetch_multiple_accounts(addresses: list[str]) -> list[dict | None]:
    """Batch-fetch accounts via getMultipleAccounts."""
    if not addresses:
        return []

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getMultipleAccounts",
        "params": [addresses, {"encoding": "base64"}],
    }
    result = _rpc_post(payload).get("result")
    if result is None or result.get("value") is None:
        return [None] * len(addresses)

    out: list[dict | None] = []
    for value in result["value"]:
        if value is None:
            out.append(None)
            continue

        raw_data = value.get("data")
        decoded: bytes | None = None
        if isinstance(raw_data, list) and len(raw_data) >= 1:
            try:
                decoded = base64.b64decode(raw_data[0])
            except Exception:
                logger.warning("Failed to base64-decode account data in batch")

        out.append({
            "data": decoded,
            "owner": value.get("owner"),
            "lamports": value.get("lamports"),
            "executable": value.get("executable", False),
        })
    return out


# ---------------------------------------------------------------------------
# PDA derivation
# ---------------------------------------------------------------------------


def derive_pda(seeds: list[bytes], program_id: str) -> str:
    """Derive a Program Derived Address using solders.

    Returns the PDA as a base58 string. The bump seed is discarded.
    """
    pid = Pubkey.from_string(program_id)
    pda, _bump = Pubkey.find_program_address(seeds, pid)
    return str(pda)


# ---------------------------------------------------------------------------
# Anchor account deserialization helper
# ---------------------------------------------------------------------------

_ANCHOR_DISCRIMINATOR_LEN = 8


def deserialize_anchor_account(
    data: bytes, expected_discriminator: bytes | None = None
) -> bytes:
    """Strip the 8-byte Anchor discriminator and return the remaining bytes.

    If expected_discriminator is provided, verifies it matches before returning.
    Raises ValueError on mismatch or insufficient data.
    """
    if len(data) < _ANCHOR_DISCRIMINATOR_LEN:
        raise ValueError(
            f"Account data too short ({len(data)} bytes), "
            f"need at least {_ANCHOR_DISCRIMINATOR_LEN} for Anchor discriminator"
        )

    disc = data[:_ANCHOR_DISCRIMINATOR_LEN]
    if expected_discriminator is not None and disc != expected_discriminator:
        raise ValueError(
            f"Discriminator mismatch: expected {expected_discriminator.hex()}, "
            f"got {disc.hex()}"
        )

    return data[_ANCHOR_DISCRIMINATOR_LEN:]
