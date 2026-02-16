"""Resilient HTTP client with retry, backoff, and endpoint health tracking.

Mirrors the TypeScript connection.ts failover pattern for the Python backend:
  - Automatic retry with exponential backoff on transient failures
  - Per-endpoint health tracking (latency, success rate, status)
  - Cooldown for repeatedly failing endpoints
  - Shared singleton for all Solana data API calls
"""

import asyncio
import atexit
import hashlib
import json as _json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from cortex.config import (
    SOLANA_HTTP_TIMEOUT,
    SOLANA_MAX_CONNECTIONS,
    SOLANA_MAX_KEEPALIVE,
)

logger = logging.getLogger(__name__)

# ── Config ──

MAX_RETRIES = int(os.environ.get("RPC_MAX_RETRIES", "3"))
BACKOFF_BASE_MS = float(os.environ.get("RPC_BACKOFF_BASE_MS", "500"))
BACKOFF_MAX_MS = float(os.environ.get("RPC_BACKOFF_MAX_MS", "5000"))
COOLDOWN_SECONDS = float(os.environ.get("RPC_COOLDOWN_SECONDS", "60"))
MAX_FAILURES_BEFORE_COOLDOWN = int(os.environ.get("RPC_MAX_FAILURES", "5"))
LATENCY_WINDOW_SIZE = int(os.environ.get("RPC_LATENCY_WINDOW", "20"))

# HTTP status codes that are retryable
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class EndpointHealth:
    """Tracks health metrics for a single API host."""

    host: str
    fail_count: int = 0
    last_failure: float = 0.0
    total_requests: int = 0
    success_count: int = 0
    last_success_at: float | None = None
    last_failure_at: float | None = None
    latency_window: deque = field(
        default_factory=lambda: deque(maxlen=LATENCY_WINDOW_SIZE)
    )

    @property
    def avg_latency_ms(self) -> float:
        if not self.latency_window:
            return 0.0
        return sum(self.latency_window) / len(self.latency_window)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return self.success_count / self.total_requests

    @property
    def status(self) -> str:
        if self.fail_count >= MAX_FAILURES_BEFORE_COOLDOWN:
            return "down"
        if self.fail_count > 0 or (self.total_requests > 10 and self.success_rate < 0.9):
            return "degraded"
        return "healthy"

    @property
    def in_cooldown(self) -> bool:
        if self.fail_count < MAX_FAILURES_BEFORE_COOLDOWN:
            return False
        return (time.time() - self.last_failure) < COOLDOWN_SECONDS

    def record_success(self, latency_ms: float) -> None:
        self.success_count += 1
        self.total_requests += 1
        self.last_success_at = time.time()
        self.fail_count = 0
        self.latency_window.append(latency_ms)
        _fire_and_forget_write(self.host, self)

    def record_failure(self, latency_ms: float) -> None:
        self.fail_count += 1
        self.last_failure = time.time()
        self.last_failure_at = time.time()
        self.total_requests += 1
        self.latency_window.append(latency_ms)
        _fire_and_forget_write(self.host, self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "status": self.status,
            "total_requests": self.total_requests,
            "success_count": self.success_count,
            "fail_count": self.fail_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "success_rate": round(self.success_rate, 3),
            "in_cooldown": self.in_cooldown,
            "last_success_at": self.last_success_at,
            "last_failure_at": self.last_failure_at,
        }


class ResilientClient:
    """Drop-in replacement for httpx.Client with retry + health tracking.

    Tracks per-host health so callers can check which APIs are degraded.
    Retries transient failures (429, 5xx, connection errors) with backoff.

    Usage:
        pool = ResilientClient()
        resp = pool.get("https://public-api.birdeye.so/defi/price", params={...})
    """

    def __init__(
        self,
        timeout: float = SOLANA_HTTP_TIMEOUT,
        max_connections: int = SOLANA_MAX_CONNECTIONS,
        max_keepalive: int = SOLANA_MAX_KEEPALIVE,
        max_retries: int = MAX_RETRIES,
    ):
        self._client = httpx.Client(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive,
            ),
        )
        self._max_retries = max_retries
        self._health: dict[str, EndpointHealth] = {}
        atexit.register(self.close)

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass

    def _get_host(self, url: str) -> str:
        try:
            parsed = httpx.URL(url)
            return str(parsed.host)
        except Exception:
            return url[:40]

    def _get_health(self, host: str) -> EndpointHealth:
        if host not in self._health:
            self._health[host] = EndpointHealth(host=host)
        return self._health[host]

    def _should_skip(self, host: str) -> bool:
        """Check if host is in cooldown or flagged down by other process."""
        if is_endpoint_flagged_down(host):
            return True
        health = self._health.get(host)
        if health is None:
            return False
        return health.in_cooldown

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        params: Any = None,
        json: Any = None,
        content: Any = None,
        max_retries: int | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """Make an HTTP request with automatic retry and backoff.

        Raises the last exception if all retries are exhausted.
        """
        host = self._get_host(url)
        health = self._get_health(host)
        retries = max_retries if max_retries is not None else self._max_retries
        last_exc: Exception | None = None

        if self._should_skip(host):
            logger.warning(
                "[RPC-PY] Host %s is in cooldown, attempting anyway", host
            )

        for attempt in range(retries + 1):
            start = time.time()
            try:
                resp = self._client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                    content=content,
                    **kwargs,
                )
                latency_ms = (time.time() - start) * 1000

                if resp.status_code in _RETRYABLE_STATUS:
                    health.record_failure(latency_ms)
                    if attempt < retries:
                        backoff = _calc_backoff(attempt)
                        logger.warning(
                            "[RPC-PY] Retryable %d from %s (attempt %d/%d, backoff %.0fms)",
                            resp.status_code,
                            host,
                            attempt + 1,
                            retries + 1,
                            backoff,
                        )
                        time.sleep(backoff / 1000)
                        continue
                    resp.raise_for_status()

                health.record_success(latency_ms)
                return resp

            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout) as exc:
                latency_ms = (time.time() - start) * 1000
                health.record_failure(latency_ms)
                last_exc = exc
                if attempt < retries:
                    backoff = _calc_backoff(attempt)
                    logger.warning(
                        "[RPC-PY] Connection error to %s (attempt %d/%d, backoff %.0fms): %s",
                        host,
                        attempt + 1,
                        retries + 1,
                        backoff,
                        type(exc).__name__,
                    )
                    time.sleep(backoff / 1000)
                    continue
                raise

            except httpx.HTTPStatusError as exc:
                latency_ms = (time.time() - start) * 1000
                health.record_failure(latency_ms)
                raise

            except Exception as exc:
                latency_ms = (time.time() - start) * 1000
                health.record_failure(latency_ms)
                last_exc = exc
                if attempt < retries:
                    backoff = _calc_backoff(attempt)
                    time.sleep(backoff / 1000)
                    continue
                raise

        raise last_exc  # type: ignore[misc]

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def get_all_health(self) -> dict[str, Any]:
        """Return health metrics for all tracked API hosts."""
        endpoints = [h.to_dict() for h in self._health.values()]
        any_down = any(e["status"] == "down" for e in endpoints)
        any_degraded = any(e["status"] == "degraded" for e in endpoints)

        if any_down:
            overall = "degraded" if not all(e["status"] == "down" for e in endpoints) else "down"
        elif any_degraded:
            overall = "degraded"
        else:
            overall = "healthy"

        return {
            "status": overall,
            "endpoints": endpoints,
            "timestamp": time.time(),
        }

    def get_host_health(self, url: str) -> dict[str, Any] | None:
        """Return health for a specific API host."""
        host = self._get_host(url)
        health = self._health.get(host)
        return health.to_dict() if health else None

    def reset_health(self) -> None:
        self._health.clear()


def _calc_backoff(attempt: int) -> float:
    """Exponential backoff: base * 2^attempt, capped at max."""
    backoff = BACKOFF_BASE_MS * (2 ** attempt)
    return min(backoff, BACKOFF_MAX_MS)


# ── Redis Shared Health Sync ──

_REDIS_KEY_PREFIX = "rpc:health:"
_REDIS_TTL = 120  # seconds

_redis_client: Any | None = None  # redis.asyncio.Redis
_shared_health_cache: dict[str, dict[str, Any]] = {}


def _url_to_key(url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:16]
    return f"{_REDIS_KEY_PREFIX}{h}"


async def init_rpc_health_redis() -> None:
    """Initialize async Redis client for RPC health sharing. No-op if unavailable."""
    global _redis_client
    redis_url = os.environ.get("REDIS_URL", os.environ.get("PERSISTENCE_REDIS_URL", ""))
    if not redis_url or os.environ.get("REDIS_ENABLED") == "false":
        logger.info("[RPC-PY] Redis disabled — shared health sync off")
        return
    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(redis_url, decode_responses=True, socket_connect_timeout=5)
        await _redis_client.ping()
        logger.info("[RPC-PY] Redis connected for RPC health sync")
    except Exception as exc:
        logger.warning("[RPC-PY] Redis init failed — shared health sync off: %s", exc)
        _redis_client = None


async def close_rpc_health_redis() -> None:
    """Close the Redis client."""
    global _redis_client
    if _redis_client:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
        _redis_client = None


def _fire_and_forget_write(host: str, health: EndpointHealth) -> None:
    """Non-blocking Redis SET of endpoint health."""
    if _redis_client is None:
        return
    value = _json.dumps({
        "url": host,
        "status": health.status,
        "failCount": health.fail_count,
        "lastFailure": health.last_failure,
        "avgLatencyMs": round(health.avg_latency_ms, 2),
        "successRate": round(health.success_rate, 3),
        "writer": "python",
        "updatedAt": int(time.time() * 1000),
    })
    key = _url_to_key(host)
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_redis_client.setex(key, _REDIS_TTL, value))
    except RuntimeError:
        pass  # No event loop — skip


async def refresh_shared_health() -> None:
    """Read all shared health entries from Redis into local cache."""
    global _shared_health_cache
    if _redis_client is None:
        return
    try:
        keys: list[str] = []
        async for key in _redis_client.scan_iter(match=f"{_REDIS_KEY_PREFIX}*", count=50):
            keys.append(key)
        if not keys:
            _shared_health_cache = {}
            return
        values = await _redis_client.mget(*keys)
        new_cache: dict[str, dict[str, Any]] = {}
        for val in values:
            if val:
                try:
                    parsed = _json.loads(val)
                    new_cache[parsed.get("url", "")] = parsed
                except Exception:
                    pass
        _shared_health_cache = new_cache
    except Exception as exc:
        logger.warning("[RPC-PY] Shared health refresh failed: %s", exc)


def is_endpoint_flagged_down(host: str) -> bool:
    """Check if the TS process flagged this endpoint as down."""
    entry = _shared_health_cache.get(host)
    if not entry:
        return False
    if entry.get("writer") == "python":
        return False
    age_ms = int(time.time() * 1000) - entry.get("updatedAt", 0)
    if age_ms > _REDIS_TTL * 1000:
        return False
    return entry.get("status") == "down"


# ── Shared singleton ──

_pool: ResilientClient | None = None


def get_resilient_pool() -> ResilientClient:
    """Get the shared resilient HTTP client for all Solana data API calls."""
    global _pool
    if _pool is None:
        _pool = ResilientClient()
    return _pool
