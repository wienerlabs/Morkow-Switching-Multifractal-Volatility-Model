"""API security middleware: authentication, rate limiting, CORS configuration."""

import logging
import os
import time
from collections import defaultdict

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("cortex_risk_engine.security")

# ── API Key Authentication ──────────────────────────────────────────

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _get_valid_keys() -> set[str]:
    raw = os.environ.get("CORTEX_API_KEYS", "")
    if not raw.strip():
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


async def verify_api_key(api_key: str | None = Security(_api_key_header)) -> str | None:
    """FastAPI dependency — validates X-API-Key header.

    If CORTEX_API_KEYS is not set, auth is disabled (open mode for dev).
    """
    valid_keys = _get_valid_keys()
    if not valid_keys:
        return None  # open mode
    if not api_key or api_key not in valid_keys:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ── Rate Limiting (sliding window) ──────────────────────────────────

_DEFAULT_READ_LIMIT = 60   # requests per window
_DEFAULT_WRITE_LIMIT = 10  # requests per window for POST (calibration)
_WINDOW_SECONDS = 60


def _get_limits() -> tuple[int, int]:
    read = int(os.environ.get("CORTEX_RATE_LIMIT_READ", _DEFAULT_READ_LIMIT))
    write = int(os.environ.get("CORTEX_RATE_LIMIT_WRITE", _DEFAULT_WRITE_LIMIT))
    return read, write


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-key sliding-window rate limiter.

    Exempt paths: /health, /, /openapi.json, /docs, /redoc
    """

    EXEMPT_PATHS = frozenset({"/health", "/", "/openapi.json", "/docs", "/redoc"})

    def __init__(self, app):
        super().__init__(app)
        self._buckets: dict[str, list[float]] = defaultdict(list)

    def _client_key(self, request: Request) -> str:
        api_key = request.headers.get("x-api-key")
        if api_key:
            return f"key:{api_key}"
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        client = request.client
        return f"ip:{client.host}" if client else "ip:unknown"

    def _prune(self, bucket: list[float], now: float) -> list[float]:
        cutoff = now - _WINDOW_SECONDS
        return [t for t in bucket if t > cutoff]

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        read_limit, write_limit = _get_limits()
        is_write = request.method in ("POST", "PUT", "PATCH", "DELETE")
        limit = write_limit if is_write else read_limit

        key = self._client_key(request)
        now = time.monotonic()

        self._buckets[key] = self._prune(self._buckets[key], now)

        if len(self._buckets[key]) >= limit:
            retry_after = int(_WINDOW_SECONDS - (now - self._buckets[key][0])) + 1
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded", "retry_after": retry_after},
                headers={"Retry-After": str(retry_after)},
            )

        self._buckets[key].append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(limit - len(self._buckets[key]))
        return response


# ── CORS Configuration ──────────────────────────────────────────────


def get_allowed_origins() -> list[str]:
    raw = os.environ.get("CORTEX_ALLOWED_ORIGINS", "")
    if not raw.strip():
        return ["http://localhost:8000", "http://localhost:3000"]
    if raw.strip() == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]

