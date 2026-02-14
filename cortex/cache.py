"""Async caching layer using cashews with Redis backend and in-memory fallback.

Usage:
    from cortex.cache import cache, setup_cache

    @cache(ttl="60s", key="mykey:{arg}")
    async def expensive_call(arg: str) -> dict:
        ...
"""
from __future__ import annotations

import logging

from cashews import cache

from cortex.config import CACHE_DEFAULT_TTL, CACHE_ENABLED, REDIS_URL

logger = logging.getLogger(__name__)

__all__ = ["cache", "setup_cache"]


async def setup_cache() -> None:
    """Initialize the cache backend.

    Tries Redis first (if REDIS_URL is set), falls back to in-memory.
    If CACHE_ENABLED is False, uses a null backend (disable).
    """
    if not CACHE_ENABLED:
        cache.setup("null://")
        logger.info("Cache disabled via CACHE_ENABLED=false")
        return

    if REDIS_URL:
        try:
            cache.setup(REDIS_URL, default_timeout=CACHE_DEFAULT_TTL)
            logger.info("Cache backend: Redis (%s)", REDIS_URL.split("@")[-1] if "@" in REDIS_URL else REDIS_URL)
            return
        except Exception:
            logger.warning("Redis connection failed, falling back to in-memory cache", exc_info=True)

    cache.setup("mem://", default_timeout=CACHE_DEFAULT_TTL)
    logger.info("Cache backend: in-memory (TTL=%ds)", CACHE_DEFAULT_TTL)

