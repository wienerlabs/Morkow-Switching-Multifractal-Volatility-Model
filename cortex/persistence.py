"""Redis-backed model state persistence with in-memory fallback.

Stores are keyed by ``{prefix}{store_name}:{token}`` in Redis.
Values are pickle-serialised (safe — only our own calibration dicts).

When Redis is unavailable the class degrades to a plain dict so the
application keeps working exactly as before.
"""

from __future__ import annotations

__all__ = ["PersistentStore", "VersionedPersistentStore"]

import json
import logging
import pickle
import time
from typing import Any, Iterator

from cortex.config import (
    MODEL_VERSION_HISTORY_SIZE,
    PERSISTENCE_ENABLED,
    PERSISTENCE_KEY_PREFIX,
    PERSISTENCE_REDIS_URL,
)

logger = logging.getLogger(__name__)

_redis_client: Any | None = None
_redis_available: bool = False


async def init_persistence() -> None:
    """Connect to Redis for persistence. Safe to call even if Redis is down."""
    global _redis_client, _redis_available

    if not PERSISTENCE_ENABLED or not PERSISTENCE_REDIS_URL:
        logger.info("Model persistence disabled (PERSISTENCE_ENABLED=%s, URL=%s)",
                     PERSISTENCE_ENABLED, bool(PERSISTENCE_REDIS_URL))
        return

    try:
        import redis.asyncio as aioredis
        _redis_client = aioredis.from_url(
            PERSISTENCE_REDIS_URL,
            decode_responses=False,
            socket_connect_timeout=5,
        )
        await _redis_client.ping()
        _redis_available = True
        logger.info("Model persistence: Redis connected (%s)",
                     PERSISTENCE_REDIS_URL.split("@")[-1] if "@" in PERSISTENCE_REDIS_URL else PERSISTENCE_REDIS_URL)
    except Exception:
        _redis_available = False
        _redis_client = None
        logger.warning("Model persistence: Redis unavailable, using in-memory only", exc_info=True)


async def close_persistence() -> None:
    global _redis_client, _redis_available
    if _redis_client is not None:
        try:
            await _redis_client.aclose()
        except Exception:
            pass
    _redis_client = None
    _redis_available = False


class PersistentStore:
    """Dict-like store that lazily persists to Redis on write.

    Read operations always hit the in-memory dict (fast path).
    Write operations update both in-memory and Redis (if available).
    On startup, ``restore()`` loads all keys from Redis into memory.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._data: dict[str, dict] = {}

    def _redis_key(self, token: str) -> str:
        return f"{PERSISTENCE_KEY_PREFIX}{self._name}:{token}"

    # ── dict interface ──

    def __contains__(self, token: str) -> bool:
        return token in self._data

    def __getitem__(self, token: str) -> dict:
        return self._data[token]

    def __setitem__(self, token: str, value: dict) -> None:
        self._data[token] = value
        self._persist(token, value)

    def __delitem__(self, token: str) -> None:
        self._data.pop(token, None)
        self._delete(token)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def get(self, token: str, default: Any = None) -> Any:
        return self._data.get(token, default)

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def pop(self, token: str, *args) -> Any:
        val = self._data.pop(token, *args)
        self._delete(token)
        return val

    # ── persistence helpers (fire-and-forget via sync wrapper) ──
    # NOTE: pickle is safe here — we only serialise our own calibration dicts
    # containing numpy arrays and pandas objects. No external/untrusted data.

    def _persist(self, token: str, value: dict) -> None:
        if not _redis_available or _redis_client is None:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_persist(token, value))
        except RuntimeError:
            pass

    async def _async_persist(self, token: str, value: dict) -> None:
        try:
            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
            await _redis_client.set(self._redis_key(token), data)
        except Exception:
            logger.warning("Failed to persist %s:%s to Redis", self._name, token, exc_info=True)

    def _delete(self, token: str) -> None:
        if not _redis_available or _redis_client is None:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_delete(token))
        except RuntimeError:
            pass

    async def _async_delete(self, token: str) -> None:
        try:
            await _redis_client.delete(self._redis_key(token))
        except Exception:
            logger.warning("Failed to delete %s:%s from Redis", self._name, token, exc_info=True)

    # ── bulk operations ──

    async def restore(self) -> int:
        """Load all keys for this store from Redis into memory. Returns count."""
        if not _redis_available or _redis_client is None:
            return 0

        pattern = f"{PERSISTENCE_KEY_PREFIX}{self._name}:*"
        prefix_len = len(f"{PERSISTENCE_KEY_PREFIX}{self._name}:")
        count = 0
        try:
            async for key in _redis_client.scan_iter(match=pattern, count=100):
                try:
                    raw = await _redis_client.get(key)
                    if raw is None:
                        continue
                    token = key.decode()[prefix_len:] if isinstance(key, bytes) else key[prefix_len:]
                    self._data[token] = pickle.loads(raw)  # noqa: S301 — trusted internal data only
                    count += 1
                except Exception:
                    logger.warning("Failed to restore key %s", key, exc_info=True)
        except Exception:
            logger.warning("Failed to scan Redis for %s", self._name, exc_info=True)
        return count

    async def persist_all(self) -> int:
        """Write all in-memory data to Redis. Returns count."""
        if not _redis_available or _redis_client is None:
            return 0

        count = 0
        for token, value in self._data.items():
            try:
                data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                await _redis_client.set(self._redis_key(token), data)
                count += 1
            except Exception:
                logger.warning("Failed to persist %s:%s", self._name, token, exc_info=True)
        return count


class VersionedPersistentStore(PersistentStore):
    """PersistentStore that keeps versioned snapshots in Redis for rollback.

    On every write, a versioned copy is saved alongside the current value.
    Up to ``max_versions`` snapshots are retained per token; older ones are pruned.
    The current/active model remains at the standard key so all existing
    read paths (``_get_model``, Guardian, etc.) are unaffected.
    """

    def __init__(self, name: str, max_versions: int = MODEL_VERSION_HISTORY_SIZE) -> None:
        super().__init__(name)
        self._max_versions = max_versions
        self._version_meta: dict[str, list[dict[str, Any]]] = {}

    def _version_meta_key(self, token: str) -> str:
        return f"{PERSISTENCE_KEY_PREFIX}versions:{self._name}:{token}:meta"

    def _version_data_key(self, token: str, version: int) -> str:
        return f"{PERSISTENCE_KEY_PREFIX}versions:{self._name}:{token}:v{version}"

    def __setitem__(self, token: str, value: dict) -> None:
        super().__setitem__(token, value)
        self._snapshot_version(token, value)

    def _snapshot_version(self, token: str, value: dict) -> None:
        """Create a versioned snapshot in Redis."""
        if not _redis_available or _redis_client is None:
            return
        try:
            import asyncio
            loop = asyncio.get_running_loop()
            loop.create_task(self._async_snapshot(token, value))
        except RuntimeError:
            pass

    async def _async_snapshot(self, token: str, value: dict) -> None:
        try:
            meta = await self._load_meta(token)

            next_version = (meta[-1]["version"] + 1) if meta else 1
            ts = time.time()
            calibrated_at = ""
            if hasattr(value.get("calibrated_at", ""), "isoformat"):
                calibrated_at = value["calibrated_at"].isoformat()

            entry = {
                "version": next_version,
                "timestamp": ts,
                "calibrated_at": calibrated_at,
            }

            data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)  # noqa: S301
            await _redis_client.set(self._version_data_key(token, next_version), data)

            meta.append(entry)

            # Prune old versions
            while len(meta) > self._max_versions:
                old = meta.pop(0)
                try:
                    await _redis_client.delete(self._version_data_key(token, old["version"]))
                except Exception:
                    pass

            await _redis_client.set(
                self._version_meta_key(token),
                json.dumps(meta).encode(),
            )
            self._version_meta[token] = meta

        except Exception:
            logger.warning("Failed to snapshot version for %s:%s", self._name, token, exc_info=True)

    async def _load_meta(self, token: str) -> list[dict[str, Any]]:
        """Load version metadata from Redis (or in-memory cache)."""
        if token in self._version_meta:
            return list(self._version_meta[token])
        if not _redis_available or _redis_client is None:
            return []
        try:
            raw = await _redis_client.get(self._version_meta_key(token))
            if raw is None:
                return []
            meta = json.loads(raw)
            self._version_meta[token] = meta
            return list(meta)
        except Exception:
            return []

    async def get_versions(self, token: str) -> list[dict[str, Any]]:
        """Return version metadata for a token."""
        return await self._load_meta(token)

    async def get_all_versions(self) -> dict[str, list[dict[str, Any]]]:
        """Return version metadata for all tokens currently in the store."""
        result: dict[str, list[dict[str, Any]]] = {}
        for token in self._data:
            result[token] = await self.get_versions(token)
        return result

    async def get_version_data(self, token: str, version: int) -> dict | None:
        """Load a specific versioned snapshot from Redis."""
        if not _redis_available or _redis_client is None:
            return None
        try:
            raw = await _redis_client.get(self._version_data_key(token, version))
            if raw is None:
                return None
            return pickle.loads(raw)  # noqa: S301 — trusted internal data only
        except Exception:
            logger.warning("Failed to load version %d for %s:%s", version, self._name, token, exc_info=True)
            return None

    async def restore_version(self, token: str, version: int) -> bool:
        """Rollback to a specific version. Returns True on success."""
        data = await self.get_version_data(token, version)
        if data is None:
            return False
        self[token] = data
        return True

    async def restore(self) -> int:
        """Restore current models + version metadata from Redis."""
        count = await super().restore()
        # Also restore version metadata for each token
        for token in list(self._data.keys()):
            await self._load_meta(token)
        return count
