"""Runtime Config Hot-Reload — restart-less config updates via file or Redis pub/sub.

Watches a JSON override file (``HOT_CONFIG_PATH``) and optionally subscribes
to a Redis pub/sub channel for push-based updates.  When a change is detected,
the matching ``cortex.config`` module-level attributes are patched in-place so
every running component picks up the new values without a restart.

Only a curated allowlist of keys can be hot-reloaded (safety first).
"""
from __future__ import annotations

__all__ = ["HotConfigReloader", "get_reloader"]

import asyncio
import json
import logging
import os
import time
from typing import Any

import cortex.config as _cfg

logger = logging.getLogger(__name__)

# Keys that are safe to hot-reload at runtime.
# Anything not on this list is silently ignored.
RELOADABLE_KEYS: frozenset[str] = frozenset({
    # Guardian thresholds
    "APPROVAL_THRESHOLD",
    "CIRCUIT_BREAKER_THRESHOLD",
    # Scoring
    "EVT_SCORE_FLOOR",
    "EVT_SCORE_RANGE",
    "SVJ_BASE_CAP",
    "REGIME_BASE_MAX",
    "CRISIS_REGIME_HAIRCUT",
    "NEAR_CRISIS_REGIME_HAIRCUT",
    # Circuit breaker
    "CB_THRESHOLD",
    "CB_COOLDOWN_SECONDS",
    # Kelly
    "GUARDIAN_KELLY_FRACTION",
    # Portfolio risk
    "MAX_DAILY_DRAWDOWN",
    "MAX_WEEKLY_DRAWDOWN",
    "MAX_CORRELATED_EXPOSURE",
    # Execution
    "JUPITER_SLIPPAGE_BPS",
    "EXECUTION_MAX_SLIPPAGE_BPS",
    # Heartbeat
    "HEARTBEAT_INTERVAL_SECONDS",
    "HEARTBEAT_DRAWDOWN_WARN_PCT",
    "HEARTBEAT_CB_PROXIMITY_PCT",
    # Cognitive state
    "COGNITIVE_STATE_SMOOTHING",
    # Feature flags (bool)
    "COGNITIVE_STATE_ENABLED",
    "HEARTBEAT_ENABLED",
    "EXECUTION_PIPELINE_ENABLED",
    "TRADE_LEDGER_ENABLED",
    "NARRATOR_ENABLED",
    "EXECUTION_ENABLED",
    "SIMULATION_MODE",
})

# Redis pub/sub channel for push-based hot-reload
HOT_CONFIG_CHANNEL = "cortex:hot_config"
HOT_CONFIG_PATH = os.environ.get("HOT_CONFIG_PATH", "")


def _coerce(key: str, value: Any) -> Any:
    """Coerce a JSON value to the expected Python type for *key*."""
    current = getattr(_cfg, key, None)
    if current is None:
        return value
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes")
        return bool(value)
    if isinstance(current, int):
        return int(value)
    if isinstance(current, float):
        return float(value)
    return value


class HotConfigReloader:
    """Polls a JSON file and/or listens on Redis pub/sub for config overrides."""

    def __init__(
        self,
        poll_interval: float | None = None,
        config_path: str | None = None,
    ) -> None:
        self._poll_interval = poll_interval or _cfg.HOT_CONFIG_POLL_INTERVAL
        self._config_path = config_path or HOT_CONFIG_PATH
        self._last_mtime: float = 0.0
        self._applied: dict[str, Any] = {}
        self._change_count: int = 0
        self._running: bool = False
        self._tasks: list[asyncio.Task] = []

    # ── Public API ──

    def apply(self, overrides: dict[str, Any]) -> dict[str, Any]:
        """Apply a dict of config overrides. Returns {key: new_value} for applied keys."""
        applied = {}
        for key, raw_value in overrides.items():
            if key not in RELOADABLE_KEYS:
                logger.debug("hot_config: ignoring non-reloadable key %s", key)
                continue

            if not hasattr(_cfg, key):
                logger.warning("hot_config: key %s not found in cortex.config", key)
                continue

            value = _coerce(key, raw_value)
            old = getattr(_cfg, key)
            if old == value:
                continue

            setattr(_cfg, key, value)
            self._applied[key] = value
            self._change_count += 1
            applied[key] = value
            logger.info("hot_config: %s = %r (was %r)", key, value, old)

        return applied

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": _cfg.HOT_CONFIG_ENABLED,
            "running": self._running,
            "poll_interval": self._poll_interval,
            "config_path": self._config_path,
            "total_changes": self._change_count,
            "active_overrides": dict(self._applied),
        }

    def get_active_overrides(self) -> dict[str, Any]:
        return dict(self._applied)

    def reset(self) -> dict[str, Any]:
        """Remove all overrides by re-reading original env vars.
        Returns the keys that were reset."""
        reset_keys = {}
        for key in list(self._applied.keys()):
            env_val = os.environ.get(key)
            if env_val is not None:
                value = _coerce(key, env_val)
            else:
                # Can't recover original default without re-importing; skip
                continue
            old = getattr(_cfg, key)
            setattr(_cfg, key, value)
            reset_keys[key] = {"from": old, "to": value}
            logger.info("hot_config RESET: %s = %r (was %r)", key, value, old)

        self._applied.clear()
        return reset_keys

    # ── File watcher ──

    def _read_config_file(self) -> dict[str, Any] | None:
        if not self._config_path:
            return None
        try:
            mtime = os.path.getmtime(self._config_path)
            if mtime <= self._last_mtime:
                return None
            self._last_mtime = mtime
            with open(self._config_path) as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning("hot_config: config file is not a JSON object")
                return None
            return data
        except FileNotFoundError:
            return None
        except Exception:
            logger.warning("hot_config: failed to read config file", exc_info=True)
            return None

    async def _file_poll_loop(self) -> None:
        logger.info("hot_config: file poll loop started (path=%s, interval=%.1fs)",
                     self._config_path, self._poll_interval)
        while self._running:
            overrides = self._read_config_file()
            if overrides:
                applied = self.apply(overrides)
                if applied:
                    logger.info("hot_config: applied %d changes from file", len(applied))
            await asyncio.sleep(self._poll_interval)

    # ── Redis pub/sub listener ──

    async def _redis_listen_loop(self) -> None:
        from cortex.config import PERSISTENCE_REDIS_URL
        if not PERSISTENCE_REDIS_URL:
            logger.info("hot_config: Redis not configured, skipping pub/sub listener")
            return

        try:
            import redis.asyncio as aioredis
        except ImportError:
            logger.info("hot_config: redis package not available, skipping pub/sub")
            return

        while self._running:
            try:
                client = aioredis.from_url(
                    PERSISTENCE_REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                )
                pubsub = client.pubsub()
                await pubsub.subscribe(HOT_CONFIG_CHANNEL)
                logger.info("hot_config: subscribed to Redis channel %s", HOT_CONFIG_CHANNEL)

                async for message in pubsub.listen():
                    if not self._running:
                        break
                    if message["type"] != "message":
                        continue
                    try:
                        data = json.loads(message["data"])
                        if isinstance(data, dict):
                            applied = self.apply(data)
                            if applied:
                                logger.info("hot_config: applied %d changes from Redis", len(applied))
                    except json.JSONDecodeError:
                        logger.warning("hot_config: invalid JSON on Redis channel")

                await pubsub.unsubscribe(HOT_CONFIG_CHANNEL)
                await client.aclose()
            except Exception:
                if self._running:
                    logger.warning("hot_config: Redis pub/sub error, retrying in 10s", exc_info=True)
                    await asyncio.sleep(10)

    # ── Lifecycle ──

    async def start(self) -> None:
        if not _cfg.HOT_CONFIG_ENABLED:
            logger.info("hot_config: disabled (HOT_CONFIG_ENABLED=false)")
            return

        self._running = True

        if self._config_path:
            self._tasks.append(asyncio.create_task(self._file_poll_loop()))

        self._tasks.append(asyncio.create_task(self._redis_listen_loop()))
        logger.info("hot_config: reloader started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        logger.info("hot_config: reloader stopped")


# ── Singleton ──

_reloader: HotConfigReloader | None = None


def get_reloader() -> HotConfigReloader:
    global _reloader
    if _reloader is None:
        _reloader = HotConfigReloader()
    return _reloader
