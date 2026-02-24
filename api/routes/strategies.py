"""Strategy configuration endpoint — serves live strategy config to the dashboard."""

import copy
import json
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cortex.config import PERSISTENCE_KEY_PREFIX

router = APIRouter(tags=["strategies"])

logger = logging.getLogger(__name__)

_TRADE_MODE: str = "autonomous"
VALID_TRADE_MODES = {"autonomous", "semi-auto", "manual"}

# Redis keys for persistence
_REDIS_KEY_TRADE_MODE = f"{PERSISTENCE_KEY_PREFIX}strategy:trade_mode"
_REDIS_KEY_STRATEGY_TOGGLES = f"{PERSISTENCE_KEY_PREFIX}strategy:toggles"


def _redis_client():
    """Get persistence Redis client if available."""
    try:
        from cortex.persistence import _redis_available, _redis_client as client
        if _redis_available and client is not None:
            return client
    except Exception:
        pass
    return None


def _persist_trade_mode(mode: str) -> None:
    """Fire-and-forget: save trade mode to Redis."""
    client = _redis_client()
    if not client:
        return
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        loop.create_task(client.set(_REDIS_KEY_TRADE_MODE, mode))
    except RuntimeError:
        pass


def _persist_strategy_toggle(key: str, enabled: bool) -> None:
    """Fire-and-forget: save strategy toggle state to Redis hash."""
    client = _redis_client()
    if not client:
        return
    try:
        import asyncio
        loop = asyncio.get_running_loop()
        loop.create_task(client.hset(_REDIS_KEY_STRATEGY_TOGGLES, key, json.dumps(enabled)))
    except RuntimeError:
        pass


async def restore_strategy_state() -> None:
    """Restore strategy toggles and trade mode from Redis on startup."""
    global _TRADE_MODE
    client = _redis_client()
    if not client:
        return

    try:
        mode = await client.get(_REDIS_KEY_TRADE_MODE)
        if mode:
            decoded = mode.decode() if isinstance(mode, bytes) else mode
            if decoded in VALID_TRADE_MODES:
                _TRADE_MODE = decoded
                logger.info("Restored trade mode from Redis: %s", _TRADE_MODE)
    except Exception:
        logger.warning("Failed to restore trade mode from Redis", exc_info=True)

    try:
        from cortex.config import STRATEGY_CONFIG
        toggles = await client.hgetall(_REDIS_KEY_STRATEGY_TOGGLES)
        if not toggles:
            return
        count = 0
        for raw_key, raw_val in toggles.items():
            key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
            val = json.loads(raw_val)
            for strat in STRATEGY_CONFIG:
                if strat.get("key") == key:
                    strat["enabled"] = val
                    strat["status"] = "running" if val else "paused"
                    count += 1
                    break
        if count:
            logger.info("Restored %d strategy toggle(s) from Redis", count)
    except Exception:
        logger.warning("Failed to restore strategy toggles from Redis", exc_info=True)


class TradeModeUpdate(BaseModel):
    mode: str


@router.get("/strategies/config", summary="Get strategy configuration")
def get_strategy_config():
    """Return strategy definitions enriched with live circuit breaker state.

    The base config comes from cortex.config.STRATEGY_CONFIG (env-overridable).
    Each strategy is enriched with its circuit breaker state when available.
    """
    from cortex.config import STRATEGY_CONFIG

    strategies = copy.deepcopy(STRATEGY_CONFIG)

    cb_map: dict[str, dict] = {}
    try:
        from cortex.circuit_breaker import get_outcome_states
        for ob in get_outcome_states():
            key = ob.get("strategy", "")
            if key:
                cb_map[key] = ob
    except Exception:
        logger.debug("Circuit breaker state unavailable", exc_info=True)

    for strat in strategies:
        key = strat.get("key", "")
        cb = cb_map.get(key)
        if cb:
            strat["circuit_breaker"] = cb
            if cb.get("state") == "open":
                strat["status"] = "paused"
                strat["enabled"] = False

    active_count = sum(1 for s in strategies if s.get("enabled", True))

    return {
        "strategies": strategies,
        "active_count": active_count,
        "total_count": len(strategies),
        "timestamp": time.time(),
    }




def _do_toggle(name: str) -> dict:
    """Shared logic: flip a strategy's enabled flag and return the updated record."""
    from cortex.config import STRATEGY_CONFIG

    for strat in STRATEGY_CONFIG:
        if strat.get("key") == name or strat.get("name") == name:
            strat["enabled"] = not strat.get("enabled", True)
            strat["status"] = "running" if strat["enabled"] else "paused"
            _persist_strategy_toggle(strat["key"], strat["enabled"])
            return {
                "strategy": strat["key"],
                "name": strat["name"],
                "enabled": strat["enabled"],
                "status": strat["status"],
                "timestamp": time.time(),
            }

    raise HTTPException(status_code=404, detail=f"Strategy '{name}' not found")


@router.put("/strategies/{name}/toggle", summary="Toggle strategy enabled state (PUT)")
def toggle_strategy_put(name: str):
    """Enable or disable a strategy by its key name (PUT variant).

    Flips the 'enabled' flag and updates 'status' accordingly.
    The change is in-memory only (resets on restart).
    """
    return _do_toggle(name)


@router.post("/strategies/{name}/toggle", summary="Toggle strategy enabled state (POST)")
def toggle_strategy_post(name: str):
    """Enable or disable a strategy by its key name (POST variant for browser clients).

    Identical behaviour to the PUT variant; provided so browser fetch calls
    (which use POST for mutations) work without requiring a CORS preflight for PUT.
    The change is in-memory only (resets on restart).
    """
    return _do_toggle(name)


# ── Trade Mode ───────────────────────────────────────────────────────────────


@router.get("/strategies/trade-mode", summary="Get current trade mode")
def get_trade_mode():
    """Return the current trade mode (autonomous | semi-auto | manual)."""
    return {"mode": _TRADE_MODE, "timestamp": time.time()}


@router.post("/strategies/trade-mode", summary="Set trade mode")
def set_trade_mode(body: TradeModeUpdate):
    """Set the trade mode.  Accepted values: autonomous, semi-auto, manual."""
    global _TRADE_MODE
    mode = body.mode.lower().strip()
    if mode not in VALID_TRADE_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(VALID_TRADE_MODES))}",
        )
    _TRADE_MODE = mode
    _persist_trade_mode(mode)
    logger.info("Trade mode changed to %s", mode)
    return {"mode": _TRADE_MODE, "timestamp": time.time()}