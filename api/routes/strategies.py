"""Strategy configuration endpoint — serves live strategy config to the dashboard."""

import copy
import logging
import time

from fastapi import APIRouter

router = APIRouter(tags=["strategies"])

logger = logging.getLogger(__name__)


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

