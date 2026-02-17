"""In-memory async task store for background calibration jobs + recalibration scheduler."""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


_task_store: dict[str, dict[str, Any]] = {}


def create_task(endpoint: str) -> str:
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    _task_store[task_id] = {
        "task_id": task_id,
        "status": TaskStatus.PENDING,
        "endpoint": endpoint,
        "result": None,
        "error": None,
        "created_at": now,
        "started_at": None,
        "completed_at": None,
    }
    return task_id


def get_task(task_id: str) -> dict[str, Any] | None:
    return _task_store.get(task_id)


async def run_in_background(task_id: str, sync_fn, *args, **kwargs) -> None:
    task = _task_store[task_id]
    task["status"] = TaskStatus.RUNNING
    task["started_at"] = datetime.now(timezone.utc).isoformat()
    try:
        result = await asyncio.to_thread(sync_fn, *args, **kwargs)
        task["status"] = TaskStatus.COMPLETED
        task["result"] = result
    except Exception as exc:
        task["status"] = TaskStatus.FAILED
        task["error"] = str(exc)
    finally:
        task["completed_at"] = datetime.now(timezone.utc).isoformat()


# ── Periodic Recalibration Scheduler ─────────────────────────────────

_recalib_task: asyncio.Task | None = None


def _recalibrate_token(token: str, model_data: dict) -> dict | None:
    """Re-run MSM calibration for a single token using stored parameters."""
    from cortex import msm

    returns = model_data.get("returns")
    if returns is None or len(returns) < 30:
        return None

    cal = model_data.get("calibration", {})
    num_states = cal.get("num_states", 4)
    leverage_gamma = model_data.get("leverage_gamma", 0.0)

    new_cal = msm.calibrate_msm_advanced(
        returns, num_states=num_states, method=cal.get("method", "mle"),
        target_var_breach=cal.get("target_var_breach", 0.05),
        verbose=False, leverage_gamma=leverage_gamma,
    )

    sigma_f, sigma_filt, fprobs, sigma_states, P = msm.msm_vol_forecast(
        returns, num_states=new_cal["num_states"],
        sigma_low=new_cal["sigma_low"], sigma_high=new_cal["sigma_high"],
        p_stay=new_cal["p_stay"], leverage_gamma=leverage_gamma,
    )

    return {
        "calibration": new_cal,
        "returns": returns,
        "sigma_forecast": sigma_f,
        "sigma_filtered": sigma_filt,
        "filter_probs": fprobs,
        "sigma_states": sigma_states,
        "P_matrix": P,
        "use_student_t": model_data.get("use_student_t", False),
        "nu": model_data.get("nu", 5.0),
        "leverage_gamma": leverage_gamma,
        "calibrated_at": datetime.now(timezone.utc),
    }


async def _recalibration_loop() -> None:
    """Periodically recalibrate all known tokens in the model store."""
    from cortex.config import RECALIBRATION_INTERVAL_HOURS

    interval_seconds = RECALIBRATION_INTERVAL_HOURS * 3600

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            from api.stores import _model_store

            tokens = list(_model_store.keys())
            if not tokens:
                logger.debug("Recalibration: no tokens in model store")
                continue

            logger.info("Recalibration starting for %d token(s)", len(tokens))
            recalibrated = 0
            for token in tokens:
                model_data = _model_store.get(token)
                if not model_data:
                    continue
                try:
                    updated = await asyncio.to_thread(_recalibrate_token, token, model_data)
                    if updated:
                        _model_store[token] = updated
                        recalibrated += 1
                except Exception:
                    logger.warning("Recalibration failed for %s", token, exc_info=True)

            logger.info("Recalibration complete: %d/%d tokens updated", recalibrated, len(tokens))
        except Exception:
            logger.error("Recalibration loop error", exc_info=True)


def start_recalibration_scheduler() -> None:
    """Start the periodic recalibration background task."""
    from cortex.config import RECALIBRATION_ENABLED

    global _recalib_task
    if not RECALIBRATION_ENABLED:
        logger.info("Recalibration scheduler disabled")
        return
    if _recalib_task is not None:
        return
    _recalib_task = asyncio.create_task(_recalibration_loop())
    logger.info("Recalibration scheduler started")


def stop_recalibration_scheduler() -> None:
    """Cancel the recalibration background task."""
    global _recalib_task
    if _recalib_task is not None:
        _recalib_task.cancel()
        _recalib_task = None
        logger.info("Recalibration scheduler stopped")


# ── Background News Collector ─────────────────────────────────────

_news_collector_task: asyncio.Task | None = None


async def _news_collector_loop() -> None:
    """Periodically fetch news and populate the global buffer."""
    from cortex.config import NEWS_COLLECTOR_INTERVAL_SECONDS, NEWS_BUFFER_MAX_ITEMS
    from cortex.news import fetch_news_intelligence, news_buffer

    news_buffer._max_items = NEWS_BUFFER_MAX_ITEMS

    # Initial fetch immediately on startup
    try:
        result = await asyncio.to_thread(
            fetch_news_intelligence, regime_state=_get_regime_state(), max_items=NEWS_BUFFER_MAX_ITEMS,
        )
        news_buffer.update(result)
        logger.info("News collector: initial fetch OK (%d items)", len(result.get("items", [])))
    except Exception:
        logger.warning("News collector: initial fetch failed", exc_info=True)

    while True:
        await asyncio.sleep(NEWS_COLLECTOR_INTERVAL_SECONDS)
        try:
            result = await asyncio.to_thread(
                fetch_news_intelligence,
                regime_state=_get_regime_state(),
                max_items=NEWS_BUFFER_MAX_ITEMS,
            )
            news_buffer.update(result)
            logger.debug("News collector: refreshed (%d items)", len(result.get("items", [])))
        except Exception as exc:
            news_buffer.record_error(str(exc))
            logger.warning("News collector: fetch failed: %s", exc)


def _get_regime_state() -> int:
    """Get current regime state from model store, default 3."""
    try:
        from api.stores import _model_store
        for model_data in _model_store.values():
            fprobs = model_data.get("filter_probs")
            if fprobs is not None and len(fprobs) > 0:
                import numpy as np
                return int(np.argmax(fprobs[-1])) + 1
    except Exception:
        pass
    return 3


def start_news_collector() -> None:
    from cortex.config import NEWS_COLLECTOR_ENABLED

    global _news_collector_task
    if not NEWS_COLLECTOR_ENABLED:
        logger.info("News collector disabled")
        return
    if _news_collector_task is not None:
        return
    from cortex.config import NEWS_COLLECTOR_INTERVAL_SECONDS
    _news_collector_task = asyncio.create_task(_news_collector_loop())
    logger.info("News collector started (interval=%ds)", NEWS_COLLECTOR_INTERVAL_SECONDS)


def stop_news_collector() -> None:
    global _news_collector_task
    if _news_collector_task is not None:
        _news_collector_task.cancel()
        _news_collector_task = None
        logger.info("News collector stopped")


# ── Liquidity Snapshot Collector ─────────────────────────────────────

_liquidity_snapshot_task: asyncio.Task | None = None


async def _liquidity_snapshot_loop() -> None:
    """Periodically snapshot TVL/liquidity for watched pools."""
    from cortex.config import LIQUIDITY_SNAPSHOT_INTERVAL_SECONDS, LIQUIDITY_WATCHLIST
    from cortex.liquidity_history import store_snapshot, take_snapshot

    while True:
        await asyncio.sleep(LIQUIDITY_SNAPSHOT_INTERVAL_SECONDS)
        pools = LIQUIDITY_WATCHLIST
        if not pools:
            logger.debug("Liquidity snapshot: no pools in watchlist")
            continue

        captured = 0
        for pool_address in pools:
            try:
                snapshot = await asyncio.to_thread(take_snapshot, pool_address)
                if snapshot:
                    store_snapshot(pool_address, snapshot)
                    captured += 1
            except Exception:
                logger.warning("Liquidity snapshot failed for %s", pool_address, exc_info=True)

        logger.debug("Liquidity snapshot: captured %d/%d pools", captured, len(pools))


def start_liquidity_snapshot_collector() -> None:
    from cortex.config import LIQUIDITY_SNAPSHOT_ENABLED

    global _liquidity_snapshot_task
    if not LIQUIDITY_SNAPSHOT_ENABLED:
        logger.info("Liquidity snapshot collector disabled")
        return
    if _liquidity_snapshot_task is not None:
        return
    from cortex.config import LIQUIDITY_SNAPSHOT_INTERVAL_SECONDS
    _liquidity_snapshot_task = asyncio.create_task(_liquidity_snapshot_loop())
    logger.info("Liquidity snapshot collector started (interval=%ds)", LIQUIDITY_SNAPSHOT_INTERVAL_SECONDS)


def stop_liquidity_snapshot_collector() -> None:
    global _liquidity_snapshot_task
    if _liquidity_snapshot_task is not None:
        _liquidity_snapshot_task.cancel()
        _liquidity_snapshot_task = None
        logger.info("Liquidity snapshot collector stopped")

