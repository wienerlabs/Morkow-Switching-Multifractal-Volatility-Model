"""Wave 9 — Trade Execution endpoints."""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(tags=["execution"])


class PreflightRequest(BaseModel):
    token: str
    trade_size_usd: float = Field(gt=0)
    direction: str = Field(pattern="^(buy|sell)$")


class ExecuteTradeRequest(BaseModel):
    private_key: str
    token_mint: str
    direction: str = Field(pattern="^(buy|sell)$")
    amount: float = Field(gt=0)
    trade_size_usd: float = Field(gt=0)
    slippage_bps: int | None = None
    force: bool = False


class PreflightResponse(BaseModel):
    approved: bool
    token: str
    trade_size_usd: float
    direction: str
    guardian_score: float | None = None
    circuit_breaker_ok: bool = True
    veto_reasons: list[str] = []
    timestamp_iso: str | None = None


class ExecuteTradeResponse(BaseModel):
    success: bool
    tx_hash: str | None = None
    token_mint: str
    direction: str
    amount: float
    slippage_bps: int | None = None
    price_usd: float | None = None
    error: str | None = None
    timestamp_iso: str | None = None


class ExecutionLogEntry(BaseModel):
    tx_hash: str | None = None
    token: str | None = None
    direction: str | None = None
    amount: float | None = None
    price_usd: float | None = None
    timestamp: str | None = None


class ExecutionLogResponse(BaseModel):
    entries: list[dict]


class ExecutionStatsResponse(BaseModel):
    total_trades: int = 0
    successful: int = 0
    failed: int = 0
    avg_slippage_bps: float | None = None
    total_volume_usd: float = 0.0
    timestamp: str | None = None


@router.post("/execution/preflight", summary="Preflight trade check", response_model=PreflightResponse)
def preflight(req: PreflightRequest):
    """Run preflight checks (Guardian veto, circuit breakers, portfolio limits) before execution."""
    from cortex.execution import preflight_check

    try:
        result = preflight_check(
            token=req.token,
            trade_size_usd=req.trade_size_usd,
            direction=req.direction,
        )
        result["timestamp_iso"] = datetime.now(timezone.utc).isoformat()
        return result
    except Exception as exc:
        logger.exception("Preflight check failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/execution/trade", summary="Execute trade", response_model=ExecuteTradeResponse)
def execute_trade(req: ExecuteTradeRequest):
    """Execute a swap on Solana via Jupiter aggregator after preflight validation."""
    from cortex.execution import execute_trade as _execute

    try:
        result = _execute(
            private_key=req.private_key,
            token_mint=req.token_mint,
            direction=req.direction,
            amount=req.amount,
            trade_size_usd=req.trade_size_usd,
            slippage_bps=req.slippage_bps,
            force=req.force,
        )
        result["timestamp_iso"] = datetime.now(timezone.utc).isoformat()
        return result
    except Exception as exc:
        logger.exception("Trade execution failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/execution/log", summary="Get execution log", response_model=ExecutionLogResponse)
def get_execution_log(limit: int = Query(50, ge=1, le=500)):
    """Return recent trade execution log entries."""
    from cortex.execution import get_execution_log as _log

    return {"entries": _log(limit=limit)}


@router.get("/execution/stats", summary="Get execution statistics", response_model=ExecutionStatsResponse)
def get_execution_stats():
    """Return aggregate execution statistics (fill rates, slippage, PnL)."""
    from cortex.execution import get_execution_stats as _stats

    result = _stats()
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result


@router.get("/execution/pipeline-status", summary="Get execution pipeline status for dashboard monitor")
def get_pipeline_status():
    """Return the current execution pipeline state for the dashboard execution monitor.

    Maps the latest execution to a structured view with step progress,
    guardian checks, TX metrics, and overall status. Returns IDLE when
    no recent executions exist.
    """
    from cortex.execution import get_pipeline_status as _status

    try:
        return _status()
    except Exception as exc:
        logger.exception("Pipeline status fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/execution/status", summary="Get current execution status")
def get_execution_status():
    """Return the current execution pipeline state.

    Alias for /execution/pipeline-status — used by the dashboard UI
    to poll execution state (idle, executing, confirming, etc.).
    """
    from cortex.execution import get_pipeline_status as _status

    try:
        return _status()
    except Exception as exc:
        logger.exception("Execution status fetch failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/execution/stream", summary="Execution events SSE stream")
async def execution_sse_stream():
    """SSE endpoint streaming execution pipeline events in real time.

    Connect with: curl -N /api/v1/execution/stream
    Events are pushed after each execution log entry.
    """
    from cortex.execution import subscribe_execution_events, unsubscribe_execution_events

    queue = subscribe_execution_events()

    async def event_generator():
        try:
            while True:
                try:
                    if queue:
                        event = queue.popleft()
                        yield f"data: {json.dumps(event, default=str)}\n\n"
                    else:
                        await asyncio.sleep(0.5)
                except IndexError:
                    await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass
        finally:
            unsubscribe_execution_events(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

