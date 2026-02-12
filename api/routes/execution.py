"""Wave 9 — Trade Execution endpoints."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query
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


@router.post("/execution/preflight")
def preflight(req: PreflightRequest):
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


@router.post("/execution/trade")
def execute_trade(req: ExecuteTradeRequest):
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


@router.get("/execution/log")
def get_execution_log(limit: int = Query(50, ge=1, le=500)):
    from cortex.execution import get_execution_log as _log

    return {"entries": _log(limit=limit)}


@router.get("/execution/stats")
def get_execution_stats():
    from cortex.execution import get_execution_stats as _stats

    result = _stats()
    result["timestamp"] = datetime.now(timezone.utc).isoformat()
    return result

