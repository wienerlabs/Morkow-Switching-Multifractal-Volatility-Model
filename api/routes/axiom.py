"""Axiom Trade DEX aggregator endpoints."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

router = APIRouter(tags=["axiom"])


@router.get("/axiom/price/{token_address}")
def get_axiom_price(token_address: str):
    from cortex.data.axiom import get_token_price

    try:
        data = get_token_price(token_address)
        data["timestamp_iso"] = datetime.now(timezone.utc).isoformat()
        return data
    except Exception as exc:
        logger.exception("Axiom price fetch failed")
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/axiom/pair/{pair_address}")
def get_axiom_pair(pair_address: str):
    from cortex.data.axiom import get_pair_liquidity

    try:
        return get_pair_liquidity(pair_address)
    except Exception as exc:
        logger.exception("Axiom pair fetch failed")
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/axiom/liquidity-metrics/{pair_address}")
def get_axiom_liquidity_metrics(pair_address: str):
    from cortex.data.axiom import extract_liquidity_metrics

    try:
        return extract_liquidity_metrics(pair_address)
    except Exception as exc:
        logger.exception("Axiom liquidity metrics failed")
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/axiom/holders/{pair_address}")
def get_axiom_holders(pair_address: str):
    from cortex.data.axiom import get_holder_data

    try:
        return get_holder_data(pair_address)
    except Exception as exc:
        logger.exception("Axiom holder data failed")
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/axiom/token-analysis")
def get_axiom_token_analysis(
    dev_address: str = Query(...),
    token_ticker: str = Query(...),
):
    from cortex.data.axiom import get_token_analysis

    try:
        return get_token_analysis(dev_address, token_ticker)
    except Exception as exc:
        logger.exception("Axiom token analysis failed")
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/axiom/new-tokens")
def get_axiom_new_tokens(
    limit: int = Query(20, ge=1, le=100),
    min_liquidity: bool = Query(True),
):
    from cortex.data.axiom import get_new_tokens

    return {"tokens": get_new_tokens(limit=limit, min_liquidity=min_liquidity)}


@router.get("/axiom/ws-status")
def get_axiom_ws_status():
    from cortex.data.axiom import get_ws_status

    return get_ws_status()


@router.get("/axiom/wallet/{wallet_address}")
def get_axiom_wallet_balance(wallet_address: str):
    from cortex.data.axiom import get_wallet_balance

    try:
        return get_wallet_balance(wallet_address)
    except Exception as exc:
        logger.exception("Axiom wallet balance failed")
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/axiom/status")
def get_axiom_status():
    from cortex.data.axiom import get_ws_status, is_available

    return {
        "available": is_available(),
        "websocket": get_ws_status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

