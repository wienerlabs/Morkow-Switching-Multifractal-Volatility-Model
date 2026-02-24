"""DexScreener DEX data endpoints."""

import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.models import (
    DexLiquidityMetricsResponse,
    DexNewTokensResponse,
    DexPairLiquidityResponse,
    DexStatusResponse,
    DexTokenPriceResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dex"])

_SOLANA_ADDR_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")


def _validate_solana_address(address: str, label: str = "address") -> None:
    """Validate a Solana base58 address format."""
    if not _SOLANA_ADDR_RE.match(address):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid Solana {label}: {address!r}",
        )


@router.get("/dex/price/{token_address}", summary="Get token price", response_model=DexTokenPriceResponse)
def get_dex_price(token_address: str):
    """Fetch current token price from DexScreener."""
    _validate_solana_address(token_address, "token address")
    from cortex.data.dexscreener import get_token_price

    try:
        data = get_token_price(token_address)
        data["timestamp_iso"] = datetime.now(timezone.utc).isoformat()
        return data
    except Exception as exc:
        logger.exception("DexScreener price fetch failed")
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/dex/pair/{pair_address}", summary="Get pair liquidity", response_model=DexPairLiquidityResponse)
def get_dex_pair(pair_address: str):
    """Fetch liquidity data for a trading pair from DexScreener."""
    _validate_solana_address(pair_address, "pair address")
    from cortex.data.dexscreener import get_pair_liquidity

    try:
        return get_pair_liquidity(pair_address)
    except Exception as exc:
        logger.exception("DexScreener pair fetch failed")
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/dex/liquidity-metrics/{pair_address}", summary="Get liquidity metrics", response_model=DexLiquidityMetricsResponse)
def get_dex_liquidity_metrics(pair_address: str):
    """Extract structured liquidity metrics (TVL, depth, concentration) for a pair."""
    _validate_solana_address(pair_address, "pair address")
    from cortex.data.dexscreener import extract_liquidity_metrics

    try:
        return extract_liquidity_metrics(pair_address)
    except Exception as exc:
        logger.exception("DexScreener liquidity metrics failed")
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/dex/new-tokens", summary="List new tokens", response_model=DexNewTokensResponse)
def get_dex_new_tokens(
    limit: int = Query(20, ge=1, le=100),
    min_liquidity: bool = Query(True),
):
    """List recently launched tokens, optionally filtered by minimum liquidity."""
    from cortex.data.dexscreener import get_new_tokens

    return {"tokens": get_new_tokens(limit=limit, min_liquidity=min_liquidity)}


@router.get("/dex/status", summary="DexScreener service status", response_model=DexStatusResponse)
def get_dex_status():
    """Return DexScreener integration status and availability."""
    from cortex.data.dexscreener import is_available

    return {
        "available": is_available(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
