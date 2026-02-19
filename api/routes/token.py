"""Token info and supply endpoints."""

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from api.models import TokenInfoResponse, TokenSupplyResponse
from cortex.data.rpc_failover import get_resilient_pool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["token"])

_EXPRESS_BASE = os.environ.get("EXPRESS_BACKEND_URL", "http://localhost:3001")


@router.get("/token/info/{address}", response_model=TokenInfoResponse)
def get_token_info(address: str):
    """Fetch token metadata (name, symbol, logo, price, market cap, etc.)."""
    from cortex.data.solana import get_token_metadata

    try:
        data = get_token_metadata(address)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Token metadata fetch failed for %s", address)
        raise HTTPException(status_code=502, detail=f"Birdeye API error: {exc}")

    return TokenInfoResponse(**data, timestamp=datetime.now(timezone.utc))


@router.get("/token/supply", response_model=TokenSupplyResponse)
def get_token_supply():
    """Fetch on-chain token supply, staking, and treasury data.

    Proxies to the Express backend which has direct Solana RPC access.
    Falls back to design constants if the Express backend is unreachable.
    """
    pool = get_resilient_pool()
    try:
        resp = pool.get(f"{_EXPRESS_BASE}/api/solana/tokenomics", max_retries=1)
        resp.raise_for_status()
        data = resp.json()

        return TokenSupplyResponse(
            symbol=data["token"]["symbol"],
            decimals=data["token"]["decimals"],
            total_supply=data["token"]["totalSupply"],
            total_supply_formatted=data["token"]["totalSupplyFormatted"],
            mint=data["token"]["mint"],
            staking={
                "total_staked": data["staking"]["totalStaked"],
                "total_staked_formatted": data["staking"]["totalStakedFormatted"],
                "reward_rate": data["staking"]["rewardRate"],
                "reward_rate_formatted": data["staking"]["rewardRateFormatted"],
            },
            treasury={
                "sol_balance": data["treasury"]["solBalance"],
                "address": data["treasury"]["address"],
            },
            programs=data.get("programs", {}),
            timestamp=datetime.now(timezone.utc),
        )
    except Exception as exc:
        logger.warning("Express tokenomics proxy failed, returning defaults: %s", exc)
        return TokenSupplyResponse(
            timestamp=datetime.now(timezone.utc),
        )

