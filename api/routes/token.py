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
        logger.warning("Express tokenomics proxy failed, trying direct RPC: %s", exc)
        return _fallback_solana_rpc(pool)


# Known CRTX program addresses (mirrors backend/src/services/solana.ts)
_CRTX_MINT = "HAUqFj3uYsFt6PhMgztaVkRi5RC3mFWxyLceJzCRDevg"
_TREASURY_ADDR = "GsMtBFGq3DWFGDqtJ6Fb4LjdB6sBADKL2Lix9xaYV7GS"
_SOLANA_RPC = os.environ.get(
    "SOLANA_RPC_URL",
    os.environ.get("HELIUS_RPC_URL", "https://api.mainnet-beta.solana.com"),
)


def _fallback_solana_rpc(pool) -> TokenSupplyResponse:
    """Direct Solana JSON-RPC fallback when Express backend is down."""
    import httpx

    now = datetime.now(timezone.utc)
    total_supply_formatted = 100_000_000.0
    total_supply_raw = "100000000000000000"
    decimals = 9
    treasury_sol = 0.0

    try:
        with httpx.Client(timeout=10) as client:
            # Fetch token supply via getTokenSupply RPC
            supply_resp = client.post(
                _SOLANA_RPC,
                json={
                    "jsonrpc": "2.0", "id": 1,
                    "method": "getTokenSupply",
                    "params": [_CRTX_MINT],
                },
            )
            if supply_resp.status_code == 200:
                result = supply_resp.json().get("result", {}).get("value", {})
                if result:
                    total_supply_raw = result.get("amount", total_supply_raw)
                    decimals = result.get("decimals", decimals)
                    total_supply_formatted = result.get("uiAmount", total_supply_formatted) or total_supply_formatted

            # Fetch treasury SOL balance
            bal_resp = client.post(
                _SOLANA_RPC,
                json={
                    "jsonrpc": "2.0", "id": 2,
                    "method": "getBalance",
                    "params": [_TREASURY_ADDR],
                },
            )
            if bal_resp.status_code == 200:
                lamports = bal_resp.json().get("result", {}).get("value", 0)
                treasury_sol = lamports / 1e9

    except Exception as e:
        logger.warning("Direct Solana RPC fallback also failed: %s", e)

    return TokenSupplyResponse(
        symbol="CRTX",
        decimals=decimals,
        total_supply=total_supply_raw,
        total_supply_formatted=total_supply_formatted,
        mint=_CRTX_MINT,
        treasury={"sol_balance": treasury_sol, "address": _TREASURY_ADDR},
        programs={
            "token": _CRTX_MINT,
            "treasury": _TREASURY_ADDR,
        },
        timestamp=now,
    )


@router.get("/token/holders", tags=["token"])
def get_token_holders(mint: str = _CRTX_MINT):
    """Fetch holder distribution and concentration risk for a token."""
    from cortex.data.helius_holders import get_holder_data

    data = get_holder_data(mint)
    return {
        "token_mint": data.get("token_mint", mint),
        "total_holders": data.get("total_holders", 0),
        "top10_pct": data.get("top10_pct", 0.0),
        "top50_pct": data.get("top50_pct", 0.0),
        "hhi": data.get("hhi", 0.0),
        "concentration_risk": data.get("concentration_risk", "unknown"),
        "top_holders": data.get("holders", [])[:20],
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

