"""Read-only Solana staking endpoints — pool state, user positions, tiers."""

import logging
import struct

from fastapi import APIRouter, HTTPException
from solders.pubkey import Pubkey  # type: ignore[import-untyped]

from api.models import (
    StakingPoolResponse,
    StakingTier,
    StakingTiersResponse,
    StakingUserResponse,
)
from api.solana_rpc import (
    cached,
    derive_pda,
    deserialize_anchor_account,
    fetch_account_info,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/staking", tags=["staking"])

STAKING_PROGRAM_ID = "rYantWFyB4PsL36r9XB7nUb8TQ1pAhn9A87S6TbpMsr"


def _fetch_pool_data() -> dict:
    """Fetch and deserialize the staking pool account."""
    pool_pda = derive_pda([b"pool"], STAKING_PROGRAM_ID)
    account = fetch_account_info(pool_pda)
    if account is None or account.get("data") is None:
        raise HTTPException(status_code=404, detail="Staking pool account not found")

    try:
        body = deserialize_anchor_account(account["data"])
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Deserialization error: {exc}")

    # Minimum required: 32 + 32 + 8 + 8 + 8 + 2 = 90 bytes
    if len(body) < 90:
        raise HTTPException(
            status_code=500,
            detail=f"Pool account data too short ({len(body)} bytes, need >=90)",
        )

    authority = str(Pubkey.from_bytes(body[0:32]))
    reward_mint = str(Pubkey.from_bytes(body[32:64]))
    reward_rate = struct.unpack_from("<Q", body, 64)[0]
    total_staked = struct.unpack_from("<Q", body, 72)[0]
    last_update_time = struct.unpack_from("<q", body, 80)[0]
    num_tiers = struct.unpack_from("<H", body, 88)[0]

    tiers: list[StakingTier] = []
    offset = 90
    for _ in range(num_tiers):
        if offset + 10 > len(body):
            break
        min_amount = struct.unpack_from("<Q", body, offset)[0]
        multiplier_bps = struct.unpack_from("<H", body, offset + 8)[0]
        tiers.append(StakingTier(min_amount=min_amount, multiplier_bps=multiplier_bps))
        offset += 10

    return {
        "pool_pda": pool_pda,
        "authority": authority,
        "reward_mint": reward_mint,
        "reward_rate": reward_rate,
        "total_staked": total_staked,
        "last_update_time": last_update_time,
        "tiers": tiers,
    }


@cached(ttl_seconds=60)
def _get_pool_cached() -> dict:
    return _fetch_pool_data()


@router.get("/pool", response_model=StakingPoolResponse)
def get_staking_pool() -> StakingPoolResponse:
    """Return staking pool state: authority, reward rate, total staked, tiers."""
    return StakingPoolResponse(**_get_pool_cached())


@router.get("/user/{wallet}", response_model=StakingUserResponse)
def get_staking_user(wallet: str) -> StakingUserResponse:
    """Return a user's staking position and computed pending rewards."""
    try:
        wallet_pubkey = Pubkey.from_string(wallet)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid wallet address: {wallet}")

    stake_pda = derive_pda(
        [b"stake", bytes(wallet_pubkey)], STAKING_PROGRAM_ID
    )
    account = fetch_account_info(stake_pda)
    if account is None or account.get("data") is None:
        raise HTTPException(status_code=404, detail="User stake account not found")

    try:
        body = deserialize_anchor_account(account["data"])
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=f"Deserialization error: {exc}")

    # 32 + 8 + 8 + 2 + 8 = 58 bytes minimum
    if len(body) < 58:
        raise HTTPException(
            status_code=500,
            detail=f"User stake data too short ({len(body)} bytes, need >=58)",
        )

    owner = str(Pubkey.from_bytes(body[0:32]))
    amount_staked = struct.unpack_from("<Q", body, 32)[0]
    reward_debt = struct.unpack_from("<Q", body, 40)[0]
    tier_index = struct.unpack_from("<H", body, 48)[0]
    stake_timestamp = struct.unpack_from("<q", body, 50)[0]

    # Compute pending rewards: amount_staked * pool.reward_rate / 1e9 - reward_debt
    pending_rewards = 0.0
    try:
        pool = _get_pool_cached()
        pending_rewards = (amount_staked * pool["reward_rate"] / 1e9) - reward_debt
        if pending_rewards < 0:
            pending_rewards = 0.0
    except Exception:
        logger.warning("Could not compute pending rewards for %s", wallet)

    return StakingUserResponse(
        wallet=wallet,
        stake_pda=stake_pda,
        owner=owner,
        amount_staked=amount_staked,
        reward_debt=reward_debt,
        tier_index=tier_index,
        stake_timestamp=stake_timestamp,
        pending_rewards=pending_rewards,
    )


@router.get("/tiers", response_model=StakingTiersResponse)
def get_staking_tiers() -> StakingTiersResponse:
    """Return just the tier definitions from the pool."""
    pool = _get_pool_cached()
    return StakingTiersResponse(tiers=pool["tiers"])
