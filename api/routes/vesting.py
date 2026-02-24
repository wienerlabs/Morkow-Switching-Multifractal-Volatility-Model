"""Read-only Solana vesting schedule endpoints."""

import logging
import struct
import time

from fastapi import APIRouter, HTTPException
from solders.pubkey import Pubkey  # type: ignore[import-untyped]

from api.models import (
    VestingCategoriesResponse,
    VestingCategoryItem,
    VestingScheduleResponse,
)
from api.solana_rpc import cached, derive_pda, deserialize_anchor_account, fetch_account_info

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vesting", tags=["vesting"])

VESTING_PROGRAM_ID = "5PDicSrsh9zyVMwDjL61WXHuNkzQTk6rpCs5CnGzpXns"

CATEGORY_MAP: dict[int, tuple[str, str]] = {
    0: ("Team", "Core team allocation"),
    1: ("Advisor", "Advisory board allocation"),
    2: ("Community", "Community rewards"),
    3: ("Ecosystem", "Ecosystem development fund"),
}


@router.get("/categories", response_model=VestingCategoriesResponse)
def get_vesting_categories() -> VestingCategoriesResponse:
    return VestingCategoriesResponse(
        categories=[
            VestingCategoryItem(id=cid, name=name, description=desc)
            for cid, (name, desc) in sorted(CATEGORY_MAP.items())
        ]
    )


@router.get("/schedule/{wallet}", response_model=VestingScheduleResponse)
@cached(ttl_seconds=60)
def get_vesting_schedule(wallet: str) -> VestingScheduleResponse:
    try:
        wallet_pubkey = Pubkey.from_string(wallet)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid wallet address: {wallet}")

    pda = derive_pda(
        seeds=[b"vesting", bytes(wallet_pubkey)],
        program_id=VESTING_PROGRAM_ID,
    )

    account = fetch_account_info(pda)
    if account is None or account.get("data") is None:
        raise HTTPException(status_code=404, detail="No vesting schedule found for wallet")

    try:
        raw = deserialize_anchor_account(account["data"])
    except ValueError as e:
        logger.error("Vesting deserialization failed for %s: %s", wallet, e)
        raise HTTPException(status_code=500, detail="Failed to deserialize vesting account")

    # Expected layout after 8-byte Anchor discriminator:
    #   0-31:  beneficiary (Pubkey, 32 bytes)
    #  32-63:  mint (Pubkey, 32 bytes)
    #  64-71:  total_amount (u64 LE)
    #  72-79:  released_amount (u64 LE)
    #  80-87:  start_time (i64 LE)
    #  88-95:  end_time (i64 LE)
    #  96:     category (u8)
    MIN_DATA_LEN = 97
    if len(raw) < MIN_DATA_LEN:
        logger.error("Vesting account data too short: %d bytes (need %d)", len(raw), MIN_DATA_LEN)
        raise HTTPException(status_code=500, detail="Vesting account data too short")

    beneficiary = str(Pubkey.from_bytes(raw[0:32]))
    mint = str(Pubkey.from_bytes(raw[32:64]))
    total_amount = struct.unpack_from("<Q", raw, 64)[0]
    released_amount = struct.unpack_from("<Q", raw, 72)[0]
    start_time = struct.unpack_from("<q", raw, 80)[0]
    end_time = struct.unpack_from("<q", raw, 88)[0]
    category = raw[96]

    now = int(time.time())
    if now < start_time:
        claimable = 0
    elif now >= end_time:
        claimable = total_amount - released_amount
    else:
        elapsed = now - start_time
        duration = end_time - start_time
        vested = total_amount * elapsed // duration
        claimable = max(0, vested - released_amount)

    if total_amount > 0:
        duration = end_time - start_time
        if duration > 0:
            elapsed_clamped = min(max(now - start_time, 0), duration)
            percent_vested = round(elapsed_clamped / duration * 100, 2)
        else:
            percent_vested = 100.0
    else:
        percent_vested = 0.0

    category_name = CATEGORY_MAP.get(category, (f"Unknown({category})", ""))[0]

    return VestingScheduleResponse(
        beneficiary=beneficiary,
        mint=mint,
        total_amount=total_amount,
        released_amount=released_amount,
        start_time=start_time,
        end_time=end_time,
        category=category,
        category_name=category_name,
        claimable_amount=claimable,
        percent_vested=percent_vested,
    )
