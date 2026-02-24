"""Read-only Solana vault endpoints — vault state and user position."""

import logging
import struct

import base58
from fastapi import APIRouter, HTTPException, Query
from solders.pubkey import Pubkey  # type: ignore[import-untyped]

from api.models import VaultInfoResponse, VaultUserResponse
from api.solana_rpc import (
    _rpc_post,
    cached,
    derive_pda,
    deserialize_anchor_account,
    fetch_account_info,
    get_rpc_url,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vault", tags=["vault"])

VAULT_PROGRAM_ID = "5Rkn4B2CAcAiizUyHrxxBTRcAsZcRaLSMi8gdzXUW1nX"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
TOKEN_PROGRAM_ID = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"

VAULT_STATE_MAP = {0: "Active", 1: "Locked", 2: "Deprecated"}


def _parse_vault_fields(data: bytes) -> dict:
    """Parse vault account fields from raw bytes (after discriminator strip)."""
    if len(data) < 147:
        raise ValueError(f"Vault data too short: {len(data)} bytes, need >= 147")

    authority = base58.b58encode(data[0:32]).decode()
    asset_mint = base58.b58encode(data[32:64]).decode()
    share_mint = base58.b58encode(data[64:96]).decode()
    total_assets = struct.unpack_from("<Q", data, 96)[0]
    total_shares = struct.unpack_from("<Q", data, 104)[0]
    performance_fee_bps = struct.unpack_from("<H", data, 112)[0]
    state_byte = data[114]
    name_raw = data[115:147]
    name = name_raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace").strip()

    return {
        "authority": authority,
        "asset_mint": asset_mint,
        "share_mint": share_mint,
        "total_assets": total_assets,
        "total_shares": total_shares,
        "performance_fee_bps": performance_fee_bps,
        "state": VAULT_STATE_MAP.get(state_byte, f"Unknown({state_byte})"),
        "name": name,
    }


@router.get("/info", response_model=VaultInfoResponse)
@cached(ttl_seconds=30)
def get_vault_info(
    mint: str = Query(default=USDC_MINT, description="Asset mint address"),
) -> VaultInfoResponse:
    mint_bytes = bytes(Pubkey.from_string(mint))
    vault_pda = derive_pda([b"vault", mint_bytes], VAULT_PROGRAM_ID)

    account = fetch_account_info(vault_pda)
    if account is None or account.get("data") is None:
        raise HTTPException(status_code=404, detail="Vault account not found")

    try:
        body = deserialize_anchor_account(account["data"])
        fields = _parse_vault_fields(body)
    except (ValueError, struct.error) as exc:
        logger.error("Vault deserialization failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Vault deserialization error: {exc}"
        )

    return VaultInfoResponse(vault_pda=vault_pda, **fields)


@router.get("/user/{wallet}", response_model=VaultUserResponse)
def get_vault_user(
    wallet: str,
    mint: str = Query(default=USDC_MINT, description="Asset mint address"),
) -> VaultUserResponse:
    # First get vault info to find share_mint and compute share price
    mint_bytes = bytes(Pubkey.from_string(mint))
    vault_pda = derive_pda([b"vault", mint_bytes], VAULT_PROGRAM_ID)

    account = fetch_account_info(vault_pda)
    if account is None or account.get("data") is None:
        raise HTTPException(status_code=404, detail="Vault account not found")

    try:
        body = deserialize_anchor_account(account["data"])
        fields = _parse_vault_fields(body)
    except (ValueError, struct.error) as exc:
        logger.error("Vault deserialization failed: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Vault deserialization error: {exc}"
        )

    share_mint = fields["share_mint"]
    total_assets = fields["total_assets"]
    total_shares = fields["total_shares"]

    # Fetch user's share token accounts
    share_balance = _fetch_token_balance(wallet, share_mint)

    estimated_assets = 0.0
    if total_shares > 0 and share_balance > 0:
        estimated_assets = share_balance * total_assets / total_shares

    return VaultUserResponse(
        wallet=wallet,
        share_balance=share_balance,
        estimated_assets=estimated_assets,
    )


def _fetch_token_balance(owner: str, mint: str) -> int:
    """Get the total token balance for an owner+mint via getTokenAccountsByOwner."""
    get_rpc_url()  # ensure RPC is configured
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getTokenAccountsByOwner",
        "params": [
            owner,
            {"mint": mint},
            {"encoding": "jsonParsed"},
        ],
    }
    data = _rpc_post(payload)
    result = data.get("result")
    if result is None or result.get("value") is None:
        return 0

    total = 0
    for acct in result["value"]:
        try:
            info = acct["account"]["data"]["parsed"]["info"]
            amount = int(info["tokenAmount"]["amount"])
            total += amount
        except (KeyError, TypeError, ValueError):
            continue
    return total
