"""Webacy token safety and wallet risk endpoints for standalone UI screening."""

import logging

from fastapi import APIRouter, HTTPException

from api.models import (
    WebacySanctionsResponse,
    WebacyStatusResponse,
    WebacyTokenSafetyResponse,
    WebacyWalletRiskResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webacy", tags=["webacy"])


@router.get("/status", response_model=WebacyStatusResponse)
def get_webacy_status() -> WebacyStatusResponse:
    """Health check — returns whether the Webacy integration is enabled."""
    try:
        from cortex.webacy import is_webacy_enabled
    except ImportError:
        return WebacyStatusResponse(enabled=False)

    return WebacyStatusResponse(enabled=is_webacy_enabled())


@router.get("/token/{mint_address}", response_model=WebacyTokenSafetyResponse)
async def get_token_safety(mint_address: str) -> WebacyTokenSafetyResponse:
    """Run Webacy token safety analysis for a Solana mint address."""
    try:
        from cortex.webacy import check_token_safety
    except ImportError:
        raise HTTPException(status_code=503, detail="Webacy module not available")

    try:
        result = await check_token_safety(mint_address)
    except Exception as exc:
        logger.error("Webacy token safety check failed for %s: %s", mint_address, exc)
        raise HTTPException(status_code=503, detail=f"Webacy API unavailable: {exc}")

    return WebacyTokenSafetyResponse(**result)


@router.get("/wallet/{address}", response_model=WebacyWalletRiskResponse)
async def get_wallet_risk(address: str) -> WebacyWalletRiskResponse:
    """Run Webacy wallet threat risk analysis for a Solana address."""
    try:
        from cortex.webacy import check_wallet_risk
    except ImportError:
        raise HTTPException(status_code=503, detail="Webacy module not available")

    try:
        result = await check_wallet_risk(address)
    except Exception as exc:
        logger.error("Webacy wallet risk check failed for %s: %s", address, exc)
        raise HTTPException(status_code=503, detail=f"Webacy API unavailable: {exc}")

    return WebacyWalletRiskResponse(**result)


@router.get("/sanctions/{address}", response_model=WebacySanctionsResponse)
async def get_sanctions_status(address: str) -> WebacySanctionsResponse:
    """Check OFAC/sanctions status for a Solana address via Webacy."""
    try:
        from cortex.webacy import check_wallet_sanctions
    except ImportError:
        raise HTTPException(status_code=503, detail="Webacy module not available")

    try:
        result = await check_wallet_sanctions(address)
    except Exception as exc:
        logger.error("Webacy sanctions check failed for %s: %s", address, exc)
        raise HTTPException(status_code=503, detail=f"Webacy API unavailable: {exc}")

    return WebacySanctionsResponse(**result)
