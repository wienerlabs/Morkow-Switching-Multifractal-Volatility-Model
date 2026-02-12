"""Macro market indicator endpoints."""

import logging

from fastapi import APIRouter, HTTPException

from api.models import (
    BtcDominanceItem,
    FearGreedItem,
    MacroIndicatorsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["macro"])


@router.get("/macro/indicators", response_model=MacroIndicatorsResponse)
def get_macro_indicators():
    """Return macro market indicators (fear/greed, BTC dominance)."""
    from cortex.data.macro import get_macro_indicators as _indicators

    try:
        result = _indicators()
    except Exception as exc:
        logger.exception("Macro indicators fetch failed")
        raise HTTPException(status_code=502, detail=f"Macro error: {exc}")

    return MacroIndicatorsResponse(
        fear_greed=FearGreedItem(**result["fear_greed"]),
        btc_dominance=BtcDominanceItem(**result["btc_dominance"]),
        risk_level=result["risk_level"],
        timestamp=result["timestamp"],
    )

