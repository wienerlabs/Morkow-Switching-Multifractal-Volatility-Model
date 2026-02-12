"""Oracle price feed endpoints — Pyth Network integration."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.models import OraclePriceItem, OraclePricesResponse, OracleStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["oracle"])


@router.get("/oracle/prices", response_model=OraclePricesResponse)
def get_oracle_prices():
    """Return latest oracle prices for all tracked tokens."""
    from cortex.data.oracle import get_latest_prices

    try:
        raw = get_latest_prices()
    except Exception as exc:
        logger.exception("Oracle price fetch failed")
        raise HTTPException(status_code=502, detail=f"Oracle error: {exc}")

    prices = {
        token: OraclePriceItem(token=token, **data)
        for token, data in raw.items()
    }
    return OraclePricesResponse(
        prices=prices,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/oracle/status", response_model=OracleStatusResponse)
def get_oracle_status():
    """Return oracle subsystem health status."""
    from cortex.data.oracle import get_oracle_status as _status

    return OracleStatusResponse(**_status())

