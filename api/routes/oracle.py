"""Oracle price feed endpoints — fully dynamic Pyth Network integration.

Endpoints:
  GET /oracle/feeds       — list all 585+ crypto feeds (with optional search)
  GET /oracle/search      — search feeds by token name
  GET /oracle/prices      — fetch live prices for any tokens (by symbol or feed ID)
  GET /oracle/history     — fetch historical prices at a past timestamp
  GET /oracle/buffer      — get price ring-buffer for a feed
  GET /oracle/status      — oracle subsystem health
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.models import (
    OracleHistoricalResponse,
    OraclePriceItem,
    OraclePricesResponse,
    OracleStatusResponse,
    OracleStreamStatus,
    PythFeedAttributes,
    PythFeedItem,
    PythFeedListResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["oracle"])


@router.get("/oracle/feeds", response_model=PythFeedListResponse)
def get_oracle_feeds(
    query: str = Query(None, description="Filter feeds by name (e.g. 'SOL', 'bitcoin')"),
    asset_type: str = Query("crypto", description="Asset type filter"),
):
    """List all available Pyth price feeds. Supports search filtering."""
    from cortex.data.oracle import list_feeds

    try:
        feeds = list_feeds(asset_type=asset_type, query=query)
    except Exception as exc:
        logger.exception("Feed listing failed")
        raise HTTPException(status_code=502, detail=f"Feed list error: {exc}")

    return PythFeedListResponse(
        feeds=[
            PythFeedItem(id=f["id"], attributes=PythFeedAttributes(**f.get("attributes", {})))
            for f in feeds
        ],
        total=len(feeds),
        query=query,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/oracle/search", response_model=PythFeedListResponse)
def search_oracle_feeds(
    q: str = Query(..., description="Search query (e.g. 'SOL', 'ethereum', 'BONK')"),
):
    """Search Pyth feeds by token name — server-side Hermes search."""
    from cortex.data.oracle import search_feeds

    try:
        feeds = search_feeds(query=q)
    except Exception as exc:
        logger.exception("Feed search failed")
        raise HTTPException(status_code=502, detail=f"Search error: {exc}")

    return PythFeedListResponse(
        feeds=[
            PythFeedItem(id=f["id"], attributes=PythFeedAttributes(**f.get("attributes", {})))
            for f in feeds
        ],
        total=len(feeds),
        query=q,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/oracle/prices", response_model=OraclePricesResponse)
def get_oracle_prices(
    symbols: str = Query(
        None,
        description="Comma-separated token symbols (e.g. 'SOL,BTC,ETH,BONK,JUP')",
    ),
    ids: str = Query(
        None,
        description="Comma-separated Pyth feed IDs (hex, no 0x prefix ok)",
    ),
):
    """Fetch live prices for any tokens. Pass symbols OR feed IDs."""
    from cortex.data.oracle import fetch_prices, fetch_prices_by_symbols, get_latest_prices

    try:
        if symbols:
            sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
            results = fetch_prices_by_symbols(sym_list)
        elif ids:
            id_list = [i.strip() for i in ids.split(",") if i.strip()]
            results = fetch_prices(id_list)
        else:
            cached = get_latest_prices()
            results = list(cached.values())
    except Exception as exc:
        logger.exception("Oracle price fetch failed")
        raise HTTPException(status_code=502, detail=f"Oracle error: {exc}")

    return OraclePricesResponse(
        prices=[OraclePriceItem(**r) for r in results],
        count=len(results),
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/oracle/history", response_model=OracleHistoricalResponse)
def get_oracle_history(
    timestamp: int = Query(..., description="Unix timestamp to query"),
    symbols: str = Query(
        None, description="Comma-separated symbols (e.g. 'SOL,BTC')",
    ),
    ids: str = Query(
        None, description="Comma-separated Pyth feed IDs",
    ),
):
    """Fetch historical prices at a specific past unix timestamp."""
    from cortex.data.oracle import fetch_historical, resolve_feed_ids

    if not symbols and not ids:
        raise HTTPException(status_code=400, detail="Provide symbols or ids parameter")

    try:
        if symbols:
            sym_list = [s.strip() for s in symbols.split(",") if s.strip()]
            feed_ids = resolve_feed_ids(sym_list)
        else:
            feed_ids = [i.strip() for i in ids.split(",") if i.strip()]

        results = fetch_historical(feed_ids=feed_ids, timestamp=timestamp)
    except Exception as exc:
        logger.exception("Historical price fetch failed")
        raise HTTPException(status_code=502, detail=f"History error: {exc}")

    return OracleHistoricalResponse(
        prices=[OraclePriceItem(**r) for r in results],
        query_timestamp=timestamp,
        count=len(results),
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/oracle/buffer")
def get_oracle_buffer(
    feed_id: str = Query(..., description="Pyth feed ID to get buffer for"),
):
    """Return the price ring-buffer history for a specific feed."""
    from cortex.data.oracle import get_price_buffer

    buf = get_price_buffer(feed_id)
    return {"feed_id": feed_id, "entries": buf, "count": len(buf)}


@router.get("/oracle/status", response_model=OracleStatusResponse)
def get_oracle_status():
    """Return oracle subsystem health status."""
    from cortex.data.oracle import get_oracle_status as _status

    raw = _status()
    raw["stream"] = OracleStreamStatus(**raw["stream"])
    return OracleStatusResponse(**raw)

