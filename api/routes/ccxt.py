"""ccxt multi-exchange data feed endpoints."""

import asyncio
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.models import (
    CcxtExchangesResponse,
    CcxtOHLCVRequest,
    CcxtOHLCVResponse,
    CcxtOrderBookResponse,
    CcxtTickerResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ccxt"])


@router.post("/ccxt/ohlcv", response_model=CcxtOHLCVResponse)
async def get_ohlcv(req: CcxtOHLCVRequest):
    """Fetch OHLCV candlestick data from an exchange via ccxt."""
    from cortex.data.ccxt_feed import fetch_ohlcv

    try:
        df = await asyncio.to_thread(
            fetch_ohlcv, req.symbol, req.timeframe, req.limit, req.exchange,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Exchange error: {e}")

    return CcxtOHLCVResponse(
        symbol=req.symbol,
        exchange=req.exchange or "default",
        timeframe=req.timeframe,
        n_candles=len(df),
        first_timestamp=str(df.index[0]) if len(df) > 0 else "",
        last_timestamp=str(df.index[-1]) if len(df) > 0 else "",
        last_close=float(df["close"].iloc[-1]) if len(df) > 0 else 0.0,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ccxt/orderbook", response_model=CcxtOrderBookResponse)
async def get_order_book(
    symbol: str = Query(..., description="Trading pair, e.g. BTC/USDT"),
    limit: int = Query(20, ge=1, le=100),
    exchange: str | None = Query(None),
):
    """Fetch order book depth from an exchange."""
    from cortex.data.ccxt_feed import fetch_order_book

    try:
        result = await asyncio.to_thread(fetch_order_book, symbol, limit, exchange)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Exchange error: {e}")

    return CcxtOrderBookResponse(
        symbol=result["symbol"],
        exchange=result["exchange"],
        best_bid=result["best_bid"],
        best_ask=result["best_ask"],
        mid_price=result["mid_price"],
        spread=result["spread"],
        spread_bps=result["spread_bps"],
        bid_depth=result["bid_depth"],
        ask_depth=result["ask_depth"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ccxt/ticker", response_model=CcxtTickerResponse)
async def get_ticker(
    symbol: str = Query(..., description="Trading pair, e.g. BTC/USDT"),
    exchange: str | None = Query(None),
):
    """Fetch 24h ticker summary from an exchange."""
    from cortex.data.ccxt_feed import fetch_ticker

    try:
        result = await asyncio.to_thread(fetch_ticker, symbol, exchange)
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"Exchange error: {e}")

    return CcxtTickerResponse(
        symbol=result["symbol"],
        exchange=result["exchange"],
        last=result["last"],
        high=result["high"],
        low=result["low"],
        volume=result["volume"],
        quote_volume=result["quote_volume"],
        change_pct=result["change_pct"],
        vwap=result["vwap"],
        bid=result["bid"],
        ask=result["ask"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/ccxt/exchanges", response_model=CcxtExchangesResponse)
async def list_exchanges():
    """List all supported exchanges."""
    from cortex.data.ccxt_feed import list_exchanges as _list

    try:
        exchanges = _list()
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    return CcxtExchangesResponse(exchanges=exchanges, count=len(exchanges))
