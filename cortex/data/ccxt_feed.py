"""Multi-exchange cryptocurrency data feed via ccxt.

Provides unified OHLCV, order-book, and ticker data from 100+ exchanges
through a single interface. Falls back gracefully when ccxt is not installed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from cortex.config import CCXT_DEFAULT_EXCHANGE

logger = logging.getLogger(__name__)

_CCXT_AVAILABLE = False
try:
    import ccxt as _ccxt
    _CCXT_AVAILABLE = True
except ImportError:
    _ccxt = None  # type: ignore[assignment]

_exchange_cache: dict[str, Any] = {}


def _get_exchange(exchange_id: str | None = None) -> Any:
    """Get or create a ccxt exchange instance (cached)."""
    if not _CCXT_AVAILABLE:
        raise RuntimeError("ccxt not installed. Install with: pip install ccxt")

    eid = exchange_id or CCXT_DEFAULT_EXCHANGE
    if eid not in _exchange_cache:
        cls = getattr(_ccxt, eid, None)
        if cls is None:
            raise ValueError(f"Unknown exchange: {eid}. Available: {_ccxt.exchanges[:10]}...")
        _exchange_cache[eid] = cls({"enableRateLimit": True})
    return _exchange_cache[eid]


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "1d",
    limit: int = 500,
    exchange_id: str | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV candlestick data from an exchange.

    Args:
        symbol: Trading pair (e.g. "BTC/USDT", "SOL/USDT").
        timeframe: Candle interval ("1m", "5m", "1h", "4h", "1d").
        limit: Number of candles to fetch.
        exchange_id: Exchange name (default: config CCXT_DEFAULT_EXCHANGE).

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume.
    """
    exchange = _get_exchange(exchange_id)
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df


def fetch_order_book(
    symbol: str,
    limit: int = 20,
    exchange_id: str | None = None,
) -> dict[str, Any]:
    """Fetch order book depth.

    Returns:
        Dict with bids, asks (each list of [price, size]), spread, mid_price.
    """
    exchange = _get_exchange(exchange_id)
    ob = exchange.fetch_order_book(symbol, limit=limit)
    bids = ob.get("bids", [])
    asks = ob.get("asks", [])

    best_bid = bids[0][0] if bids else 0.0
    best_ask = asks[0][0] if asks else 0.0
    mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else 0.0
    spread = (best_ask - best_bid) if (best_bid and best_ask) else 0.0
    spread_bps = (spread / mid * 10_000) if mid > 0 else 0.0

    return {
        "symbol": symbol,
        "exchange": exchange_id or CCXT_DEFAULT_EXCHANGE,
        "bids": bids[:limit],
        "asks": asks[:limit],
        "best_bid": best_bid,
        "best_ask": best_ask,
        "mid_price": mid,
        "spread": spread,
        "spread_bps": round(spread_bps, 2),
        "bid_depth": sum(b[1] for b in bids[:limit]),
        "ask_depth": sum(a[1] for a in asks[:limit]),
    }


def fetch_ticker(
    symbol: str,
    exchange_id: str | None = None,
) -> dict[str, Any]:
    """Fetch 24h ticker summary.

    Returns:
        Dict with last, high, low, volume, change_pct, vwap.
    """
    exchange = _get_exchange(exchange_id)
    t = exchange.fetch_ticker(symbol)
    return {
        "symbol": symbol,
        "exchange": exchange_id or CCXT_DEFAULT_EXCHANGE,
        "last": t.get("last"),
        "high": t.get("high"),
        "low": t.get("low"),
        "volume": t.get("baseVolume"),
        "quote_volume": t.get("quoteVolume"),
        "change_pct": t.get("percentage"),
        "vwap": t.get("vwap"),
        "bid": t.get("bid"),
        "ask": t.get("ask"),
    }


def fetch_multi_exchange_ohlcv(
    symbol: str,
    exchanges: list[str] | None = None,
    timeframe: str = "1d",
    limit: int = 100,
) -> dict[str, pd.DataFrame]:
    """Fetch OHLCV from multiple exchanges for cross-venue comparison.

    Args:
        symbol: Trading pair.
        exchanges: List of exchange IDs. Default: binance, kraken, coinbase.
        timeframe: Candle interval.
        limit: Number of candles.

    Returns:
        Dict mapping exchange_id to DataFrame.
    """
    if exchanges is None:
        exchanges = ["binance", "kraken", "coinbasepro"]

    results: dict[str, pd.DataFrame] = {}
    for eid in exchanges:
        try:
            results[eid] = fetch_ohlcv(symbol, timeframe, limit, exchange_id=eid)
        except Exception as e:
            logger.warning("Failed to fetch %s from %s: %s", symbol, eid, e)
    return results


def ohlcv_to_returns(df: pd.DataFrame, pct: bool = True) -> pd.Series:
    """Convert OHLCV DataFrame to log-return series.

    Args:
        df: DataFrame with 'close' column.
        pct: If True, returns in % (consistent with cortex convention).

    Returns:
        pd.Series of log-returns.
    """
    close = df["close"]
    log_ret = np.log(close / close.shift(1)).dropna()
    if pct:
        log_ret *= 100
    return log_ret


def list_exchanges() -> list[str]:
    """Return list of all supported exchange IDs."""
    if not _CCXT_AVAILABLE:
        raise RuntimeError("ccxt not installed")
    return list(_ccxt.exchanges)
