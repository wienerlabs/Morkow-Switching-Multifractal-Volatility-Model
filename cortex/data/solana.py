"""
Solana DeFi data adapter for MSM-VaR model.
Fetches OHLCV, funding rates, and liquidity metrics from on-chain sources.
Outputs DataFrames compatible with the existing yfinance-based pipeline.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import httpx

logger = logging.getLogger(__name__)

BIRDEYE_BASE = "https://public-api.birdeye.so"
DRIFT_DATA_API = "https://data.api.drift.trade"
RAYDIUM_API = "https://api-v3.raydium.io"

# Well-known SPL token mint addresses
TOKEN_REGISTRY: dict[str, str] = {
    "SOL": "So11111111111111111111111111111111111111112",
    "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
    "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
    "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
    "USDC": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    "USDT": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
    "ORCA": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
    "MNDE": "MNDEFzGvMt87ueuHvVU9VcTqsAP5b3fTGPsHuuPA5ey",
    "MSOL": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So",
    "JITOSOL": "J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn",
}

# Drift perp market symbols
DRIFT_PERP_MARKETS: dict[str, int] = {
    "SOL-PERP": 0,
    "BTC-PERP": 1,
    "ETH-PERP": 2,
    "APT-PERP": 3,
    "BONK-PERP": 4,
    "MATIC-PERP": 5,
    "ARB-PERP": 6,
    "DOGE-PERP": 7,
    "BNB-PERP": 8,
    "SUI-PERP": 9,
    "JUP-PERP": 21,
    "JTO-PERP": 22,
    "WIF-PERP": 23,
}


def _resolve_token_address(token: str) -> str:
    if len(token) > 20:
        return token
    upper = token.upper()
    if upper in TOKEN_REGISTRY:
        return TOKEN_REGISTRY[upper]
    raise ValueError(
        f"Unknown token symbol '{token}'. "
        f"Pass a mint address or use one of: {list(TOKEN_REGISTRY.keys())}"
    )


def _birdeye_headers() -> dict[str, str]:
    api_key = os.environ.get("BIRDEYE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "BIRDEYE_API_KEY environment variable is required. "
            "Get one at https://birdeye.so"
        )
    return {"X-API-KEY": api_key, "x-chain": "solana", "accept": "application/json"}


def _to_unix(dt: datetime | str) -> int:
    if isinstance(dt, str):
        dt = datetime.fromisoformat(dt.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def get_token_ohlcv(
    token: str,
    start_date: str | datetime,
    end_date: str | datetime,
    interval: str = "1D",
    quote_token: str = "USDC",
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Birdeye API.

    Returns DataFrame with columns [Open, High, Low, Close, Volume]
    and DatetimeIndex — same shape as yfinance output.

    Args:
        token: Symbol (SOL, RAY, JUP, BONK) or mint address.
        start_date: ISO date string or datetime.
        end_date: ISO date string or datetime.
        interval: Candle size — 1m, 5m, 15m, 1H, 4H, 1D, 1W.
        quote_token: Quote currency for the pair (default USDC).
    """
    address = _resolve_token_address(token)
    time_from = _to_unix(start_date)
    time_to = _to_unix(end_date)

    params = {
        "address": address,
        "type": interval,
        "time_from": time_from,
        "time_to": time_to,
    }

    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{BIRDEYE_BASE}/defi/ohlcv",
            headers=_birdeye_headers(),
            params=params,
        )
        resp.raise_for_status()
        data = resp.json()

    items = data.get("data", {}).get("items", [])
    if not items:
        raise ValueError(f"No OHLCV data returned for {token} in the given range")

    records = []
    for candle in items:
        records.append({
            "Date": pd.Timestamp(candle["unixTime"], unit="s", tz="UTC"),
            "Open": float(candle["o"]),
            "High": float(candle["h"]),
            "Low": float(candle["l"]),
            "Close": float(candle["c"]),
            "Volume": float(candle["v"]),
        })

    df = pd.DataFrame(records).set_index("Date").sort_index()
    df.index.name = "Date"
    return df


def get_funding_rates(
    perp_market: str,
    start_date: Optional[str | datetime] = None,
    end_date: Optional[str | datetime] = None,
) -> pd.DataFrame:
    """
    Fetch funding rate history from Drift Protocol Data API.

    Returns DataFrame with columns [fundingRate, oracleTwap, markTwap]
    and DatetimeIndex.

    Args:
        perp_market: Market name (e.g. 'SOL-PERP') or market index as int.
        start_date: Optional start filter.
        end_date: Optional end filter.
    """
    if isinstance(perp_market, str) and perp_market in DRIFT_PERP_MARKETS:
        market_index = DRIFT_PERP_MARKETS[perp_market]
        market_name = perp_market
    elif isinstance(perp_market, int):
        market_index = perp_market
        market_name = next(
            (k for k, v in DRIFT_PERP_MARKETS.items() if v == perp_market),
            f"MARKET-{perp_market}",
        )
    else:
        raise ValueError(
            f"Unknown perp market '{perp_market}'. "
            f"Use one of: {list(DRIFT_PERP_MARKETS.keys())}"
        )

    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{DRIFT_DATA_API}/fundingRates",
            params={"marketIndex": market_index, "marketType": "perp"},
        )
        resp.raise_for_status()
        data = resp.json()

    records = data if isinstance(data, list) else data.get("data", [])
    if not records:
        raise ValueError(f"No funding rate data for {market_name}")

    rows = []
    for rec in records:
        ts = rec.get("ts") or rec.get("timestamp")
        rows.append({
            "Date": pd.Timestamp(int(ts), unit="s", tz="UTC"),
            "fundingRate": float(rec.get("fundingRate", 0)) / 1e9,
            "oracleTwap": float(rec.get("oraclePriceTwap", 0)) / 1e6,
            "markTwap": float(rec.get("markPriceTwap", 0)) / 1e6,
        })

    df = pd.DataFrame(rows).set_index("Date").sort_index()

    if start_date:
        df = df[df.index >= pd.Timestamp(_to_unix(start_date), unit="s", tz="UTC")]
    if end_date:
        df = df[df.index <= pd.Timestamp(_to_unix(end_date), unit="s", tz="UTC")]

    return df


def get_liquidity_metrics(
    pool_address: str,
) -> dict:
    """
    Fetch liquidity depth metrics from Raydium API.

    Returns dict with pool TVL, volume, and depth info.

    Args:
        pool_address: Raydium AMM pool address.
    """
    with httpx.Client(timeout=30) as client:
        resp = client.get(
            f"{RAYDIUM_API}/pools/info/ids",
            params={"ids": pool_address},
        )
        resp.raise_for_status()
        data = resp.json()

    pools = data.get("data", [])
    if not pools:
        raise ValueError(f"No pool data found for {pool_address}")

    pool = pools[0] if isinstance(pools, list) else pools

    return {
        "pool_address": pool_address,
        "tvl": float(pool.get("tvl", 0)),
        "volume_24h": float(pool.get("day", {}).get("volume", 0)),
        "fee_24h": float(pool.get("day", {}).get("volumeFee", 0)),
        "apr_24h": float(pool.get("day", {}).get("apr", 0)),
        "price": float(pool.get("price", 0)),
        "mint_a": pool.get("mintA", {}).get("symbol", ""),
        "mint_b": pool.get("mintB", {}).get("symbol", ""),
        "liquidity": float(pool.get("lpAmount", 0)),
    }


def ohlcv_to_returns(df: pd.DataFrame) -> pd.Series:
    """
    Convert OHLCV DataFrame to log-returns in % (same format as MSM-VaR pipeline).
    Uses Close prices, computes 100 * diff(log(close)).
    """
    close = df["Close"].dropna()
    rets = 100 * np.diff(np.log(close.values))
    return pd.Series(rets, index=close.index[1:], name="r")

