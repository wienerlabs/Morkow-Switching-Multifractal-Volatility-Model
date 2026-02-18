"""
Solana DeFi data adapter for MSM-VaR model.
Fetches OHLCV, funding rates, and liquidity metrics from on-chain sources.
Outputs DataFrames compatible with the existing yfinance-based pipeline.
"""

__all__ = [
    "is_available",
    "TOKEN_REGISTRY",
    "fetch_ohlcv",
    "fetch_funding_rates",
    "fetch_liquidity",
]

import os
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from cortex.config import (
    BIRDEYE_BASE,
    DRIFT_DATA_API,
    RAYDIUM_API,
)
from cortex.data.rpc_failover import get_resilient_pool

logger = logging.getLogger(__name__)

_pool = get_resilient_pool()


def is_available() -> bool:
    """Check if Birdeye API key is configured."""
    return bool(os.environ.get("BIRDEYE_API_KEY", ""))

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

# Well-known DEX program IDs (Solana mainnet)
DEX_PROGRAMS: dict[str, str] = {
    "raydium_amm_v4": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    "raydium_clmm": "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK",
    "orca_whirlpool": "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",
    "meteora_dlmm": "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo",
}

# Lending/perp protocol program IDs (for liquidation events)
LENDING_PROGRAMS: dict[str, str] = {
    "So1endEVFfnCjSQnx7qoiMajcpyBrm2kRg4rJNQer6": "Solend",
    "dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH": "Drift",
    "4MangoMjqJ2firMokCjjGPuH8rk3EyL6zmuDAsgzgz": "Mango v4",
    "MFv2hWf31Z9kbCa1snEPYctwafyhdvnV7FZnsebVacA": "MarginFi",
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

    resp = _pool.get(
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

    resp = _pool.get(
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
    resp = _pool.get(
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



def get_pool_liquidity(pool_address: str) -> dict:
    """
    Fetch detailed liquidity metrics from a Raydium pool for LVaR calculations.

    Returns TVL, volume, fee tier, and derived liquidity depth metrics
    needed for market impact and spread estimation.

    Args:
        pool_address: Raydium AMM pool address.
    """
    raw = get_liquidity_metrics(pool_address)

    tvl = raw["tvl"]
    volume_24h = raw["volume_24h"]

    depth_ratio = tvl / volume_24h if volume_24h > 0 else float("inf")
    depth_score = min(100.0, depth_ratio * 10.0)

    fee_rate = raw["fee_24h"] / volume_24h if volume_24h > 0 else 0.003
    estimated_spread_pct = fee_rate * 100.0 + (1.0 / (1.0 + tvl / 1e6)) * 0.5

    return {
        "pool_address": pool_address,
        "tvl": tvl,
        "volume_24h": volume_24h,
        "depth_ratio": float(depth_ratio) if depth_ratio != float("inf") else None,
        "depth_score": float(depth_score),
        "fee_rate": float(fee_rate),
        "estimated_spread_pct": float(estimated_spread_pct),
        "mint_a": raw["mint_a"],
        "mint_b": raw["mint_b"],
    }


def get_market_depth(
    token: str,
    trade_size_usd: float = 10_000.0,
) -> dict:
    """
    Estimate market depth and price impact for a given trade size.

    Uses Birdeye price + volume data to estimate how much a trade
    would move the price. For AMM pools, impact ≈ trade_size / (2 * TVL).

    Args:
        token: Token symbol or mint address.
        trade_size_usd: Proposed trade size in USD.
    """
    address = _resolve_token_address(token)

    resp = _pool.get(
        f"{BIRDEYE_BASE}/defi/price",
        headers=_birdeye_headers(),
        params={"address": address},
    )
    resp.raise_for_status()
    price_data = resp.json().get("data", {})
    current_price = float(price_data.get("value", 0))

    resp2 = _pool.get(
        f"{BIRDEYE_BASE}/defi/token_overview",
        headers=_birdeye_headers(),
        params={"address": address},
    )
    resp2.raise_for_status()
    overview = resp2.json().get("data", {})

    volume_24h = float(overview.get("v24hUSD", 0))
    liquidity = float(overview.get("liquidity", 0))

    if liquidity > 0:
        impact_pct = (trade_size_usd / (2.0 * liquidity)) * 100.0
    elif volume_24h > 0:
        participation = trade_size_usd / volume_24h
        impact_pct = np.sqrt(participation) * 100.0
    else:
        impact_pct = float("nan")

    return {
        "token": token,
        "current_price": current_price,
        "volume_24h_usd": volume_24h,
        "liquidity_usd": liquidity,
        "trade_size_usd": trade_size_usd,
        "estimated_impact_pct": float(impact_pct),
        "estimated_impact_usd": float(impact_pct / 100.0 * trade_size_usd) if not np.isnan(impact_pct) else None,
        "participation_rate": float(trade_size_usd / volume_24h) if volume_24h > 0 else None,
    }



# Display name overrides for well-known wrapped/native tokens
_DISPLAY_NAMES: dict[str, str] = {
    "So11111111111111111111111111111111111111112": "Solana",
}


def get_token_metadata(token: str) -> dict:
    """Fetch token metadata from Birdeye /defi/token_overview.

    Returns name, symbol, logo, price, 24h change, market cap, volume,
    liquidity, holder count, deployer address, creation date, and DEX info.
    """
    address = _resolve_token_address(token)

    resp = _pool.get(
        f"{BIRDEYE_BASE}/defi/token_overview",
        headers=_birdeye_headers(),
        params={"address": address},
    )
    resp.raise_for_status()
    d = resp.json().get("data", {})

    price = float(d.get("price", 0) or 0)
    price_change_24h = float(d.get("priceChange24hPercent", 0) or 0)
    mc = float(d.get("marketCap", 0) or d.get("fdv", 0) or 0)
    volume_24h = float(d.get("v24hUSD", 0) or 0)
    liquidity = float(d.get("liquidity", 0) or 0)
    holder = int(d.get("holder", 0) or 0)

    extensions = d.get("extensions", {}) or {}

    # createdAt is not in token_overview — only use it if actually present
    raw_created = d.get("createdAt")
    created_at = raw_created if raw_created else None

    raw_name = d.get("name") or d.get("symbol") or address[:8]
    display_name = _DISPLAY_NAMES.get(address, raw_name)

    return {
        "address": address,
        "name": display_name,
        "symbol": d.get("symbol", ""),
        "logo_uri": d.get("logoURI") or d.get("icon") or extensions.get("imageUrl") or "",
        "decimals": int(d.get("decimals", 0) or 0),
        "price_usd": price,
        "price_change_24h_pct": price_change_24h,
        "market_cap": mc,
        "volume_24h_usd": volume_24h,
        "liquidity_usd": liquidity,
        "holder_count": holder,
        "deployer": d.get("owner") or d.get("creator") or "",
        "created_at": created_at,
        "dex_platform": d.get("lastTradeHumanTime") and "Solana DEX" or "",
    }


# ── DexScreener-enhanced price feeds ──


def get_dexscreener_price(token: str) -> dict | None:
    """Fetch token price from DexScreener as alternative source.

    Returns dict with price, source, timestamp or None if unavailable.
    """
    try:
        from cortex.data.dexscreener import get_token_price, is_available

        if not is_available():
            return None

        address = _resolve_token_address(token)
        data = get_token_price(address)
        price = float(data.get("price_usd") or 0)
        if price <= 0:
            return None

        return {
            "price": price,
            "source": "dexscreener",
            "token": token,
            "address": address,
            "timestamp": data.get("timestamp"),
        }
    except Exception as e:
        logger.warning("DexScreener price fetch failed for %s: %s", token, e)
        return None


def get_multi_source_price(token: str) -> dict:
    """Fetch price from multiple sources (Birdeye + DexScreener) and return best.

    Returns dict with price, source, all_sources for cross-validation.
    Prefers Birdeye as primary, uses DexScreener for validation/fallback.
    """
    address = _resolve_token_address(token)
    sources: list[dict] = []

    # Primary: Birdeye
    try:
        resp = _pool.get(
            f"{BIRDEYE_BASE}/defi/price",
            headers=_birdeye_headers(),
            params={"address": address},
        )
        resp.raise_for_status()
        birdeye_price = float(resp.json().get("data", {}).get("value", 0))
        if birdeye_price > 0:
            sources.append({"source": "birdeye", "price": birdeye_price})
    except Exception as e:
        logger.warning("Birdeye price failed for %s: %s", token, e)

    # Secondary: DexScreener
    dex_data = get_dexscreener_price(token)
    if dex_data:
        sources.append({"source": "dexscreener", "price": dex_data["price"]})

    if not sources:
        raise ValueError(f"No price data available for {token} from any source")

    primary = sources[0]
    deviation = None
    if len(sources) > 1:
        prices = [s["price"] for s in sources]
        mean_price = sum(prices) / len(prices)
        deviation = max(abs(p - mean_price) / mean_price * 100 for p in prices) if mean_price > 0 else None

    return {
        "token": token,
        "price": primary["price"],
        "source": primary["source"],
        "all_sources": sources,
        "price_deviation_pct": deviation,
        "cross_validated": len(sources) > 1,
    }