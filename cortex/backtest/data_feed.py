"""Unified historical data loader for the backtesting engine.

Loads OHLCV data from Birdeye API or local CSV cache, computes returns,
and extracts large-move events for Hawkes process input.
"""

import asyncio
import os
import warnings
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import structlog

from cortex.config import BIRDEYE_BASE
from cortex.data.solana import TOKEN_REGISTRY

logger = structlog.get_logger(__name__)

# Birdeye interval mapping: our timeframe labels → API type param
_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "4h": "4H",
    "1d": "1D",
    "1w": "1W",
}

# Birdeye caps responses at ~1000 candles. For large ranges we need to chunk.
_MAX_CANDLES_PER_REQUEST = 1000

# Approximate seconds per candle for chunking calculation
_INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}


def _resolve_address(token: str) -> str:
    """Resolve token symbol to mint address, or pass through if already an address."""
    if len(token) > 20:
        return token
    upper = token.upper()
    if upper in TOKEN_REGISTRY:
        return TOKEN_REGISTRY[upper]
    raise ValueError(
        f"Unknown token '{token}'. Use a mint address or one of: {list(TOKEN_REGISTRY.keys())}"
    )


def _birdeye_headers() -> dict[str, str]:
    api_key = os.environ.get("BIRDEYE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "BIRDEYE_API_KEY environment variable is required. Get one at https://birdeye.so"
        )
    return {"X-API-KEY": api_key, "x-chain": "solana", "accept": "application/json"}


def _to_unix(dt_str: str) -> int:
    """Parse ISO date string to unix timestamp."""
    dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def _cache_filename(token: str, timeframe: str, start: str, end: str) -> str:
    """Generate deterministic cache filename."""
    safe_token = token.replace("/", "_")
    safe_start = start.replace(":", "-").replace("+", "")
    safe_end = end.replace(":", "-").replace("+", "")
    return f"{safe_token}_{timeframe}_{safe_start}_{safe_end}.csv"


class HistoricalDataFeed:
    """Loads and manages historical OHLCV data for backtesting."""

    def __init__(self, cache_dir: str = "data/backtest_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def load_ohlcv(
        self,
        token: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """Load OHLCV from Birdeye API with CSV cache.

        Returns DataFrame with DatetimeIndex (UTC) and columns:
        open, high, low, close, volume.
        """
        cache_path = self.cache_dir / _cache_filename(token, timeframe, start_date, end_date)

        if cache_path.exists():
            logger.info("cache_hit", token=token, path=str(cache_path))
            return self.load_from_csv(str(cache_path))

        logger.info("fetching_ohlcv", token=token, start=start_date, end=end_date, timeframe=timeframe)
        df = await self._fetch_birdeye_ohlcv(token, start_date, end_date, timeframe)

        self.save_to_csv(df, str(cache_path))
        logger.info("cached_ohlcv", token=token, rows=len(df), path=str(cache_path))
        return df

    async def _fetch_birdeye_ohlcv(
        self,
        token: str,
        start_date: str,
        end_date: str,
        timeframe: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV from Birdeye, chunking if the range exceeds API limits."""
        address = _resolve_address(token)
        interval = _INTERVAL_MAP.get(timeframe.lower())
        if interval is None:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Use one of: {list(_INTERVAL_MAP.keys())}")

        time_from = _to_unix(start_date)
        time_to = _to_unix(end_date)
        interval_secs = _INTERVAL_SECONDS[timeframe.lower()]
        headers = _birdeye_headers()

        all_items: list[dict] = []

        # Chunk the range to stay within Birdeye's per-request candle limit
        chunk_start = time_from
        async with httpx.AsyncClient(timeout=30) as client:
            while chunk_start < time_to:
                chunk_end = min(
                    chunk_start + _MAX_CANDLES_PER_REQUEST * interval_secs,
                    time_to,
                )
                for attempt in range(5):
                    resp = await client.get(
                        f"{BIRDEYE_BASE}/defi/ohlcv",
                        headers=headers,
                        params={
                            "address": address,
                            "type": interval,
                            "time_from": chunk_start,
                            "time_to": chunk_end,
                        },
                    )
                    if resp.status_code == 429:
                        wait = 2 ** attempt
                        logger.info("rate_limited", retry_in=wait, attempt=attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()
                    break
                else:
                    resp.raise_for_status()
                items = resp.json().get("data", {}).get("items", [])
                all_items.extend(items)
                chunk_start = chunk_end

        if not all_items:
            raise ValueError(f"No OHLCV data returned for {token} ({start_date} → {end_date})")

        records = []
        for candle in all_items:
            records.append({
                "timestamp": pd.Timestamp(candle["unixTime"], unit="s", tz="UTC"),
                "open": float(candle["o"]),
                "high": float(candle["h"]),
                "low": float(candle["l"]),
                "close": float(candle["c"]),
                "volume": float(candle["v"]),
            })

        df = pd.DataFrame(records).set_index("timestamp").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        # Handle small gaps via forward-fill, warn on large ones
        expected_freq = pd.Timedelta(seconds=interval_secs)
        gaps = df.index.to_series().diff()
        large_gaps = gaps[gaps > expected_freq * 3].dropna()
        if not large_gaps.empty:
            warnings.warn(
                f"{len(large_gaps)} gap(s) larger than 3x expected interval in {token} data. "
                f"Largest gap: {large_gaps.max()}. Forward-filling small gaps only.",
                stacklevel=2,
            )
        df = df.ffill(limit=2)

        return df

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Load OHLCV from a local CSV file."""
        df = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    def save_to_csv(self, df: pd.DataFrame, path: str) -> None:
        """Save OHLCV DataFrame to CSV for caching."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=True, index_label="timestamp")

    @staticmethod
    def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Add pct_return and log_return columns to OHLCV DataFrame."""
        df = df.copy()
        df["pct_return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        return df

    @staticmethod
    def extract_events(
        df: pd.DataFrame,
        threshold: float = 3.0,
        window: int = 20,
    ) -> np.ndarray:
        """Detect large price moves for Hawkes process input.

        Returns array of Unix timestamps where |log_return| > threshold * rolling_std.
        Requires compute_returns() to have been called first (or 'log_return' column present).
        """
        if "log_return" not in df.columns:
            df = HistoricalDataFeed.compute_returns(df)

        rolling_std = df["log_return"].rolling(window).std()
        mask = abs(df["log_return"]) > threshold * rolling_std
        large_moves = df[mask].dropna(subset=["log_return"])

        return large_moves.index.astype(np.int64) / 1e9  # Unix timestamps

    @staticmethod
    def get_window(
        df: pd.DataFrame,
        current_ts: pd.Timestamp,
        lookback_days: int,
    ) -> pd.DataFrame:
        """Get data window up to current_ts (NO look-ahead bias).

        Returns df[start:current_ts] where start = current_ts - lookback_days.
        Uses <= (inclusive) on both ends.
        """
        start = current_ts - pd.Timedelta(days=lookback_days)
        return df[(df.index >= start) & (df.index <= current_ts)]
