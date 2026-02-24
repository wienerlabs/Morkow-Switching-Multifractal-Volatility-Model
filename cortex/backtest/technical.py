"""Technical analysis indicators for backtesting signal confirmation."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TechnicalIndicators:
    """Calculate RSI, MACD, and Bollinger Bands from OHLCV data."""

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index using Wilder's smoothed moving average."""
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        both_zero = (avg_gain == 0) & (avg_loss == 0)
        all_gain = (avg_gain > 0) & (avg_loss == 0)
        all_loss = (avg_gain == 0) & (avg_loss > 0)

        rsi = rsi.mask(both_zero, 50.0)
        rsi = rsi.mask(all_gain, 100.0)
        rsi = rsi.mask(all_loss, 0.0)

        return rsi

    @staticmethod
    def macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """MACD line, signal line, and histogram."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(
        close: pd.Series,
        period: int = 20,
        std_dev: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands: upper, middle, lower."""
        middle = close.rolling(window=period).mean()
        rolling_std = close.rolling(window=period).std()

        upper = middle + (std_dev * rolling_std)
        lower = middle - (std_dev * rolling_std)

        return upper, middle, lower

    @staticmethod
    def compute_all(df: pd.DataFrame) -> pd.DataFrame:
        """Add all TA indicator columns to a DataFrame.

        Expects df to have a 'close' column.
        Adds: rsi, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower
        """
        close = df["close"].astype(float)

        df["rsi"] = TechnicalIndicators.rsi(close)

        macd_line, signal_line, histogram = TechnicalIndicators.macd(close)
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = histogram

        upper, middle, lower = TechnicalIndicators.bollinger_bands(close)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower

        return df
