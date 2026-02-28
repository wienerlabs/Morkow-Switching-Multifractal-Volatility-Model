"""Macro Analyst agent — BTC correlation, Fear & Greed, BTC dominance."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from cortex.agents.base import AgentSignal, BaseAgent

logger = logging.getLogger(__name__)


class MacroAnalystAgent(BaseAgent):
    """Analyzes macro market conditions: Fear/Greed, BTC dominance, SOL/BTC correlation."""

    name = "macro_analyst"
    weight = 0.10

    def __init__(self, correlation_window: int = 168) -> None:
        self.correlation_window = correlation_window  # 168 hours = 7 days for 1h bars
        self._btc_cache: pd.Series | None = None

    def analyze(self, token: str, data: pd.DataFrame, context: dict[str, Any]) -> AgentSignal:
        """Live mode: fetch real macro data."""
        try:
            from cortex.data.macro import get_fear_greed, get_btc_dominance
            fg = get_fear_greed()
            btc_dom = get_btc_dominance()
        except Exception:
            logger.warning("macro_data_fetch_failed, using neutral")
            fg = {"value": 50, "classification": "Neutral"}
            btc_dom = {"btc_dominance": 0.0}

        fg_value = fg.get("value", 50)
        btc_dominance = btc_dom.get("btc_dominance", 0.0)

        # SOL/BTC correlation from context (if available)
        sol_btc_corr = context.get("sol_btc_correlation", 0.5)
        btc_trend = context.get("btc_trend", "neutral")

        score, direction, confidence = self._compute_score(
            fg_value, btc_dominance, sol_btc_corr, btc_trend
        )

        reasoning = (
            f"Macro: F&G={fg_value} ({fg.get('classification', '?')}), "
            f"BTC_dom={btc_dominance:.1f}%, "
            f"SOL/BTC_corr={sol_btc_corr:.2f}, "
            f"BTC_trend={btc_trend} → score={score:.1f}"
        )

        return AgentSignal(
            agent_name=self.name,
            score=score,
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            metadata={
                "fear_greed": fg_value,
                "fear_greed_class": fg.get("classification", "Neutral"),
                "btc_dominance": btc_dominance,
                "sol_btc_correlation": round(sol_btc_corr, 4),
                "btc_trend": btc_trend,
            },
        )

    def analyze_backtest(
        self, token: str, data: pd.DataFrame, bar_idx: int, context: dict[str, Any]
    ) -> AgentSignal:
        """Backtest mode: use BTC price data from context as macro proxy."""
        btc_prices = context.get("btc_close")
        sol_close = data["close"].astype(float)

        if btc_prices is None and self._btc_cache is not None:
            btc_prices = self._btc_cache

        if btc_prices is None and data.index is not None and len(data) > 0:
            try:
                from cortex.backtest.data_feed import load_btc_ohlcv
                start_str = str(data.index[0].date())
                end_str = str(data.index[-1].date())
                btc_df = load_btc_ohlcv(start_str, end_str)
                btc_prices = btc_df["close"]
                self._btc_cache = btc_prices
                logger.info("btc_loaded_on_fly: %d rows", len(btc_prices))
            except Exception as e:
                logger.warning("btc_load_failed_falling_back_to_neutral: %s", e)

        # Default neutral values
        fg_proxy = 50
        sol_btc_corr = 0.5
        btc_trend = "neutral"

        if btc_prices is not None and len(btc_prices) > bar_idx:
            btc_slice = btc_prices.iloc[: bar_idx + 1].astype(float)

            # Fear/Greed proxy: BTC 14-bar rolling return
            if len(btc_slice) >= 14:
                btc_ret_14 = (btc_slice.iloc[-1] / btc_slice.iloc[-14] - 1) * 100
                # Map: -15% → 25 (fear), +15% → 75 (greed), linear between
                fg_proxy = int(max(0, min(100, 50 + btc_ret_14 * (25 / 15))))

            # BTC trend: 7-bar momentum
            if len(btc_slice) >= 7:
                btc_ret_7 = (btc_slice.iloc[-1] / btc_slice.iloc[-7] - 1) * 100
                if btc_ret_7 > 2:
                    btc_trend = "up"
                elif btc_ret_7 < -2:
                    btc_trend = "down"

            # SOL/BTC correlation
            window = min(self.correlation_window, bar_idx + 1)
            if window >= 20:
                sol_slice = sol_close.iloc[bar_idx + 1 - window: bar_idx + 1]
                btc_window = btc_slice.iloc[len(btc_slice) - window:]
                sol_ret = sol_slice.pct_change().dropna()
                btc_ret = btc_window.pct_change().dropna()
                min_len = min(len(sol_ret), len(btc_ret))
                if min_len >= 10:
                    sol_btc_corr = float(sol_ret.iloc[-min_len:].corr(btc_ret.iloc[-min_len:]))
                    if np.isnan(sol_btc_corr):
                        sol_btc_corr = 0.5

        score, direction, confidence = self._compute_score(
            fg_proxy, 0.0, sol_btc_corr, btc_trend
        )

        # Phase 8: Cointegration adjustment
        coint_signal = context.get("cointegration_signal") if context else None
        if coint_signal is not None and hasattr(coint_signal, "is_valid") and coint_signal.is_valid:
            coint_adj = float(np.clip(coint_signal.z_score * -10.0, -25.0, 25.0))
            score = float(np.clip(score + coint_adj, 0.0, 100.0))

            # Re-derive direction from adjusted score
            if score < 35:
                direction = "long"
            elif score > 65:
                direction = "short"
            else:
                direction = None

            # Strong cointegration overrides direction
            if abs(coint_signal.z_score) >= 2.0 and coint_signal.direction:
                direction = coint_signal.direction
                confidence = min(0.8, confidence + 0.2)

        fg_class = "Neutral"
        if fg_proxy <= 25:
            fg_class = "Extreme Fear"
        elif fg_proxy <= 40:
            fg_class = "Fear"
        elif fg_proxy >= 75:
            fg_class = "Extreme Greed"
        elif fg_proxy >= 60:
            fg_class = "Greed"

        reasoning = (
            f"Macro(backtest): F&G_proxy={fg_proxy} ({fg_class}), "
            f"SOL/BTC_corr={sol_btc_corr:.2f}, "
            f"BTC_trend={btc_trend} → score={score:.1f}"
        )

        return AgentSignal(
            agent_name=self.name,
            score=score,
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            metadata={
                "fear_greed": fg_proxy,
                "fear_greed_class": fg_class,
                "btc_dominance": 0.0,
                "sol_btc_correlation": round(sol_btc_corr, 4),
                "btc_trend": btc_trend,
                "backtest_mode": True,
            },
        )

    def _compute_score(
        self, fg_value: int, btc_dominance: float, sol_btc_corr: float, btc_trend: str
    ) -> tuple[float, str | None, float]:
        """Compute macro risk score, direction, and confidence."""
        base_score = 50.0

        # Fear & Greed adjustment
        if fg_value <= 25:
            fg_adj = 30.0 * (25 - fg_value) / 25  # extreme fear → high risk
        elif fg_value >= 75:
            fg_adj = 20.0 * (fg_value - 75) / 25  # greed → moderate risk (bubble)
        elif fg_value < 50:
            fg_adj = -10.0 * (50 - fg_value) / 25  # mild fear → slightly lower risk
        else:
            fg_adj = -5.0 * (1 - (fg_value - 50) / 25)  # mild greed → slightly lower

        # BTC correlation + trend adjustment
        corr_adj = 0.0
        if sol_btc_corr > 0.7:
            if btc_trend == "down":
                corr_adj = 15.0
            elif btc_trend == "up":
                corr_adj = -15.0

        score = max(0.0, min(100.0, base_score + fg_adj + corr_adj))

        # Direction
        direction = None
        if score < 35:
            direction = "long"
        elif score > 65:
            direction = "short"

        # Confidence: higher at extremes
        fg_extremity = abs(fg_value - 50) / 50  # 0-1
        confidence = 0.3 + fg_extremity * 0.4  # 0.3-0.7

        return score, direction, confidence
