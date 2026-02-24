"""Technical Analyst agent — RSI, MACD, Bollinger Bands signal confirmation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from cortex.agents.base import AgentSignal, BaseAgent
from cortex.backtest.technical import TechnicalIndicators


class TechnicalAnalystAgent(BaseAgent):
    """Analyzes price action using RSI, MACD, and Bollinger Bands."""

    name = "technical_analyst"
    weight = 0.15

    def __init__(self, rsi_oversold: float = 30.0, rsi_overbought: float = 70.0) -> None:
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def analyze(self, token: str, data: pd.DataFrame, context: dict[str, Any]) -> AgentSignal:
        if len(data) < 30:
            return AgentSignal(agent_name=self.name, score=50.0, reasoning="Insufficient data for TA")

        close = data["close"].astype(float)

        rsi = TechnicalIndicators.rsi(close)
        macd_line, signal_line, histogram = TechnicalIndicators.macd(close)
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(close)

        current_rsi = rsi.iloc[-1]
        current_macd_hist = histogram.iloc[-1]
        current_close = close.iloc[-1]
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]
        current_bb_middle = bb_middle.iloc[-1]

        # Score individual indicators (0 = very bullish, 100 = very bearish)
        # RSI scoring
        if pd.isna(current_rsi):
            rsi_score = 50.0
        elif current_rsi <= self.rsi_oversold:
            rsi_score = max(0.0, current_rsi / self.rsi_oversold * 30.0)  # 0-30
        elif current_rsi >= self.rsi_overbought:
            rsi_score = min(100.0, 70.0 + (current_rsi - self.rsi_overbought) / (100.0 - self.rsi_overbought) * 30.0)
        else:
            rsi_score = 30.0 + (current_rsi - self.rsi_oversold) / (self.rsi_overbought - self.rsi_oversold) * 40.0

        # MACD scoring
        if pd.isna(current_macd_hist):
            macd_score = 50.0
        else:
            hist_std = histogram.dropna().std()
            if hist_std == 0:
                macd_score = 50.0
            else:
                z = current_macd_hist / hist_std
                macd_score = max(0.0, min(100.0, 50.0 - z * 20.0))

        # Bollinger scoring
        if pd.isna(current_bb_upper) or pd.isna(current_bb_lower):
            bb_score = 50.0
        else:
            bb_width = current_bb_upper - current_bb_lower
            if bb_width == 0:
                bb_score = 50.0
            else:
                bb_position = (current_close - current_bb_lower) / bb_width
                bb_score = max(0.0, min(100.0, bb_position * 100.0))

        # Composite: equal weight
        score = (rsi_score + macd_score + bb_score) / 3.0

        # Direction
        bullish_count = sum([
            rsi_score < 35,
            macd_score < 40,
            bb_score < 30,
        ])
        bearish_count = sum([
            rsi_score > 65,
            macd_score > 60,
            bb_score > 70,
        ])

        if bullish_count >= 2:
            direction = "long"
        elif bearish_count >= 2:
            direction = "short"
        else:
            direction = None

        # Confidence: higher when indicators agree
        agreement = max(bullish_count, bearish_count) / 3.0
        confidence = 0.3 + agreement * 0.5  # 0.3 to 0.8

        reasoning_parts = []
        if not pd.isna(current_rsi):
            reasoning_parts.append(f"RSI={current_rsi:.1f}")
        reasoning_parts.append(f"MACD_hist={current_macd_hist:.4f}" if not pd.isna(current_macd_hist) else "MACD=N/A")
        if not pd.isna(current_bb_lower):
            reasoning_parts.append(f"BB_pos={bb_score:.0f}%")
        reasoning = f"TA: {', '.join(reasoning_parts)} → score={score:.1f}"

        return AgentSignal(
            agent_name=self.name,
            score=score,
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            metadata={
                "rsi": None if pd.isna(current_rsi) else round(current_rsi, 2),
                "rsi_score": round(rsi_score, 2),
                "macd_hist": None if pd.isna(current_macd_hist) else round(current_macd_hist, 6),
                "macd_score": round(macd_score, 2),
                "bb_position_pct": round(bb_score, 2),
                "bb_score": round(bb_score, 2),
                "close": round(current_close, 4),
                "bb_upper": None if pd.isna(current_bb_upper) else round(current_bb_upper, 4),
                "bb_lower": None if pd.isna(current_bb_lower) else round(current_bb_lower, 4),
            },
        )
