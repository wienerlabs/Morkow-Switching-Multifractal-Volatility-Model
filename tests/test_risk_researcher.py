"""Tests for RiskResearcherAgent."""

import numpy as np
import pandas as pd
import pytest
from cortex.agents.risk_researcher import RiskResearcherAgent


def _make_test_data(n=200):
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    np.random.seed(42)
    close = np.cumsum(np.random.normal(0, 1, n)) + 150
    return pd.DataFrame(
        {"open": close - 1, "high": close + 2, "low": close - 2, "close": close, "volume": 1e6},
        index=dates,
    )


class TestRiskResearcher:
    def test_init(self):
        agent = RiskResearcherAgent()
        assert agent.name == "risk_researcher"

    def test_analyze_backtest(self):
        agent = RiskResearcherAgent()
        data = _make_test_data()
        signal = agent.analyze_backtest("SOL", data, 199, {})
        assert signal.agent_name == "risk_researcher"
        assert 0 <= signal.score <= 100
        assert "regime_state" in signal.metadata
