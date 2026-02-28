"""Tests for MacroAnalystAgent with cointegration integration."""

import numpy as np
import pandas as pd
import pytest
from cortex.agents.macro_analyst import MacroAnalystAgent


def _make_test_data(n=200):
    dates = pd.date_range("2026-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": 150.0, "high": 155.0, "low": 145.0, "close": 150.0, "volume": 1e6},
        index=dates,
    )


class TestMacroAnalystAgent:
    def test_init(self):
        agent = MacroAnalystAgent()
        assert agent.name == "macro_analyst"
        assert agent.weight == 0.10

    def test_analyze_backtest_no_btc(self):
        agent = MacroAnalystAgent()
        data = _make_test_data()
        signal = agent.analyze_backtest("SOL", data, 199, {})
        assert signal.agent_name == "macro_analyst"
        assert 0 <= signal.score <= 100

    def test_analyze_backtest_with_btc(self):
        agent = MacroAnalystAgent()
        data = _make_test_data(200)
        btc_prices = pd.Series(
            np.linspace(40000, 45000, 200),
            index=data.index,
        )
        context = {"btc_close": btc_prices}
        signal = agent.analyze_backtest("SOL", data, 199, context)
        assert signal.agent_name == "macro_analyst"
        assert signal.metadata["backtest_mode"] is True

    def test_without_btc_data_neutral(self):
        """Previously failing test — stdlib logger kwargs fix."""
        agent = MacroAnalystAgent()
        data = _make_test_data(50)
        signal = agent.analyze_backtest("SOL", data, 49, {})
        assert signal.agent_name == "macro_analyst"

    def test_compute_score_extreme_fear(self):
        agent = MacroAnalystAgent()
        score, direction, conf = agent._compute_score(10, 0, 0.5, "neutral")
        assert score > 50  # extreme fear = high risk

    def test_compute_score_extreme_greed(self):
        agent = MacroAnalystAgent()
        score, direction, conf = agent._compute_score(90, 0, 0.5, "neutral")
        assert score > 50  # extreme greed = risk

    def test_compute_score_high_corr_btc_down(self):
        agent = MacroAnalystAgent()
        score, direction, conf = agent._compute_score(50, 0, 0.9, "down")
        assert score >= 60  # high corr + btc down = risk

    def test_compute_score_high_corr_btc_up(self):
        agent = MacroAnalystAgent()
        score, direction, conf = agent._compute_score(50, 0, 0.9, "up")
        assert score < 40  # high corr + btc up = low risk

    def test_direction_thresholds(self):
        agent = MacroAnalystAgent()
        # Extreme fear (fg=5) + BTC correlation up => competing adjustments
        score_low, dir_low, _ = agent._compute_score(5, 0, 0.9, "up")
        # fg=5: fg_adj = 30*(25-5)/25 = 24, corr_adj = -15 => score=59
        assert score_low > 50  # still elevated due to extreme fear

        score_high, dir_high, _ = agent._compute_score(90, 0, 0.9, "down")
        # fg=90: fg_adj = 20*(90-75)/25 = 12, corr_adj = 15 => score=77
        assert dir_high == "short" or score_high > 65

    def test_cointegration_strong_signal_overrides_direction(self):
        agent = MacroAnalystAgent()
        data = _make_test_data(200)

        class MockCoint:
            z_score = -2.5
            direction = "long"
            confidence = 0.6
            is_valid = True

        context = {"cointegration_signal": MockCoint()}
        signal = agent.analyze_backtest("SOL", data, 199, context)
        assert signal.direction == "long"

    def test_cointegration_invalid_signal_no_effect(self):
        agent = MacroAnalystAgent()
        data = _make_test_data(200)

        class MockCoint:
            z_score = -2.5
            direction = "long"
            confidence = 0.6
            is_valid = False

        context = {"cointegration_signal": MockCoint()}
        signal_with = agent.analyze_backtest("SOL", data, 199, context)
        signal_without = agent.analyze_backtest("SOL", data, 199, {})
        # Invalid coint should not change the score
        assert abs(signal_with.score - signal_without.score) < 0.01
