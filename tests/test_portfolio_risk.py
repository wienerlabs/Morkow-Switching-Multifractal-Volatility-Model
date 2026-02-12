"""Tests for cortex/portfolio_risk.py — Portfolio-level risk tracking."""

import time

import pytest

from cortex import portfolio_risk as pr


@pytest.fixture(autouse=True)
def clean_state():
    """Reset module-level state before each test."""
    pr._positions.clear()
    pr._pnl_history.clear()
    pr._portfolio_value = 100_000.0
    yield
    pr._positions.clear()
    pr._pnl_history.clear()
    pr._portfolio_value = 100_000.0


class TestPositions:
    def test_update_and_get(self):
        pr.update_position("SOL", 5_000, "long", 150.0)
        positions = pr.get_positions()
        assert len(positions) == 1
        assert positions[0]["token"] == "SOL"
        assert positions[0]["size_usd"] == 5_000

    def test_close_removes_position(self):
        pr.update_position("SOL", 5_000, "long")
        pr.close_position("SOL", pnl=200.0)
        assert pr.get_positions() == []

    def test_close_records_pnl(self):
        pr.update_position("SOL", 5_000, "long")
        pr.close_position("SOL", pnl=-100.0)
        assert len(pr._pnl_history) == 1
        assert pr._pnl_history[0]["pnl"] == -100.0

    def test_multiple_positions(self):
        pr.update_position("SOL", 5_000, "long")
        pr.update_position("RAY", 3_000, "short")
        pr.update_position("BTC", 10_000, "long")
        assert len(pr.get_positions()) == 3

    def test_update_overwrites(self):
        pr.update_position("SOL", 5_000, "long")
        pr.update_position("SOL", 8_000, "long")
        positions = pr.get_positions()
        assert len(positions) == 1
        assert positions[0]["size_usd"] == 8_000

    def test_pnl_history_capped(self):
        for i in range(10_100):
            pr._pnl_history.append({"token": "X", "pnl": 1.0, "ts": time.time()})
        pr.close_position("Y", pnl=1.0)
        assert len(pr._pnl_history) <= 5_001


class TestDrawdown:
    def test_no_pnl_zero_drawdown(self):
        dd = pr.get_drawdown()
        assert dd["daily_drawdown_pct"] == 0.0
        assert dd["weekly_drawdown_pct"] == 0.0
        assert dd["daily_breached"] is False
        assert dd["weekly_breached"] is False

    def test_negative_pnl_creates_drawdown(self):
        pr._pnl_history.append({"token": "SOL", "pnl": -6_000, "ts": time.time()})
        dd = pr.get_drawdown()
        assert dd["daily_drawdown_pct"] == pytest.approx(0.06, abs=0.001)
        assert dd["daily_breached"] is True  # 6% > 5% limit

    def test_positive_pnl_no_drawdown(self):
        pr._pnl_history.append({"token": "SOL", "pnl": 5_000, "ts": time.time()})
        dd = pr.get_drawdown()
        assert dd["daily_drawdown_pct"] == 0.0

    def test_weekly_breach(self):
        pr._pnl_history.append({"token": "SOL", "pnl": -11_000, "ts": time.time()})
        dd = pr.get_drawdown()
        assert dd["weekly_breached"] is True  # 11% > 10% limit

    def test_portfolio_value_setter(self):
        pr.set_portfolio_value(200_000.0)
        assert pr.get_portfolio_value() == 200_000.0
        pr._pnl_history.append({"token": "SOL", "pnl": -6_000, "ts": time.time()})
        dd = pr.get_drawdown()
        assert dd["daily_drawdown_pct"] == pytest.approx(0.03, abs=0.001)  # 6k/200k = 3%
        assert dd["daily_breached"] is False


class TestCorrelatedExposure:
    def test_sol_ecosystem_group(self):
        pr.update_position("SOL", 10_000, "long")
        pr.update_position("RAY", 5_000, "long")
        corr = pr.get_correlated_exposure("SOL")
        assert corr["group"] == "sol_ecosystem"
        assert corr["group_exposure_usd"] == 15_000
        assert corr["exposure_pct"] == pytest.approx(0.15, abs=0.001)
        assert corr["breached"] is True  # 15% >= 15% limit

    def test_unknown_token_no_group(self):
        corr = pr.get_correlated_exposure("UNKNOWN_TOKEN")
        assert corr["group"] is None
        assert corr["breached"] is False

    def test_case_insensitive(self):
        pr.update_position("SOL", 5_000, "long")
        corr = pr.get_correlated_exposure("sol")
        assert corr["group"] == "sol_ecosystem"


class TestCheckLimits:
    def test_no_breach(self):
        result = pr.check_limits("SOL")
        assert result["blocked"] is False
        assert result["blockers"] == []

    def test_drawdown_breach_blocks(self):
        pr._pnl_history.append({"token": "SOL", "pnl": -6_000, "ts": time.time()})
        result = pr.check_limits("SOL")
        assert result["blocked"] is True
        assert "daily_drawdown" in result["blockers"]

    def test_correlation_breach_blocks(self):
        pr.update_position("SOL", 8_000, "long")
        pr.update_position("RAY", 8_000, "long")
        result = pr.check_limits("JUP")
        assert result["blocked"] is True
        assert "correlated_exposure" in result["blockers"]

    def test_multiple_breaches(self):
        pr._pnl_history.append({"token": "SOL", "pnl": -6_000, "ts": time.time()})
        pr.update_position("SOL", 8_000, "long")
        pr.update_position("RAY", 8_000, "long")
        result = pr.check_limits("JUP")
        assert result["blocked"] is True
        assert len(result["blockers"]) >= 2

