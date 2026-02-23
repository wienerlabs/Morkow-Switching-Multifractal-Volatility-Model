"""Tests for cortex/heartbeat.py — Heartbeat Self-Check Pipeline."""
import time
from unittest.mock import patch, MagicMock

import pytest

from cortex.heartbeat import (
    AlertSeverity,
    HeartbeatAlert,
    HeartbeatPipeline,
)


@pytest.fixture
def pipeline():
    return HeartbeatPipeline(alert_cooldown=0.0)  # no cooldown for tests


def test_run_returns_alerts(pipeline):
    alerts = pipeline.run()
    assert isinstance(alerts, list)


def test_active_hours_outside(pipeline):
    pipeline._active_hours = (99, 100)  # always outside
    alerts = pipeline._check_active_hours()
    assert len(alerts) == 1
    assert alerts[0].check == "active_hours"


def test_active_hours_inside(pipeline):
    pipeline._active_hours = (0, 24)  # always inside
    alerts = pipeline._check_active_hours()
    assert len(alerts) == 0


def test_open_positions_alert(pipeline):
    with patch("cortex.trade_ledger.get_trade_ledger") as mock_ledger:
        mock_ledger.return_value.stats.return_value = {
            "pending_tokens": ["SOL", "ETH"],
            "total": 10,
        }
        alerts = pipeline._check_open_positions()
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING
        assert "SOL" in alerts[0].message


def test_open_positions_no_pending(pipeline):
    with patch("cortex.trade_ledger.get_trade_ledger") as mock_ledger:
        mock_ledger.return_value.stats.return_value = {
            "pending_tokens": [],
            "total": 5,
        }
        alerts = pipeline._check_open_positions()
        assert len(alerts) == 0


def test_circuit_breaker_proximity(pipeline):
    with patch("cortex.circuit_breaker.get_all_states") as mock_states:
        mock_states.return_value = [
            {"name": "global", "last_score": 85.0, "threshold": 90.0},
            {"name": "lp", "last_score": 30.0, "threshold": 90.0},
        ]
        alerts = pipeline._check_circuit_breaker_proximity()
        assert len(alerts) == 1
        assert alerts[0].check == "circuit_breaker_proximity"
        assert alerts[0].details["breaker"] == "global"


def test_cognitive_state_extreme(pipeline):
    with patch("cortex.cognitive_state.get_cognitive_state") as mock_csm:
        from cortex.cognitive_state import EmotionLevel
        mock_csm.return_value.level = EmotionLevel.EXTREME_GREED
        mock_csm.return_value.get_adjustments.return_value = {
            "smoothed_score": 0.75,
            "threshold_delta": 10.0,
            "size_multiplier": 0.6,
        }
        alerts = pipeline._check_cognitive_state()
        assert len(alerts) == 1
        assert "extreme_greed" in alerts[0].message


def test_cognitive_state_neutral(pipeline):
    with patch("cortex.cognitive_state.get_cognitive_state") as mock_csm:
        from cortex.cognitive_state import EmotionLevel
        mock_csm.return_value.level = EmotionLevel.NEUTRAL
        alerts = pipeline._check_cognitive_state()
        assert len(alerts) == 0


def test_dedup_within_cooldown():
    pipeline = HeartbeatPipeline(alert_cooldown=600.0)  # 10 min cooldown
    alert = HeartbeatAlert(
        check="test_check",
        severity=AlertSeverity.WARNING,
        message="test",
        details={},
        timestamp=time.time(),
    )
    # First call: passes through
    result1 = pipeline._dedup([alert])
    assert len(result1) == 1

    # Second call within cooldown: filtered out
    result2 = pipeline._dedup([alert])
    assert len(result2) == 0


def test_alert_serialization():
    alert = HeartbeatAlert(
        check="portfolio_drawdown",
        severity=AlertSeverity.CRITICAL,
        message="Drawdown at 4.5%",
        details={"drawdown_pct": 4.5},
        timestamp=1000.0,
    )
    d = alert.to_dict()
    assert d["severity"] == "critical"
    assert d["check"] == "portfolio_drawdown"


def test_get_status(pipeline):
    status = pipeline.get_status()
    assert "enabled" in status
    assert "total_alerts" in status
    assert "active_hours" in status


def test_get_history(pipeline):
    alerts = pipeline.run()
    history = pipeline.get_history()
    assert isinstance(history, list)


def test_disabled():
    with patch("cortex.heartbeat.HEARTBEAT_ENABLED", False):
        pipeline = HeartbeatPipeline()
        alerts = pipeline.run()
        assert alerts == []
