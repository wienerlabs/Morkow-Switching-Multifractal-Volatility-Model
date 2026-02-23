"""Heartbeat Self-Check Pipeline — 5-stage agent health monitoring.

Periodically checks:
1. Active hours guard (only run during configured hours)
2. Open position risk assessment
3. Portfolio drawdown proximity
4. Circuit breaker proximity
5. Cognitive state anomalies

Each check produces a HeartbeatAlert. Alerts are deduped within a cooldown window.
"""
from __future__ import annotations

__all__ = [
    "HeartbeatPipeline",
    "HeartbeatAlert",
    "AlertSeverity",
    "get_heartbeat",
]

import enum
import logging
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable

from cortex.config import (
    HEARTBEAT_ENABLED,
    HEARTBEAT_DRAWDOWN_WARN_PCT,
    HEARTBEAT_CB_PROXIMITY_PCT,
)

logger = logging.getLogger(__name__)


class AlertSeverity(str, enum.Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class HeartbeatAlert:
    check: str
    severity: AlertSeverity
    message: str
    details: dict[str, Any]
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


class HeartbeatPipeline:
    """5-stage periodic health check for the trading agent."""

    def __init__(
        self,
        alert_cooldown: float = 300.0,
        active_hours: tuple[int, int] = (0, 24),
    ) -> None:
        self._alert_cooldown = alert_cooldown
        self._active_hours = active_hours
        self._last_alerts: dict[str, float] = {}
        self._alert_history: list[HeartbeatAlert] = []
        self._checks: list[Callable[[], list[HeartbeatAlert]]] = [
            self._check_active_hours,
            self._check_open_positions,
            self._check_portfolio_drawdown,
            self._check_circuit_breaker_proximity,
            self._check_cognitive_state,
        ]

    def run(self) -> list[HeartbeatAlert]:
        """Run all checks and return deduped alerts."""
        if not HEARTBEAT_ENABLED:
            return []

        all_alerts: list[HeartbeatAlert] = []
        for check_fn in self._checks:
            try:
                alerts = check_fn()
                all_alerts.extend(alerts)
            except Exception as e:
                logger.debug("Heartbeat check %s failed: %s", check_fn.__name__, e)

        deduped = self._dedup(all_alerts)
        self._alert_history.extend(deduped)
        if len(self._alert_history) > 500:
            self._alert_history = self._alert_history[-500:]

        if deduped:
            logger.info("heartbeat: %d alerts (%s)",
                        len(deduped),
                        ", ".join(f"{a.check}:{a.severity.value}" for a in deduped))
        return deduped

    def _dedup(self, alerts: list[HeartbeatAlert]) -> list[HeartbeatAlert]:
        """Filter out alerts that fired within cooldown window."""
        now = time.time()
        result = []
        for alert in alerts:
            key = f"{alert.check}:{alert.severity.value}"
            last = self._last_alerts.get(key, 0.0)
            if (now - last) >= self._alert_cooldown:
                self._last_alerts[key] = now
                result.append(alert)
        return result

    def _check_active_hours(self) -> list[HeartbeatAlert]:
        """Check if we're within active trading hours."""
        import datetime
        hour = datetime.datetime.now(datetime.timezone.utc).hour
        start, end = self._active_hours
        if start <= hour < end:
            return []
        return [HeartbeatAlert(
            check="active_hours",
            severity=AlertSeverity.INFO,
            message=f"Outside active hours ({start}-{end} UTC), current={hour}",
            details={"current_hour": hour, "active_start": start, "active_end": end},
            timestamp=time.time(),
        )]

    def _check_open_positions(self) -> list[HeartbeatAlert]:
        """Check trade ledger for pending/stuck entries."""
        alerts = []
        try:
            from cortex.trade_ledger import get_trade_ledger
            ledger = get_trade_ledger()
            stats = ledger.stats()
            pending = stats.get("pending_tokens", [])
            if pending:
                alerts.append(HeartbeatAlert(
                    check="open_positions",
                    severity=AlertSeverity.WARNING,
                    message=f"{len(pending)} pending trade(s): {', '.join(pending[:5])}",
                    details={"pending_tokens": pending, "count": len(pending)},
                    timestamp=time.time(),
                ))
        except Exception:
            pass
        return alerts

    def _check_portfolio_drawdown(self) -> list[HeartbeatAlert]:
        """Check if portfolio drawdown is approaching limits."""
        alerts = []
        try:
            from cortex.portfolio_risk import get_current_drawdown
            dd = get_current_drawdown()
            if dd is not None:
                dd_pct = dd * 100.0
                if dd_pct >= HEARTBEAT_DRAWDOWN_WARN_PCT:
                    severity = AlertSeverity.CRITICAL if dd_pct >= HEARTBEAT_DRAWDOWN_WARN_PCT * 1.5 else AlertSeverity.WARNING
                    alerts.append(HeartbeatAlert(
                        check="portfolio_drawdown",
                        severity=severity,
                        message=f"Drawdown at {dd_pct:.1f}% (warn threshold: {HEARTBEAT_DRAWDOWN_WARN_PCT}%)",
                        details={"drawdown_pct": round(dd_pct, 2), "threshold_pct": HEARTBEAT_DRAWDOWN_WARN_PCT},
                        timestamp=time.time(),
                    ))
        except ImportError:
            pass
        except Exception:
            logger.debug("Drawdown check failed", exc_info=True)
        return alerts

    def _check_circuit_breaker_proximity(self) -> list[HeartbeatAlert]:
        """Check if any circuit breaker is close to triggering."""
        alerts = []
        try:
            from cortex.circuit_breaker import get_all_states
            states = get_all_states()
            for state in states:
                name = state.get("name", "unknown")
                score = state.get("last_score", 0.0)
                threshold = state.get("threshold", 90.0)
                if threshold > 0:
                    proximity = (score / threshold) * 100.0
                    if proximity >= HEARTBEAT_CB_PROXIMITY_PCT:
                        severity = AlertSeverity.CRITICAL if proximity >= 95.0 else AlertSeverity.WARNING
                        alerts.append(HeartbeatAlert(
                            check="circuit_breaker_proximity",
                            severity=severity,
                            message=f"CB '{name}' at {proximity:.0f}% of threshold (score={score:.1f}/{threshold:.0f})",
                            details={
                                "breaker": name,
                                "score": round(score, 2),
                                "threshold": threshold,
                                "proximity_pct": round(proximity, 1),
                            },
                            timestamp=time.time(),
                        ))
        except ImportError:
            pass
        except Exception:
            logger.debug("CB proximity check failed", exc_info=True)
        return alerts

    def _check_cognitive_state(self) -> list[HeartbeatAlert]:
        """Report if cognitive state is at extremes."""
        alerts = []
        try:
            from cortex.cognitive_state import get_cognitive_state, EmotionLevel
            csm = get_cognitive_state()
            level = csm.level
            if level in (EmotionLevel.EXTREME_FEAR, EmotionLevel.EXTREME_GREED):
                adj = csm.get_adjustments()
                alerts.append(HeartbeatAlert(
                    check="cognitive_state",
                    severity=AlertSeverity.WARNING,
                    message=f"Agent emotion at {level.value} (score={adj['smoothed_score']:.2f})",
                    details={
                        "emotion_level": level.value,
                        "smoothed_score": adj["smoothed_score"],
                        "threshold_delta": adj["threshold_delta"],
                        "size_multiplier": adj["size_multiplier"],
                    },
                    timestamp=time.time(),
                ))
        except ImportError:
            pass
        except Exception:
            logger.debug("Cognitive state check failed", exc_info=True)
        return alerts

    def get_history(self, limit: int = 50) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self._alert_history[-limit:]]

    def get_status(self) -> dict[str, Any]:
        return {
            "enabled": HEARTBEAT_ENABLED,
            "total_alerts": len(self._alert_history),
            "recent_alerts": len([a for a in self._alert_history if time.time() - a.timestamp < 3600]),
            "active_hours": self._active_hours,
            "cooldown_seconds": self._alert_cooldown,
        }


_heartbeat: HeartbeatPipeline | None = None


def get_heartbeat() -> HeartbeatPipeline:
    global _heartbeat
    if _heartbeat is None:
        _heartbeat = HeartbeatPipeline()
    return _heartbeat
