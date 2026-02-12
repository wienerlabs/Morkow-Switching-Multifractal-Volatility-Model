"""Portfolio-level risk tracking — drawdown monitoring and correlated exposure limits.

In-memory implementation (Redis upgrade planned for Wave 9).
"""
from __future__ import annotations

import logging
import time
from typing import Any

from cortex.config import MAX_CORRELATED_EXPOSURE, MAX_DAILY_DRAWDOWN, MAX_WEEKLY_DRAWDOWN

logger = logging.getLogger(__name__)

_positions: dict[str, dict[str, Any]] = {}
_pnl_history: list[dict[str, Any]] = []
_portfolio_value: float = 100_000.0

CORRELATION_GROUPS: dict[str, list[str]] = {
    "sol_ecosystem": ["SOL", "RAY", "JUP", "ORCA", "MNGO", "SRM", "STEP"],
    "memecoins": ["BONK", "WIF", "POPCAT", "MYRO", "BOME", "MEW"],
    "defi_blue": ["BTC", "ETH", "AVAX", "LINK", "UNI"],
    "stablecoins": ["USDC", "USDT", "PYUSD"],
}


def set_portfolio_value(value: float) -> None:
    global _portfolio_value
    _portfolio_value = value


def get_portfolio_value() -> float:
    return _portfolio_value


def update_position(token: str, size_usd: float, direction: str, entry_price: float = 0.0) -> None:
    _positions[token] = {
        "token": token,
        "size_usd": size_usd,
        "direction": direction,
        "entry_price": entry_price,
        "opened_at": time.time(),
    }


def close_position(token: str, pnl: float) -> None:
    _positions.pop(token, None)
    _pnl_history.append({"token": token, "pnl": pnl, "ts": time.time()})
    if len(_pnl_history) > 10_000:
        _pnl_history[:] = _pnl_history[-5_000:]


def get_positions() -> list[dict[str, Any]]:
    return list(_positions.values())


def get_drawdown() -> dict[str, Any]:
    now = time.time()
    day_ago = now - 86_400
    week_ago = now - 604_800

    daily_pnl = sum(e["pnl"] for e in _pnl_history if e["ts"] >= day_ago)
    weekly_pnl = sum(e["pnl"] for e in _pnl_history if e["ts"] >= week_ago)

    pv = _portfolio_value if _portfolio_value > 0 else 1.0
    daily_dd = abs(min(0.0, daily_pnl)) / pv
    weekly_dd = abs(min(0.0, weekly_pnl)) / pv

    return {
        "daily_pnl": round(daily_pnl, 2),
        "weekly_pnl": round(weekly_pnl, 2),
        "daily_drawdown_pct": round(daily_dd, 6),
        "weekly_drawdown_pct": round(weekly_dd, 6),
        "daily_limit_pct": MAX_DAILY_DRAWDOWN,
        "weekly_limit_pct": MAX_WEEKLY_DRAWDOWN,
        "daily_breached": daily_dd >= MAX_DAILY_DRAWDOWN,
        "weekly_breached": weekly_dd >= MAX_WEEKLY_DRAWDOWN,
        "portfolio_value": _portfolio_value,
    }


def get_correlated_exposure(token: str) -> dict[str, Any]:
    token_upper = token.upper()
    group_name = None
    group_tokens: list[str] = []
    for gname, tokens in CORRELATION_GROUPS.items():
        if token_upper in tokens:
            group_name = gname
            group_tokens = tokens
            break

    if not group_name:
        return {"group": None, "group_tokens": [], "group_exposure_usd": 0.0,
                "exposure_pct": 0.0, "limit_pct": MAX_CORRELATED_EXPOSURE, "breached": False}

    exposure = sum(
        p["size_usd"] for p in _positions.values() if p["token"].upper() in group_tokens
    )
    pv = _portfolio_value if _portfolio_value > 0 else 1.0
    pct = exposure / pv

    return {
        "group": group_name,
        "group_tokens": group_tokens,
        "group_exposure_usd": round(exposure, 2),
        "exposure_pct": round(pct, 6),
        "limit_pct": MAX_CORRELATED_EXPOSURE,
        "breached": pct >= MAX_CORRELATED_EXPOSURE,
    }


def check_limits(token: str) -> dict[str, Any]:
    dd = get_drawdown()
    corr = get_correlated_exposure(token)
    blockers: list[str] = []
    if dd["daily_breached"]:
        blockers.append("daily_drawdown")
    if dd["weekly_breached"]:
        blockers.append("weekly_drawdown")
    if corr["breached"]:
        blockers.append("correlated_exposure")
    return {"blocked": len(blockers) > 0, "blockers": blockers, "drawdown": dd, "correlation": corr}

