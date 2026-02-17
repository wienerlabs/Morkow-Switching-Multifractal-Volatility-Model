"""Wave 9 — Trade Execution Layer for the Cortex Risk Engine.

Safety hierarchy (all gates must pass):
  1. EXECUTION_ENABLED must be True
  2. SIMULATION_MODE check (if True, log but don't execute)
  3. Guardian approval (assess_trade must approve)
  4. Slippage cap (EXECUTION_MAX_SLIPPAGE_BPS)
  5. MEV protection (EXECUTION_MEV_PROTECTION)
  6. Jupiter Swap API execution (buy/sell)

Private keys never leave the local process — they are passed through
from the caller (API layer or agent) and used only for local signing.
"""

import logging
import time
from typing import Any

import structlog.contextvars

from cortex.config import (
    EXECUTION_ENABLED,
    EXECUTION_MAX_SLIPPAGE_BPS,
    EXECUTION_MEV_PROTECTION,
    SIMULATION_MODE,
    TRADING_MODE,
)

logger = logging.getLogger(__name__)

_execution_log: list[dict[str, Any]] = []


def _get_request_id() -> str | None:
    return structlog.contextvars.get_contextvars().get("request_id")


def _log_execution(entry: dict[str, Any]) -> None:
    request_id = _get_request_id()
    if request_id:
        entry["request_id"] = request_id
    _execution_log.append(entry)
    if len(_execution_log) > 500:
        _execution_log.pop(0)


def preflight_check(
    token: str,
    trade_size_usd: float,
    direction: str,
    model_data: dict | None = None,
    evt_data: dict | None = None,
    svj_data: dict | None = None,
    hawkes_data: dict | None = None,
    news_data: dict | None = None,
) -> dict[str, Any]:
    """Run all safety gates without executing. Returns approval status."""
    gates: list[dict[str, Any]] = []

    # Gate 1: Execution enabled
    gates.append({"gate": "execution_enabled", "passed": EXECUTION_ENABLED, "value": EXECUTION_ENABLED})

    # Gate 2: Simulation mode
    gates.append({"gate": "simulation_mode", "passed": True, "value": SIMULATION_MODE, "note": "sim=True means dry-run only"})

    # Gate 3: Guardian approval
    guardian_result = None
    try:
        from cortex.guardian import assess_trade
        guardian_result = assess_trade(
            token=token, trade_size_usd=trade_size_usd, direction=direction,
            model_data=model_data, evt_data=evt_data, svj_data=svj_data,
            hawkes_data=hawkes_data, news_data=news_data,
        )
        approved = guardian_result.get("approved", False)
        gates.append({"gate": "guardian_approval", "passed": approved, "risk_score": guardian_result.get("risk_score"), "veto_reasons": guardian_result.get("veto_reasons", [])})
    except Exception as e:
        gates.append({"gate": "guardian_approval", "passed": False, "error": str(e)})

    # Gate 4: Slippage cap
    gates.append({"gate": "slippage_cap", "passed": True, "max_bps": EXECUTION_MAX_SLIPPAGE_BPS})

    # Gate 5: MEV protection
    gates.append({"gate": "mev_protection", "passed": True, "enabled": EXECUTION_MEV_PROTECTION})

    all_passed = all(g["passed"] for g in gates)
    can_execute = all_passed and EXECUTION_ENABLED and not SIMULATION_MODE

    return {
        "token": token,
        "direction": direction,
        "trade_size_usd": trade_size_usd,
        "trading_mode": TRADING_MODE,
        "all_gates_passed": all_passed,
        "can_execute": can_execute,
        "simulation_mode": SIMULATION_MODE,
        "gates": gates,
        "guardian": guardian_result,
        "recommended_size": guardian_result.get("recommended_size") if guardian_result else None,
        "timestamp": time.time(),
    }


def execute_trade(
    private_key: str,
    token_mint: str,
    direction: str,
    amount: float,
    trade_size_usd: float,
    model_data: dict | None = None,
    evt_data: dict | None = None,
    svj_data: dict | None = None,
    hawkes_data: dict | None = None,
    news_data: dict | None = None,
    slippage_bps: int | None = None,
    force: bool = False,
    strategy: str | None = None,
    post_execution_callback: Any = None,
) -> dict[str, Any]:
    """Execute a trade through the full safety pipeline.

    Args:
        private_key: Solana wallet private key (passed to Axiom SDK).
        token_mint: SPL token mint address.
        direction: "buy" or "sell".
        amount: SOL amount (buy) or percentage (sell).
        trade_size_usd: Estimated trade value in USD.
        model_data..news_data: Risk model outputs for Guardian.
        slippage_bps: Override slippage (capped at EXECUTION_MAX_SLIPPAGE_BPS).
        force: Skip Guardian check (DANGEROUS, for emergency use only).
        strategy: Strategy name (spot, lp, arb, perp, lending) for CB feedback.
        post_execution_callback: Optional callback(entry_dict) after execution.
    """
    token_short = token_mint[:8] + "..."
    slippage = min(slippage_bps or EXECUTION_MAX_SLIPPAGE_BPS, EXECUTION_MAX_SLIPPAGE_BPS)

    entry: dict[str, Any] = {
        "token_mint": token_mint,
        "direction": direction,
        "amount": amount,
        "trade_size_usd": trade_size_usd,
        "slippage_bps": slippage,
        "strategy": strategy,
        "timestamp": time.time(),
    }

    if not EXECUTION_ENABLED:
        entry.update({"status": "blocked", "reason": "execution_disabled"})
        _log_execution(entry)
        return entry

    # Guardian check (unless forced)
    if not force:
        preflight = preflight_check(
            token=token_mint, trade_size_usd=trade_size_usd, direction=direction,
            model_data=model_data, evt_data=evt_data, svj_data=svj_data,
            hawkes_data=hawkes_data, news_data=news_data,
        )
        if not preflight["all_gates_passed"]:
            failed = [g["gate"] for g in preflight["gates"] if not g["passed"]]
            entry.update({"status": "rejected", "reason": "gates_failed", "failed_gates": failed, "preflight": preflight})
            _log_execution(entry)
            logger.warning("Trade rejected for %s: %s", token_short, failed)
            return entry

        recommended = preflight.get("recommended_size")
        if recommended and recommended < trade_size_usd:
            amount = amount * (recommended / trade_size_usd)
            entry["size_adjusted"] = True
            entry["original_amount"] = entry["amount"]
            entry["amount"] = amount
            logger.info("Trade size adjusted to %.2f for %s", amount, token_short)

    if SIMULATION_MODE:
        entry.update({"status": "simulated", "reason": "simulation_mode"})
        _log_execution(entry)
        logger.info("[SIM] %s %s %.4f (slippage=%dbps)", direction, token_short, amount, slippage)
        return entry

    # Execute via Jupiter Swap API
    try:
        from cortex.data.jupiter import execute_buy, execute_sell

        if direction == "buy":
            result = execute_buy(private_key, token_mint, amount, slippage_bps=slippage, mev_protection=EXECUTION_MEV_PROTECTION)
        elif direction == "sell":
            result = execute_sell(private_key, token_mint, amount, slippage_bps=slippage, mev_protection=EXECUTION_MEV_PROTECTION)
        else:
            raise ValueError(f"Invalid direction: {direction}")

        entry.update({"status": "executed", "jupiter_result": result})
        logger.info("Trade executed: %s %s %.4f", direction, token_short, amount)
    except Exception as e:
        entry.update({"status": "failed", "error": str(e)})
        logger.error("Trade execution failed for %s: %s", token_short, e)

    # Auto-record to circuit breaker when strategy is known and no callback overrides
    if strategy and post_execution_callback is None:
        _auto_record_to_cb(entry)

    if post_execution_callback is not None:
        try:
            post_execution_callback(entry)
        except Exception as cb_err:
            logger.warning("Post-execution callback error: %s", cb_err)

    _log_execution(entry)
    return entry


def _auto_record_to_cb(entry: dict[str, Any]) -> None:
    """Record execution result to circuit breaker and guardian outcome trackers."""
    strategy = entry.get("strategy")
    if not strategy:
        return
    status = entry.get("status", "")
    success = status == "executed"
    loss_type = "execution_failure" if status == "failed" else ""

    try:
        from cortex.circuit_breaker import record_trade_outcome
        record_trade_outcome(
            strategy=strategy, success=success, pnl=0.0,
            loss_type=loss_type,
            details=f"auto-recorded from execute_trade: {status}",
        )
    except Exception as e:
        logger.warning("CB auto-record failed: %s", e)

    try:
        from cortex.guardian import record_trade_outcome as guardian_record
        guardian_record(pnl=0.0, size=entry.get("trade_size_usd", 0.0),
                        token=entry.get("token_mint", ""))
    except Exception as e:
        logger.warning("Guardian auto-record failed: %s", e)


def record_execution_result(
    token_mint: str,
    strategy: str,
    success: bool,
    pnl: float = 0.0,
    trade_size_usd: float = 0.0,
    loss_type: str = "",
    details: str = "",
) -> dict[str, Any]:
    """Record a post-exit trade result to both circuit breaker and guardian.

    Use this for recording realized PnL after a position is closed,
    separate from the initial execution auto-recording (which records pnl=0).
    """
    from cortex.circuit_breaker import record_trade_outcome
    from cortex.guardian import record_trade_outcome as guardian_record

    cb_status = record_trade_outcome(
        strategy=strategy, success=success, pnl=pnl,
        loss_type=loss_type, details=details,
    )
    guardian_record(pnl=pnl, size=trade_size_usd, token=token_mint)
    request_id = _get_request_id()
    logger.info(
        "Execution result recorded: %s strategy=%s success=%s pnl=%.2f",
        token_mint[:8], strategy, success, pnl,
    )
    if request_id:
        cb_status["request_id"] = request_id
    return cb_status


def get_execution_log(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent execution log entries."""
    return _execution_log[-limit:]


def get_execution_stats() -> dict[str, Any]:
    """Return execution statistics."""
    if not _execution_log:
        return {"total": 0, "executed": 0, "simulated": 0, "rejected": 0, "failed": 0}

    statuses = [e.get("status", "unknown") for e in _execution_log]
    return {
        "total": len(statuses),
        "executed": statuses.count("executed"),
        "simulated": statuses.count("simulated"),
        "rejected": statuses.count("rejected"),
        "failed": statuses.count("failed"),
        "blocked": statuses.count("blocked"),
        "execution_enabled": EXECUTION_ENABLED,
        "simulation_mode": SIMULATION_MODE,
        "trading_mode": TRADING_MODE,
    }

