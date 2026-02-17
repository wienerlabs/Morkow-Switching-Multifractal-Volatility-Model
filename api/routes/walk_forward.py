"""Walk-forward backtesting API routes."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

import asyncio
from fastapi import APIRouter, HTTPException

from api.models import (
    AsyncTaskResponse,
    HistoricalExportResponse,
    TaskStatusResponse,
    WalkForwardRegimeResult,
    WalkForwardReportResponse,
    WalkForwardRequest,
)
from api.stores import _model_store
from api.tasks import create_task, get_task, run_in_background
from cortex.backtest.export import export_historical_data
from cortex.backtest.report import REGIME_NAMES, generate_report
from cortex.backtest.walk_forward import WalkForwardConfig, run_walk_forward

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Walk-Forward Backtesting"])


def _run_walk_forward_sync(token: str, req: WalkForwardRequest) -> dict:
    """Synchronous walk-forward runner for background execution."""
    model_data = _model_store.get(token)
    if not model_data:
        raise ValueError(f"Token '{token}' not found in model store")

    returns = model_data.get("returns")
    if returns is None or len(returns) < req.min_train_window + 10:
        raise ValueError(
            f"Insufficient data for token '{token}': "
            f"need {req.min_train_window + 10}, have {len(returns) if returns is not None else 0}"
        )

    leverage_gamma = model_data.get("leverage_gamma", 0.0)

    cfg = WalkForwardConfig(
        min_train_window=req.min_train_window,
        step_size=req.step_size,
        refit_interval=req.refit_interval,
        expanding=req.expanding,
        max_train_window=req.max_train_window,
        confidence=req.confidence,
        num_states=req.num_states,
        method=req.method,
        use_student_t=req.use_student_t,
        nu=req.nu,
        leverage_gamma=leverage_gamma,
    )

    result = run_walk_forward(returns, cfg)
    report = generate_report(result)
    report["token"] = token
    return report


@router.post("/backtest/walk-forward", response_model=AsyncTaskResponse)
async def start_walk_forward(req: WalkForwardRequest):
    """Launch a walk-forward backtest as a background task.

    Walk-forward backtesting re-calibrates the MSM model on expanding (or rolling)
    windows and produces 1-step-ahead VaR forecasts, validated with Kupiec and
    Christoffersen tests overall and per-regime.
    """
    if req.token not in _model_store:
        raise HTTPException(404, f"Token '{req.token}' not calibrated. POST /msm/calibrate first.")

    task_id = create_task("backtest/walk-forward")
    asyncio.create_task(run_in_background(task_id, _run_walk_forward_sync, req.token, req))

    return AsyncTaskResponse(
        task_id=task_id,
        status="pending",
        endpoint="backtest/walk-forward",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/backtest/walk-forward/{task_id}", response_model=TaskStatusResponse)
async def get_walk_forward_result(task_id: str):
    """Poll the result of a walk-forward backtest task."""
    task = get_task(task_id)
    if task is None:
        raise HTTPException(404, f"Task '{task_id}' not found")
    return TaskStatusResponse(**task)


@router.post("/backtest/walk-forward/sync", response_model=WalkForwardReportResponse)
async def run_walk_forward_sync(req: WalkForwardRequest):
    """Run walk-forward backtest synchronously (blocks until complete).

    Use the async POST /backtest/walk-forward for large datasets.
    """
    if req.token not in _model_store:
        raise HTTPException(404, f"Token '{req.token}' not calibrated. POST /msm/calibrate first.")

    try:
        report = await asyncio.to_thread(_run_walk_forward_sync, req.token, req)
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    per_regime = []
    for r in report.get("per_regime", []):
        per_regime.append(WalkForwardRegimeResult(
            regime=r["regime"],
            regime_name=r.get("regime_name", REGIME_NAMES.get(r["regime"], f"State {r['regime']}")),
            n_obs=r.get("n_obs", 0),
            n_violations=r.get("n_violations", 0),
            violation_rate=r.get("violation_rate", 0.0),
            mean_return=r.get("mean_return"),
            volatility=r.get("volatility"),
            sharpe=r.get("sharpe"),
            insufficient_data=r.get("insufficient_data", False),
        ))

    return WalkForwardReportResponse(
        token=req.token,
        overall=report.get("overall", {}),
        per_regime=per_regime,
        parameter_stability=report.get("parameter_stability", {"n_refits": 0, "stable": True}),
        health=report.get("health", {"pass": True, "flags": []}),
        n_calibration_snapshots=len(report.get("calibration_snapshots", [])),
        elapsed_ms=report.get("overall", {}).get("elapsed_ms", 0.0),
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/backtest/export/{token}", response_model=HistoricalExportResponse)
async def export_token_data(token: str):
    """Export historical regime, VaR, and volatility data for a calibrated token."""
    if token not in _model_store:
        raise HTTPException(404, f"Token '{token}' not calibrated. POST /msm/calibrate first.")

    model_data = _model_store[token]
    data = export_historical_data(model_data, token)

    return HistoricalExportResponse(
        token=data["token"],
        n_observations=data["n_observations"],
        regime_timeline=data["regime_timeline"],
        regime_statistics=data["regime_statistics"],
        calibration=data["calibration"],
        calibrated_at=data["calibrated_at"],
        timestamp=data["timestamp"],
    )
