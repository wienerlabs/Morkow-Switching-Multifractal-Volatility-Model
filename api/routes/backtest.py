"""API endpoint for running Guardian backtests."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cortex.backtest.data_feed import HistoricalDataFeed
from cortex.backtest.guardian_backtester import BacktestConfig, GuardianBacktester
from cortex.backtest.analytics import PerformanceAnalyzer

router = APIRouter(prefix="/backtest", tags=["backtest"])


class BacktestRequest(BaseModel):
    token: str = "SOL"
    start_date: str
    end_date: str
    timeframe: str = "1h"
    initial_capital: float = 10000.0
    trade_size_pct: float = 0.05
    approval_threshold: float = 75.0
    recalibration_interval: int = 24
    signal_strategy: str = "regime"


@router.post("/run")
async def run_backtest(req: BacktestRequest) -> dict:
    """Run a Guardian backtest and return metrics."""
    config = BacktestConfig(
        token=req.token,
        start_date=req.start_date,
        end_date=req.end_date,
        timeframe=req.timeframe,
        initial_capital=req.initial_capital,
        trade_size_pct=req.trade_size_pct,
        approval_threshold=req.approval_threshold,
        recalibration_interval=req.recalibration_interval,
        signal_strategy=req.signal_strategy,
    )

    try:
        feed = HistoricalDataFeed()
        data = await feed.load_ohlcv(
            token=config.token,
            start_date=config.start_date,
            end_date=config.end_date,
            timeframe=config.timeframe,
        )
    except (ValueError, EnvironmentError) as exc:
        raise HTTPException(status_code=400, detail=f"Data loading failed: {exc}")

    try:
        backtester = GuardianBacktester(config)
        result = backtester.run(data=data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Backtest failed: {exc}")

    analyzer = PerformanceAnalyzer(result)
    metrics = analyzer.compute_all()

    return {
        "metrics": metrics,
        "trades_count": len(result.trades),
        "equity_start": float(result.equity_curve.iloc[0]) if len(result.equity_curve) > 0 else config.initial_capital,
        "equity_end": float(result.equity_curve.iloc[-1]) if len(result.equity_curve) > 0 else config.initial_capital,
    }
