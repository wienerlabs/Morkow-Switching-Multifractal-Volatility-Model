"""CLI runner for Guardian backtesting engine.

Usage:
    python -m cortex.backtest.run --token SOL --start 2026-01-01 --end 2026-01-31 --timeframe 1h --capital 10000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from cortex.backtest.data_feed import HistoricalDataFeed
from cortex.backtest.guardian_backtester import BacktestConfig, GuardianBacktester
from cortex.backtest.analytics import PerformanceAnalyzer


def parse_args(argv: list[str] | None = None) -> tuple[BacktestConfig, str | None]:
    parser = argparse.ArgumentParser(description="Guardian Backtesting Engine")
    parser.add_argument("--token", default="SOL", help="Token symbol (default: SOL)")
    parser.add_argument("--start", required=True, help="Start date (ISO format, e.g. 2026-01-01)")
    parser.add_argument("--end", required=True, help="End date (ISO format)")
    parser.add_argument(
        "--timeframe", default="1h",
        choices=["1m", "5m", "15m", "1h", "4h", "1d"],
        help="Candle timeframe",
    )
    parser.add_argument("--capital", type=float, default=10000.0, help="Initial capital in USD")
    parser.add_argument(
        "--trade-size-pct", type=float, default=0.05,
        help="Trade size as fraction of portfolio (default: 0.05)",
    )
    parser.add_argument("--threshold", type=float, default=75.0, help="Guardian approval threshold (default: 75)")
    parser.add_argument("--recalibrate", type=int, default=24, help="Recalibration interval in bars (default: 24)")
    parser.add_argument(
        "--strategy", default="regime",
        choices=["regime", "always_long", "mean_revert"],
        help="Signal strategy",
    )
    parser.add_argument("--output", default=None, help="Output JSON file path (optional)")

    args = parser.parse_args(argv)

    config = BacktestConfig(
        token=args.token,
        start_date=args.start,
        end_date=args.end,
        timeframe=args.timeframe,
        initial_capital=args.capital,
        trade_size_pct=args.trade_size_pct,
        approval_threshold=args.threshold,
        recalibration_interval=args.recalibrate,
        signal_strategy=args.strategy,
    )
    return config, args.output


def main(argv: list[str] | None = None) -> None:
    config, output_path = parse_args(argv)

    print(f"Running Guardian backtest: {config.token} {config.start_date} → {config.end_date} ({config.timeframe})")
    print(f"Capital: ${config.initial_capital:,.2f} | Strategy: {config.signal_strategy} | Threshold: {config.approval_threshold}")
    print()

    # Load OHLCV data (async API, run synchronously here)
    feed = HistoricalDataFeed()
    data = asyncio.run(feed.load_ohlcv(
        token=config.token,
        start_date=config.start_date,
        end_date=config.end_date,
        timeframe=config.timeframe,
    ))

    backtester = GuardianBacktester(config)
    result = backtester.run(data=data)

    analyzer = PerformanceAnalyzer(result)
    report = analyzer.generate_report()
    print(report)

    if output_path:
        metrics = analyzer.compute_all()
        output = {
            "config": {
                "token": config.token,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "timeframe": config.timeframe,
                "initial_capital": config.initial_capital,
                "strategy": config.signal_strategy,
            },
            "metrics": metrics,
            "trades": result.trades,
        }
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
