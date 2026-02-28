"""Focused parameter sweep + out-of-sample validation.

Phase 1: Sweep key params on Jan 2026 (in-sample)
Phase 2: Validate top configs on Oct-Dec 2025 (out-of-sample)

Uses bar-indexed calibration cache: first run builds cache, subsequent
runs reuse cached model calibrations (EVT/SVJ/Hawkes/MSM are data-dependent
only, not affected by trade params or approval_threshold).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")

from cortex.backtest.data_feed import HistoricalDataFeed
from cortex.backtest.guardian_backtester import BacktestConfig, GuardianBacktester
from cortex.backtest.analytics import PerformanceAnalyzer


def run_single(config: BacktestConfig, data: pd.DataFrame, cal_cache: dict | None = None, btc_data: pd.Series | None = None) -> tuple[dict | None, dict]:
    """Run backtest, return (metrics, calibration_cache)."""
    try:
        bt = GuardianBacktester(config, calibration_cache=cal_cache)
        result = bt.run(data=data.copy(), btc_data=btc_data)
        new_cache = bt.export_calibration_cache()
        analyzer = PerformanceAnalyzer(result)
        metrics = analyzer.compute_all()

        first_close = float(data["close"].iloc[0])
        last_close = float(data["close"].iloc[-1])
        bh_return = (last_close - first_close) / first_close * 100

        return {
            "sharpe": metrics.get("sharpe_ratio", 0),
            "sortino": metrics.get("sortino_ratio", 0),
            "total_return_pct": metrics.get("total_return_pct", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "win_rate": metrics.get("win_rate", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "total_trades": metrics.get("total_trades", 0),
            "expectancy": metrics.get("expectancy", 0),
            "buy_hold_return_pct": bh_return,
            "alpha_pct": metrics.get("total_return_pct", 0) - bh_return,
        }, new_cache
    except Exception as e:
        print(f"  FAILED: {e}")
        return None, cal_cache or {}


def run_sweep(base_config: BacktestConfig, data: pd.DataFrame, grid: dict) -> pd.DataFrame:
    keys = list(grid.keys())
    combos = list(itertools.product(*grid.values()))
    total = len(combos)
    print(f"\nSweeping {total} combinations...")
    print("=" * 80)

    cal_cache = None
    results = []
    for idx, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        config = replace(base_config, **params)

        t0 = time.time()
        metrics, cal_cache = run_single(config, data, cal_cache)
        elapsed = time.time() - t0

        if idx == 1:
            print(f"  [calibration cached — remaining {total-1} runs will be ~5-10x faster]")

        if metrics:
            row = {**params, **metrics}
            results.append(row)
            print(
                f"  [{idx}/{total}] {elapsed:.0f}s | "
                f"Sharpe={row['sharpe']:.3f} Return={row['total_return_pct']:.2f}% "
                f"Alpha={row['alpha_pct']:.2f}% Trades={int(row['total_trades'])} "
                f"| {params}"
            )
        else:
            print(f"  [{idx}/{total}] FAILED | {params}")

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)


def print_results(df: pd.DataFrame, title: str, top_n: int = 10):
    if df.empty:
        print(f"\n{title}: No results.")
        return

    top = df.head(top_n)
    print(f"\n{'=' * 90}")
    print(f"  {title} (Top {min(top_n, len(top))})")
    print(f"{'=' * 90}")

    param_cols = [c for c in df.columns if c not in {
        "sharpe", "sortino", "total_return_pct", "max_drawdown",
        "win_rate", "profit_factor", "total_trades", "expectancy",
        "buy_hold_return_pct", "alpha_pct",
    }]

    for i, row in top.iterrows():
        params_str = ", ".join(f"{c}={row[c]}" for c in param_cols)
        print(
            f"  #{i+1}: Sharpe={row['sharpe']:+.3f} | Return={row['total_return_pct']:+.2f}% "
            f"| Alpha={row['alpha_pct']:+.2f}% | DD={row['max_drawdown']:.2f}% "
            f"| WR={row['win_rate']*100:.0f}% | PF={row['profit_factor']:.2f} "
            f"| Trades={int(row['total_trades'])}"
        )
        print(f"       {params_str}")
    print(f"{'=' * 90}")


def fetch_with_retry(feed: HistoricalDataFeed, token: str, start: str, end: str, timeframe: str, max_retries: int = 3) -> pd.DataFrame:
    for attempt in range(max_retries):
        try:
            return asyncio.run(feed.load_ohlcv(token=token, start_date=start, end_date=end, timeframe=timeframe))
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parameter sweep + out-of-sample validation")
    parser.add_argument("--sharpe-enabled", action="store_true", help="Enable Sharpe contribution weights (Phase 6)")
    parser.add_argument("--hmm-enabled", action="store_true", help="Enable HMM regime detection (Phase 7)")
    parser.add_argument("--coint-enabled", action="store_true", help="Enable SOL/BTC cointegration signal (Phase 8)")
    parser.add_argument("--compare-features", action="store_true", help="Run 8-combo feature toggle sweep")
    return parser.parse_args()


def main():
    args = parse_args()
    feed = HistoricalDataFeed()

    # ── Phase 1: In-sample sweep on Jan 2026 ──
    print("=" * 90)
    print("  PHASE 1: Parameter Sweep — Jan 2026 (in-sample)")
    print("=" * 90)

    is_data = fetch_with_retry(feed, "SOL", "2026-01-01", "2026-01-31", "1h")

    base = BacktestConfig(
        token="SOL", start_date="2026-01-01", end_date="2026-01-31",
        timeframe="1h", initial_capital=10000,
        sharpe_weights_enabled=args.sharpe_enabled,
        hmm_regime_enabled=args.hmm_enabled,
        cointegration_enabled=args.coint_enabled,
    )

    # Focused grid: most impactful params
    # approval_threshold controls trade count (biggest lever)
    # SL/TP control risk/reward profile
    # trailing_stop adds downside protection
    grid = {
        "approval_threshold": [40.0, 50.0, 60.0],
        "stop_loss_pct": [0.03, 0.05, 0.08],
        "take_profit_pct": [0.08, 0.12, 0.15],
        "trailing_stop_pct": [0.03, 0.05],
    }

    total_combos = 1
    for v in grid.values():
        total_combos *= len(v)
    print(f"Grid size: {total_combos} combinations")

    is_results = run_sweep(base, is_data, grid)
    print_results(is_results, "IN-SAMPLE RESULTS (Jan 2026)")

    if is_results.empty:
        print("No results to validate. Exiting.")
        return

    # Filter to configs that actually traded
    traded = is_results[is_results["total_trades"] > 0]
    if traded.empty:
        print("No configs produced trades. Reporting all results.")
        traded = is_results

    # ── Phase 2: Out-of-sample validation on Oct-Dec 2025 ──
    print("\n" + "=" * 90)
    print("  PHASE 2: Out-of-Sample Validation — Oct-Dec 2025")
    print("=" * 90)

    oos_data = fetch_with_retry(feed, "SOL", "2025-10-01", "2025-12-31", "1h")

    top_3 = traded.head(3)
    param_cols = [c for c in traded.columns if c not in {
        "sharpe", "sortino", "total_return_pct", "max_drawdown",
        "win_rate", "profit_factor", "total_trades", "expectancy",
        "buy_hold_return_pct", "alpha_pct",
    }]

    oos_rows = []
    oos_cache = None
    for rank, (_, row) in enumerate(top_3.iterrows(), 1):
        params = {col: row[col] for col in param_cols}
        config = replace(base,
            start_date="2025-10-01", end_date="2025-12-31",
            **{k: v for k, v in params.items() if k in BacktestConfig.__dataclass_fields__},
        )

        print(f"\n  Config #{rank}: {params}")
        t0 = time.time()
        metrics, oos_cache = run_single(config, oos_data, oos_cache)
        elapsed = time.time() - t0

        if metrics:
            oos_row = {**params, **metrics, "rank_in_sample": rank}
            oos_rows.append(oos_row)
            print(
                f"    -> {elapsed:.0f}s | Sharpe={metrics['sharpe']:.3f} "
                f"Return={metrics['total_return_pct']:.2f}% "
                f"Alpha={metrics['alpha_pct']:.2f}%"
            )

    oos_df = pd.DataFrame(oos_rows) if oos_rows else pd.DataFrame()

    # ── Final comparison ──
    print("\n\n" + "=" * 90)
    print("  FINAL COMPARISON: In-Sample vs Out-of-Sample")
    print("=" * 90)
    print(f"{'Config':>8} | {'IS Sharpe':>10} {'IS Return':>10} {'IS Alpha':>10} | {'OOS Sharpe':>10} {'OOS Return':>10} {'OOS Alpha':>10}")
    print("-" * 90)

    for rank in range(min(3, len(top_3))):
        is_row = top_3.iloc[rank]
        oos_row = oos_df[oos_df["rank_in_sample"] == rank + 1].iloc[0] if not oos_df.empty and rank < len(oos_df) else None

        is_str = f"{is_row['sharpe']:+.3f}   {is_row['total_return_pct']:+.2f}%   {is_row['alpha_pct']:+.2f}%"
        if oos_row is not None:
            oos_str = f"{oos_row['sharpe']:+.3f}   {oos_row['total_return_pct']:+.2f}%   {oos_row['alpha_pct']:+.2f}%"
        else:
            oos_str = "    N/A         N/A         N/A"

        print(f"  #{rank+1:>5} | {is_str} | {oos_str}")

        params = {col: is_row[col] for col in param_cols}
        print(f"          Params: {params}")

    print("=" * 90)

    # Save results
    output_dir = _project_root / "data" / "sweep_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    is_results.to_csv(output_dir / "sweep_jan2026_insample.csv", index=False)
    if not oos_df.empty:
        oos_df.to_csv(output_dir / "sweep_oct_dec2025_oos.csv", index=False)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
