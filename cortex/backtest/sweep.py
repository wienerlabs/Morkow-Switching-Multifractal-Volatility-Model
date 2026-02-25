"""Parameter sweep runner for Guardian backtesting engine."""

from __future__ import annotations

import itertools
from dataclasses import replace

import pandas as pd

from cortex.backtest.guardian_backtester import BacktestConfig, GuardianBacktester
from cortex.backtest.analytics import PerformanceAnalyzer


class ParameterSweep:
    """Run backtester across parameter grid to find optimal configuration."""

    def __init__(self, base_config: BacktestConfig, data: pd.DataFrame, btc_data: pd.Series | None = None):
        self.base_config = base_config
        self.data = data
        self.btc_data = btc_data

    def sweep(self, param_grid: dict[str, list]) -> pd.DataFrame:
        """Run all parameter combinations, return results sorted by Sharpe.

        Uses calibration caching: first run calibrates models, subsequent
        runs reuse cached calibration — typically 5-10x faster.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(itertools.product(*values))

        # Params that don't affect model calibration (only trade execution)
        trade_only_params = {
            "stop_loss_pct", "take_profit_pct", "trailing_stop_pct",
            "use_trailing_stop", "trade_size_pct", "position_hold_bars",
            "rsi_oversold", "rsi_overbought", "use_ta_filter",
            "momentum_threshold", "use_momentum_filter", "momentum_window",
            "agent_approval_threshold", "agent_veto_score",
        }

        # Check if sweep only varies trade params (can reuse calibration)
        varies_model_params = any(k not in trade_only_params for k in keys)
        calibration_cache = None

        results = []
        total = len(combos)

        for idx, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
            config = replace(self.base_config, **params)

            try:
                bt = GuardianBacktester(config, calibration_cache=calibration_cache)
                result = bt.run(data=self.data.copy(), btc_data=self.btc_data)

                # Cache calibration from first run
                if calibration_cache is None and not varies_model_params:
                    calibration_cache = bt.export_calibration_cache()
                    print(f"  [cached calibration from run 1 — remaining runs will be faster]")

                analyzer = PerformanceAnalyzer(result)
                metrics = analyzer.compute_all()

                row = {**params}
                row["sharpe"] = metrics.get("sharpe_ratio", 0)
                row["sortino"] = metrics.get("sortino_ratio", 0)
                row["total_return_pct"] = metrics.get("total_return_pct", 0)
                row["max_drawdown"] = metrics.get("max_drawdown", 0)
                row["win_rate"] = metrics.get("win_rate", 0)
                row["profit_factor"] = metrics.get("profit_factor", 0)
                row["total_trades"] = metrics.get("total_trades", 0)
                row["expectancy"] = metrics.get("expectancy", 0)
                results.append(row)

                print(f"  [{idx}/{total}] Sharpe={row['sharpe']:.3f} Return={row['total_return_pct']:.2f}% Trades={row['total_trades']} | {params}")
            except Exception as e:
                print(f"  [{idx}/{total}] FAILED: {e} | {params}")
                continue

        if not results:
            return pd.DataFrame()

        df = pd.DataFrame(results).sort_values("sharpe", ascending=False).reset_index(drop=True)
        return df

    @staticmethod
    def default_grid() -> dict[str, list]:
        """Predefined parameter grid for quick sweep."""
        return {
            "stop_loss_pct": [0.02, 0.03, 0.05],
            "take_profit_pct": [0.05, 0.08, 0.12],
            "trailing_stop_pct": [0.02, 0.03, 0.05],
            "approval_threshold": [50.0, 60.0, 70.0],
            "trade_size_pct": [0.10, 0.15],
            "momentum_threshold": [0.3, 0.5, 0.8],
            "rsi_oversold": [25.0, 30.0, 35.0],
            "rsi_overbought": [65.0, 70.0, 75.0],
        }

    @staticmethod
    def top_n(results_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
        return results_df.head(n)

    @staticmethod
    def format_results(results_df: pd.DataFrame, top_n: int = 10) -> str:
        if results_df.empty:
            return "No results."
        top = results_df.head(top_n)
        lines = [
            "=" * 80,
            "PARAMETER SWEEP RESULTS (Top {})".format(min(top_n, len(top))),
            "=" * 80,
            "",
        ]
        for i, row in top.iterrows():
            param_cols = [c for c in row.index if c not in {
                "sharpe", "sortino", "total_return_pct", "max_drawdown",
                "win_rate", "profit_factor", "total_trades", "expectancy",
            }]
            params_str = ", ".join(f"{c}={row[c]}" for c in param_cols)
            lines.append(
                f"#{i+1}: Sharpe={row['sharpe']:.3f} | Return={row['total_return_pct']:.2f}% "
                f"| DD={row['max_drawdown']:.2f}% | WR={row['win_rate']*100:.0f}% "
                f"| PF={row['profit_factor']:.2f} | Trades={int(row['total_trades'])}"
            )
            lines.append(f"     {params_str}")
            lines.append("")
        lines.append("=" * 80)
        return "\n".join(lines)
