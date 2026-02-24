"""Performance analytics for Guardian backtesting results.

Computes risk-adjusted returns, drawdown analysis, trade quality metrics,
Guardian-specific approval/regime stats, VaR validation, and benchmark comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from cortex.backtest.guardian_backtester import BacktestResult


class PerformanceAnalyzer:
    """Post-processes a BacktestResult into a complete performance report."""

    def __init__(self, result: BacktestResult) -> None:
        self.result = result
        self.trades_df = pd.DataFrame(result.trades) if result.trades else pd.DataFrame()

    def compute_all(self) -> dict:
        metrics: dict = {}
        metrics.update(self.risk_adjusted_returns())
        metrics.update(self.drawdown_analysis())
        metrics.update(self.trade_quality())
        metrics.update(self.guardian_metrics())
        metrics["exit_reasons"] = self.exit_reason_breakdown()
        metrics["risk_management"] = self.risk_management_effectiveness()
        metrics.update(self.benchmark_comparison())
        return metrics

    # ── Risk-Adjusted Returns ──────────────────────────────────────────

    def risk_adjusted_returns(self) -> dict:
        returns = self.result.daily_returns.dropna()
        if len(returns) < 2:
            return {
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "total_return_pct": 0.0,
                "annual_return_pct": 0.0,
            }

        mean_r = returns.mean()
        std_r = returns.std()

        sharpe = np.sqrt(252) * mean_r / std_r if std_r > 0 else 0.0

        downside = returns[returns < 0]
        down_std = downside.std() if len(downside) > 1 else 0.0
        sortino = np.sqrt(252) * mean_r / down_std if down_std > 0 else 0.0

        eq = self.result.equity_curve
        total_return = (eq.iloc[-1] / eq.iloc[0]) - 1 if len(eq) > 1 else 0.0
        days = len(returns)
        annual_return = (1 + total_return) ** (365 / max(days, 1)) - 1

        max_dd = self.max_drawdown()
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0

        return {
            "sharpe_ratio": round(float(sharpe), 3),
            "sortino_ratio": round(float(sortino), 3),
            "calmar_ratio": round(float(calmar), 3),
            "total_return_pct": round(float(total_return * 100), 2),
            "annual_return_pct": round(float(annual_return * 100), 2),
        }

    # ── Drawdown Analysis ──────────────────────────────────────────────

    def max_drawdown(self) -> float:
        eq = self.result.equity_curve
        if len(eq) < 2:
            return 0.0
        running_max = eq.cummax()
        dd = (eq - running_max) / running_max
        return round(float(dd.min()), 4)

    def drawdown_series(self) -> pd.Series:
        eq = self.result.equity_curve
        if len(eq) < 2:
            return pd.Series(dtype=float)
        running_max = eq.cummax()
        return (eq - running_max) / running_max

    def max_drawdown_duration(self) -> int:
        eq = self.result.equity_curve
        if len(eq) < 2:
            return 0
        running_max = eq.cummax()
        in_drawdown = eq < running_max
        if not in_drawdown.any():
            return 0
        groups = (~in_drawdown).cumsum()
        dd_groups = groups[in_drawdown]
        if len(dd_groups) == 0:
            return 0
        return int(dd_groups.value_counts().max())

    def recovery_time(self) -> int:
        """Bars from deepest trough back to previous peak (0 if never recovered)."""
        eq = self.result.equity_curve
        if len(eq) < 2:
            return 0
        running_max = eq.cummax()
        dd = (eq - running_max) / running_max
        trough_idx = dd.argmin()
        post_trough = eq.iloc[trough_idx:]
        peak_before = running_max.iloc[trough_idx]
        recovered = post_trough[post_trough >= peak_before]
        if len(recovered) == 0:
            return 0
        return int(recovered.index.get_loc(recovered.index[0]) if hasattr(recovered.index, "get_loc") else 0) or (
            list(post_trough.index).index(recovered.index[0])
        )

    def drawdown_analysis(self) -> dict:
        return {
            "max_drawdown": self.max_drawdown(),
            "max_drawdown_duration_bars": self.max_drawdown_duration(),
            "recovery_time_bars": self.recovery_time(),
        }

    # ── Trade Quality ──────────────────────────────────────────────────

    def trade_quality(self) -> dict:
        if self.trades_df.empty or "pnl" not in self.trades_df.columns:
            return {
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "avg_trade_pnl": 0.0,
                "total_trades": 0,
                "total_wins": 0,
                "total_losses": 0,
            }

        pnl = self.trades_df["pnl"]
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]

        win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0.0
        gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
        gross_loss = abs(float(losses.sum())) if len(losses) > 0 else 0.0

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = abs(float(losses.mean())) if len(losses) > 0 else 0.0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return {
            "win_rate": round(win_rate, 3),
            "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else float("inf"),
            "expectancy": round(expectancy, 2),
            "avg_trade_pnl": round(float(pnl.mean()), 2),
            "total_trades": len(pnl),
            "total_wins": len(wins),
            "total_losses": len(losses),
        }

    # ── Guardian-Specific ──────────────────────────────────────────────

    def guardian_metrics(self) -> dict:
        r = self.result
        total_signals = r.signals_generated
        approval_rate = r.signals_approved / total_signals if total_signals > 0 else 0.0

        # Component attribution: avg score per component across assessments
        component_avgs: dict[str, list[float]] = {}
        if r.component_score_history:
            for entry in r.component_score_history:
                for cs in entry.get("component_scores", []):
                    name = cs.get("component", "unknown")
                    score = cs.get("score", 0)
                    component_avgs.setdefault(name, []).append(score)
        component_attribution = {k: round(float(np.mean(v)), 2) for k, v in component_avgs.items()}

        # Per-regime performance
        regime_performance: dict = {}
        if not self.trades_df.empty and "regime_state" in self.trades_df.columns:
            for regime, group in self.trades_df.groupby("regime_state"):
                regime_performance[str(regime)] = {
                    "trades": len(group),
                    "win_rate": round(float((group["pnl"] > 0).mean()), 3),
                    "avg_pnl": round(float(group["pnl"].mean()), 2),
                    "total_pnl": round(float(group["pnl"].sum()), 2),
                }

        return {
            "approval_rate": round(approval_rate, 3),
            "signals_generated": total_signals,
            "signals_approved": r.signals_approved,
            "signals_rejected": r.signals_rejected,
            "component_attribution": component_attribution,
            "regime_performance": regime_performance,
        }

    # ── Exit Reason Breakdown ───────────────────────────────────────────

    def exit_reason_breakdown(self) -> dict:
        if self.trades_df.empty or "exit_reason" not in self.trades_df.columns:
            return {}

        counts = self.trades_df["exit_reason"].value_counts().to_dict()
        total = len(self.trades_df)

        breakdown = {}
        for reason, count in counts.items():
            reason_trades = self.trades_df[self.trades_df["exit_reason"] == reason]
            avg_pnl = reason_trades["pnl"].mean()
            win_rate = (reason_trades["pnl"] > 0).mean()
            breakdown[reason] = {
                "count": int(count),
                "pct": round(count / total * 100, 1),
                "avg_pnl": round(float(avg_pnl), 4),
                "win_rate": round(float(win_rate), 3),
            }

        return breakdown

    # ── Risk Management Effectiveness ───────────────────────────────────

    def risk_management_effectiveness(self) -> dict:
        if self.trades_df.empty or "exit_reason" not in self.trades_df.columns:
            return {}

        df = self.trades_df
        sl_trades = df[df["exit_reason"] == "stop_loss"]
        tp_trades = df[df["exit_reason"] == "take_profit"]
        trail_trades = df[df["exit_reason"] == "trailing_stop"]
        hold_trades = df[df["exit_reason"] == "hold_limit"]

        # Avg loss on SL exits vs hold_limit exits
        sl_losses = sl_trades["pnl"][sl_trades["pnl"] < 0]
        hold_losses = hold_trades["pnl"][hold_trades["pnl"] < 0]
        avg_sl_loss = round(float(sl_losses.mean()), 4) if len(sl_losses) > 0 else 0.0
        avg_hold_loss = round(float(hold_losses.mean()), 4) if len(hold_losses) > 0 else 0.0

        # Avg win on TP/trailing exits vs hold_limit exits
        managed = pd.concat([tp_trades, trail_trades]) if not (tp_trades.empty and trail_trades.empty) else pd.DataFrame()
        managed_wins = managed["pnl"][managed["pnl"] > 0] if not managed.empty and "pnl" in managed.columns else pd.Series(dtype=float)
        hold_wins = hold_trades["pnl"][hold_trades["pnl"] > 0]
        avg_managed_win = round(float(managed_wins.mean()), 4) if len(managed_wins) > 0 else 0.0
        avg_hold_win = round(float(hold_wins.mean()), 4) if len(hold_wins) > 0 else 0.0

        # sl_loss_reduction: positive means SL capped losses tighter than hold_limit
        # (avg_hold_loss is more negative, so hold_loss - sl_loss = how much worse hold is)
        sl_loss_reduction = round(avg_sl_loss - avg_hold_loss, 4) if avg_hold_loss != 0.0 else 0.0

        return {
            "avg_sl_loss": avg_sl_loss,
            "avg_hold_loss": avg_hold_loss,
            "sl_loss_reduction": sl_loss_reduction,
            "avg_managed_win": avg_managed_win,
            "avg_hold_win": avg_hold_win,
            "sl_count": len(sl_trades),
            "tp_count": len(tp_trades),
            "trail_count": len(trail_trades),
            "hold_count": len(hold_trades),
        }

    # ── VaR Validation ─────────────────────────────────────────────────

    def var_validation(
        self,
        var_forecasts: np.ndarray,
        confidence: float = 95.0,
    ) -> dict:
        """Run Kupiec + Christoffersen tests against VaR forecasts.

        Reuses the implementations in cortex.backtesting.
        """
        from cortex.backtesting import backtest_var

        returns = self.result.daily_returns.dropna().values
        return backtest_var(returns, var_forecasts, confidence)

    # ── Benchmark Comparison ───────────────────────────────────────────

    def benchmark_comparison(self, benchmark_prices: pd.Series | None = None) -> dict:
        eq = self.result.equity_curve
        if len(eq) < 2:
            return {"strategy_return_pct": 0.0}

        strategy_return = (eq.iloc[-1] / eq.iloc[0]) - 1

        result: dict = {
            "strategy_return_pct": round(float(strategy_return * 100), 2),
        }

        if benchmark_prices is not None and len(benchmark_prices) >= 2:
            bh_return = (benchmark_prices.iloc[-1] / benchmark_prices.iloc[0]) - 1
            result["buy_hold_return_pct"] = round(float(bh_return * 100), 2)
            result["vs_buy_hold_pct"] = round(float((strategy_return - bh_return) * 100), 2)

            # Information ratio
            strat_daily = eq.pct_change().dropna()
            bench_daily = benchmark_prices.pct_change().dropna()
            common = strat_daily.index.intersection(bench_daily.index)
            if len(common) > 1:
                active = strat_daily.loc[common] - bench_daily.loc[common]
                tracking_error = active.std()
                info_ratio = float(np.sqrt(252) * active.mean() / tracking_error) if tracking_error > 0 else 0.0
                result["information_ratio"] = round(info_ratio, 3)

        return result

    # ── Text Report ────────────────────────────────────────────────────

    def generate_report(self) -> str:
        m = self.compute_all()
        cfg = self.result.config
        eq = self.result.equity_curve

        final_equity = f"${eq.iloc[-1]:,.2f}" if len(eq) > 0 else "N/A"

        pf = m.get("profit_factor", 0)
        pf_str = "inf" if pf == float("inf") else f"{pf:.3f}"

        lines = [
            "=" * 60,
            "  GUARDIAN BACKTESTING REPORT",
            "=" * 60,
            f"  Token:           {cfg.token}",
            f"  Period:          {cfg.start_date} -> {cfg.end_date}",
            f"  Timeframe:       {cfg.timeframe}",
            f"  Initial Capital: ${cfg.initial_capital:,.2f}",
            f"  Final Equity:    {final_equity}",
            "",
            "-- Risk-Adjusted Returns --",
            f"  Total Return:    {m.get('total_return_pct', 0):.2f}%",
            f"  Annual Return:   {m.get('annual_return_pct', 0):.2f}%",
            f"  Sharpe Ratio:    {m.get('sharpe_ratio', 0):.3f}",
            f"  Sortino Ratio:   {m.get('sortino_ratio', 0):.3f}",
            f"  Calmar Ratio:    {m.get('calmar_ratio', 0):.3f}",
            "",
            "-- Drawdown --",
            f"  Max Drawdown:    {m.get('max_drawdown', 0) * 100:.2f}%",
            f"  Max DD Duration: {m.get('max_drawdown_duration_bars', 0)} bars",
            f"  Recovery Time:   {m.get('recovery_time_bars', 0)} bars",
            "",
            "-- Trade Quality --",
            f"  Total Trades:    {m.get('total_trades', 0)}",
            f"  Win Rate:        {m.get('win_rate', 0) * 100:.1f}%",
            f"  Profit Factor:   {pf_str}",
            f"  Expectancy:      ${m.get('expectancy', 0):.2f}",
            f"  Avg Trade PnL:   ${m.get('avg_trade_pnl', 0):.2f}",
            "",
            "-- Guardian --",
            f"  Signals:         {m.get('signals_generated', 0)} generated, "
            f"{m.get('signals_approved', 0)} approved, "
            f"{m.get('signals_rejected', 0)} rejected",
            f"  Approval Rate:   {m.get('approval_rate', 0) * 100:.1f}%",
            "",
            "=" * 60,
        ]
        return "\n".join(lines)
