"""Guardian replay engine — replays the full Guardian risk pipeline against historical data.

Ties together HistoricalDataFeed, ExecutionSimulator, risk models (EVT, SVJ,
Hawkes, MSM), TradeLedger, and assess_trade() to produce a complete
backtest with equity curve, trade log, and component-level scoring history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import structlog

from cortex.backtest.data_feed import HistoricalDataFeed
from cortex.backtest.execution import ExecutionSimulator
from cortex.config import APPROVAL_THRESHOLD, GUARDIAN_WEIGHTS
from cortex.trade_ledger import TradeLedger

logger = structlog.get_logger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    token: str = "SOL"
    token_mint: str = "So11111111111111111111111111111111111111112"
    timeframe: str = "1h"
    start_date: str = ""
    end_date: str = ""
    trade_size_pct: float = 0.10
    recalibration_interval: int = 24
    min_calibration_bars: int = 100
    guardian_weights: dict | None = None
    approval_threshold: float = 60.0
    webacy_enabled: bool = False
    news_enabled: bool = False
    signal_strategy: str = "regime"
    position_hold_bars: int = 24
    mean_revert_threshold: float = 2.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    trades: list[dict] = field(default_factory=list)
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    regime_history: list[dict] = field(default_factory=list)
    component_score_history: list[dict] = field(default_factory=list)
    signals_generated: int = 0
    signals_approved: int = 0
    signals_rejected: int = 0
    config: BacktestConfig = field(default_factory=BacktestConfig)


class GuardianBacktester:
    """Replays Guardian pipeline on historical OHLCV data bar-by-bar."""

    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.feed = HistoricalDataFeed()
        self.executor = ExecutionSimulator()
        self.ledger = TradeLedger()
        self.equity = config.initial_capital
        self.position: dict | None = None
        self.equity_curve: list[tuple] = []
        self.trades: list[dict] = []
        self.component_scores: list[dict] = []
        self.regime_history: list[dict] = []

        self._evt_params: dict | None = None
        self._svj_params: dict | None = None
        self._hawkes_params: dict | None = None
        self._msm_params: dict | None = None
        self._filter_probs = None
        self._sigma_states = None
        self._P_matrix = None
        self._last_calibration_bar: int = -1

    def run(self, data: pd.DataFrame | None = None) -> BacktestResult:
        """Main bar-by-bar event loop."""
        if data is None:
            raise ValueError(
                "Provide OHLCV DataFrame directly (use HistoricalDataFeed.load_ohlcv() externally)."
            )

        df = HistoricalDataFeed.compute_returns(data)
        if len(df) < self.config.min_calibration_bars + 10:
            raise ValueError(
                f"Need at least {self.config.min_calibration_bars + 10} bars, got {len(df)}"
            )

        returns_pct = df["log_return"].dropna() * 100.0
        start_bar = self.config.min_calibration_bars

        signals_generated = 0
        signals_approved = 0
        signals_rejected = 0

        for i in range(start_bar, len(df)):
            bar = df.iloc[i]
            ts = df.index[i]
            returns_window = returns_pct.iloc[:i + 1]

            # Recalibrate models at configured intervals
            if self._needs_recalibration(i):
                events = self._extract_hawkes_events(returns_window)
                self._calibrate_models(returns_window, events)
                self._last_calibration_bar = i

            # Determine current regime
            regime_state = self._current_regime()
            self.regime_history.append({
                "timestamp": str(ts),
                "bar": i,
                "regime": regime_state,
            })

            # Close position if hold limit exceeded or regime changed
            if self.position is not None:
                bars_held = i - self.position["entry_bar"]
                should_close = bars_held >= self.config.position_hold_bars
                if not should_close and len(self.regime_history) >= 2:
                    prev_regime = self.regime_history[-2]["regime"]
                    should_close = regime_state != prev_regime
                if should_close:
                    pnl = self._close_position(bar)
                    self.equity += pnl

            # Generate signal
            signal = self._build_signal(bar, regime_state, returns_window)
            if signal is not None and self.position is None:
                signals_generated += 1
                trade_size_usd = self.equity * self.config.trade_size_pct

                # Build model data dicts for assess_trade()
                model_data = self._build_model_data(returns_window)
                evt_data = self._build_evt_data()
                svj_data = self._build_svj_data(returns_window)
                hawkes_data = self._build_hawkes_data(
                    self._extract_hawkes_events(returns_window)
                )

                try:
                    from cortex.guardian import assess_trade

                    decision = assess_trade(
                        token=self.config.token,
                        trade_size_usd=trade_size_usd,
                        direction=signal,
                        model_data=model_data,
                        evt_data=evt_data,
                        svj_data=svj_data,
                        hawkes_data=hawkes_data,
                        news_data=None,
                        alams_data=None,
                        strategy=None,
                        run_debate=False,
                        agent_confidence=None,
                    )
                except Exception:
                    logger.warning("assess_trade_failed", bar=i, exc_info=True)
                    decision = None

                if decision is not None:
                    self.component_scores.append({
                        "timestamp": str(ts),
                        "bar": i,
                        "risk_score": decision.get("risk_score"),
                        "approved": decision.get("approved"),
                        "component_scores": decision.get("component_scores", []),
                    })

                    if decision.get("approved"):
                        signals_approved += 1
                        exec_result = self._execute_trade(
                            bar, signal, decision.get("recommended_size", trade_size_usd)
                        )
                        if exec_result is not None:
                            self.position = {
                                "direction": signal,
                                "size_usd": exec_result["size_usd"],
                                "entry_price": exec_result["execution_price"],
                                "entry_bar": i,
                                "entry_ts": str(ts),
                            }
                    else:
                        signals_rejected += 1

            # Update equity curve: cash + unrealized PnL
            unrealized = 0.0
            if self.position is not None:
                unrealized = self._unrealized_pnl(bar)
            self.equity_curve.append((ts, self.equity + unrealized))

        # Close any open position at end
        if self.position is not None:
            pnl = self._close_position(df.iloc[-1])
            self.equity += pnl
            self.equity_curve[-1] = (df.index[-1], self.equity)

        eq_series = pd.Series(
            [e[1] for e in self.equity_curve],
            index=pd.DatetimeIndex([e[0] for e in self.equity_curve]),
            name="equity",
        )
        daily_returns = eq_series.resample("1D").last().pct_change().dropna()

        return BacktestResult(
            equity_curve=eq_series,
            trades=self.trades,
            daily_returns=daily_returns,
            regime_history=self.regime_history,
            component_score_history=self.component_scores,
            signals_generated=signals_generated,
            signals_approved=signals_approved,
            signals_rejected=signals_rejected,
            config=self.config,
        )

    def _needs_recalibration(self, current_bar: int) -> bool:
        if self._last_calibration_bar < 0:
            return True
        return (current_bar - self._last_calibration_bar) >= self.config.recalibration_interval

    def _calibrate_models(self, returns_pct: pd.Series, events: np.ndarray) -> None:
        """Recalibrate all risk models on current data window (no look-ahead)."""
        # EVT
        try:
            from cortex.evt import fit_gpd, select_threshold

            losses = -returns_pct.values / 100.0
            losses = np.abs(losses)
            thresh_info = select_threshold(returns_pct.values / 100.0)
            self._evt_params = fit_gpd(losses, thresh_info["threshold"])
        except Exception:
            logger.debug("evt_calibration_failed", exc_info=True)

        # SVJ (last 252 bars or all available)
        try:
            from cortex.svj import calibrate_svj

            window = returns_pct.iloc[-252:] if len(returns_pct) > 252 else returns_pct
            self._svj_params = calibrate_svj(window)
        except Exception:
            logger.debug("svj_calibration_failed", exc_info=True)

        # Hawkes
        try:
            from cortex.hawkes import fit_hawkes as _fit_hawkes

            if len(events) >= 5:
                T = float(len(returns_pct))
                self._hawkes_params = _fit_hawkes(events, T)
        except Exception:
            logger.debug("hawkes_calibration_failed", exc_info=True)

        # MSM
        try:
            from cortex.msm import calibrate_msm_advanced, msm_vol_forecast

            self._msm_params = calibrate_msm_advanced(
                returns_pct, num_states=5, method="mle", verbose=False
            )
            _, _, filter_probs, sigma_states, P_matrix = msm_vol_forecast(
                returns_pct,
                num_states=self._msm_params["num_states"],
                sigma_low=self._msm_params["sigma_low"],
                sigma_high=self._msm_params["sigma_high"],
                p_stay=self._msm_params["p_stay"],
            )
            self._filter_probs = filter_probs
            self._sigma_states = sigma_states
            self._P_matrix = P_matrix
        except Exception:
            logger.debug("msm_calibration_failed", exc_info=True)

    def _current_regime(self) -> int:
        """Return the most likely regime state (0-indexed)."""
        if self._filter_probs is None:
            return 2  # default: middle regime
        last_probs = self._filter_probs.iloc[-1].values
        return int(np.argmax(last_probs))

    def _build_signal(
        self, bar: pd.Series, regime_state: int, returns_pct: pd.Series
    ) -> str | None:
        strategy = self.config.signal_strategy

        if strategy == "always_long":
            return "long"

        if strategy == "regime":
            if regime_state <= 1:
                return "long"
            if regime_state >= 3:
                return "short"
            return None

        if strategy == "mean_revert":
            if len(returns_pct) < 2:
                return None
            last_ret = returns_pct.iloc[-1]
            roll_std = returns_pct.iloc[-20:].std() if len(returns_pct) >= 20 else returns_pct.std()
            if roll_std == 0:
                return None
            z = last_ret / roll_std
            if z < -self.config.mean_revert_threshold:
                return "long"
            if z > self.config.mean_revert_threshold:
                return "short"
            return None

        return None

    def _build_model_data(self, returns_pct: pd.Series) -> dict | None:
        if self._filter_probs is None or self._msm_params is None:
            return None
        return {
            "filter_probs": self._filter_probs,
            "calibration": {
                "num_states": self._msm_params["num_states"],
                "sigma_states": (
                    self._sigma_states.tolist()
                    if self._sigma_states is not None
                    else []
                ),
            },
        }

    def _build_evt_data(self) -> dict | None:
        if self._evt_params is None:
            return None
        return {
            "xi": self._evt_params["xi"],
            "beta": self._evt_params["beta"],
            "threshold": self._evt_params["threshold"],
            "n_total": self._evt_params["n_total"],
            "n_exceedances": self._evt_params["n_exceedances"],
        }

    def _build_svj_data(self, returns_pct: pd.Series) -> dict | None:
        if self._svj_params is None:
            return None
        return {
            "returns": returns_pct,
            "calibration": self._svj_params,
        }

    def _build_hawkes_data(self, events: np.ndarray) -> dict | None:
        if self._hawkes_params is None:
            return None
        return {
            "event_times": events,
            "mu": self._hawkes_params["mu"],
            "alpha": self._hawkes_params["alpha"],
            "beta": self._hawkes_params["beta"],
        }

    def _extract_hawkes_events(self, returns_pct: pd.Series) -> np.ndarray:
        """Extract extreme-move event indices from returns for Hawkes input."""
        try:
            from cortex.hawkes import extract_events

            result = extract_events(returns_pct, threshold_percentile=5.0, use_absolute=True)
            return result["event_times"]
        except Exception:
            return np.array([])

    def _execute_trade(self, bar: pd.Series, direction: str, size_usd: float) -> dict | None:
        """Simulate execution and record in TradeLedger."""
        price = float(bar["close"])
        volume = float(bar.get("volume", 1_000_000))

        result = self.executor.simulate_execution(
            price=price,
            size_usd=size_usd,
            volume_24h=volume * 24,
            direction=direction,
        )

        if not result.filled:
            return None

        entry_hash = self.ledger.stage(
            token=self.config.token,
            direction=direction,
            trade_size_usd=size_usd,
            strategy="backtest",
            guardian_score=None,
        )
        self.ledger.commit(entry_hash, f"backtest_{direction}_{price:.2f}")
        self.ledger.push(entry_hash, result={
            "execution_price": result.execution_price,
            "slippage_bps": result.slippage_bps,
            "fee_usd": result.fee_usd,
        })

        self.equity -= result.net_cost_usd

        return {
            "execution_price": result.execution_price,
            "size_usd": size_usd,
            "fee_usd": result.fee_usd,
            "net_cost_usd": result.net_cost_usd,
        }

    def _unrealized_pnl(self, bar: pd.Series) -> float:
        if self.position is None:
            return 0.0
        current_price = float(bar["close"])
        entry_price = self.position["entry_price"]
        size_usd = self.position["size_usd"]
        if self.position["direction"] == "long":
            return size_usd * (current_price - entry_price) / entry_price
        else:
            return size_usd * (entry_price - current_price) / entry_price

    def _close_position(self, bar: pd.Series) -> float:
        """Close current position, return realized PnL."""
        if self.position is None:
            return 0.0

        current_price = float(bar["close"])
        entry_price = self.position["entry_price"]
        size_usd = self.position["size_usd"]
        direction = self.position["direction"]

        if direction == "long":
            pnl = size_usd * (current_price - entry_price) / entry_price
        else:
            pnl = size_usd * (entry_price - current_price) / entry_price

        exit_cost = self.executor.get_transaction_cost(size_usd)
        pnl -= exit_cost

        self.trades.append({
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": current_price,
            "size_usd": size_usd,
            "pnl": round(pnl, 4),
            "entry_bar": self.position["entry_bar"],
            "entry_ts": self.position["entry_ts"],
            "exit_ts": str(bar.name) if hasattr(bar, "name") else "",
        })

        self.position = None
        return pnl
