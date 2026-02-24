"""Guardian replay engine — replays the full Guardian risk pipeline against historical data.

Ties together HistoricalDataFeed, ExecutionSimulator, risk models (EVT, SVJ,
Hawkes, MSM), TradeLedger, and assess_trade() to produce a complete
backtest with equity curve, trade log, and component-level scoring history.
"""

from __future__ import annotations

import copy
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
    position_hold_bars: int = 48
    mean_revert_threshold: float = 2.0
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.08
    trailing_stop_pct: float = 0.03
    use_trailing_stop: bool = True
    momentum_window: int = 10
    momentum_threshold: float = 0.5
    use_momentum_filter: bool = True
    use_ta_filter: bool = True
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    use_macd_filter: bool = True
    use_bb_filter: bool = True
    use_agents: bool = False
    agent_approval_threshold: float = 60.0
    agent_veto_score: float = 85.0


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

    def __init__(self, config: BacktestConfig, calibration_cache: dict | None = None) -> None:
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

        # Bar-indexed calibration cache: {bar_number: calibration_snapshot}
        # Populated on first run, reused on subsequent runs (deterministic)
        self._calibration_bar_cache: dict[int, dict] = {}
        if calibration_cache:
            self._calibration_bar_cache = calibration_cache

        # Multi-agent coordinator (lazy-init when use_agents=True)
        self._coordinator = None
        if config.use_agents:
            self._coordinator = self._init_coordinator()

    def run(self, data: pd.DataFrame | None = None, btc_data: pd.Series | None = None) -> BacktestResult:
        """Main bar-by-bar event loop.

        Args:
            data: OHLCV DataFrame with datetime index.
            btc_data: Optional BTC close prices (Series) for macro agent backtest mode.
        """
        if data is None:
            raise ValueError(
                "Provide OHLCV DataFrame directly (use HistoricalDataFeed.load_ohlcv() externally)."
            )

        self._btc_data = btc_data

        df = HistoricalDataFeed.compute_returns(data)

        # Pre-compute TA indicators once (adds rsi, macd_hist, bb_upper, bb_lower columns)
        if self.config.use_ta_filter:
            from cortex.backtest.technical import TechnicalIndicators
            df = TechnicalIndicators.compute_all(df)

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
                if i in self._calibration_bar_cache:
                    self._restore_calibration(self._calibration_bar_cache[i])
                else:
                    events = self._extract_hawkes_events(returns_window)
                    self._calibrate_models(returns_window, events)
                    self._calibration_bar_cache[i] = self._snapshot_calibration()
                self._last_calibration_bar = i

            # Determine current regime
            regime_state = self._current_regime()
            self.regime_history.append({
                "timestamp": str(ts),
                "bar": i,
                "regime": regime_state,
            })

            # Risk management exits (SL/TP/trailing)
            if self.position is not None:
                current_price = float(bar["close"])
                entry_price = self.position["entry_price"]

                if self.position["direction"] == "long":
                    unrealized_pct = (current_price - entry_price) / entry_price
                else:
                    unrealized_pct = (entry_price - current_price) / entry_price

                self.position["max_favorable_pct"] = max(
                    self.position.get("max_favorable_pct", 0.0), unrealized_pct
                )

                exit_reason = None

                if self.config.stop_loss_pct and unrealized_pct <= -self.config.stop_loss_pct:
                    exit_reason = "stop_loss"
                elif self.config.take_profit_pct and unrealized_pct >= self.config.take_profit_pct:
                    exit_reason = "take_profit"
                elif (
                    self.config.use_trailing_stop
                    and self.config.trailing_stop_pct
                    and self.position["max_favorable_pct"] > 0
                ):
                    drawdown_from_peak = self.position["max_favorable_pct"] - unrealized_pct
                    if drawdown_from_peak >= self.config.trailing_stop_pct:
                        exit_reason = "trailing_stop"

                if exit_reason:
                    pnl = self._close_position(bar, exit_reason=exit_reason)
                    self.equity += pnl

            # Close position if hold limit exceeded or regime changed
            if self.position is not None:
                bars_held = i - self.position["entry_bar"]
                should_close = bars_held >= self.config.position_hold_bars
                if not should_close and len(self.regime_history) >= 2:
                    prev_regime = self.regime_history[-2]["regime"]
                    should_close = regime_state != prev_regime
                if should_close:
                    exit_reason = "hold_limit"
                    if not (bars_held >= self.config.position_hold_bars):
                        exit_reason = "regime_change"
                    pnl = self._close_position(bar, exit_reason=exit_reason)
                    self.equity += pnl

            # Generate signal
            signal = self._build_signal(bar, regime_state, returns_window)
            if signal is not None and self.position is None:
                signals_generated += 1
                trade_size_usd = self.equity * self.config.trade_size_pct

                # Build model data dicts
                model_data = self._build_model_data(returns_window)
                evt_data = self._build_evt_data()
                svj_data = self._build_svj_data(returns_window)
                hawkes_data = self._build_hawkes_data(
                    self._extract_hawkes_events(returns_window)
                )

                if self._coordinator is not None:
                    # Multi-agent path: coordinator dispatches to all agents
                    decision = self._evaluate_agents(
                        df, i, trade_size_usd, model_data, evt_data, svj_data, hawkes_data
                    )
                else:
                    # Legacy path: monolithic assess_trade()
                    decision = self._evaluate_guardian(
                        signal, trade_size_usd, model_data, evt_data, svj_data, hawkes_data
                    )

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
                                "max_favorable_pct": 0.0,
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
            pnl = self._close_position(df.iloc[-1], exit_reason="end_of_data")
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

    def export_calibration_cache(self) -> dict[int, dict]:
        """Export bar-indexed calibration cache for reuse across sweep runs.

        Returns a shallow copy of the cache dict — individual snapshots are
        already deep-copied at creation time, and _restore_calibration deep-copies
        on read, so this is safe for cross-run sharing.
        """
        return dict(self._calibration_bar_cache)

    def _snapshot_calibration(self) -> dict:
        """Snapshot current calibration state (deep copy to avoid mutation)."""
        return {
            "evt_params": copy.deepcopy(self._evt_params),
            "svj_params": copy.deepcopy(self._svj_params),
            "hawkes_params": copy.deepcopy(self._hawkes_params),
            "msm_params": copy.deepcopy(self._msm_params),
            "filter_probs": self._filter_probs.copy(deep=True) if self._filter_probs is not None else None,
            "sigma_states": self._sigma_states.copy() if self._sigma_states is not None else None,
            "P_matrix": self._P_matrix.copy() if self._P_matrix is not None else None,
        }

    def _restore_calibration(self, snapshot: dict) -> None:
        """Restore calibration from a snapshot (deep copy to isolate runs)."""
        self._evt_params = copy.deepcopy(snapshot.get("evt_params"))
        self._svj_params = copy.deepcopy(snapshot.get("svj_params"))
        self._hawkes_params = copy.deepcopy(snapshot.get("hawkes_params"))
        self._msm_params = copy.deepcopy(snapshot.get("msm_params"))
        fp = snapshot.get("filter_probs")
        self._filter_probs = fp.copy(deep=True) if fp is not None else None
        ss = snapshot.get("sigma_states")
        self._sigma_states = ss.copy() if ss is not None else None
        pm = snapshot.get("P_matrix")
        self._P_matrix = pm.copy() if pm is not None else None

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

    def _confirm_momentum(self, returns_pct: pd.Series, direction: str) -> bool:
        """Check if recent momentum confirms the signal direction."""
        if not self.config.use_momentum_filter:
            return True
        window = min(self.config.momentum_window, len(returns_pct))
        if window < 3:
            return True
        recent = returns_pct.iloc[-window:]
        std = recent.std()
        momentum = recent.mean() / std if std > 0 else 0.0
        if direction == "long":
            return momentum > -self.config.momentum_threshold
        return momentum < self.config.momentum_threshold

    def _confirm_ta(self, bar: pd.Series, direction: str) -> bool:
        """Check if at least 1 TA indicator confirms the signal direction."""
        if not self.config.use_ta_filter:
            return True

        score = 0
        rsi = bar.get("rsi")
        macd_hist = bar.get("macd_hist")
        bb_upper = bar.get("bb_upper")
        bb_lower = bar.get("bb_lower")
        close = float(bar["close"])

        if direction == "long":
            if rsi is not None and not pd.isna(rsi) and rsi < self.config.rsi_oversold:
                score += 1
            if self.config.use_macd_filter and macd_hist is not None and not pd.isna(macd_hist) and macd_hist > 0:
                score += 1
            if self.config.use_bb_filter and bb_lower is not None and not pd.isna(bb_lower) and close <= bb_lower * 1.01:
                score += 1
        else:
            if rsi is not None and not pd.isna(rsi) and rsi > self.config.rsi_overbought:
                score += 1
            if self.config.use_macd_filter and macd_hist is not None and not pd.isna(macd_hist) and macd_hist < 0:
                score += 1
            if self.config.use_bb_filter and bb_upper is not None and not pd.isna(bb_upper) and close >= bb_upper * 0.99:
                score += 1

        return score >= 1

    def _build_signal(
        self, bar: pd.Series, regime_state: int, returns_pct: pd.Series
    ) -> str | None:
        strategy = self.config.signal_strategy

        if strategy == "always_long":
            return "long"

        if strategy == "regime":
            if regime_state <= 1:
                direction = "long"
            elif regime_state >= 3:
                direction = "short"
            else:
                return None
            if not self._confirm_momentum(returns_pct, direction):
                return None
            if not self._confirm_ta(bar, direction):
                return None
            return direction

        if strategy == "mean_revert":
            if len(returns_pct) < 2:
                return None
            last_ret = returns_pct.iloc[-1]
            roll_std = returns_pct.iloc[-20:].std() if len(returns_pct) >= 20 else returns_pct.std()
            if roll_std == 0:
                return None
            z = last_ret / roll_std
            if z < -self.config.mean_revert_threshold:
                direction = "long"
            elif z > self.config.mean_revert_threshold:
                direction = "short"
            else:
                return None
            if not self._confirm_momentum(returns_pct, direction):
                return None
            if not self._confirm_ta(bar, direction):
                return None
            return direction

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

    def _init_coordinator(self):
        """Initialize multi-agent coordinator with all specialist agents."""
        from cortex.agents.coordinator import AgentCoordinator
        from cortex.agents.technical_analyst import TechnicalAnalystAgent
        from cortex.agents.macro_analyst import MacroAnalystAgent
        from cortex.agents.risk_researcher import RiskResearcherAgent

        agents = [
            TechnicalAnalystAgent(
                rsi_oversold=self.config.rsi_oversold,
                rsi_overbought=self.config.rsi_overbought,
            ),
            MacroAnalystAgent(),
            RiskResearcherAgent(),
        ]
        return AgentCoordinator(
            agents=agents,
            approval_threshold=self.config.agent_approval_threshold,
            veto_score=self.config.agent_veto_score,
        )

    def _evaluate_agents(
        self,
        df: pd.DataFrame,
        bar_idx: int,
        trade_size_usd: float,
        model_data: dict | None,
        evt_data: dict | None,
        svj_data: dict | None,
        hawkes_data: dict | None,
    ) -> dict | None:
        """Run multi-agent coordinator for trade approval."""
        context = {
            "evt_data": evt_data,
            "svj_data": svj_data,
            "hawkes_data": hawkes_data,
            "model_data": model_data,
            "btc_close": self._btc_data,
        }
        try:
            decision = self._coordinator.evaluate_backtest(
                token=self.config.token,
                data=df,
                bar_idx=bar_idx,
                context=context,
                trade_size_usd=trade_size_usd,
            )
            return self._coordinator.to_guardian_format(decision)
        except Exception:
            logger.warning("agent_coordinator_failed", bar=bar_idx, exc_info=True)
            return None

    def _evaluate_guardian(
        self,
        signal: str,
        trade_size_usd: float,
        model_data: dict | None,
        evt_data: dict | None,
        svj_data: dict | None,
        hawkes_data: dict | None,
    ) -> dict | None:
        """Run legacy monolithic assess_trade() for trade approval."""
        neutral_news = {"n_items": 0}
        neutral_alams = {
            "var_total": 0.04375,
            "current_regime": 0,
            "delta": 0.0,
            "regime_probs": [],
        }
        try:
            from cortex.guardian import assess_trade

            return assess_trade(
                token=self.config.token,
                trade_size_usd=trade_size_usd,
                direction=signal,
                model_data=model_data,
                evt_data=evt_data,
                svj_data=svj_data,
                hawkes_data=hawkes_data,
                news_data=neutral_news,
                alams_data=neutral_alams,
                strategy=None,
                run_debate=False,
                agent_confidence=None,
            )
        except Exception:
            logger.warning("assess_trade_failed", exc_info=True)
            return None

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

    def _close_position(self, bar: pd.Series, exit_reason: str = "hold_limit") -> float:
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
            "exit_reason": exit_reason,
        })

        self.position = None
        return pnl
