"""Walk-forward validation runner for MSM-based VaR models.

Implements anchored (expanding) and rolling walk-forward backtesting with
periodic model recalibration. At each step, produces a 1-step-ahead VaR
forecast, then scores the full out-of-sample series with Kupiec (1995)
and Christoffersen (1998) tests, both overall and per-regime.

References:
  - Kupiec (1995) "Techniques for Verifying the Accuracy of Risk Measurement Models"
  - Christoffersen (1998) "Evaluating Interval Forecasts"
  - Hansen & Lunde (2005) walk-forward model evaluation
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from cortex import msm
from cortex.backtesting import backtest_var

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    min_train_window: int = 120
    step_size: int = 1
    refit_interval: int = 20
    expanding: bool = True
    max_train_window: int | None = None
    confidence: float = 95.0
    num_states: int = 5
    method: str = "empirical"
    use_student_t: bool = False
    nu: float = 5.0
    leverage_gamma: float = 0.0

    def to_dict(self) -> dict:
        return {
            "min_train_window": self.min_train_window,
            "step_size": self.step_size,
            "refit_interval": self.refit_interval,
            "expanding": self.expanding,
            "max_train_window": self.max_train_window,
            "confidence": self.confidence,
            "num_states": self.num_states,
            "method": self.method,
            "use_student_t": self.use_student_t,
            "nu": self.nu,
            "leverage_gamma": self.leverage_gamma,
        }


@dataclass
class WalkForwardResult:
    forecasts: pd.DataFrame
    calibrations: list[dict] = field(default_factory=list)
    backtest: dict = field(default_factory=dict)
    regime_backtests: dict = field(default_factory=dict)
    parameter_stability: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0


def _calibrate_and_forecast(
    train_returns: pd.Series,
    cfg: WalkForwardConfig,
) -> tuple[dict, pd.DataFrame, np.ndarray, np.ndarray]:
    """Run calibration + filter on a training window. Returns (calibration, filter_probs, sigma_states, P_matrix)."""
    cal = msm.calibrate_msm_advanced(
        train_returns,
        num_states=cfg.num_states,
        method=cfg.method,
        verbose=False,
        leverage_gamma=cfg.leverage_gamma,
    )
    _, _, filter_probs, sigma_states, P_matrix = msm.msm_vol_forecast(
        train_returns,
        num_states=cal["num_states"],
        sigma_low=cal["sigma_low"],
        sigma_high=cal["sigma_high"],
        p_stay=cal["p_stay"],
        leverage_gamma=cfg.leverage_gamma,
    )
    return cal, filter_probs, sigma_states, P_matrix


def _forecast_var(
    filter_probs: pd.DataFrame,
    sigma_states: np.ndarray,
    P_matrix: np.ndarray,
    cfg: WalkForwardConfig,
    last_return: float = 0.0,
    p_stay=None,
) -> float:
    """Produce a 1-step-ahead VaR forecast from current filter state."""
    alpha = 1.0 - cfg.confidence / 100.0
    var_t1, _, _, _ = msm.msm_var_forecast_next_day(
        filter_probs,
        sigma_states,
        P_matrix,
        alpha=alpha,
        use_student_t=cfg.use_student_t,
        nu=cfg.nu,
        leverage_gamma=cfg.leverage_gamma,
        last_return=last_return,
        p_stay=p_stay,
    )
    return float(var_t1)


def run_walk_forward(
    returns: pd.Series,
    config: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    """Run walk-forward VaR backtest over a return series.

    Args:
        returns: Daily log-returns in %, with DatetimeIndex.
        config: Walk-forward configuration. Defaults to WalkForwardConfig().

    Returns:
        WalkForwardResult with forecasts DataFrame, backtest stats,
        per-regime backtests, parameter stability, and calibration snapshots.

    Raises:
        ValueError: If returns too short for the minimum training window.
    """
    t_start = time.monotonic()
    cfg = config or WalkForwardConfig()
    n = len(returns)

    if n < cfg.min_train_window + 10:
        raise ValueError(
            f"Need at least {cfg.min_train_window + 10} observations, got {n}"
        )

    records: list[dict] = []
    calibrations: list[dict] = []

    cal = None
    filter_probs = None
    sigma_states = None
    P_matrix = None
    steps_since_refit = cfg.refit_interval  # force initial calibration

    step_indices = list(range(cfg.min_train_window, n, cfg.step_size))

    for step_num, train_end in enumerate(step_indices):
        if train_end >= n:
            break

        # Determine training window
        if cfg.expanding:
            train_start = 0
        else:
            window = cfg.max_train_window or cfg.min_train_window
            train_start = max(0, train_end - window)

        train_returns = returns.iloc[train_start:train_end]

        # Recalibrate if needed
        if steps_since_refit >= cfg.refit_interval or cal is None:
            try:
                cal, filter_probs, sigma_states, P_matrix = _calibrate_and_forecast(
                    train_returns, cfg
                )
                calibrations.append({
                    "step": step_num,
                    "train_end_idx": train_end,
                    "date": str(returns.index[train_end]) if hasattr(returns.index, 'date') else train_end,
                    "sigma_low": float(cal["sigma_low"]),
                    "sigma_high": float(cal["sigma_high"]),
                    "p_stay": [float(p) for p in cal["p_stay"]] if isinstance(cal["p_stay"], (list, np.ndarray)) else float(cal["p_stay"]),
                    "num_states": cal["num_states"],
                })
                steps_since_refit = 0
            except Exception:
                logger.warning("Calibration failed at step %d, reusing previous", step_num)
                if cal is None:
                    continue
        else:
            # Re-run filter with existing parameters on updated training data
            try:
                _, _, filter_probs, sigma_states, P_matrix = msm.msm_vol_forecast(
                    train_returns,
                    num_states=cal["num_states"],
                    sigma_low=cal["sigma_low"],
                    sigma_high=cal["sigma_high"],
                    p_stay=cal["p_stay"],
                    leverage_gamma=cfg.leverage_gamma,
                )
            except Exception:
                logger.warning("Filter update failed at step %d", step_num)
                continue

        steps_since_refit += 1

        # Produce VaR forecast for t+1
        last_return = float(train_returns.iloc[-1]) if len(train_returns) > 0 else 0.0
        p_stay = cal.get("p_stay") if cal else None
        var_forecast = _forecast_var(
            filter_probs, sigma_states, P_matrix, cfg,
            last_return=last_return, p_stay=p_stay,
        )

        # Realized return at t+1
        test_idx = train_end
        realized = float(returns.iloc[test_idx])

        # Current regime (argmax of last filter probs row)
        last_probs = np.asarray(filter_probs.iloc[-1], dtype=float)
        regime = int(np.argmax(last_probs)) + 1  # 1-based

        records.append({
            "date": returns.index[test_idx],
            "var_forecast": var_forecast,
            "realized_return": realized,
            "violation": realized < var_forecast,
            "regime": regime,
            "regime_prob": float(last_probs[regime - 1]),
            "sigma_forecast": float(sigma_states[regime - 1]) if sigma_states is not None else np.nan,
        })

    if not records:
        return WalkForwardResult(
            forecasts=pd.DataFrame(),
            config=cfg.to_dict(),
            elapsed_ms=(time.monotonic() - t_start) * 1000,
        )

    forecasts = pd.DataFrame(records)
    if "date" in forecasts.columns:
        forecasts = forecasts.set_index("date")

    # Overall backtest
    bt = backtest_var(
        forecasts["realized_return"].values,
        forecasts["var_forecast"].values,
        confidence=cfg.confidence,
    )

    # Per-regime backtests
    regime_bts: dict[int, dict] = {}
    for regime_id in sorted(forecasts["regime"].unique()):
        mask = forecasts["regime"] == regime_id
        regime_df = forecasts[mask]
        if len(regime_df) < 5:
            regime_bts[int(regime_id)] = {
                "n_obs": len(regime_df),
                "n_violations": int(regime_df["violation"].sum()),
                "violation_rate": float(regime_df["violation"].mean()),
                "insufficient_data": True,
            }
            continue
        regime_bt = backtest_var(
            regime_df["realized_return"].values,
            regime_df["var_forecast"].values,
            confidence=cfg.confidence,
        )
        regime_bt["mean_return"] = round(float(regime_df["realized_return"].mean()), 4)
        regime_bt["volatility"] = round(float(regime_df["realized_return"].std()), 4)
        vol = regime_df["realized_return"].std()
        mean_r = regime_df["realized_return"].mean()
        regime_bt["sharpe"] = round(float(mean_r / vol * np.sqrt(252)), 4) if vol > 1e-12 else 0.0
        regime_bts[int(regime_id)] = regime_bt

    # Parameter stability across calibration snapshots
    param_stability = _compute_parameter_stability(calibrations)

    elapsed = (time.monotonic() - t_start) * 1000

    return WalkForwardResult(
        forecasts=forecasts,
        calibrations=calibrations,
        backtest=bt,
        regime_backtests=regime_bts,
        parameter_stability=param_stability,
        config=cfg.to_dict(),
        elapsed_ms=round(elapsed, 1),
    )


def _compute_parameter_stability(calibrations: list[dict]) -> dict:
    """Compute stability metrics for model parameters across recalibration windows."""
    if len(calibrations) < 2:
        return {"n_refits": len(calibrations), "stable": True}

    sigma_lows = [c["sigma_low"] for c in calibrations]
    sigma_highs = [c["sigma_high"] for c in calibrations]

    def _cv(values: list[float]) -> float:
        arr = np.array(values)
        mean = arr.mean()
        return float(arr.std() / mean) if mean > 1e-12 else 0.0

    return {
        "n_refits": len(calibrations),
        "sigma_low": {
            "mean": round(float(np.mean(sigma_lows)), 6),
            "std": round(float(np.std(sigma_lows)), 6),
            "cv": round(_cv(sigma_lows), 4),
        },
        "sigma_high": {
            "mean": round(float(np.mean(sigma_highs)), 6),
            "std": round(float(np.std(sigma_highs)), 6),
            "cv": round(_cv(sigma_highs), 4),
        },
        "stable": _cv(sigma_lows) < 0.5 and _cv(sigma_highs) < 0.5,
    }
