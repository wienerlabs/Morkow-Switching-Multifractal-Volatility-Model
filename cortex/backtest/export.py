"""Data export helpers for historical regime and signal data."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from cortex.regime import extract_regime_history, compute_regime_statistics


def export_historical_data(model_data: dict, token: str) -> dict:
    """Export historical regime, VaR, and volatility data for a calibrated token.

    Args:
        model_data: The stored model dict from _model_store[token].
        token: Token identifier.

    Returns:
        Dict with regime_timeline, regime_statistics, filter_probs_series,
        var_series, sigma_series, and calibration metadata.
    """
    returns: pd.Series = model_data.get("returns", pd.Series(dtype=float))
    filter_probs: pd.DataFrame = model_data.get("filter_probs", pd.DataFrame())
    sigma_states: np.ndarray = model_data.get("sigma_states", np.array([]))
    sigma_forecast: pd.Series = model_data.get("sigma_forecast", pd.Series(dtype=float))
    sigma_filtered: pd.Series = model_data.get("sigma_filtered", pd.Series(dtype=float))
    calibration: dict = model_data.get("calibration", {})

    # Regime timeline
    regime_timeline: list[dict] = []
    if not filter_probs.empty and len(sigma_states) > 0 and len(returns) > 0:
        history_df = extract_regime_history(filter_probs, returns, sigma_states)
        for _, row in history_df.iterrows():
            regime_timeline.append({
                "start": str(row["start"]),
                "end": str(row["end"]),
                "regime": int(row["regime"]),
                "duration": int(row["duration"]),
                "cumulative_return": float(row["cumulative_return"]),
                "volatility": float(row["volatility"]),
                "max_drawdown": float(row["max_drawdown"]),
            })

    # Regime statistics
    regime_stats: list[dict] = []
    if not filter_probs.empty and len(sigma_states) > 0 and len(returns) > 0:
        stats_df = compute_regime_statistics(returns, filter_probs, sigma_states)
        for _, row in stats_df.iterrows():
            regime_stats.append({
                "regime": int(row["regime"]),
                "mean_return": float(row["mean_return"]),
                "volatility": float(row["volatility"]),
                "sharpe_ratio": float(row["sharpe_ratio"]),
                "max_drawdown": float(row["max_drawdown"]),
                "days_in_regime": int(row["days_in_regime"]),
                "frequency": float(row["frequency"]),
            })

    # Time series data (filter_probs as list of dicts for JSON)
    filter_probs_series: list[dict] = []
    if not filter_probs.empty:
        for idx, row in filter_probs.iterrows():
            entry = {"date": str(idx)}
            for col in filter_probs.columns:
                entry[col] = round(float(row[col]), 6)
            filter_probs_series.append(entry)

    # VaR and sigma series
    def _series_to_list(s: pd.Series) -> list[dict]:
        if s.empty:
            return []
        return [
            {"date": str(idx), "value": round(float(v), 6)}
            for idx, v in s.items()
            if not (isinstance(v, float) and np.isnan(v))
        ]

    return {
        "token": token,
        "n_observations": len(returns),
        "regime_timeline": regime_timeline,
        "regime_statistics": regime_stats,
        "filter_probs_series": filter_probs_series,
        "sigma_forecast_series": _series_to_list(sigma_forecast),
        "sigma_filtered_series": _series_to_list(sigma_filtered),
        "calibration": {
            "num_states": calibration.get("num_states"),
            "sigma_low": calibration.get("sigma_low"),
            "sigma_high": calibration.get("sigma_high"),
            "p_stay": calibration.get("p_stay"),
            "method": calibration.get("method"),
        },
        "calibrated_at": str(model_data.get("calibrated_at", "")),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
