"""Performance report generation for walk-forward backtest results."""
from __future__ import annotations

from cortex.backtest.walk_forward import WalkForwardResult

REGIME_NAMES: dict[int, str] = {
    1: "Very Low Vol",
    2: "Low Vol",
    3: "Normal",
    4: "High Vol",
    5: "Crisis",
}


def generate_report(result: WalkForwardResult) -> dict:
    """Generate a structured performance report from walk-forward results.

    Returns a dict with sections: overall, per_regime, parameter_stability, health.
    """
    if result.forecasts.empty:
        return {"error": "No forecast data available", "config": result.config}

    df = result.forecasts
    expected_rate = 1.0 - result.config.get("confidence", 95.0) / 100.0

    overall = {
        "n_observations": result.backtest.get("n_obs", len(df)),
        "n_violations": result.backtest.get("n_violations", int(df["violation"].sum())),
        "violation_rate": result.backtest.get("violation_rate", float(df["violation"].mean())),
        "expected_rate": expected_rate,
        "kupiec": result.backtest.get("kupiec", {}),
        "christoffersen": result.backtest.get("christoffersen", {}),
        "mean_return": round(float(df["realized_return"].mean()), 4),
        "volatility": round(float(df["realized_return"].std()), 4),
        "elapsed_ms": result.elapsed_ms,
    }

    per_regime: list[dict] = []
    for regime_id, bt in sorted(result.regime_backtests.items()):
        entry = {
            "regime": regime_id,
            "regime_name": REGIME_NAMES.get(regime_id, f"State {regime_id}"),
            **bt,
        }
        per_regime.append(entry)

    # Health flags
    violation_rate = overall["violation_rate"]
    kupiec_pass = result.backtest.get("kupiec", {}).get("pass", True)
    christoffersen_pass = result.backtest.get("christoffersen", {}).get("pass", True)
    params_stable = result.parameter_stability.get("stable", True)

    health_flags: list[str] = []
    if violation_rate > 2 * expected_rate:
        health_flags.append(f"violation_rate ({violation_rate:.3f}) > 2× expected ({expected_rate:.3f})")
    if not kupiec_pass:
        health_flags.append("kupiec_test_failed")
    if not christoffersen_pass:
        health_flags.append("christoffersen_test_failed (violation clustering)")
    if not params_stable:
        health_flags.append("parameter_instability_detected")

    return {
        "overall": overall,
        "per_regime": per_regime,
        "parameter_stability": result.parameter_stability,
        "calibration_snapshots": result.calibrations,
        "health": {
            "pass": len(health_flags) == 0,
            "flags": health_flags,
        },
        "config": result.config,
    }
