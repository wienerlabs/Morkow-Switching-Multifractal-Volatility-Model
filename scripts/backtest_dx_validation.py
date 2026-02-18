#!/usr/bin/env python3
"""DX Research Validation: Backtest DX modules' impact on VaR accuracy.

Runs walk-forward VaR backtest twice — once with all DX flags OFF (baseline)
and once with all DX flags ON — then compares violation rates, Kupiec/Christoffersen
test results, and per-regime accuracy.

Uses synthetic data (seeded random walk with regime shifts) for reproducibility.
No external API calls required.

Usage:
    python3 scripts/backtest_dx_validation.py
    python3 scripts/backtest_dx_validation.py --n-obs 500 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from cortex.backtest.walk_forward import WalkForwardConfig, run_walk_forward
from cortex.backtesting import backtest_var

# DX feature flag env var names and their ON/OFF values
DX_FLAGS = {
    "PROSPECT_THEORY_NEWS_ENABLED": ("true", "false"),
    "DEBATE_INFO_ASYMMETRY_ENABLED": ("true", "false"),
    "STIGMERGY_ENABLED": ("true", "false"),
    "ISING_CASCADE_ENABLED": ("true", "false"),
    "PERSONA_DIVERSITY_ENABLED": ("true", "false"),
    "VAULT_DELTA_ENABLED": ("true", "false"),
    "HUMAN_OVERRIDE_ENABLED": ("true", "false"),
}

RESULTS_DIR = PROJECT_ROOT / "data" / "backtest_results"


def generate_synthetic_returns(
    n_obs: int = 1000,
    n_assets: int = 5,
    seed: int = 42,
) -> dict[str, pd.Series]:
    """Generate synthetic log-return series with regime shifts and fat tails.

    Returns dict mapping asset name -> pd.Series of daily log-returns (%).
    """
    rng = np.random.default_rng(seed)
    asset_names = ["SOL", "BTC", "ETH", "AVAX", "MATIC"][:n_assets]

    dates = pd.bdate_range(end=pd.Timestamp.now(), periods=n_obs, freq="B")

    # Regime parameters: (mean_daily_pct, vol_daily_pct, weight)
    regimes = [
        (0.05, 1.5, 0.50),   # calm
        (0.02, 2.5, 0.25),   # moderate
        (-0.10, 5.0, 0.15),  # stressed
        (-0.30, 10.0, 0.07), # crisis
        (-0.50, 15.0, 0.03), # extreme crisis
    ]

    # Generate regime sequence with realistic persistence
    regime_labels = _generate_regime_sequence(n_obs, regimes, rng)

    series = {}
    for i, asset in enumerate(asset_names):
        asset_rng = np.random.default_rng(seed + i)
        returns = np.zeros(n_obs)

        for t in range(n_obs):
            regime_idx = regime_labels[t]
            mu, sigma, _ = regimes[regime_idx]

            # Add asset-specific scaling
            asset_scale = 1.0 + 0.2 * i
            sigma_t = sigma * asset_scale

            # Student-t innovations for fat tails (df=5)
            innovation = asset_rng.standard_t(df=5)
            returns[t] = mu + sigma_t * innovation

        series[asset] = pd.Series(returns, index=dates, name=asset)

    return series, regime_labels


def _generate_regime_sequence(
    n_obs: int,
    regimes: list[tuple],
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Markov chain regime sequence with realistic persistence."""
    n_regimes = len(regimes)

    # Transition matrix: high self-persistence, gradual transitions
    P = np.zeros((n_regimes, n_regimes))
    for i in range(n_regimes):
        P[i, i] = 0.95  # 95% stay in same regime
        remaining = 0.05
        for j in range(n_regimes):
            if j != i:
                # Closer regimes more likely
                distance = abs(i - j)
                P[i, j] = remaining / (2 ** distance)
        # Normalize row
        P[i] /= P[i].sum()

    regime_labels = np.zeros(n_obs, dtype=int)
    regime_labels[0] = 0  # start calm

    for t in range(1, n_obs):
        regime_labels[t] = rng.choice(n_regimes, p=P[regime_labels[t - 1]])

    return regime_labels


def set_dx_flags(enabled: bool) -> dict[str, str]:
    """Set all DX feature flags via environment variables. Returns previous values."""
    previous = {}
    for flag, (on_val, off_val) in DX_FLAGS.items():
        previous[flag] = os.environ.get(flag, "")
        os.environ[flag] = on_val if enabled else off_val
    return previous


def restore_flags(previous: dict[str, str]) -> None:
    """Restore environment variables to previous values."""
    for flag, val in previous.items():
        if val:
            os.environ[flag] = val
        else:
            os.environ.pop(flag, None)


def reload_config() -> None:
    """Force-reload cortex.config to pick up env var changes."""
    import importlib
    import cortex.config
    importlib.reload(cortex.config)


def run_backtest_pass(
    returns: pd.Series,
    label: str,
    config: WalkForwardConfig,
) -> dict:
    """Run a single walk-forward backtest pass and return structured results."""
    result = run_walk_forward(returns, config)

    if result.forecasts.empty:
        return {
            "label": label,
            "n_obs": 0,
            "error": "No forecasts produced",
        }

    bt = result.backtest
    regime_bts = result.regime_backtests

    # Extract per-regime violation rates
    regime_detail = {}
    for regime_id, rbt in regime_bts.items():
        regime_detail[str(regime_id)] = {
            "n_obs": rbt.get("n_obs", 0),
            "n_violations": rbt.get("n_violations", 0),
            "violation_rate": round(rbt.get("violation_rate", 0.0), 4),
            "kupiec_p": round(rbt.get("kupiec", {}).get("p_value", 0.0), 4)
            if "kupiec" in rbt
            else None,
        }

    return {
        "label": label,
        "n_obs": bt.get("n_obs", 0),
        "n_violations": bt.get("n_violations", 0),
        "violation_rate": round(bt.get("violation_rate", 0.0), 4),
        "expected_rate": round(1.0 - config.confidence / 100.0, 4),
        "kupiec": {
            "statistic": round(bt.get("kupiec", {}).get("statistic", 0.0), 4),
            "p_value": round(bt.get("kupiec", {}).get("p_value", 0.0), 4),
            "pass": bt.get("kupiec", {}).get("pass", False),
        },
        "christoffersen": {
            "statistic": round(bt.get("christoffersen", {}).get("statistic", 0.0), 4),
            "p_value": round(bt.get("christoffersen", {}).get("p_value", 0.0), 4),
            "pass": bt.get("christoffersen", {}).get("pass", False),
        },
        "per_regime": regime_detail,
        "parameter_stability": result.parameter_stability,
        "elapsed_ms": result.elapsed_ms,
    }


def compute_guardian_scores(
    returns: pd.Series,
    regime_labels: np.ndarray,
    dx_enabled: bool,
) -> dict:
    """Run Guardian scoring pipeline on synthetic data to compare composite scores.

    Simulates the Guardian pipeline by generating mock component inputs
    from the synthetic return series. This captures how DX modules
    (prospect theory, stigmergy, etc.) affect the final risk assessment.
    """
    from cortex.backtesting import simple_var_forecast

    n = len(returns)
    window = 120
    scores = []

    for t in range(window, min(n, window + 200)):
        train = returns.iloc[:t]
        var_forecast = simple_var_forecast(train.values, confidence=95.0)
        realized = float(returns.iloc[t])

        # Build minimal news signal to exercise prospect theory path
        ewma = float(train.iloc[-20:].mean()) / max(float(train.std()), 0.01)
        ewma = max(-1.0, min(1.0, ewma))

        news_signal = {
            "sentiment_ewma": ewma,
            "strength": abs(ewma),
            "confidence": 0.7,
            "entropy": 0.8,
            "direction": "LONG" if ewma > 0 else "SHORT",
            "n_items": 5,
        }

        # Score news component directly to measure prospect theory impact
        from cortex.guardian import _score_news
        news_score = _score_news(news_signal, "long")

        regime_idx = int(regime_labels[t]) if t < len(regime_labels) else 0

        scores.append({
            "t": t,
            "regime": regime_idx,
            "var_forecast": round(var_forecast, 4),
            "realized": round(realized, 4),
            "violation": realized < var_forecast,
            "news_risk_score": news_score["score"],
            "prospect_applied": news_score["details"].get("prospect_theory_applied", False),
        })

    if not scores:
        return {"n_samples": 0}

    news_scores = [s["news_risk_score"] for s in scores]
    prospect_count = sum(1 for s in scores if s["prospect_applied"])

    return {
        "n_samples": len(scores),
        "mean_news_risk_score": round(float(np.mean(news_scores)), 2),
        "std_news_risk_score": round(float(np.std(news_scores)), 2),
        "prospect_theory_applications": prospect_count,
        "prospect_theory_pct": round(100.0 * prospect_count / len(scores), 1),
    }


def compare_results(baseline: dict, dx_enhanced: dict) -> dict:
    """Compute comparison metrics between baseline and DX-enhanced runs."""
    if baseline.get("error") or dx_enhanced.get("error"):
        return {"error": "One or both runs failed"}

    vr_baseline = baseline["violation_rate"]
    vr_dx = dx_enhanced["violation_rate"]
    expected = baseline["expected_rate"]

    # How close each is to the expected violation rate
    vr_gap_baseline = abs(vr_baseline - expected)
    vr_gap_dx = abs(vr_dx - expected)

    return {
        "violation_rate_delta": round(vr_dx - vr_baseline, 4),
        "violation_rate_gap_baseline": round(vr_gap_baseline, 4),
        "violation_rate_gap_dx": round(vr_gap_dx, 4),
        "dx_closer_to_expected": vr_gap_dx < vr_gap_baseline,
        "kupiec_p_baseline": baseline["kupiec"]["p_value"],
        "kupiec_p_dx": dx_enhanced["kupiec"]["p_value"],
        "kupiec_improvement": dx_enhanced["kupiec"]["p_value"] > baseline["kupiec"]["p_value"],
        "christoffersen_p_baseline": baseline["christoffersen"]["p_value"],
        "christoffersen_p_dx": dx_enhanced["christoffersen"]["p_value"],
        "both_kupiec_pass": baseline["kupiec"]["pass"] and dx_enhanced["kupiec"]["pass"],
        "speedup_ms": round(baseline["elapsed_ms"] - dx_enhanced["elapsed_ms"], 1),
    }


def print_summary(results: dict) -> None:
    """Print human-readable summary table."""
    baseline = results["baseline"]
    dx = results["dx_enhanced"]
    comp = results["comparison"]
    guardian = results.get("guardian_comparison", {})

    print("\n" + "=" * 70)
    print("  DX Research Validation — Walk-Forward VaR Backtest")
    print("=" * 70)

    print(f"\n  Asset: {results['metadata']['asset']}")
    print(f"  Observations: {results['metadata']['n_obs']}")
    print(f"  Confidence: {results['metadata']['confidence']}%")
    print(f"  Expected VaR violation rate: {baseline['expected_rate']:.2%}")

    print(f"\n{'':4}{'Metric':<32} {'Baseline':>12} {'DX-Enhanced':>12} {'Delta':>10}")
    print("  " + "-" * 66)

    def row(label, b, d, fmt=".4f", pct=False):
        delta = d - b
        suffix = "%" if pct else ""
        f = f">{12}{fmt}" if not pct else f">{11}{fmt}"
        print(f"    {label:<32} {b:{fmt}}{suffix:>1} {d:{fmt}}{suffix:>1} {delta:>+10{fmt}}")

    row("Violation rate", baseline["violation_rate"], dx["violation_rate"])
    row("Kupiec p-value", baseline["kupiec"]["p_value"], dx["kupiec"]["p_value"])
    row("Christoffersen p-value", baseline["christoffersen"]["p_value"], dx["christoffersen"]["p_value"])
    row("Elapsed (ms)", baseline["elapsed_ms"], dx["elapsed_ms"], ".1f")

    print(f"\n    {'Kupiec pass':<32} {'PASS' if baseline['kupiec']['pass'] else 'FAIL':>12} {'PASS' if dx['kupiec']['pass'] else 'FAIL':>12}")
    print(f"    {'Christoffersen pass':<32} {'PASS' if baseline['christoffersen']['pass'] else 'FAIL':>12} {'PASS' if dx['christoffersen']['pass'] else 'FAIL':>12}")

    # Per-regime breakdown
    print(f"\n  Per-Regime Violation Rates:")
    print(f"    {'Regime':<10} {'Baseline':>12} {'DX-Enhanced':>12} {'N (base)':>10} {'N (dx)':>10}")
    print("    " + "-" * 54)

    b_regimes = baseline.get("per_regime", {})
    d_regimes = dx.get("per_regime", {})
    all_regimes = sorted(set(list(b_regimes.keys()) + list(d_regimes.keys())))

    for r in all_regimes:
        b_vr = b_regimes.get(r, {}).get("violation_rate", 0.0)
        d_vr = d_regimes.get(r, {}).get("violation_rate", 0.0)
        b_n = b_regimes.get(r, {}).get("n_obs", 0)
        d_n = d_regimes.get(r, {}).get("n_obs", 0)
        print(f"    {r:<10} {b_vr:>12.4f} {d_vr:>12.4f} {b_n:>10} {d_n:>10}")

    # Guardian comparison
    if guardian.get("baseline") and guardian.get("dx_enhanced"):
        gb = guardian["baseline"]
        gd = guardian["dx_enhanced"]
        print(f"\n  Guardian News Scoring (prospect theory impact):")
        print(f"    {'Mean news risk score':<32} {gb['mean_news_risk_score']:>12.2f} {gd['mean_news_risk_score']:>12.2f}")
        print(f"    {'Prospect theory applied':<32} {gb['prospect_theory_pct']:>11.1f}% {gd['prospect_theory_pct']:>11.1f}%")

    # Verdict
    print(f"\n  {'Verdict':}")
    if comp.get("dx_closer_to_expected"):
        print("    DX-Enhanced violation rate is CLOSER to expected rate")
    else:
        print("    Baseline violation rate is closer to expected rate")

    if comp.get("kupiec_improvement"):
        print("    Kupiec p-value IMPROVED with DX modules")
    else:
        print("    Kupiec p-value did not improve with DX modules")

    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="DX Research Validation Backtest")
    parser.add_argument("--n-obs", type=int, default=1000, help="Number of observations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--confidence", type=float, default=95.0, help="VaR confidence level")
    parser.add_argument("--asset", type=str, default="SOL", help="Asset to backtest")
    parser.add_argument("--min-train", type=int, default=120, help="Minimum training window")
    parser.add_argument("--refit-interval", type=int, default=20, help="Recalibration interval")
    args = parser.parse_args()

    print(f"Generating synthetic data: {args.n_obs} observations, seed={args.seed}")
    series, regime_labels = generate_synthetic_returns(
        n_obs=args.n_obs, seed=args.seed,
    )

    if args.asset not in series:
        print(f"Asset {args.asset} not found. Available: {list(series.keys())}")
        sys.exit(1)

    returns = series[args.asset]

    wf_config = WalkForwardConfig(
        min_train_window=args.min_train,
        step_size=1,
        refit_interval=args.refit_interval,
        expanding=True,
        confidence=args.confidence,
        num_states=5,
        method="empirical",
    )

    # --- Pass 1: Baseline (all DX flags OFF) ---
    print("\n[1/4] Running BASELINE backtest (DX flags OFF)...")
    prev_flags = set_dx_flags(enabled=False)
    reload_config()
    baseline_result = run_backtest_pass(returns, "baseline", wf_config)

    # Guardian comparison: baseline
    print("[2/4] Running Guardian scoring (baseline)...")
    guardian_baseline = compute_guardian_scores(returns, regime_labels, dx_enabled=False)

    # --- Pass 2: DX-Enhanced (all DX flags ON) ---
    print("[3/4] Running DX-ENHANCED backtest (DX flags ON)...")
    restore_flags(prev_flags)
    set_dx_flags(enabled=True)
    reload_config()
    dx_result = run_backtest_pass(returns, "dx_enhanced", wf_config)

    # Guardian comparison: DX-enhanced
    print("[4/4] Running Guardian scoring (DX-enhanced)...")
    guardian_dx = compute_guardian_scores(returns, regime_labels, dx_enabled=True)

    # Restore original flags
    restore_flags(prev_flags)

    # Compare
    comparison = compare_results(baseline_result, dx_result)

    # Assemble output
    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_obs": args.n_obs,
            "seed": args.seed,
            "confidence": args.confidence,
            "asset": args.asset,
            "min_train_window": args.min_train,
            "refit_interval": args.refit_interval,
            "dx_flags": list(DX_FLAGS.keys()),
        },
        "baseline": baseline_result,
        "dx_enhanced": dx_result,
        "comparison": comparison,
        "guardian_comparison": {
            "baseline": guardian_baseline,
            "dx_enhanced": guardian_dx,
        },
    }

    # Write JSON results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"dx_validation_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults written to: {out_path}")
    print_summary(output)


if __name__ == "__main__":
    main()
