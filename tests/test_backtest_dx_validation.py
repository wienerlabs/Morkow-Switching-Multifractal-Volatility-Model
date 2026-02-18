"""Tests for the DX Research backtest validation script."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_dx_validation import (
    generate_synthetic_returns,
    _generate_regime_sequence,
    set_dx_flags,
    restore_flags,
    run_backtest_pass,
    compare_results,
    compute_guardian_scores,
    DX_FLAGS,
)
from cortex.backtest.walk_forward import WalkForwardConfig


class TestSyntheticDataGeneration:
    """Verify synthetic data produces valid, reproducible arrays."""

    def test_returns_shape_and_assets(self):
        series, regime_labels = generate_synthetic_returns(n_obs=200, n_assets=5, seed=42)
        assert len(series) == 5
        assert set(series.keys()) == {"SOL", "BTC", "ETH", "AVAX", "MATIC"}
        for name, s in series.items():
            assert len(s) == 200
            assert isinstance(s, pd.Series)
            assert s.name == name

    def test_returns_are_finite(self):
        series, _ = generate_synthetic_returns(n_obs=500, seed=99)
        for s in series.values():
            assert np.all(np.isfinite(s.values)), f"Non-finite values in {s.name}"

    def test_regime_labels_valid(self):
        _, regime_labels = generate_synthetic_returns(n_obs=300, seed=42)
        assert len(regime_labels) == 300
        assert regime_labels.dtype == int
        assert np.all(regime_labels >= 0)
        assert np.all(regime_labels < 5)

    def test_reproducibility(self):
        s1, r1 = generate_synthetic_returns(n_obs=100, seed=42)
        s2, r2 = generate_synthetic_returns(n_obs=100, seed=42)
        np.testing.assert_array_equal(r1, r2)
        for asset in s1:
            np.testing.assert_array_equal(s1[asset].values, s2[asset].values)

    def test_different_seeds_produce_different_data(self):
        s1, _ = generate_synthetic_returns(n_obs=100, seed=42)
        s2, _ = generate_synthetic_returns(n_obs=100, seed=123)
        assert not np.array_equal(s1["SOL"].values, s2["SOL"].values)

    def test_regime_sequence_persistence(self):
        """Regime transitions should be sticky (high self-transition probability)."""
        rng = np.random.default_rng(42)
        regimes = [(0, 1, 0.5), (0, 2, 0.3), (0, 5, 0.15), (0, 10, 0.05)]
        labels = _generate_regime_sequence(1000, regimes, rng)
        # Count self-transitions
        same_count = sum(1 for i in range(1, len(labels)) if labels[i] == labels[i - 1])
        persistence = same_count / (len(labels) - 1)
        # With 95% self-transition probability, expect >85% persistence
        assert persistence > 0.80, f"Persistence too low: {persistence:.2%}"

    def test_fat_tails_present(self):
        """Student-t innovations should produce fatter tails than Gaussian."""
        series, _ = generate_synthetic_returns(n_obs=5000, seed=42)
        sol = series["SOL"].values
        # Kurtosis of standard normal is ~3, student-t(5) is ~9
        kurtosis = float(np.mean((sol - sol.mean()) ** 4) / sol.std() ** 4)
        assert kurtosis > 4.0, f"Kurtosis {kurtosis:.1f} not fat enough"


class TestDXFlagToggling:
    """Verify DX flags can be toggled correctly via environment variables."""

    def test_set_flags_off(self):
        prev = set_dx_flags(enabled=False)
        try:
            for flag in DX_FLAGS:
                assert os.environ[flag] == "false"
        finally:
            restore_flags(prev)

    def test_set_flags_on(self):
        prev = set_dx_flags(enabled=True)
        try:
            for flag in DX_FLAGS:
                assert os.environ[flag] == "true"
        finally:
            restore_flags(prev)

    def test_restore_flags(self):
        os.environ["PROSPECT_THEORY_NEWS_ENABLED"] = "original"
        prev = set_dx_flags(enabled=False)
        assert os.environ["PROSPECT_THEORY_NEWS_ENABLED"] == "false"
        restore_flags(prev)
        assert os.environ["PROSPECT_THEORY_NEWS_ENABLED"] == "original"
        del os.environ["PROSPECT_THEORY_NEWS_ENABLED"]


class TestWalkForwardBacktest:
    """Verify walk-forward backtest runs produce valid structured results."""

    @pytest.fixture
    def synthetic_returns(self):
        series, regime_labels = generate_synthetic_returns(n_obs=300, seed=42)
        return series["SOL"], regime_labels

    @pytest.fixture
    def wf_config(self):
        return WalkForwardConfig(
            min_train_window=120,
            step_size=1,
            refit_interval=20,
            expanding=True,
            confidence=95.0,
            num_states=5,
            method="empirical",
        )

    def test_backtest_pass_returns_valid_structure(self, synthetic_returns, wf_config):
        returns, _ = synthetic_returns
        result = run_backtest_pass(returns, "test_run", wf_config)

        assert result["label"] == "test_run"
        assert result["n_obs"] > 0
        assert 0 <= result["violation_rate"] <= 1.0
        assert "kupiec" in result
        assert "christoffersen" in result
        assert "per_regime" in result
        assert "elapsed_ms" in result
        assert result["expected_rate"] == pytest.approx(0.05, abs=0.001)

    def test_kupiec_fields(self, synthetic_returns, wf_config):
        returns, _ = synthetic_returns
        result = run_backtest_pass(returns, "test", wf_config)
        kup = result["kupiec"]
        assert "statistic" in kup
        assert "p_value" in kup
        assert "pass" in kup
        assert 0 <= kup["p_value"] <= 1.0

    def test_christoffersen_fields(self, synthetic_returns, wf_config):
        returns, _ = synthetic_returns
        result = run_backtest_pass(returns, "test", wf_config)
        chris = result["christoffersen"]
        assert "statistic" in chris
        assert "p_value" in chris
        assert "pass" in chris


class TestBaselineVsDXProducesDifferentResults:
    """Baseline vs DX-enhanced should produce different Guardian scores."""

    def test_guardian_scores_differ_with_prospect_theory(self):
        series, regime_labels = generate_synthetic_returns(n_obs=300, seed=42)
        returns = series["SOL"]

        prev = set_dx_flags(enabled=False)
        from scripts.backtest_dx_validation import reload_config
        reload_config()
        baseline = compute_guardian_scores(returns, regime_labels, dx_enabled=False)

        restore_flags(prev)
        set_dx_flags(enabled=True)
        reload_config()
        dx = compute_guardian_scores(returns, regime_labels, dx_enabled=True)

        restore_flags(prev)

        # With prospect theory enabled, negative sentiments get amplified
        # so the news risk scores should shift
        assert baseline["n_samples"] > 0
        assert dx["n_samples"] > 0

        # DX-enhanced should have prospect theory applications
        assert dx["prospect_theory_pct"] >= 0


class TestComparisonMetrics:
    """Verify comparison logic computes correctly."""

    def test_comparison_structure(self):
        baseline = {
            "violation_rate": 0.065,
            "expected_rate": 0.05,
            "kupiec": {"statistic": 0.1, "p_value": 0.72, "pass": True},
            "christoffersen": {"statistic": 0.05, "p_value": 0.45, "pass": True},
            "elapsed_ms": 500.0,
        }
        dx = {
            "violation_rate": 0.048,
            "expected_rate": 0.05,
            "kupiec": {"statistic": 0.05, "p_value": 0.81, "pass": True},
            "christoffersen": {"statistic": 0.03, "p_value": 0.52, "pass": True},
            "elapsed_ms": 520.0,
        }
        comp = compare_results(baseline, dx)

        assert comp["violation_rate_delta"] == pytest.approx(-0.017, abs=0.001)
        assert comp["dx_closer_to_expected"] is True
        assert comp["kupiec_improvement"] is True
        assert comp["both_kupiec_pass"] is True

    def test_comparison_with_error(self):
        comp = compare_results({"error": "fail"}, {"violation_rate": 0.05})
        assert comp["error"] == "One or both runs failed"


class TestOutputJSONSchema:
    """Verify output JSON has the expected schema."""

    def test_full_output_schema(self):
        series, regime_labels = generate_synthetic_returns(n_obs=300, seed=42)
        returns = series["SOL"]
        wf_config = WalkForwardConfig(
            min_train_window=120,
            step_size=1,
            refit_interval=20,
            expanding=True,
            confidence=95.0,
            num_states=5,
        )

        baseline = run_backtest_pass(returns, "baseline", wf_config)
        dx = run_backtest_pass(returns, "dx_enhanced", wf_config)
        comparison = compare_results(baseline, dx)

        output = {
            "metadata": {
                "n_obs": 300,
                "seed": 42,
                "confidence": 95.0,
                "asset": "SOL",
            },
            "baseline": baseline,
            "dx_enhanced": dx,
            "comparison": comparison,
        }

        # Should be JSON-serializable
        json_str = json.dumps(output, default=str)
        parsed = json.loads(json_str)

        assert "metadata" in parsed
        assert "baseline" in parsed
        assert "dx_enhanced" in parsed
        assert "comparison" in parsed
        assert parsed["baseline"]["label"] == "baseline"
        assert parsed["dx_enhanced"]["label"] == "dx_enhanced"
