"""Tests for walk-forward backtesting integration."""
import numpy as np
import pandas as pd
import pytest

from cortex.backtest.walk_forward import WalkForwardConfig, WalkForwardResult, run_walk_forward
from cortex.backtest.report import generate_report
from cortex.backtest.export import export_historical_data


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def long_returns() -> pd.Series:
    """500-point synthetic return series with regime-like volatility clustering."""
    rng = np.random.RandomState(42)
    n = 500
    rets = np.empty(n)
    # Simulate 3 volatility regimes
    regime_vols = [0.5, 1.5, 3.0]
    regime = 0
    for i in range(n):
        if rng.random() < 0.02:  # 2% chance of regime switch
            regime = rng.randint(0, 3)
        rets[i] = rng.randn() * regime_vols[regime]

    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(rets, index=dates, name="synthetic_regimes")


@pytest.fixture(scope="module")
def short_returns() -> pd.Series:
    """50-point series (too short for default config)."""
    rng = np.random.RandomState(99)
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    return pd.Series(rng.randn(50) * 1.5, index=dates, name="short")


# ── Walk-Forward Runner Tests ─────────────────────────────────────────


class TestWalkForwardRunner:
    def test_expanding_window(self, long_returns):
        cfg = WalkForwardConfig(
            min_train_window=60,
            refit_interval=50,
            method="empirical",
            num_states=3,
        )
        result = run_walk_forward(long_returns, cfg)

        assert isinstance(result, WalkForwardResult)
        assert not result.forecasts.empty
        assert "var_forecast" in result.forecasts.columns
        assert "realized_return" in result.forecasts.columns
        assert "violation" in result.forecasts.columns
        assert "regime" in result.forecasts.columns
        assert len(result.calibrations) >= 2
        assert result.elapsed_ms > 0

    def test_rolling_window(self, long_returns):
        cfg = WalkForwardConfig(
            min_train_window=60,
            expanding=False,
            max_train_window=120,
            refit_interval=50,
            method="empirical",
            num_states=3,
        )
        result = run_walk_forward(long_returns, cfg)
        assert not result.forecasts.empty
        assert len(result.calibrations) >= 2

    def test_backtest_stats_populated(self, long_returns):
        cfg = WalkForwardConfig(
            min_train_window=60,
            refit_interval=50,
            method="empirical",
            num_states=3,
        )
        result = run_walk_forward(long_returns, cfg)

        assert "n_obs" in result.backtest
        assert "kupiec" in result.backtest
        assert "christoffersen" in result.backtest
        assert result.backtest["n_obs"] > 0

    def test_regime_backtests(self, long_returns):
        cfg = WalkForwardConfig(
            min_train_window=60,
            refit_interval=50,
            method="empirical",
            num_states=3,
        )
        result = run_walk_forward(long_returns, cfg)

        assert len(result.regime_backtests) > 0
        for regime_id, bt in result.regime_backtests.items():
            assert isinstance(regime_id, int)
            assert "n_obs" in bt or "n_violations" in bt

    def test_parameter_stability(self, long_returns):
        cfg = WalkForwardConfig(
            min_train_window=60,
            refit_interval=30,
            method="empirical",
            num_states=3,
        )
        result = run_walk_forward(long_returns, cfg)

        ps = result.parameter_stability
        assert "n_refits" in ps
        assert ps["n_refits"] >= 2
        assert "stable" in ps
        assert "sigma_low" in ps
        assert "sigma_high" in ps

    def test_too_short_raises(self, short_returns):
        cfg = WalkForwardConfig(min_train_window=120)
        with pytest.raises(ValueError, match="at least"):
            run_walk_forward(short_returns, cfg)

    def test_step_size(self, long_returns):
        cfg = WalkForwardConfig(
            min_train_window=60,
            step_size=5,
            refit_interval=20,
            method="empirical",
            num_states=3,
        )
        result = run_walk_forward(long_returns, cfg)
        # With step_size=5, we get roughly (500-60)/5 = 88 forecast points
        assert len(result.forecasts) > 50
        assert len(result.forecasts) < 200

    def test_var_forecasts_are_negative(self, long_returns):
        """VaR forecasts at 95% should be negative (loss threshold)."""
        cfg = WalkForwardConfig(
            min_train_window=60,
            refit_interval=50,
            method="empirical",
            num_states=3,
        )
        result = run_walk_forward(long_returns, cfg)
        # VaR forecasts should generally be negative for 95% confidence
        assert (result.forecasts["var_forecast"] < 0).mean() > 0.8

    def test_violation_rate_reasonable(self, long_returns):
        """Violation rate should be in the ballpark of expected rate."""
        cfg = WalkForwardConfig(
            min_train_window=60,
            refit_interval=50,
            method="empirical",
            num_states=3,
            confidence=95.0,
        )
        result = run_walk_forward(long_returns, cfg)
        vr = result.backtest.get("violation_rate", 0)
        # With synthetic data, tolerance is wide: 0-20%
        assert 0.0 <= vr <= 0.20


# ── Report Generator Tests ────────────────────────────────────────────


class TestReportGenerator:
    def test_generate_report(self, long_returns):
        cfg = WalkForwardConfig(
            min_train_window=60,
            refit_interval=50,
            method="empirical",
            num_states=3,
        )
        result = run_walk_forward(long_returns, cfg)
        report = generate_report(result)

        assert "overall" in report
        assert "per_regime" in report
        assert "parameter_stability" in report
        assert "health" in report
        assert "config" in report

    def test_health_flags(self, long_returns):
        cfg = WalkForwardConfig(
            min_train_window=60,
            refit_interval=50,
            method="empirical",
            num_states=3,
        )
        result = run_walk_forward(long_returns, cfg)
        report = generate_report(result)

        health = report["health"]
        assert isinstance(health["pass"], bool)
        assert isinstance(health["flags"], list)

    def test_per_regime_has_names(self, long_returns):
        cfg = WalkForwardConfig(
            min_train_window=60,
            refit_interval=50,
            method="empirical",
            num_states=3,
        )
        result = run_walk_forward(long_returns, cfg)
        report = generate_report(result)

        for regime_entry in report["per_regime"]:
            assert "regime_name" in regime_entry
            assert "regime" in regime_entry

    def test_empty_result(self):
        result = WalkForwardResult(
            forecasts=pd.DataFrame(),
            config={"confidence": 95.0},
        )
        report = generate_report(result)
        assert "error" in report


# ── Data Export Tests ─────────────────────────────────────────────────


class TestDataExport:
    def test_export_with_calibrated_model(self, calibrated_model):
        data = export_historical_data(calibrated_model, "TEST")

        assert data["token"] == "TEST"
        assert data["n_observations"] > 0
        assert isinstance(data["regime_timeline"], list)
        assert isinstance(data["regime_statistics"], list)
        assert len(data["regime_timeline"]) > 0
        assert len(data["regime_statistics"]) > 0
        assert "calibration" in data

    def test_export_has_calibration_metadata(self, calibrated_model):
        data = export_historical_data(calibrated_model, "TEST")
        cal = data["calibration"]
        assert "num_states" in cal
        assert "sigma_low" in cal
        assert "sigma_high" in cal

    def test_export_regime_statistics_structure(self, calibrated_model):
        data = export_historical_data(calibrated_model, "TEST")
        for stat in data["regime_statistics"]:
            assert "regime" in stat
            assert "mean_return" in stat
            assert "volatility" in stat
            assert "sharpe_ratio" in stat
            assert "frequency" in stat

    def test_export_empty_model(self):
        data = export_historical_data({}, "EMPTY")
        assert data["token"] == "EMPTY"
        assert data["n_observations"] == 0
        assert data["regime_timeline"] == []
        assert data["regime_statistics"] == []


# ── Config Tests ──────────────────────────────────────────────────────


class TestWalkForwardConfig:
    def test_defaults(self):
        cfg = WalkForwardConfig()
        assert cfg.min_train_window == 120
        assert cfg.expanding is True
        assert cfg.confidence == 95.0

    def test_to_dict(self):
        cfg = WalkForwardConfig(min_train_window=60, method="mle")
        d = cfg.to_dict()
        assert d["min_train_window"] == 60
        assert d["method"] == "mle"
        assert "confidence" in d
