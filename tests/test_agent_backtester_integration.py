"""Integration tests for agent-mode backtester with Phase 6-8 features."""

import pytest
from cortex.backtest.guardian_backtester import BacktestConfig


class TestBacktestConfigPhase68:
    def test_default_flags_off(self):
        cfg = BacktestConfig()
        assert cfg.sharpe_weights_enabled is False
        assert cfg.hmm_regime_enabled is False
        assert cfg.cointegration_enabled is False

    def test_coint_lookback_default(self):
        cfg = BacktestConfig()
        assert cfg.coint_lookback == 168

    def test_flags_can_be_enabled(self):
        cfg = BacktestConfig(
            sharpe_weights_enabled=True,
            hmm_regime_enabled=True,
            cointegration_enabled=True,
        )
        assert cfg.sharpe_weights_enabled is True
        assert cfg.hmm_regime_enabled is True
        assert cfg.cointegration_enabled is True

    def test_hmm_min_bars_default(self):
        cfg = BacktestConfig()
        assert cfg.hmm_min_bars == 100
