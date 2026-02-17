"""Tests for all 8 win-rate enhancement features (Tasks 1-8).

Each feature is opt-in via environment variable (default disabled).
Tests verify both disabled (passthrough) and enabled behavior.
"""
import json
import math
import time

import numpy as np
import pandas as pd
import pytest


# ── Task 1: Platt Calibration Bridge ────────────────────────────────


class TestCalibrationBridge:
    """Tests for cortex/calibration_bridge.py"""

    def setup_method(self):
        import cortex.calibration_bridge as cb
        cb.reset()

    def test_disabled_returns_raw(self, monkeypatch):
        import cortex.calibration_bridge as cb
        cb.reset()
        monkeypatch.setattr(cb, "CALIBRATION_ENABLED", False)
        assert cb.calibrate_probability(0.8) == 0.8
        assert cb.calibrate_probability(0.2) == 0.2

    def test_enabled_no_params_returns_raw(self, monkeypatch):
        import cortex.calibration_bridge as cb
        cb.reset()
        monkeypatch.setattr(cb, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(cb, "CALIBRATION_PARAMS_PATH", "")
        assert cb.calibrate_probability(0.8) == 0.8

    def test_enabled_with_params(self, monkeypatch, tmp_path):
        params = {"A": 2.0, "B": -1.0, "model_name": "test"}
        params_file = tmp_path / "cal.json"
        params_file.write_text(json.dumps(params))

        import cortex.calibration_bridge as cb
        cb.reset()
        monkeypatch.setattr(cb, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(cb, "CALIBRATION_PARAMS_PATH", str(params_file))

        result = cb.calibrate_probability(0.8)
        # sigmoid(2.0 * 0.8 + (-1.0)) = sigmoid(0.6)
        expected = 1.0 / (1.0 + math.exp(-0.6))
        assert abs(result - expected) < 1e-6

    def test_calibrate_bounds(self, monkeypatch, tmp_path):
        params = {"A": 50.0, "B": 0.0, "model_name": "extreme"}
        params_file = tmp_path / "cal.json"
        params_file.write_text(json.dumps(params))

        import cortex.calibration_bridge as cb
        cb.reset()
        monkeypatch.setattr(cb, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(cb, "CALIBRATION_PARAMS_PATH", str(params_file))

        result = cb.calibrate_probability(0.99)
        assert 0.0 <= result <= 1.0

    def test_get_calibration_info(self, monkeypatch, tmp_path):
        params = {"A": 1.5, "B": -0.5, "model_name": "info_test"}
        params_file = tmp_path / "cal.json"
        params_file.write_text(json.dumps(params))

        import cortex.calibration_bridge as cb
        cb.reset()
        monkeypatch.setattr(cb, "CALIBRATION_ENABLED", True)
        monkeypatch.setattr(cb, "CALIBRATION_PARAMS_PATH", str(params_file))

        info = cb.get_calibration_info()
        assert info["enabled"] is True
        assert info["loaded"] is True
        assert info["A"] == 1.5
        assert info["model_name"] == "info_test"


# ── Task 2: Regime-Conditional Kelly Criterion ──────────────────────


class TestRegimeKelly:
    """Tests for regime-aware Kelly in guardian.py"""

    def setup_method(self):
        import cortex.guardian as g
        g._trade_history.clear()

    def test_kelly_basic_unchanged_when_disabled(self, monkeypatch):
        import cortex.guardian as g
        g._trade_history.clear()
        monkeypatch.setattr(g, "KELLY_REGIME_AWARE", False)
        monkeypatch.setattr(g, "GUARDIAN_KELLY_MIN_TRADES", 5)

        for i in range(6):
            g._trade_history.append({"pnl": 10.0 if i % 2 == 0 else -5.0, "size": 100, "token": "SOL", "regime": 1, "ts": time.time()})

        result = g._compute_kelly()
        assert result["active"] is True
        assert "regime_stats" not in result

    def test_kelly_regime_aware(self, monkeypatch):
        import cortex.guardian as g
        g._trade_history.clear()
        monkeypatch.setattr(g, "KELLY_REGIME_AWARE", True)
        monkeypatch.setattr(g, "GUARDIAN_KELLY_MIN_TRADES", 5)
        monkeypatch.setattr(g, "KELLY_REGIME_FRACTIONS", {"0": 0.30, "1": 0.25, "2": 0.20, "3": 0.10, "4": 0.05})

        for i in range(20):
            g._trade_history.append({
                "pnl": 10.0 if i % 3 != 0 else -5.0,
                "size": 100, "token": "SOL",
                "regime": 4, "ts": time.time(),
            })

        result = g._compute_kelly(current_regime=4)
        assert result["active"] is True
        assert result["fraction_used"] == 0.05
        assert "regime_stats" in result
        assert result["regime_stats"]["regime"] == 4

    def test_kelly_crisis_reduces_fraction(self, monkeypatch):
        import cortex.guardian as g
        g._trade_history.clear()
        monkeypatch.setattr(g, "KELLY_REGIME_AWARE", True)
        monkeypatch.setattr(g, "GUARDIAN_KELLY_MIN_TRADES", 5)
        monkeypatch.setattr(g, "KELLY_REGIME_FRACTIONS", {"0": 0.30, "4": 0.05})
        monkeypatch.setattr(g, "GUARDIAN_KELLY_FRACTION", 0.25)

        for i in range(10):
            g._trade_history.append({"pnl": 10.0 if i % 2 == 0 else -5.0, "size": 100, "token": "SOL", "regime": 0, "ts": time.time()})

        calm = g._compute_kelly(current_regime=0)
        crisis = g._compute_kelly(current_regime=4)
        assert crisis["kelly_fraction"] < calm["kelly_fraction"]


# ── Task 3: Adaptive Guardian Weights ───────────────────────────────


class TestAdaptiveWeights:
    """Tests for cortex/adaptive_weights.py"""

    def setup_method(self):
        import cortex.adaptive_weights as aw
        aw.reset()

    def test_disabled_returns_static(self, monkeypatch):
        import cortex.adaptive_weights as aw
        aw.reset()
        monkeypatch.setattr(aw, "ADAPTIVE_WEIGHTS_ENABLED", False)
        from cortex.config import GUARDIAN_WEIGHTS
        assert aw.get_weights() == GUARDIAN_WEIGHTS

    def test_needs_min_samples(self, monkeypatch):
        import cortex.adaptive_weights as aw
        aw.reset()
        monkeypatch.setattr(aw, "ADAPTIVE_WEIGHTS_ENABLED", True)
        monkeypatch.setattr(aw, "ADAPTIVE_WEIGHTS_MIN_SAMPLES", 5)

        scores = [{"component": "evt", "score": 70.0}]
        result = aw.record_outcome(scores, 60.0, -10.0)
        assert result.get("active") is False

    def test_weights_update_after_min_samples(self, monkeypatch):
        import cortex.adaptive_weights as aw
        aw.reset()
        monkeypatch.setattr(aw, "ADAPTIVE_WEIGHTS_ENABLED", True)
        monkeypatch.setattr(aw, "ADAPTIVE_WEIGHTS_MIN_SAMPLES", 3)

        scores = [
            {"component": "evt", "score": 80.0},
            {"component": "svj", "score": 30.0},
        ]

        for _ in range(3):
            aw.record_outcome(scores, 60.0, -10.0)

        result = aw.record_outcome(scores, 60.0, -10.0)
        assert result["active"] is True
        w = aw.get_weights()
        assert w["evt"] > w["svj"]

    def test_weights_sum_to_one(self, monkeypatch):
        import cortex.adaptive_weights as aw
        aw.reset()
        monkeypatch.setattr(aw, "ADAPTIVE_WEIGHTS_ENABLED", True)
        monkeypatch.setattr(aw, "ADAPTIVE_WEIGHTS_MIN_SAMPLES", 2)

        scores = [
            {"component": "evt", "score": 80.0},
            {"component": "svj", "score": 20.0},
            {"component": "hawkes", "score": 50.0},
        ]
        for _ in range(5):
            aw.record_outcome(scores, 50.0, 10.0)

        w = aw.get_weights()
        assert abs(sum(w.values()) - 1.0) < 1e-6


# ── Task 4: Hawkes Intensity-Gated Trade Timing ────────────────────


class TestHawkesTimingGate:
    """Tests for Hawkes timing gate in guardian.py assess_trade"""

    def test_gate_disabled_no_veto(self, monkeypatch):
        import cortex.guardian as g
        g._cache.clear()
        monkeypatch.setattr(g, "HAWKES_TIMING_GATE_ENABLED", False)
        result = g.assess_trade(
            token="HAWKES_OFF", trade_size_usd=100, direction="long",
            model_data=None, evt_data=None, svj_data=None,
            hawkes_data={"event_times": np.array([1.0, 2.0]), "mu": 0.5, "alpha": 0.3, "beta": 1.0},
        )
        assert result["hawkes_deferred"] is False
        assert "hawkes_intensity_gate_defer" not in result["veto_reasons"]

    def test_gate_structure_in_result(self):
        from cortex.guardian import assess_trade
        result = assess_trade(
            token="HAWKES_STRUCT", trade_size_usd=100, direction="long",
            model_data=None, evt_data=None, svj_data=None, hawkes_data=None,
        )
        assert "hawkes_deferred" in result
        assert isinstance(result["hawkes_deferred"], bool)

    def test_gate_fires_on_high_intensity(self, monkeypatch):
        """When many events cluster near the evaluation point, intensity >> baseline.
        Gate should defer the trade."""
        import cortex.guardian as g
        g._cache.clear()
        monkeypatch.setattr(g, "HAWKES_TIMING_GATE_ENABLED", True)
        monkeypatch.setattr(g, "HAWKES_TIMING_KAPPA", 1.0)

        # Cluster 20 events in the last 0.5 time units → high excitation
        events = np.concatenate([
            np.linspace(0, 8, 10),       # sparse early events
            np.linspace(9.5, 10.0, 20),  # dense cluster at the end
        ])
        hawkes_data = {
            "event_times": events,
            "mu": 0.5,     # low baseline
            "alpha": 0.8,  # strong excitation
            "beta": 1.0,   # moderate decay
        }
        result = g.assess_trade(
            token="HAWKES_HIGH", trade_size_usd=100, direction="long",
            model_data=None, evt_data=None, svj_data=None,
            hawkes_data=hawkes_data,
        )
        assert result["hawkes_deferred"] is True
        assert "hawkes_intensity_gate_defer" in result["veto_reasons"]

    def test_gate_allows_on_low_intensity(self, monkeypatch):
        """When events are sparse and excitation is weak, gate should not fire."""
        import cortex.guardian as g
        g._cache.clear()
        monkeypatch.setattr(g, "HAWKES_TIMING_GATE_ENABLED", True)
        # High kappa = very lenient gate: only triggers at extreme spikes
        monkeypatch.setattr(g, "HAWKES_TIMING_KAPPA", 10.0)

        # One event at t=0, evaluated at t=100.01 → excitation fully decayed
        events = np.array([0.0, 100.0])
        hawkes_data = {
            "event_times": events,
            "mu": 1.0,     # baseline
            "alpha": 0.1,  # weak excitation (peak spike = 0.1 above mu)
            "beta": 5.0,   # fast decay → steady_state = 1.02
            # gate_threshold = 1.0 + 10.0 * 0.02 = 1.2 — spike of 0.095 won't reach it
        }
        result = g.assess_trade(
            token="HAWKES_LOW", trade_size_usd=100, direction="long",
            model_data=None, evt_data=None, svj_data=None,
            hawkes_data=hawkes_data,
        )
        assert result["hawkes_deferred"] is False
        assert "hawkes_intensity_gate_defer" not in result["veto_reasons"]

    def test_gate_uses_t_eval_not_linspace(self, monkeypatch):
        """Verify the fix: intensity is evaluated at events[-1] via t_eval,
        not via a 500-point linspace (which was the bug)."""
        import cortex.guardian as g
        from unittest.mock import patch, MagicMock
        g._cache.clear()
        monkeypatch.setattr(g, "HAWKES_TIMING_GATE_ENABLED", True)
        monkeypatch.setattr(g, "HAWKES_TIMING_KAPPA", 1.0)

        events = np.array([1.0, 2.0, 3.0])
        hawkes_data = {"event_times": events, "mu": 0.5, "alpha": 0.3, "beta": 1.0}

        with patch("cortex.hawkes.hawkes_intensity") as mock_hi:
            mock_hi.return_value = {
                "current_intensity": 0.6,
                "baseline": 0.5,
                "t_eval": [3.0],
                "intensity": [0.6],
                "intensity_ratio": 1.2,
                "peak_intensity": 0.6,
                "mean_intensity": 0.6,
            }
            g.assess_trade(
                token="HAWKES_TEVAL", trade_size_usd=100, direction="long",
                model_data=None, evt_data=None, svj_data=None,
                hawkes_data=hawkes_data,
            )
            mock_hi.assert_called_once()
            call_kwargs = mock_hi.call_args
            # Verify t_eval was explicitly passed (the fix)
            assert call_kwargs.kwargs.get("t_eval") is not None
            np.testing.assert_array_almost_equal(call_kwargs.kwargs["t_eval"], [3.01], decimal=2)


# ── Task 5: Regime-Conditional Entry Thresholds ────────────────────


class TestRegimeThresholds:
    """Tests for regime-conditional threshold scaling in guardian.py"""

    def test_disabled_uses_default(self, monkeypatch):
        import cortex.guardian as g
        monkeypatch.setattr(g, "REGIME_THRESHOLD_SCALING_ENABLED", False)
        result = g.assess_trade(
            token="SOL", trade_size_usd=100, direction="long",
            model_data=None, evt_data=None, svj_data=None, hawkes_data=None,
        )
        assert result["effective_threshold"] == 75.0

    def test_high_vol_regime_tightens_threshold(self, monkeypatch):
        import cortex.guardian as g
        g._cache.clear()  # Clear TTL cache from previous test
        monkeypatch.setattr(g, "REGIME_THRESHOLD_SCALING_ENABLED", True)
        monkeypatch.setattr(g, "REGIME_THRESHOLD_LAMBDA", 0.5)
        monkeypatch.setattr(g, "APPROVAL_THRESHOLD", 75.0)

        # Create model_data with high-vol regime active
        K = 5
        sigma_states = [0.01, 0.02, 0.04, 0.08, 0.16]
        filter_probs = np.zeros(K)
        filter_probs[4] = 0.9  # crisis regime dominant (regime 5, 1-based)
        filter_probs[0] = 0.1

        model_data = {
            "filter_probs": pd.DataFrame([filter_probs], columns=[f"state_{i+1}" for i in range(K)]),
            "calibration": {"num_states": K, "sigma_states": sigma_states},
        }

        result = g.assess_trade(
            token="SOL", trade_size_usd=100, direction="long",
            model_data=model_data, evt_data=None, svj_data=None, hawkes_data=None,
        )
        # Current regime is 5 (1-based), sigma=0.16, median sigma=0.04
        # ratio=4.0, effective_threshold = 75 * (1 + 0.5*(4-1)) = 75 * 2.5 = 187.5 → capped at 95
        assert result["effective_threshold"] > 75.0


# ── Task 6: Multi-Timeframe Regime Confirmation ────────────────────


class TestMultiTimeframeConfirmation:
    """Tests for cortex/regime.py multi_timeframe_agreement"""

    def test_all_agree(self):
        from cortex.regime import multi_timeframe_agreement
        returns = pd.Series(np.random.randn(200) * 0.01, index=pd.date_range("2024-01-01", periods=200))

        def mock_fp(slice_):
            n = len(slice_)
            K = 3
            probs = np.zeros((n, K))
            probs[:, 0] = 0.8
            probs[:, 1] = 0.1
            probs[:, 2] = 0.1
            fp = pd.DataFrame(probs, columns=[f"state_{i+1}" for i in range(K)], index=slice_.index)
            return fp, {"num_states": K}

        result = multi_timeframe_agreement(returns, mock_fp, windows=[30, 90, 180], min_agreement=2)
        assert result["confirmed"] is True
        assert result["dominant_regime"] == 1
        assert result["agreement_count"] == 3

    def test_disagreement(self):
        from cortex.regime import multi_timeframe_agreement
        returns = pd.Series(np.random.randn(200) * 0.01, index=pd.date_range("2024-01-01", periods=200))

        call_count = [0]

        def mock_fp(slice_):
            n = len(slice_)
            K = 3
            probs = np.zeros((n, K))
            regime_idx = call_count[0] % 3
            probs[:, regime_idx] = 0.8
            for j in range(K):
                if j != regime_idx:
                    probs[:, j] = 0.1
            call_count[0] += 1
            fp = pd.DataFrame(probs, columns=[f"state_{i+1}" for i in range(K)], index=slice_.index)
            return fp, {"num_states": K}

        result = multi_timeframe_agreement(returns, mock_fp, windows=[30, 90, 180], min_agreement=3)
        assert result["confirmed"] is False

    def test_insufficient_data(self):
        from cortex.regime import multi_timeframe_agreement
        returns = pd.Series(np.random.randn(20) * 0.01, index=pd.date_range("2024-01-01", periods=20))

        def mock_fp(slice_):
            n = len(slice_)
            K = 3
            probs = np.zeros((n, K))
            probs[:, 0] = 0.9
            probs[:, 1] = 0.05
            probs[:, 2] = 0.05
            fp = pd.DataFrame(probs, columns=[f"state_{i+1}" for i in range(K)], index=slice_.index)
            return fp, {"num_states": K}

        result = multi_timeframe_agreement(returns, mock_fp, windows=[30, 90, 180], min_agreement=2)
        assert result["confirmed"] is False


# ── Task 7: Empirical Bayes Debate Prior ────────────────────────────


class TestEmpiricalBayesPrior:
    """Tests for empirical Bayes prior in debate.py"""

    def setup_method(self):
        import cortex.debate as d
        d._debate_outcomes.clear()

    def test_disabled_returns_0_5(self, monkeypatch):
        import cortex.debate as d
        monkeypatch.setattr(d, "DEBATE_EMPIRICAL_PRIOR_ENABLED", False)
        assert d._compute_empirical_prior("spot") == 0.5

    def test_insufficient_data_returns_0_5(self, monkeypatch):
        import cortex.debate as d
        d._debate_outcomes.clear()
        monkeypatch.setattr(d, "DEBATE_EMPIRICAL_PRIOR_ENABLED", True)
        assert d._compute_empirical_prior("spot") == 0.5

    def test_high_win_rate_raises_prior(self, monkeypatch):
        import cortex.debate as d
        d._debate_outcomes.clear()
        monkeypatch.setattr(d, "DEBATE_EMPIRICAL_PRIOR_ENABLED", True)
        monkeypatch.setattr(d, "DEBATE_PRIOR_MIN", 0.3)
        monkeypatch.setattr(d, "DEBATE_PRIOR_MAX", 0.7)

        for i in range(10):
            d.record_debate_outcome("arb", True, 10.0 if i < 8 else -5.0)

        prior = d._compute_empirical_prior("arb")
        # (8+1)/(10+2) = 0.75 → clamped to 0.7
        assert prior == 0.7

    def test_low_win_rate_lowers_prior(self, monkeypatch):
        import cortex.debate as d
        d._debate_outcomes.clear()
        monkeypatch.setattr(d, "DEBATE_EMPIRICAL_PRIOR_ENABLED", True)
        monkeypatch.setattr(d, "DEBATE_PRIOR_MIN", 0.3)
        monkeypatch.setattr(d, "DEBATE_PRIOR_MAX", 0.7)

        for i in range(10):
            d.record_debate_outcome("lp", True, 10.0 if i < 2 else -5.0)

        prior = d._compute_empirical_prior("lp")
        # (2+1)/(10+2) = 0.25 → clamped to 0.3
        assert prior == 0.3

    def test_strategy_isolation(self, monkeypatch):
        import cortex.debate as d
        d._debate_outcomes.clear()
        monkeypatch.setattr(d, "DEBATE_EMPIRICAL_PRIOR_ENABLED", True)
        monkeypatch.setattr(d, "DEBATE_PRIOR_MIN", 0.3)
        monkeypatch.setattr(d, "DEBATE_PRIOR_MAX", 0.7)

        for _ in range(10):
            d.record_debate_outcome("arb", True, 10.0)

        for _ in range(10):
            d.record_debate_outcome("lp", True, -5.0)

        arb_prior = d._compute_empirical_prior("arb")
        lp_prior = d._compute_empirical_prior("lp")
        assert arb_prior > lp_prior


# ── Task 8: Copula Tail-Dependence Risk Gate ────────────────────────


class TestCopulaRiskGate:
    """Tests for copula risk gate integration in guardian.py"""

    def test_gate_disabled_no_veto(self, monkeypatch):
        import cortex.guardian as g
        monkeypatch.setattr(g, "COPULA_RISK_GATE_ENABLED", False)
        result = g.assess_trade(
            token="SOL", trade_size_usd=100, direction="long",
            model_data=None, evt_data=None, svj_data=None, hawkes_data=None,
        )
        assert not any("copula_tail" in v for v in result["veto_reasons"])

    def test_gate_structure_in_result(self):
        from cortex.guardian import assess_trade
        result = assess_trade(
            token="SOL", trade_size_usd=100, direction="long",
            model_data=None, evt_data=None, svj_data=None, hawkes_data=None,
        )
        assert "copula_gate_triggered" in result
        assert isinstance(result["copula_gate_triggered"], bool)

    def test_tail_dependence_computation(self):
        from cortex.copula import _tail_dependence
        td = _tail_dependence("clayton", {"theta": 2.0})
        expected = 2 ** (-1 / 2.0)
        assert abs(td["lambda_lower"] - expected) < 1e-6
        assert td["lambda_upper"] == 0.0

    def test_gumbel_upper_tail(self):
        from cortex.copula import _tail_dependence
        td = _tail_dependence("gumbel", {"theta": 2.0})
        expected = 2 - 2 ** (1 / 2.0)
        assert abs(td["lambda_upper"] - expected) < 1e-6
        assert td["lambda_lower"] == 0.0


# ── Integration: assess_trade returns new fields ────────────────────


class TestAssessTradeIntegration:
    """Verify assess_trade includes all new fields from Tasks 1-8."""

    def test_new_fields_present(self):
        from cortex.guardian import assess_trade
        result = assess_trade(
            token="SOL", trade_size_usd=100, direction="long",
            model_data=None, evt_data=None, svj_data=None, hawkes_data=None,
        )
        assert "calibrated_confidence" in result
        assert "effective_threshold" in result
        assert "hawkes_deferred" in result
        assert "copula_gate_triggered" in result
        assert "approved" in result
        assert "risk_score" in result

    def test_all_disabled_is_backward_compatible(self, monkeypatch):
        import cortex.guardian as g
        g._cache.clear()
        monkeypatch.setattr(g, "CALIBRATION_ENABLED", False)
        monkeypatch.setattr(g, "KELLY_REGIME_AWARE", False)
        monkeypatch.setattr(g, "ADAPTIVE_WEIGHTS_ENABLED", False)
        monkeypatch.setattr(g, "HAWKES_TIMING_GATE_ENABLED", False)
        monkeypatch.setattr(g, "REGIME_THRESHOLD_SCALING_ENABLED", False)
        monkeypatch.setattr(g, "COPULA_RISK_GATE_ENABLED", False)

        result = g.assess_trade(
            token="SOL", trade_size_usd=100, direction="long",
            model_data=None, evt_data=None, svj_data=None, hawkes_data=None,
        )
        assert result["calibrated_confidence"] is None
        assert result["effective_threshold"] == 75.0
        assert result["hawkes_deferred"] is False
        assert result["copula_gate_triggered"] is False


# ── Trade Outcome Endpoint & Feedback Loop ──────────────────────────


class TestTradeOutcomeEndpoint:
    """Tests for the upgraded POST /guardian/trade-outcome endpoint."""

    def setup_method(self):
        import cortex.guardian as g
        import cortex.debate as d
        g._trade_history.clear()
        d._debate_outcomes.clear()

    def test_backward_compat_minimal_payload(self):
        """Old callers sending only pnl+size still work."""
        from cortex.guardian import record_trade_outcome
        record_trade_outcome(pnl=10.0, size=100.0)
        # Should not raise

    def test_new_fields_recorded(self):
        """New fields (regime, component_scores, risk_score) are accepted."""
        from cortex.guardian import record_trade_outcome
        import cortex.guardian as g
        scores = [{"component": "evt", "score": 60.0}]
        record_trade_outcome(
            pnl=-5.0, size=200.0, token="SOL",
            regime=3, component_scores=scores, risk_score=55.0,
        )
        assert len(g._trade_history) == 1
        assert g._trade_history[-1]["regime"] == 3

    def test_adaptive_weights_called_when_enabled(self, monkeypatch):
        """When ADAPTIVE_WEIGHTS_ENABLED, record_outcome is called."""
        import cortex.guardian as g
        from unittest.mock import MagicMock
        monkeypatch.setattr(g, "ADAPTIVE_WEIGHTS_ENABLED", True)

        mock_record = MagicMock(return_value={"active": False})
        monkeypatch.setattr("cortex.adaptive_weights.record_outcome", mock_record)

        scores = [{"component": "evt", "score": 70.0}]
        g.record_trade_outcome(
            pnl=10.0, size=100.0, token="SOL",
            regime=1, component_scores=scores, risk_score=65.0,
        )
        mock_record.assert_called_once_with(scores, 65.0, 10.0)

    def test_adaptive_weights_not_called_when_disabled(self, monkeypatch):
        """When disabled, adaptive weights are not touched."""
        import cortex.guardian as g
        from unittest.mock import MagicMock
        monkeypatch.setattr(g, "ADAPTIVE_WEIGHTS_ENABLED", False)

        mock_record = MagicMock()
        monkeypatch.setattr("cortex.adaptive_weights.record_outcome", mock_record)

        g.record_trade_outcome(pnl=10.0, size=100.0)
        mock_record.assert_not_called()

    def test_api_route_calls_debate_outcome(self, monkeypatch):
        """The API route handler calls debate.record_debate_outcome when strategy is set."""
        import cortex.debate as d
        monkeypatch.setattr(d, "DEBATE_EMPIRICAL_PRIOR_ENABLED", True)

        from api.routes.guardian import record_trade_outcome as route_handler
        from api.models import TradeOutcomeRequest

        req = TradeOutcomeRequest(
            pnl=15.0, size=200.0, token="ETH",
            regime=2, strategy="arb", risk_score=40.0,
        )
        result = route_handler(req)
        assert result["status"] == "recorded"
        assert result["strategy"] == "arb"
        # Debate outcome should have been recorded
        assert len(d._debate_outcomes) == 1
        assert d._debate_outcomes[0]["strategy"] == "arb"

    def test_api_route_no_debate_without_strategy(self):
        """Without a strategy, debate outcome is not recorded."""
        import cortex.debate as d

        from api.routes.guardian import record_trade_outcome as route_handler
        from api.models import TradeOutcomeRequest

        req = TradeOutcomeRequest(pnl=5.0, size=100.0, token="SOL")
        result = route_handler(req)
        assert result["status"] == "recorded"
        assert len(d._debate_outcomes) == 0

    def test_api_route_returns_all_fields(self):
        """The response includes token, regime, and strategy."""
        from api.routes.guardian import record_trade_outcome as route_handler
        from api.models import TradeOutcomeRequest

        req = TradeOutcomeRequest(
            pnl=-3.0, size=50.0, token="RAY",
            regime=4, strategy="lp",
        )
        result = route_handler(req)
        assert result["pnl"] == -3.0
        assert result["size"] == 50.0
        assert result["token"] == "RAY"
        assert result["regime"] == 4
        assert result["strategy"] == "lp"
