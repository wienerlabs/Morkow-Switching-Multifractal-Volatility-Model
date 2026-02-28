"""Tests for AgentCoordinator with dynamic Sharpe weights and HMM regime."""

import pytest
from unittest.mock import MagicMock
from cortex.agents.base import AgentSignal, BaseAgent
from cortex.agents.coordinator import AgentCoordinator, CoordinatorDecision


class MockAgent(BaseAgent):
    name = "mock"
    weight = 0.5

    def __init__(self, name, weight, signal):
        self.name = name
        self.weight = weight
        self._signal = signal

    def analyze(self, token, data, context):
        return self._signal

    def analyze_backtest(self, token, data, bar_idx, context):
        return self._signal


class TestCoordinatorWeights:
    def _make_signal(self, name, score, direction="long", confidence=0.7):
        return AgentSignal(
            agent_name=name, score=score, confidence=confidence,
            direction=direction, reasoning="test", metadata={},
        )

    def test_static_weights(self):
        sig1 = self._make_signal("ta", 40)
        sig2 = self._make_signal("macro", 60)
        agents = [
            MockAgent("ta", 0.6, sig1),
            MockAgent("macro", 0.4, sig2),
        ]
        coord = AgentCoordinator(agents)
        dec = coord.evaluate("SOL", None, {})
        assert isinstance(dec, CoordinatorDecision)

    def test_with_pnl_tracker(self):
        mock_tracker = MagicMock()
        mock_tracker.compute_sharpe_weights.return_value = {"ta": 0.7, "macro": 0.3}

        sig1 = self._make_signal("ta", 40)
        sig2 = self._make_signal("macro", 60)
        agents = [
            MockAgent("ta", 0.5, sig1),
            MockAgent("macro", 0.5, sig2),
        ]
        coord = AgentCoordinator(agents, pnl_tracker=mock_tracker)
        dec = coord.evaluate("SOL", None, {})
        assert isinstance(dec, CoordinatorDecision)

    def test_with_hmm_detector(self):
        mock_hmm = MagicMock()
        mock_hmm.get_regime_multipliers.return_value = {"regime_multiplier": 0.8}

        sig1 = self._make_signal("ta", 40)
        agents = [MockAgent("ta", 0.5, sig1)]
        coord = AgentCoordinator(agents, hmm_detector=mock_hmm)
        dec = coord.evaluate("SOL", None, {})
        assert isinstance(dec, CoordinatorDecision)

    def test_no_signals_rejected(self):
        coord = AgentCoordinator([])
        dec = coord.evaluate("SOL", None, {})
        assert not dec.approved
        assert "no_agent_signals" in dec.veto_reasons

    def test_veto_on_extreme_risk(self):
        sig = self._make_signal("ta", 90, direction="long")
        agents = [MockAgent("ta", 1.0, sig)]
        coord = AgentCoordinator(agents, veto_score=85)
        dec = coord.evaluate("SOL", None, {})
        assert not dec.approved
