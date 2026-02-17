"""Tests for request_id propagation through Guardian and Execution layers."""

import numpy as np
import pandas as pd
import structlog.contextvars

from cortex.execution import _execution_log, _log_execution, execute_trade
from cortex.guardian import _cache as guardian_cache, assess_trade


def _make_model_data():
    probs = np.array([0.8, 0.1, 0.05, 0.03, 0.02])
    fp = pd.DataFrame([probs])
    return {"filter_probs": fp, "calibration": {"num_states": 5}}


class TestGuardianRequestId:
    def test_assess_trade_includes_request_id(self):
        structlog.contextvars.bind_contextvars(request_id="test-req-001")
        try:
            result = assess_trade(
                token="SOL",
                trade_size_usd=100.0,
                direction="long",
                model_data=_make_model_data(),
                evt_data=None,
                svj_data=None,
                hawkes_data=None,
            )
            assert result["request_id"] == "test-req-001"
        finally:
            structlog.contextvars.clear_contextvars()

    def test_assess_trade_no_request_id_is_none(self):
        structlog.contextvars.clear_contextvars()
        guardian_cache.clear()
        result = assess_trade(
            token="SOL",
            trade_size_usd=100.0,
            direction="long",
            model_data=_make_model_data(),
            evt_data=None,
            svj_data=None,
            hawkes_data=None,
        )
        assert result["request_id"] is None


class TestExecutionRequestId:
    def test_log_execution_includes_request_id(self):
        structlog.contextvars.bind_contextvars(request_id="test-exec-001")
        try:
            entry = {"token": "SOL", "status": "simulated"}
            _log_execution(entry)
            assert entry["request_id"] == "test-exec-001"
        finally:
            structlog.contextvars.clear_contextvars()
            _execution_log.pop()

    def test_log_execution_no_request_id(self):
        structlog.contextvars.clear_contextvars()
        entry = {"token": "SOL", "status": "simulated"}
        _log_execution(entry)
        assert "request_id" not in entry
        _execution_log.pop()

    def test_execute_trade_propagates_request_id(self):
        structlog.contextvars.bind_contextvars(request_id="test-exec-002")
        try:
            result = execute_trade(
                private_key="fake",
                token_mint="So11111111111111111111111111111111111111112",
                direction="buy",
                amount=0.01,
                trade_size_usd=1.0,
                model_data=_make_model_data(),
            )
            # Should be simulated (SIMULATION_MODE=True by default in tests)
            # or blocked (EXECUTION_ENABLED=False)
            assert result.get("request_id") == "test-exec-002"
        finally:
            structlog.contextvars.clear_contextvars()
