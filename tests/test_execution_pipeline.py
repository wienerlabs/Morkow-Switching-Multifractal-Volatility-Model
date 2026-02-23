"""Tests for cortex/execution_pipeline.py — Three-Stage Execution Pipeline."""
import pytest

from cortex.execution_pipeline import ExecutionPipeline, PipelineStage
from cortex.trade_ledger import TradeLedger, get_trade_ledger, LedgerState


@pytest.fixture(autouse=True)
def reset_ledger():
    """Reset the global ledger between tests."""
    import cortex.trade_ledger as tl_mod
    tl_mod._ledger = TradeLedger()
    yield
    tl_mod._ledger = None


def _mock_guardian_approve(**kwargs):
    return {
        "approved": True,
        "risk_score": 35.0,
        "recommended_size": kwargs.get("trade_size_usd", 5000.0) * 0.8,
        "veto_reasons": [],
    }


def _mock_guardian_reject(**kwargs):
    return {
        "approved": False,
        "risk_score": 88.0,
        "recommended_size": 0.0,
        "veto_reasons": ["evt_extreme_tail", "regime_extreme_crisis"],
    }


def _mock_execute_success(**kwargs):
    return {"tx_hash": "0xabc123", "fill_price": 150.0, "slippage_bps": 12}


def _mock_execute_fail(**kwargs):
    raise RuntimeError("slippage exceeded 3%")


def test_full_pipeline_success():
    pipeline = ExecutionPipeline(
        guardian_fn=_mock_guardian_approve,
        execute_fn=_mock_execute_success,
    )
    result = pipeline.run("SOL", "long", 5000.0, strategy="arb")

    assert result.stage == PipelineStage.EXECUTED
    assert result.guardian_approved is True
    assert result.guardian_score == 35.0
    assert result.execution_result["tx_hash"] == "0xabc123"
    assert result.execution_error is None
    assert result.entry_hash is not None
    assert result.elapsed_ms > 0

    # Ledger entry should be pushed
    ledger = get_trade_ledger()
    entry = ledger.show(result.entry_hash)
    assert entry.state == LedgerState.PUSHED


def test_pipeline_guardian_reject():
    pipeline = ExecutionPipeline(
        guardian_fn=_mock_guardian_reject,
        execute_fn=_mock_execute_success,
    )
    result = pipeline.run("SOL", "long", 5000.0)

    assert result.stage == PipelineStage.REJECTED
    assert result.guardian_approved is False
    assert len(result.validation_errors) > 0
    assert result.execution_result is None

    # Ledger entry should be rolled back
    ledger = get_trade_ledger()
    entry = ledger.show(result.entry_hash)
    assert entry.state == LedgerState.ROLLED_BACK


def test_pipeline_execution_failure():
    pipeline = ExecutionPipeline(
        guardian_fn=_mock_guardian_approve,
        execute_fn=_mock_execute_fail,
    )
    result = pipeline.run("SOL", "long", 5000.0)

    assert result.stage == PipelineStage.FAILED
    assert result.execution_error == "slippage exceeded 3%"

    ledger = get_trade_ledger()
    entry = ledger.show(result.entry_hash)
    assert entry.state == LedgerState.FAILED


def test_pipeline_dry_run():
    pipeline = ExecutionPipeline(
        guardian_fn=_mock_guardian_approve,
        execute_fn=_mock_execute_success,
    )
    result = pipeline.run("SOL", "long", 5000.0, dry_run=True)

    assert result.stage == PipelineStage.VALIDATED
    assert result.execution_result == {"dry_run": True}
    assert result.guardian_approved is True


def test_pipeline_no_guardian():
    pipeline = ExecutionPipeline(execute_fn=_mock_execute_success)
    result = pipeline.run("SOL", "long", 5000.0)

    assert result.stage == PipelineStage.EXECUTED
    assert result.guardian_approved is True
    assert "no_guardian_fn" in result.validation_errors[0]


def test_pipeline_history():
    pipeline = ExecutionPipeline(
        guardian_fn=_mock_guardian_approve,
        execute_fn=_mock_execute_success,
    )
    pipeline.run("SOL", "long", 1000.0)
    pipeline.run("ETH", "short", 2000.0)
    pipeline.run("BTC", "long", 3000.0)

    history = pipeline.get_history()
    assert len(history) == 3
    assert history[0]["token"] == "SOL"


def test_pipeline_result_serialization():
    pipeline = ExecutionPipeline(
        guardian_fn=_mock_guardian_approve,
        execute_fn=_mock_execute_success,
    )
    result = pipeline.run("SOL", "long", 5000.0)
    d = result.to_dict()

    assert d["stage"] == "executed"
    assert isinstance(d["elapsed_ms"], float)
    assert d["token"] == "SOL"


def test_pipeline_recommended_size_used():
    """Pipeline uses Guardian's recommended_size, not the original."""
    pipeline = ExecutionPipeline(
        guardian_fn=_mock_guardian_approve,
        execute_fn=_mock_execute_success,
    )
    result = pipeline.run("SOL", "long", 10000.0)
    # Guardian scales to 80%
    assert result.trade_size_usd == pytest.approx(8000.0, abs=1.0)


def test_pipeline_intent_passed():
    pipeline = ExecutionPipeline(
        guardian_fn=_mock_guardian_approve,
        execute_fn=_mock_execute_success,
    )
    intent = {"reason": "arb spread", "spread_bps": 50}
    result = pipeline.run("SOL", "long", 5000.0, intent=intent)

    ledger = get_trade_ledger()
    entry = ledger.show(result.entry_hash)
    assert entry.intent == intent
