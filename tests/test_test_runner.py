"""Tests for cortex/api/test_runner.py — Test Runner API endpoints."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from cortex.api.test_runner import (
    DiscoverResponse,
    RunState,
    RunStatus,
    _history,
    _parse_collect_output,
    _parse_result_line,
    _runs,
    router,
)


@pytest.fixture(autouse=True)
def clear_state():
    """Reset module-level state between tests."""
    _runs.clear()
    _history.clear()
    yield
    _runs.clear()
    _history.clear()


@pytest.fixture
def app():
    """Create a FastAPI app with the test runner router mounted."""
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture
def client(app):
    """Create a synchronous test client."""
    return TestClient(app)


# ---------------------------------------------------------------------------
# _parse_collect_output
# ---------------------------------------------------------------------------

SAMPLE_COLLECT_OUTPUT = """\
tests/test_guardian.py::TestGuardian::test_init
tests/test_guardian.py::TestGuardian::test_assess_trade
tests/test_guardian.py::test_standalone_fn
tests/test_heartbeat.py::test_run_returns_alerts
tests/test_heartbeat.py::test_active_hours_outside

5 tests collected in 1.23s
"""


def test_parse_collect_output_basic():
    """Collect-only output is parsed into structured test file info."""
    result = _parse_collect_output(SAMPLE_COLLECT_OUTPUT)

    assert isinstance(result, DiscoverResponse)
    assert result.total_files == 2
    assert result.total_tests == 5

    files_by_name = {tf.file: tf for tf in result.test_files}
    assert "tests/test_guardian.py" in files_by_name
    assert "tests/test_heartbeat.py" in files_by_name

    guardian = files_by_name["tests/test_guardian.py"]
    assert guardian.test_count == 3
    assert "TestGuardian::test_init" in guardian.tests
    assert "test_standalone_fn" in guardian.tests


def test_parse_collect_output_empty():
    """Empty or whitespace-only output produces zero results."""
    result = _parse_collect_output("")
    assert result.total_files == 0
    assert result.total_tests == 0

    result2 = _parse_collect_output("\n\n  \n")
    assert result2.total_files == 0
    assert result2.total_tests == 0


def test_parse_collect_output_skips_noise():
    """Summary lines and separator lines are ignored."""
    raw = """\
============================== test session starts ==============================
tests/test_abc.py::test_one
----
12 tests collected in 0.5s
"""
    result = _parse_collect_output(raw)
    assert result.total_files == 1
    assert result.total_tests == 1
    assert result.test_files[0].tests == ["test_one"]


# ---------------------------------------------------------------------------
# _parse_result_line
# ---------------------------------------------------------------------------

def test_parse_result_line_passed():
    """PASSED line is parsed into a test_result message."""
    state = RunState(run_id="abc", target="tests/test_foo.py")
    line = "tests/test_foo.py::test_one PASSED                           [ 50%]"
    messages = _parse_result_line(line, state)

    result_msgs = [m for m in messages if m["type"] == "test_result"]
    assert len(result_msgs) == 1
    assert result_msgs[0]["test"] == "tests/test_foo.py::test_one"
    assert result_msgs[0]["status"] == "passed"
    assert state.summary["passed"] == 1
    assert state.summary["total"] == 1


def test_parse_result_line_failed():
    """FAILED line is tracked correctly."""
    state = RunState(run_id="abc", target="tests/test_foo.py")
    line = "tests/test_foo.py::test_two FAILED                           [100%]"
    messages = _parse_result_line(line, state)

    result_msgs = [m for m in messages if m["type"] == "test_result"]
    assert len(result_msgs) == 1
    assert result_msgs[0]["status"] == "failed"
    assert state.summary["failed"] == 1


def test_parse_result_line_error():
    """ERROR line is tracked correctly."""
    state = RunState(run_id="abc", target="all")
    line = "tests/test_foo.py::test_bad ERROR                            [ 33%]"
    messages = _parse_result_line(line, state)

    result_msgs = [m for m in messages if m["type"] == "test_result"]
    assert len(result_msgs) == 1
    assert result_msgs[0]["status"] == "error"
    assert state.summary["error"] == 1


def test_parse_result_line_with_duration():
    """Inline duration (0.03s) is extracted when present."""
    state = RunState(run_id="abc", target="all")
    line = "tests/test_foo.py::test_one PASSED (0.03s)                   [ 50%]"
    messages = _parse_result_line(line, state)

    result_msgs = [m for m in messages if m["type"] == "test_result"]
    assert result_msgs[0]["duration"] == pytest.approx(0.03)


def test_parse_result_line_progress():
    """Progress messages are emitted alongside test results."""
    state = RunState(run_id="abc", target="all")
    _parse_result_line(
        "tests/test_a.py::test_1 PASSED [ 50%]", state,
    )
    messages = _parse_result_line(
        "tests/test_a.py::test_2 FAILED [100%]", state,
    )
    progress = [m for m in messages if m["type"] == "progress"]
    assert len(progress) == 1
    assert progress[0]["completed"] == 2
    assert progress[0]["passed"] == 1
    assert progress[0]["failed"] == 1


def test_parse_result_line_summary():
    """Final summary line sets the run duration."""
    state = RunState(run_id="abc", target="all")
    _parse_result_line(
        "========================= 5 passed in 2.45s =========================",
        state,
    )
    assert state.summary["duration"] == pytest.approx(2.45)


def test_parse_result_line_no_match():
    """Non-result lines produce no messages."""
    state = RunState(run_id="abc", target="all")
    messages = _parse_result_line("collecting ...", state)
    assert messages == []


# ---------------------------------------------------------------------------
# GET /api/tests/discover
# ---------------------------------------------------------------------------

@patch("cortex.api.test_runner.asyncio.create_subprocess_exec")
def test_discover_endpoint(mock_exec, client):
    """Discover endpoint runs pytest --collect-only and returns parsed data."""
    # Reset cache to force a fresh subprocess call
    import cortex.api.test_runner as mod
    mod._discover_cache = None
    mod._discover_cache_ts = 0.0

    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (
        SAMPLE_COLLECT_OUTPUT.encode("utf-8"),
        b"",
    )
    mock_exec.return_value = mock_proc

    resp = client.get("/api/tests/discover")
    assert resp.status_code == 200

    data = resp.json()
    assert data["total_files"] == 2
    assert data["total_tests"] == 5
    assert len(data["test_files"]) == 2


@patch("cortex.api.test_runner.asyncio.create_subprocess_exec")
def test_discover_caching(mock_exec, client):
    """Second call within TTL returns cached result without subprocess."""
    import cortex.api.test_runner as mod
    mod._discover_cache = None
    mod._discover_cache_ts = 0.0

    mock_proc = AsyncMock()
    mock_proc.communicate.return_value = (
        SAMPLE_COLLECT_OUTPUT.encode("utf-8"),
        b"",
    )
    mock_exec.return_value = mock_proc

    resp1 = client.get("/api/tests/discover")
    assert resp1.status_code == 200

    resp2 = client.get("/api/tests/discover")
    assert resp2.status_code == 200

    # Subprocess should only be called once due to caching
    assert mock_exec.call_count == 1


# ---------------------------------------------------------------------------
# POST /api/tests/run
# ---------------------------------------------------------------------------

@patch("cortex.api.test_runner._run_subprocess", new_callable=AsyncMock)
def test_run_endpoint_returns_run_id(mock_subprocess, client):
    """Run endpoint returns a run_id and running status."""
    resp = client.post("/api/tests/run", json={"target": "tests/test_guardian.py"})
    assert resp.status_code == 200

    data = resp.json()
    assert "run_id" in data
    assert data["status"] == "running"
    assert data["run_id"] in _runs


@patch("cortex.api.test_runner._run_subprocess", new_callable=AsyncMock)
def test_run_endpoint_all_target(mock_subprocess, client):
    """Run endpoint accepts 'all' as target."""
    resp = client.post("/api/tests/run", json={"target": "all"})
    assert resp.status_code == 200

    data = resp.json()
    run_id = data["run_id"]
    assert _runs[run_id].target == "all"


# ---------------------------------------------------------------------------
# GET /api/tests/results/{run_id}
# ---------------------------------------------------------------------------

def test_results_unknown_run_id(client):
    """Results endpoint returns 404 for an unknown run_id."""
    resp = client.get("/api/tests/results/nonexistent-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_results_known_run(client):
    """Results endpoint returns data for a known run."""
    state = RunState(
        run_id="test-run-123",
        target="tests/test_guardian.py",
        status=RunStatus.COMPLETED,
        started_at=1000.0,
        finished_at=1002.5,
    )
    state.summary = {"passed": 5, "failed": 1, "error": 0, "skipped": 0, "total": 6, "duration": 2.5}
    state.test_results = [
        {"test": "tests/test_guardian.py::test_init", "status": "passed", "duration": 0.1},
    ]
    state.output_lines = ["line1", "line2"]
    _runs["test-run-123"] = state
    _history.append("test-run-123")

    resp = client.get("/api/tests/results/test-run-123")
    assert resp.status_code == 200

    data = resp.json()
    assert data["run_id"] == "test-run-123"
    assert data["status"] == "completed"
    assert data["summary"]["passed"] == 5
    assert data["summary"]["failed"] == 1
    assert len(data["test_results"]) == 1
    assert len(data["output"]) == 2


# ---------------------------------------------------------------------------
# GET /api/tests/history
# ---------------------------------------------------------------------------

def test_history_empty(client):
    """History endpoint returns empty list when no runs exist."""
    resp = client.get("/api/tests/history")
    assert resp.status_code == 200
    assert resp.json() == []


def test_history_returns_runs(client):
    """History endpoint returns past runs newest-first."""
    for i in range(3):
        state = RunState(
            run_id=f"run-{i}",
            target=f"tests/test_{i}.py",
            status=RunStatus.COMPLETED,
            started_at=1000.0 + i,
            finished_at=1001.0 + i,
        )
        _runs[f"run-{i}"] = state
        _history.append(f"run-{i}")

    resp = client.get("/api/tests/history")
    assert resp.status_code == 200

    data = resp.json()
    assert len(data) == 3
    # Newest first
    assert data[0]["run_id"] == "run-2"
    assert data[2]["run_id"] == "run-0"


def test_history_limit(client):
    """History endpoint respects the limit query parameter."""
    for i in range(5):
        state = RunState(
            run_id=f"run-{i}",
            target="all",
            status=RunStatus.COMPLETED,
            started_at=1000.0 + i,
        )
        _runs[f"run-{i}"] = state
        _history.append(f"run-{i}")

    resp = client.get("/api/tests/history?limit=2")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_history_file_filter(client):
    """History endpoint filters by file when specified."""
    for i, target in enumerate(["tests/test_a.py", "tests/test_b.py", "tests/test_a.py"]):
        state = RunState(
            run_id=f"run-{i}",
            target=target,
            status=RunStatus.COMPLETED,
            started_at=1000.0 + i,
        )
        _runs[f"run-{i}"] = state
        _history.append(f"run-{i}")

    resp = client.get("/api/tests/history?file=tests/test_a.py")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 2
    assert all(e["target"] == "tests/test_a.py" for e in data)
