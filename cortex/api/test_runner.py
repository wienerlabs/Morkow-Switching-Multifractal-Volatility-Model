"""Test Runner API — FastAPI router for discovering, executing, streaming,
and retrieving pytest results.

Provides endpoints to:
  - Discover available tests via ``pytest --collect-only``
  - Launch test runs as async subprocesses
  - Stream live output over WebSocket
  - Retrieve historical run results

Mount the router on any FastAPI ``app`` with::

    from cortex.api.test_runner import router
    app.include_router(router)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
DISCOVER_CACHE_TTL = 60  # seconds
MAX_HISTORY = 100

# Regex for parsing pytest verbose output lines like:
#   tests/test_guardian.py::TestGuardian::test_init PASSED          [ 3%]
#   tests/test_guardian.py::test_standalone FAILED                  [100%]
_RESULT_RE = re.compile(
    r"^(.*?::.*?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)"
    r"(?:\s.*?)?"
    r"(?:\s+\[\s*\d+%\])?\s*$"
)

# Duration line: "=== N passed, M failed in 2.45s ==="
_SUMMARY_RE = re.compile(
    r"=+\s*(.*?)\s+in\s+([\d.]+)s\s*=+"
)

# Individual duration from --durations or inline (e.g. " (0.03s)")
_DURATION_RE = re.compile(r"\(([\d.]+)s\)")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class RunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class RunRequest(BaseModel):
    target: str  # e.g. "tests/test_guardian.py" or "all"
    files: list[str] | None = None  # optional list of specific files to run


class TestFileInfo(BaseModel):
    file: str
    test_count: int
    tests: list[str]
    tag: str = ""


class DiscoverResponse(BaseModel):
    test_files: list[TestFileInfo]
    total_files: int
    total_tests: int


class RunResponse(BaseModel):
    run_id: str
    status: str


class TestResult(BaseModel):
    test: str
    status: str
    duration: float | None = None


class RunSummary(BaseModel):
    passed: int = 0
    failed: int = 0
    error: int = 0
    skipped: int = 0
    total: int = 0
    duration: float = 0.0


class RunResultResponse(BaseModel):
    run_id: str
    target: str
    status: str
    started_at: float
    finished_at: float | None
    summary: RunSummary
    test_results: list[TestResult]
    output: list[str]


class HistoryEntry(BaseModel):
    run_id: str
    target: str
    status: str
    started_at: float
    finished_at: float | None
    summary: RunSummary


@dataclass
class RunState:
    """In-memory state for a single test run."""

    run_id: str
    target: str
    files: list[str] | None = None
    status: RunStatus = RunStatus.RUNNING
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    process: asyncio.subprocess.Process | None = None
    output_lines: list[str] = field(default_factory=list)
    test_results: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=lambda: {
        "passed": 0, "failed": 0, "error": 0, "skipped": 0, "total": 0, "duration": 0.0,
    })
    websockets: list[WebSocket] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_runs: dict[str, RunState] = {}
_history: deque[str] = deque(maxlen=MAX_HISTORY)  # run_id order

# Discover cache
_discover_cache: dict[str, Any] | None = None
_discover_cache_ts: float = 0.0

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/tests", tags=["tests"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_collect_output(raw: str) -> DiscoverResponse:
    """Parse ``pytest --collect-only -q`` output into structured data."""
    files: dict[str, list[str]] = {}

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("=") or line.startswith("-"):
            continue
        # Lines look like: "tests/test_guardian.py::TestGuardian::test_init"
        # or "tests/test_guardian.py::test_standalone"
        if "::" in line:
            parts = line.split("::", 1)
            file_path = parts[0].strip()
            test_name = parts[1].strip()
            if file_path not in files:
                files[file_path] = []
            files[file_path].append(test_name)
        # Skip summary lines like "153 tests collected"

    test_files = [
        TestFileInfo(
            file=f,
            test_count=len(tests),
            tests=tests,
            tag="",
        )
        for f, tests in sorted(files.items())
    ]

    total_tests = sum(tf.test_count for tf in test_files)
    return DiscoverResponse(
        test_files=test_files,
        total_files=len(test_files),
        total_tests=total_tests,
    )


def _parse_result_line(line: str, state: RunState) -> list[dict[str, Any]]:
    """Parse a single pytest verbose output line, returning JSON messages to send."""
    messages: list[dict[str, Any]] = []

    m = _RESULT_RE.match(line.strip())
    if m:
        test_id = m.group(1).strip()
        raw_status = m.group(2).upper()
        status_map = {
            "PASSED": "passed",
            "FAILED": "failed",
            "ERROR": "error",
            "SKIPPED": "skipped",
            "XFAIL": "skipped",
            "XPASS": "passed",
        }
        status = status_map.get(raw_status, raw_status.lower())

        # Try to extract inline duration
        dur_match = _DURATION_RE.search(line)
        duration = float(dur_match.group(1)) if dur_match else None

        result = {"test": test_id, "status": status, "duration": duration}
        state.test_results.append(result)
        messages.append({"type": "test_result", **result})

        # Update running counts
        key = status if status in ("passed", "failed", "error", "skipped") else "failed"
        state.summary[key] = state.summary.get(key, 0) + 1
        state.summary["total"] = state.summary.get("total", 0) + 1

        messages.append({
            "type": "progress",
            "completed": state.summary["total"],
            "total": state.summary["total"],
            "passed": state.summary["passed"],
            "failed": state.summary["failed"],
        })

    # Check for final summary line
    sm = _SUMMARY_RE.search(line)
    if sm:
        state.summary["duration"] = float(sm.group(2))

    return messages


async def _broadcast(state: RunState, message: dict[str, Any]) -> None:
    """Send a JSON message to all connected WebSocket clients for a run."""
    text = json.dumps(message)
    disconnected: list[WebSocket] = []
    for ws in state.websockets:
        try:
            await ws.send_text(text)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        state.websockets.remove(ws)


async def _run_subprocess(state: RunState) -> None:
    """Execute pytest in a subprocess and stream output."""
    target = state.target
    cmd = [sys.executable, "-m", "pytest"]

    if state.files:
        cmd.extend(state.files)
    elif target == "all":
        cmd.append("tests/")
    else:
        cmd.append(target)

    cmd.extend(["-v", "--tb=short", "--no-header"])

    logger.info("Starting test run %s: %s", state.run_id, " ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=PROJECT_ROOT,
        )
        state.process = proc

        assert proc.stdout is not None
        while True:
            line_bytes = await proc.stdout.readline()
            if not line_bytes:
                break
            line = line_bytes.decode("utf-8", errors="replace").rstrip("\n")
            state.output_lines.append(line)

            # Broadcast raw output
            await _broadcast(state, {"type": "output", "line": line})

            # Parse for test results
            messages = _parse_result_line(line, state)
            for msg in messages:
                await _broadcast(state, msg)

        await proc.wait()

        if state.status == RunStatus.CANCELLED:
            pass  # already set by cancel handler
        elif proc.returncode == 0:
            state.status = RunStatus.COMPLETED
        else:
            # pytest returns non-zero on failures — still "completed"
            state.status = RunStatus.COMPLETED

    except Exception:
        logger.exception("Test run %s failed with exception", state.run_id)
        state.status = RunStatus.FAILED
    finally:
        state.finished_at = time.time()
        state.process = None

        # Broadcast completion
        await _broadcast(state, {
            "type": "complete",
            "summary": state.summary,
        })

        logger.info(
            "Test run %s finished: %s (%s)",
            state.run_id,
            state.status.value,
            state.summary,
        )


def _evict_history() -> None:
    """Remove oldest runs beyond MAX_HISTORY."""
    while len(_history) > MAX_HISTORY:
        old_id = _history.popleft()
        _runs.pop(old_id, None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/discover", response_model=DiscoverResponse)
async def discover_tests() -> DiscoverResponse:
    """Enumerate all tests via ``pytest --collect-only``."""
    global _discover_cache, _discover_cache_ts

    now = time.time()
    if _discover_cache is not None and (now - _discover_cache_ts) < DISCOVER_CACHE_TTL:
        return _discover_cache

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "pytest", "--collect-only", "-qq", "tests/",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=PROJECT_ROOT,
    )
    stdout_bytes, _ = await proc.communicate()
    raw = stdout_bytes.decode("utf-8", errors="replace")

    result = _parse_collect_output(raw)
    _discover_cache = result
    _discover_cache_ts = now
    return result


@router.post("/run", response_model=RunResponse)
async def start_run(body: RunRequest) -> RunResponse:
    """Launch a pytest run as an async subprocess."""
    run_id = str(uuid.uuid4())
    state = RunState(run_id=run_id, target=body.target, files=body.files)
    _runs[run_id] = state
    _history.append(run_id)
    _evict_history()

    # Fire-and-forget the subprocess
    asyncio.create_task(_run_subprocess(state))

    return RunResponse(run_id=run_id, status=RunStatus.RUNNING.value)


@router.websocket("/stream/{run_id}")
async def stream_run(websocket: WebSocket, run_id: str) -> None:
    """Stream live test output over WebSocket."""
    if run_id not in _runs:
        await websocket.close(code=4004, reason="Unknown run_id")
        return

    await websocket.accept()
    state = _runs[run_id]
    state.websockets.append(websocket)

    # Replay existing output
    for line in state.output_lines:
        try:
            await websocket.send_text(json.dumps({"type": "output", "line": line}))
        except Exception:
            return

    # Replay existing test results
    for result in state.test_results:
        try:
            await websocket.send_text(json.dumps({"type": "test_result", **result}))
        except Exception:
            return

    # If already finished, send completion and close
    if state.status != RunStatus.RUNNING:
        try:
            await websocket.send_text(json.dumps({
                "type": "complete",
                "summary": state.summary,
            }))
        except Exception:
            pass
        finally:
            if websocket in state.websockets:
                state.websockets.remove(websocket)
        return

    # Listen for client messages (cancel) until run finishes
    try:
        while state.status == RunStatus.RUNNING:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                continue

            if msg.get("action") == "cancel":
                logger.info("Cancel requested for run %s", run_id)
                state.status = RunStatus.CANCELLED
                if state.process is not None:
                    try:
                        state.process.kill()
                    except ProcessLookupError:
                        pass
                break

    except WebSocketDisconnect:
        logger.debug("WebSocket client disconnected from run %s", run_id)
    except Exception:
        logger.exception("WebSocket error for run %s", run_id)
    finally:
        if websocket in state.websockets:
            state.websockets.remove(websocket)


@router.get("/results/{run_id}", response_model=RunResultResponse)
async def get_results(run_id: str) -> RunResultResponse:
    """Return full results of a test run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    state = _runs[run_id]
    return RunResultResponse(
        run_id=state.run_id,
        target=state.target,
        status=state.status.value,
        started_at=state.started_at,
        finished_at=state.finished_at,
        summary=RunSummary(**state.summary),
        test_results=[TestResult(**r) for r in state.test_results],
        output=state.output_lines,
    )


@router.get("/history", response_model=list[HistoryEntry])
async def get_history(
    limit: int = Query(default=20, ge=1, le=100),
    file: str | None = Query(default=None),
) -> list[HistoryEntry]:
    """Return summaries of past test runs (newest first)."""
    entries: list[HistoryEntry] = []

    # Iterate newest-first
    for run_id in reversed(_history):
        if run_id not in _runs:
            continue
        state = _runs[run_id]

        if file is not None and state.target != file:
            continue

        entries.append(HistoryEntry(
            run_id=state.run_id,
            target=state.target,
            status=state.status.value,
            started_at=state.started_at,
            finished_at=state.finished_at,
            summary=RunSummary(**state.summary),
        ))

        if len(entries) >= limit:
            break

    return entries


# ---------------------------------------------------------------------------
# Standalone app — run with: uvicorn cortex.api.test_runner:app --reload
# ---------------------------------------------------------------------------

app = FastAPI(title="Cortex Test Runner")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)

_frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend" / "ui"
if _frontend_dir.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")
