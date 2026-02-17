"""Transaction stream endpoints — Helius Enhanced WebSocket events + Guardian SSE."""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from api.models import StreamEvent, StreamEventsResponse, StreamStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streams"])

# ── Guardian SSE broadcast infrastructure ────────────────────────────

_guardian_subscribers: list[asyncio.Queue] = []


def broadcast_guardian_score(event: dict[str, Any]) -> None:
    """Broadcast a Guardian score event to all SSE subscribers."""
    event["ts"] = time.time()
    dead: list[asyncio.Queue] = []
    for q in _guardian_subscribers:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        _guardian_subscribers.remove(q)


@router.get("/streams/events", response_model=StreamEventsResponse)
def get_stream_events(
    limit: int = Query(50, ge=1, le=200),
    severity: str = Query(None, description="Filter by severity: info, warning, critical"),
):
    """Return recent classified transaction events."""
    from cortex.data.streams import get_recent_events

    try:
        events = get_recent_events(limit=limit, severity=severity)
    except Exception as exc:
        logger.exception("Stream events fetch failed")
        raise HTTPException(status_code=502, detail=f"Stream error: {exc}")

    return StreamEventsResponse(
        events=[StreamEvent(**e) for e in events],
        total=len(events),
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/streams/status", response_model=StreamStatusResponse)
def get_stream_status():
    """Return stream connection status."""
    from cortex.data.streams import get_stream_status as _status

    return StreamStatusResponse(**_status())


@router.get("/guardian/stream")
async def guardian_sse_stream():
    """SSE endpoint streaming Guardian risk score events in real time.

    Connect with: curl -N /api/v1/guardian/stream
    Events are pushed after each /guardian/assess call.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    _guardian_subscribers.append(queue)

    async def event_generator():
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in _guardian_subscribers:
                _guardian_subscribers.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

