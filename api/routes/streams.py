"""Transaction stream endpoints — Helius Enhanced WebSocket events."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.models import StreamEvent, StreamEventsResponse, StreamStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streams"])


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

