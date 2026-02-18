"""Narrator API routes — LLM-powered narrative endpoints for Cortex.

Four endpoints:
  POST /narrator/explain    — generate trade decision narrative
  POST /narrator/news       — LLM-interpret news batch
  GET  /narrator/briefing   — periodic market briefing
  POST /narrator/ask        — interactive Q&A
  GET  /narrator/status     — narrator health and usage stats
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from api.models import (
    NarratorExplainRequest,
    NarratorExplainResponse,
    NarratorNewsRequest,
    NarratorNewsResponse,
    NarratorBriefingResponse,
    NarratorAskRequest,
    NarratorAskResponse,
    NarratorStatusResponse,
)
from cortex.config import NARRATOR_ENABLED

router = APIRouter(prefix="/narrator", tags=["narrator"])


def _check_enabled() -> None:
    if not NARRATOR_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Narrator is disabled. Set NARRATOR_ENABLED=true to enable.",
        )


@router.post("/explain", response_model=NarratorExplainResponse)
async def explain_decision(req: NarratorExplainRequest):
    """Generate an LLM narrative explaining a guardian trade assessment.

    Requires a full guardian assessment dict. If assessment is None,
    runs a live guardian assessment for the given token/direction/size.
    """
    _check_enabled()

    from cortex.narrator import explain_decision as _explain

    assessment = req.assessment

    # If no assessment provided, run a live one
    if assessment is None:
        if not req.token:
            raise HTTPException(400, "Either 'assessment' or 'token' is required.")
        try:
            from api.routes.guardian import _run_assessment
            assessment = await _run_assessment(
                token=req.token,
                trade_size_usd=req.trade_size_usd,
                direction=req.direction,
                strategy=req.strategy,
            )
        except Exception as exc:
            raise HTTPException(500, f"Failed to run guardian assessment: {exc}")

    result = await _explain(
        assessment=assessment,
        token=req.token,
        direction=req.direction,
        trade_size_usd=req.trade_size_usd,
    )

    if result.get("error"):
        raise HTTPException(500, result["error"])

    return result


@router.post("/news", response_model=NarratorNewsResponse)
async def interpret_news(req: NarratorNewsRequest):
    """LLM-powered interpretation of news items beyond lexicon matching.

    If no news items provided, reads from the news buffer.
    """
    _check_enabled()

    from cortex.narrator import interpret_news as _interpret

    result = await _interpret(
        news_items=req.news_items,
        news_signal=req.news_signal,
    )

    if result.get("error") and result.get("interpretation") is None:
        raise HTTPException(500, result["error"])

    return result


@router.get("/briefing", response_model=NarratorBriefingResponse)
async def market_briefing():
    """Generate a comprehensive market briefing from all risk model outputs."""
    _check_enabled()

    from cortex.narrator import market_briefing as _briefing

    result = await _briefing()

    if result.get("error"):
        raise HTTPException(500, result["error"])

    return result


@router.post("/ask", response_model=NarratorAskResponse)
async def ask_question(req: NarratorAskRequest):
    """Ask the narrator a question about the system's current state."""
    _check_enabled()

    from cortex.narrator import answer_question as _answer

    result = await _answer(
        question=req.question,
        context_overrides=req.context,
    )

    if result.get("error"):
        raise HTTPException(500, result["error"])

    return result


@router.get("/status", response_model=NarratorStatusResponse)
async def narrator_status():
    """Return narrator health, usage stats, and configuration."""
    from cortex.narrator import get_narrator_status
    return get_narrator_status()
