"""News intelligence endpoints: feed, sentiment, signal."""

import logging

from fastapi import APIRouter, HTTPException, Query

from api.models import NewsFeedResponse, NewsMarketSignalModel
from api.stores import _current_regime_state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["news"])


def _get_from_buffer_or_live(
    regime_state: int, max_items: int, live: bool = False,
) -> dict:
    """Return news from background buffer when available, else fetch live."""
    if not live:
        from cortex.news import news_buffer
        cached = news_buffer.get_full(max_items=max_items)
        if cached is not None:
            return cached

    from cortex.news import fetch_news_intelligence
    return fetch_news_intelligence(regime_state=regime_state, max_items=max_items)


@router.get("/news/feed", response_model=NewsFeedResponse)
def get_news_feed(
    regime_state: int = Query(None, ge=1, le=10, description="Override regime state"),
    max_items: int = Query(50, ge=1, le=200),
    live: bool = Query(False, description="Force live fetch instead of buffer"),
):
    rs = regime_state if regime_state is not None else _current_regime_state()
    try:
        result = _get_from_buffer_or_live(rs, max_items, live=live)
    except Exception as exc:
        logger.exception("News feed fetch failed")
        raise HTTPException(status_code=502, detail=f"News fetch error: {exc}")

    return NewsFeedResponse(**result)


@router.get("/news/sentiment", response_model=NewsFeedResponse)
def get_news_sentiment(
    regime_state: int = Query(None, ge=1, le=10),
    max_items: int = Query(20, ge=1, le=100),
    live: bool = Query(False, description="Force live fetch instead of buffer"),
):
    rs = regime_state if regime_state is not None else _current_regime_state()
    try:
        result = _get_from_buffer_or_live(rs, max_items, live=live)
    except Exception as exc:
        logger.exception("News sentiment fetch failed")
        raise HTTPException(status_code=502, detail=f"News fetch error: {exc}")

    return NewsFeedResponse(**result)


@router.get("/news/signal", response_model=NewsMarketSignalModel)
def get_news_signal(
    regime_state: int = Query(None, ge=1, le=10),
    live: bool = Query(False, description="Force live fetch instead of buffer"),
):
    rs = regime_state if regime_state is not None else _current_regime_state()
    try:
        result = _get_from_buffer_or_live(rs, max_items=30, live=live)
    except Exception as exc:
        logger.exception("News signal fetch failed")
        raise HTTPException(status_code=502, detail=f"News fetch error: {exc}")

    return NewsMarketSignalModel(**result["signal"])


@router.get("/news/buffer-stats")
def get_news_buffer_stats():
    """Health check for the background news collector."""
    from cortex.news import news_buffer
    return news_buffer.stats
