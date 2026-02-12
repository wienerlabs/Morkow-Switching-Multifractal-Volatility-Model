"""News intelligence endpoints: feed, sentiment, signal."""

import logging

from fastapi import APIRouter, HTTPException, Query

from api.models import NewsFeedResponse, NewsMarketSignalModel
from api.stores import _current_regime_state

logger = logging.getLogger(__name__)

router = APIRouter(tags=["news"])


@router.get("/news/feed", response_model=NewsFeedResponse)
def get_news_feed(
    regime_state: int = Query(None, ge=1, le=10, description="Override regime state"),
    max_items: int = Query(50, ge=1, le=200),
):
    from cortex.news import fetch_news_intelligence

    rs = regime_state if regime_state is not None else _current_regime_state()
    try:
        result = fetch_news_intelligence(regime_state=rs, max_items=max_items)
    except Exception as exc:
        logger.exception("News feed fetch failed")
        raise HTTPException(status_code=502, detail=f"News fetch error: {exc}")

    return NewsFeedResponse(**result)


@router.get("/news/sentiment", response_model=NewsFeedResponse)
def get_news_sentiment(
    regime_state: int = Query(None, ge=1, le=10),
    max_items: int = Query(20, ge=1, le=100),
):
    from cortex.news import fetch_news_intelligence

    rs = regime_state if regime_state is not None else _current_regime_state()
    try:
        result = fetch_news_intelligence(regime_state=rs, max_items=max_items)
    except Exception as exc:
        logger.exception("News sentiment fetch failed")
        raise HTTPException(status_code=502, detail=f"News fetch error: {exc}")

    return NewsFeedResponse(**result)


@router.get("/news/signal", response_model=NewsMarketSignalModel)
def get_news_signal(regime_state: int = Query(None, ge=1, le=10)):
    from cortex.news import fetch_news_intelligence

    rs = regime_state if regime_state is not None else _current_regime_state()
    try:
        result = fetch_news_intelligence(regime_state=rs, max_items=30)
    except Exception as exc:
        logger.exception("News signal fetch failed")
        raise HTTPException(status_code=502, detail=f"News fetch error: {exc}")

    return NewsMarketSignalModel(**result["signal"])

