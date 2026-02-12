"""Social sentiment endpoints."""

import logging

from fastapi import APIRouter, HTTPException, Query

from api.models import SocialSentimentResponse, SocialSourceItem

logger = logging.getLogger(__name__)

router = APIRouter(tags=["social"])


@router.get("/social/sentiment", response_model=SocialSentimentResponse)
def get_social_sentiment(
    token: str = Query("solana", description="Token or topic to query"),
):
    """Return aggregated social sentiment for a token."""
    from cortex.data.social import fetch_social_sentiment

    try:
        result = fetch_social_sentiment(token=token)
    except Exception as exc:
        logger.exception("Social sentiment fetch failed")
        raise HTTPException(status_code=502, detail=f"Social error: {exc}")

    return SocialSentimentResponse(
        token=result["token"],
        overall_sentiment=result["overall_sentiment"],
        sources=[
            SocialSourceItem(
                source=s["source"],
                sentiment=s["sentiment"],
                count=s["count"],
            )
            for s in result["sources"]
        ],
        timestamp=result["timestamp"],
    )

