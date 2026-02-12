"""Social feed sentiment aggregation — Twitter, Discord, Telegram.

Fetches social signals via HTTP APIs (no SDK dependencies) and scores
sentiment using the same Bayesian framework as ``cortex.news``.
"""

import logging
import os
import time
from typing import Any

import httpx

from cortex.config import SOCIAL_CACHE_TTL, TWITTER_BEARER_TOKEN

logger = logging.getLogger(__name__)

_social_cache: dict[str, Any] = {}
_cache_ts: float = 0.0

_SENTIMENT_KEYWORDS = {
    "bullish": ["bullish", "moon", "pump", "breakout", "ath", "buy", "long", "accumulate"],
    "bearish": ["bearish", "dump", "crash", "sell", "short", "rug", "scam", "liquidat"],
}


def _score_text(text: str) -> float:
    """Simple keyword sentiment scorer: returns -1.0 to 1.0."""
    lower = text.lower()
    bull = sum(1 for kw in _SENTIMENT_KEYWORDS["bullish"] if kw in lower)
    bear = sum(1 for kw in _SENTIMENT_KEYWORDS["bearish"] if kw in lower)
    total = bull + bear
    if total == 0:
        return 0.0
    return (bull - bear) / total


def fetch_twitter_sentiment(query: str = "solana", max_results: int = 20) -> dict[str, Any]:
    """Fetch recent tweets and compute aggregate sentiment."""
    if not TWITTER_BEARER_TOKEN:
        logger.debug("TWITTER_BEARER_TOKEN not set — returning neutral")
        return {"source": "twitter", "sentiment": 0.0, "count": 0, "items": []}

    url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {TWITTER_BEARER_TOKEN}"}
    params = {"query": f"{query} -is:retweet lang:en", "max_results": min(max_results, 100)}

    try:
        with httpx.Client(timeout=15) as client:
            resp = client.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        logger.warning("Twitter API request failed")
        return {"source": "twitter", "sentiment": 0.0, "count": 0, "items": []}

    tweets = data.get("data", [])
    scores = [_score_text(t.get("text", "")) for t in tweets]
    avg = sum(scores) / len(scores) if scores else 0.0

    return {
        "source": "twitter",
        "sentiment": round(avg, 4),
        "count": len(tweets),
        "items": [
            {"text": t.get("text", "")[:200], "score": round(s, 4)}
            for t, s in zip(tweets, scores)
        ],
    }


def fetch_social_sentiment(token: str = "solana") -> dict[str, Any]:
    """Aggregate sentiment across available social sources with caching."""
    global _social_cache, _cache_ts

    now = time.time()
    cache_key = token.lower()
    if cache_key in _social_cache and (now - _cache_ts) < SOCIAL_CACHE_TTL:
        return _social_cache[cache_key]

    twitter = fetch_twitter_sentiment(query=token)

    sources = [twitter]
    sentiments = [s["sentiment"] for s in sources if s["count"] > 0]
    overall = sum(sentiments) / len(sentiments) if sentiments else 0.0

    result = {
        "token": token,
        "overall_sentiment": round(overall, 4),
        "sources": sources,
        "timestamp": now,
    }

    _social_cache[cache_key] = result
    _cache_ts = now
    return result

