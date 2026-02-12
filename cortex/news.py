"""
Crypto news intelligence engine with Bayesian sentiment fusion,
regime-aware impact scoring, and market signal generation.

Sources: CryptoCompare, NewsData.io, CryptoPanic
Math: TF-IDF-weighted sentiment, exponential time decay, Jaccard dedup,
      information entropy, MSM regime amplification.
"""

import math
import os
import logging
import re
import time
from dataclasses import dataclass, field

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# ── API Configuration ──

NEWSDATA_API_KEY = os.environ.get("NEWSDATA_API_KEY", "")
CC_API_KEY = os.environ.get("CC_API_KEY", "")
CP_TOKEN = os.environ.get("CRYPTOPANIC_TOKEN", "")

NEWSDATA_BASE = "https://newsdata.io/api/1/crypto"
CC_BASE = "https://data-api.cryptocompare.com/news/v1/article/list"
CP_BASE = "https://cryptopanic.com/api/developer/v2/posts/"

CC_SOURCES = (
    "coindesk,coingape,bitcoinmagazine,blockworks,dailyhodl,cryptoslate,"
    "decrypt,cryptopotato,theblock,cryptobriefing,bitcoin.com,newsbtc,"
    "utoday,investing_comcryptonews,bitcoinist,coinpedia,cryptonomist"
)

# Source credibility priors c_s ∈ [0, 1] — higher = more trustworthy signal
SOURCE_CREDIBILITY: dict[str, float] = {
    "coindesk": 0.92, "theblock": 0.90, "blockworks": 0.88,
    "decrypt": 0.85, "bitcoinmagazine": 0.84, "cryptoslate": 0.80,
    "cryptobriefing": 0.78, "coinpedia": 0.72, "cryptopotato": 0.70,
    "dailyhodl": 0.68, "newsbtc": 0.65, "utoday": 0.65,
    "bitcoinist": 0.64, "coingape": 0.62, "cryptonomist": 0.60,
    "bitcoin.com": 0.75, "investing_comcryptonews": 0.70,
}
DEFAULT_CREDIBILITY = 0.55

HALF_LIFE_HOURS = 4.0
DECAY_LAMBDA = math.log(2) / HALF_LIFE_HOURS

# ── Sentiment Lexicon with TF-IDF-inspired weights ──
# Weight reflects term specificity: rare/strong terms get higher weight.
# Scale: 1.0 = generic, 2.0 = moderate, 3.0 = strong signal

BULL_LEXICON: dict[str, float] = {
    "surge": 2.2, "rally": 2.0, "bullish": 2.5, "soar": 2.3,
    "breakout": 2.8, "ath": 3.0, "all time high": 3.0,
    "pump": 1.8, "moon": 1.5, "accumulation": 2.4,
    "institutional": 2.6, "adoption": 2.2, "partnership": 1.8,
    "upgrade": 1.6, "launch": 1.4, "milestone": 1.8,
    "approval": 2.8, "etf": 2.5, "integration": 1.6,
    "record high": 3.0, "massive gain": 2.8, "price target": 2.0,
    "outperform": 2.2, "buy signal": 2.8, "golden cross": 3.0,
    "short squeeze": 2.6, "whale accumulation": 2.8,
    "positive": 1.2, "gain": 1.4, "rise": 1.2, "growth": 1.4,
    "profit": 1.6, "recover": 1.6, "boost": 1.4, "expand": 1.2,
    "inflow": 2.0, "net buying": 2.2, "support level": 1.8,
}

BEAR_LEXICON: dict[str, float] = {
    "crash": 2.8, "bearish": 2.5, "dump": 2.2, "plunge": 2.6,
    "liquidat": 2.8, "exploit": 3.0, "hack": 3.0,
    "ban": 2.6, "crackdown": 2.8, "lawsuit": 2.6,
    "fraud": 3.0, "scam": 2.8, "rug pull": 3.0,
    "death cross": 3.0, "capitulation": 2.8, "panic sell": 2.8,
    "market crash": 3.0, "flash crash": 3.0, "bank run": 3.0,
    "sec enforcement": 2.8, "criminal charges": 3.0,
    "vulnerability": 2.4, "warning": 1.8, "risk": 1.2,
    "decline": 1.4, "drop": 1.4, "fall": 1.2, "loss": 1.4,
    "sell": 1.2, "correction": 1.6, "downturn": 1.8,
    "fear": 1.6, "collapse": 2.6, "penalty": 2.2, "fine": 2.0,
    "outflow": 2.0, "net selling": 2.2, "resistance level": 1.4,
    "whale dump": 2.8, "insolvency": 3.0, "bankruptcy": 3.0,
}

# Negation window: if a negation word appears within N tokens before a keyword
NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "don't", "doesn't",
                  "didn't", "won't", "wouldn't", "isn't", "aren't", "wasn't"}
NEGATION_WINDOW = 3


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _has_negation(tokens: list[str], idx: int) -> bool:
    """Check if any negation word appears within NEGATION_WINDOW before idx."""
    start = max(0, idx - NEGATION_WINDOW)
    return any(tokens[j] in NEGATION_WORDS for j in range(start, idx))


def _match_lexicon(text: str, lexicon: dict[str, float]) -> float:
    """Score text against a weighted lexicon with negation detection."""
    text_lower = text.lower()
    tokens = _tokenize(text_lower)
    score = 0.0

    # Multi-word phrases first (bigrams/trigrams)
    for phrase, weight in lexicon.items():
        if " " in phrase and phrase in text_lower:
            score += weight

    # Single-word terms with negation check
    for i, tok in enumerate(tokens):
        if tok in lexicon and " " not in tok:
            w = lexicon[tok]
            if _has_negation(tokens, i):
                score -= w * 0.8  # Negation flips ~80% of the signal
            else:
                score += w

    return score


@dataclass
class SentimentResult:
    score: float          # [-1, 1] continuous sentiment
    confidence: float     # [0, 1] how certain we are
    label: str            # Bullish / Bearish / Neutral
    bull_weight: float    # raw bullish signal strength
    bear_weight: float    # raw bearish signal strength
    entropy: float        # information entropy of the distribution


@dataclass
class NewsItem:
    id: str
    source: str
    api_source: str       # cryptocompare | newsdata | cryptopanic
    title: str
    body: str
    url: str
    timestamp: float      # unix ms
    assets: list[str]
    sentiment: SentimentResult
    impact: float         # [0, 10] final impact score
    novelty: float        # [0, 1] how novel vs recent headlines
    source_credibility: float
    time_decay: float     # current decay multiplier
    regime_multiplier: float


@dataclass
class MarketSignal:
    """Aggregated market signal from all news items."""
    sentiment_ewma: float       # exponentially weighted sentiment [-1, 1]
    sentiment_momentum: float   # ΔS over last window
    entropy: float              # consensus entropy (low = agreement)
    confidence: float           # aggregate confidence [0, 1]
    direction: str              # LONG / SHORT / NEUTRAL
    strength: float             # [0, 1] signal strength
    n_sources: int
    n_items: int
    bull_pct: float
    bear_pct: float
    neutral_pct: float


def compute_sentiment(title: str, body: str, source: str = "",
                      cc_sentiment: str = "") -> SentimentResult:
    """
    Bayesian-weighted sentiment scoring.

    S_raw = (Σw_bull - Σw_bear) / (Σw_bull + Σw_bear + ε)
    S_adjusted = c_s × S_raw + (1 - c_s) × 0  (shrink toward neutral)

    Confidence from signal strength and source credibility.
    Entropy H = -Σ p_i log2(p_i) over {bull, bear, neutral} distribution.
    """
    text = f"{title} {body}"
    bull_w = _match_lexicon(text, BULL_LEXICON)
    bear_w = _match_lexicon(text, BEAR_LEXICON)

    # CryptoCompare provides its own sentiment — use as Bayesian prior
    if cc_sentiment == "POSITIVE":
        bull_w += 1.5
    elif cc_sentiment == "NEGATIVE":
        bear_w += 1.5

    total = bull_w + bear_w + 1e-9
    s_raw = (bull_w - bear_w) / total  # [-1, 1]

    # Source credibility shrinkage
    src_key = source.lower().replace(" ", "").replace(".", "")
    credibility = SOURCE_CREDIBILITY.get(src_key, DEFAULT_CREDIBILITY)
    s_adj = credibility * s_raw  # shrink toward 0 for low-credibility

    # Confidence: based on signal magnitude and credibility
    signal_strength = min(1.0, total / 8.0)  # saturates at ~8 weight units
    confidence = signal_strength * credibility

    # Entropy over soft {bull, bear, neutral} distribution
    # Map s_adj to probabilities using softmax-like transform
    p_bull = max(0.01, 0.33 + 0.5 * max(0, s_adj))
    p_bear = max(0.01, 0.33 + 0.5 * max(0, -s_adj))
    p_neut = max(0.01, 1.0 - p_bull - p_bear)
    p_sum = p_bull + p_bear + p_neut
    p_bull, p_bear, p_neut = p_bull / p_sum, p_bear / p_sum, p_neut / p_sum

    entropy = -(p_bull * math.log2(p_bull + 1e-12)
                + p_bear * math.log2(p_bear + 1e-12)
                + p_neut * math.log2(p_neut + 1e-12))

    if s_adj > 0.15:
        label = "Bullish"
    elif s_adj < -0.15:
        label = "Bearish"
    else:
        label = "Neutral"

    return SentimentResult(
        score=round(s_adj, 4),
        confidence=round(confidence, 4),
        label=label,
        bull_weight=round(bull_w, 2),
        bear_weight=round(bear_w, 2),
        entropy=round(entropy, 4),
    )


def compute_impact(
    sentiment: SentimentResult,
    timestamp_ms: float,
    source_credibility: float,
    has_assets: bool,
    novelty: float = 1.0,
    regime_state: int = 3,
    num_states: int = 5,
) -> tuple[float, float, float]:
    """
    Multi-factor impact scoring.

    I_base = f(credibility, sentiment_strength, asset_specificity)
    I_time = I_base × exp(-λ × t_hours)
    I_novel = I_time × (0.5 + 0.5 × novelty)
    I_regime = I_novel × (1 + β × regime_position)

    Returns: (impact_final, time_decay, regime_multiplier)
    """
    # Base impact from source quality and signal strength
    i_base = 3.0 + 3.0 * source_credibility  # [3, 6] range
    i_base += min(2.0, abs(sentiment.score) * 3.0)  # sentiment adds up to 2
    if has_assets:
        i_base += 0.5

    # Exponential time decay
    age_hours = max(0, (time.time() * 1000 - timestamp_ms) / 3_600_000)
    time_decay = math.exp(-DECAY_LAMBDA * age_hours)
    i_time = i_base * time_decay

    # Novelty factor: duplicate news gets halved impact
    i_novel = i_time * (0.5 + 0.5 * novelty)

    # Regime amplification: crisis regime amplifies news impact
    # β controls how much regime matters (0.3 = 30% boost at max regime)
    beta = 0.3
    regime_position = (regime_state - 1) / max(1, num_states - 1)  # [0, 1]
    regime_mult = 1.0 + beta * regime_position
    i_final = i_novel * regime_mult

    return (
        round(min(max(i_final, 0.5), 10.0), 2),
        round(time_decay, 4),
        round(regime_mult, 4),
    )


# ── Jaccard Deduplication ──

def _jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity on token sets for headline deduplication."""
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compute_novelty(title: str, recent_titles: list[str],
                    threshold: float = 0.45) -> float:
    """
    Novelty = 1 - max(Jaccard similarity against recent headlines).
    Returns [0, 1]: 0 = exact duplicate, 1 = completely novel.
    """
    if not recent_titles:
        return 1.0
    max_sim = max(_jaccard_similarity(title, t) for t in recent_titles)
    return round(max(0.0, 1.0 - max_sim), 4)


def _get_source_credibility(source_name: str) -> float:
    key = source_name.lower().replace(" ", "").replace(".", "")
    return SOURCE_CREDIBILITY.get(key, DEFAULT_CREDIBILITY)


# ── API Fetchers ──

def _fetch_cryptocompare(client: httpx.Client) -> list[dict]:
    if not CC_API_KEY:
        logger.warning("CC_API_KEY not set, skipping CryptoCompare")
        return []
    resp = client.get(CC_BASE, params={
        "lang": "EN", "limit": 100,
        "source_ids": CC_SOURCES, "api_key": CC_API_KEY,
    })
    resp.raise_for_status()
    data = resp.json()
    articles = data.get("Data", [])
    if not isinstance(articles, list):
        return []
    return [a for a in articles if a.get("STATUS") == "ACTIVE"]


def _fetch_newsdata(client: httpx.Client) -> list[dict]:
    if not NEWSDATA_API_KEY:
        logger.warning("NEWSDATA_API_KEY not set, skipping NewsData")
        return []
    resp = client.get(NEWSDATA_BASE, params={
        "apikey": NEWSDATA_API_KEY, "q": "crypto",
        "prioritydomain": "top", "size": 10,
    })
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "success":
        return []
    results = data.get("results", [])
    unique = [a for a in results if not a.get("duplicate")]
    return unique if unique else results


def _fetch_cryptopanic(client: httpx.Client) -> list[dict]:
    if not CP_TOKEN:
        logger.warning("CRYPTOPANIC_TOKEN not set, skipping CryptoPanic")
        return []
    resp = client.get(CP_BASE, params={
        "auth_token": CP_TOKEN, "public": "true",
        "currencies": "SOL", "kind": "news",
    })
    resp.raise_for_status()
    data = resp.json()
    results = data.get("results", [])
    return results if isinstance(results, list) else []


# ── Transform raw API responses → NewsItem ──

def _transform_cc(raw: dict, idx: int, recent: list[str],
                   regime_state: int, num_states: int) -> NewsItem:
    title = raw.get("TITLE", "Untitled")
    body = raw.get("BODY") or raw.get("SUBTITLE", "")
    src_data = raw.get("SOURCE_DATA", {})
    src_name = src_data.get("NAME", "CryptoCompare")
    cc_sent = raw.get("SENTIMENT", "")
    cats = [c["CATEGORY"] for c in (raw.get("CATEGORY_DATA") or [])
            if c.get("CATEGORY") and len(c["CATEGORY"]) <= 6]
    ts_ms = (raw.get("PUBLISHED_ON", 0)) * 1000

    sent = compute_sentiment(title, body, src_name, cc_sent)
    cred = _get_source_credibility(src_name)
    novelty = compute_novelty(title, recent)
    impact_val, decay, regime_mult = compute_impact(
        sent, ts_ms, cred, bool(cats), novelty, regime_state, num_states)

    return NewsItem(
        id=f"cc_{idx}", source=src_name, api_source="cryptocompare",
        title=title, body=body[:500],
        url=raw.get("URL", ""), timestamp=ts_ms,
        assets=cats[:4] if cats else ["CRYPTO"],
        sentiment=sent, impact=impact_val, novelty=novelty,
        source_credibility=cred, time_decay=decay,
        regime_multiplier=regime_mult,
    )


def _parse_ts(date_str: str) -> float:
    """Parse various date formats to unix ms."""
    if not date_str:
        return 0.0
    try:
        from datetime import datetime, timezone
        if "T" in date_str:
            dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            dt = datetime.strptime(date_str + " +0000", "%Y-%m-%d %H:%M:%S %z")
        return dt.timestamp() * 1000
    except Exception as e:
        logger.debug("Failed to parse date '%s': %s", date_str, e)
        return 0.0


def _transform_nd(raw: dict, idx: int, recent: list[str],
                   regime_state: int, num_states: int) -> NewsItem:
    title = raw.get("title", "Untitled")
    body = raw.get("description", "")
    src_name = raw.get("source_name", "Unknown")
    coins = raw.get("coin", [])
    ts_ms = _parse_ts(raw.get("pubDate", ""))

    sent = compute_sentiment(title, body, src_name)
    cred = _get_source_credibility(src_name)
    novelty = compute_novelty(title, recent)
    impact_val, decay, regime_mult = compute_impact(
        sent, ts_ms, cred, bool(coins), novelty, regime_state, num_states)

    return NewsItem(
        id=f"nd_{idx}", source=src_name, api_source="newsdata",
        title=title, body=body[:500],
        url=raw.get("link", ""), timestamp=ts_ms,
        assets=coins[:4] if coins else ["CRYPTO"],
        sentiment=sent, impact=impact_val, novelty=novelty,
        source_credibility=cred, time_decay=decay,
        regime_multiplier=regime_mult,
    )


def _transform_cp(raw: dict, idx: int, recent: list[str],
                   regime_state: int, num_states: int) -> NewsItem:
    title = raw.get("title", "Untitled")
    body = raw.get("description", "")
    src_name = (raw.get("source", {}) or {}).get("title", "CryptoPanic")
    ts_ms = 0.0
    if raw.get("published_at"):
        try:
            from datetime import datetime
            ts_ms = datetime.fromisoformat(
                raw["published_at"].replace("Z", "+00:00")).timestamp() * 1000
        except Exception as e:
            logger.debug("Failed to parse published_at timestamp: %s", e)

    sent = compute_sentiment(title, body, src_name)
    cred = _get_source_credibility(src_name)
    novelty = compute_novelty(title, recent)
    impact_val, decay, regime_mult = compute_impact(
        sent, ts_ms, cred, True, novelty, regime_state, num_states)

    return NewsItem(
        id=f"cp_{idx}", source=src_name, api_source="cryptopanic",
        title=title, body=body[:500],
        url=raw.get("url", ""), timestamp=ts_ms,
        assets=["SOL"],
        sentiment=sent, impact=impact_val, novelty=novelty,
        source_credibility=cred, time_decay=decay,
        regime_multiplier=regime_mult,
    )


# ── Bayesian Multi-Source Fusion ──

def fuse_duplicate_sentiments(items: list[NewsItem],
                              sim_threshold: float = 0.45) -> list[NewsItem]:
    """
    When the same story appears from multiple sources, fuse sentiments:
    S_fused = Σ(c_i × S_i) / Σ(c_i)  (credibility-weighted average)
    σ²_fused = 1 / Σ(1/σ²_i)  (precision-weighted variance)

    Keeps the highest-credibility version, updates its sentiment.
    """
    if len(items) < 2:
        return items

    used = set()
    fused = []

    for i, item in enumerate(items):
        if i in used:
            continue
        cluster = [item]
        for j in range(i + 1, len(items)):
            if j in used:
                continue
            if _jaccard_similarity(item.title, items[j].title) > sim_threshold:
                cluster.append(items[j])
                used.add(j)

        if len(cluster) == 1:
            fused.append(item)
            continue

        # Credibility-weighted sentiment fusion
        total_cred = sum(n.source_credibility for n in cluster)
        if total_cred < 1e-9:
            fused.append(item)
            continue

        s_fused = sum(n.source_credibility * n.sentiment.score
                      for n in cluster) / total_cred

        # Precision-weighted confidence (treat each source as independent)
        # σ²_i ≈ (1 - conf_i)², precision = 1/σ²
        precisions = [1.0 / max(0.01, (1 - n.sentiment.confidence) ** 2)
                      for n in cluster]
        fused_var = 1.0 / sum(precisions)
        fused_conf = min(0.99, 1.0 - math.sqrt(fused_var))

        # Average entropy
        avg_entropy = np.mean([n.sentiment.entropy for n in cluster])

        # Keep highest-credibility item as representative
        best = max(cluster, key=lambda n: n.source_credibility)
        best.sentiment = SentimentResult(
            score=round(s_fused, 4),
            confidence=round(fused_conf, 4),
            label="Bullish" if s_fused > 0.15 else "Bearish" if s_fused < -0.15 else "Neutral",
            bull_weight=round(np.mean([n.sentiment.bull_weight for n in cluster]), 2),
            bear_weight=round(np.mean([n.sentiment.bear_weight for n in cluster]), 2),
            entropy=round(avg_entropy, 4),
        )
        # Boost impact for multi-source confirmation
        best.impact = round(min(10.0, best.impact * (1 + 0.15 * (len(cluster) - 1))), 2)
        fused.append(best)

    return fused


# ── Market Signal Aggregation ──

def compute_market_signal(items: list[NewsItem],
                          ewma_alpha: float = 0.3) -> MarketSignal:
    """
    Aggregate all news into a single market signal.

    EWMA sentiment: S_t = α × S_new + (1-α) × S_prev
    Momentum: ΔS = S_recent_half - S_older_half
    Confidence: f(n_sources, agreement, avg_novelty)
    Entropy: H over aggregate {bull, bear, neutral} distribution
    """
    if not items:
        return MarketSignal(
            sentiment_ewma=0.0, sentiment_momentum=0.0, entropy=1.585,
            confidence=0.0, direction="NEUTRAL", strength=0.0,
            n_sources=0, n_items=0,
            bull_pct=0.33, bear_pct=0.33, neutral_pct=0.34,
        )

    # Sort by timestamp ascending for EWMA
    sorted_items = sorted(items, key=lambda n: n.timestamp)

    # EWMA of sentiment scores, weighted by impact
    ewma = 0.0
    for item in sorted_items:
        w = ewma_alpha * (item.impact / 10.0)  # higher impact = more weight
        ewma = w * item.sentiment.score + (1 - w) * ewma

    # Momentum: compare recent half vs older half
    mid = len(sorted_items) // 2
    if mid > 0:
        old_avg = np.mean([n.sentiment.score for n in sorted_items[:mid]])
        new_avg = np.mean([n.sentiment.score for n in sorted_items[mid:]])
        momentum = new_avg - old_avg
    else:
        momentum = 0.0

    # Distribution counts
    n_bull = sum(1 for n in items if n.sentiment.label == "Bullish")
    n_bear = sum(1 for n in items if n.sentiment.label == "Bearish")
    n_neut = len(items) - n_bull - n_bear
    total = len(items)

    bull_pct = n_bull / total
    bear_pct = n_bear / total
    neut_pct = n_neut / total

    # Aggregate entropy
    probs = [max(1e-12, p) for p in [bull_pct, bear_pct, neut_pct]]
    entropy = -sum(p * math.log2(p) for p in probs)

    # Confidence: multi-factor
    sources_seen = set(n.api_source for n in items)
    source_coverage = min(1.0, len(sources_seen) / 3.0)
    avg_novelty = np.mean([n.novelty for n in items])
    agreement = 1.0 - entropy / 1.585  # normalize by max entropy (log2(3))
    confidence = round(source_coverage * 0.4 + agreement * 0.4 + avg_novelty * 0.2, 4)

    # Direction and strength
    if ewma > 0.1:
        direction = "LONG"
    elif ewma < -0.1:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
    strength = round(min(1.0, abs(ewma) * confidence), 4)

    return MarketSignal(
        sentiment_ewma=round(ewma, 4),
        sentiment_momentum=round(momentum, 4),
        entropy=round(entropy, 4),
        confidence=confidence,
        direction=direction,
        strength=strength,
        n_sources=len(sources_seen),
        n_items=total,
        bull_pct=round(bull_pct, 4),
        bear_pct=round(bear_pct, 4),
        neutral_pct=round(neut_pct, 4),
    )


# ── Main Orchestrator ──

def fetch_news_intelligence(
    regime_state: int = 3,
    num_states: int = 5,
    max_items: int = 50,
    timeout: float = 30.0,
) -> dict:
    """
    Fetch from all 3 sources, apply advanced scoring pipeline, return feed + signal.

    Pipeline:
    1. Parallel fetch from CryptoCompare, NewsData.io, CryptoPanic
    2. Transform each article → NewsItem with sentiment + impact
    3. Jaccard deduplication + Bayesian multi-source fusion
    4. Sort by timestamp descending, cap at max_items
    5. Compute aggregate MarketSignal

    Args:
        regime_state: Current MSM regime (1-based). Higher = more volatile.
        num_states: Total MSM regimes (default 5).
        max_items: Max news items to return.
        timeout: HTTP timeout per source.

    Returns:
        Dict with 'items' (list of NewsItem as dicts), 'signal' (MarketSignal),
        'source_counts', and 'meta'.
    """
    t0 = time.time()
    errors: list[str] = []

    with httpx.Client(timeout=timeout) as client:
        # Fetch from all sources
        try:
            cc_raw = _fetch_cryptocompare(client)
        except Exception as e:
            cc_raw = []
            errors.append(f"CryptoCompare: {e}")

        try:
            nd_raw = _fetch_newsdata(client)
        except Exception as e:
            nd_raw = []
            errors.append(f"NewsData: {e}")

        try:
            cp_raw = _fetch_cryptopanic(client)
        except Exception as e:
            cp_raw = []
            errors.append(f"CryptoPanic: {e}")

    if not cc_raw and not nd_raw and not cp_raw:
        logger.error("All news sources failed: %s", errors)
        return {
            "items": [],
            "signal": _signal_to_dict(compute_market_signal([])),
            "source_counts": {"cryptocompare": 0, "newsdata": 0, "cryptopanic": 0},
            "meta": {"errors": errors, "elapsed_ms": 0, "total": 0},
        }

    # Transform with progressive novelty tracking
    recent_titles: list[str] = []
    all_items: list[NewsItem] = []

    for i, raw in enumerate(cc_raw):
        item = _transform_cc(raw, i, recent_titles, regime_state, num_states)
        all_items.append(item)
        recent_titles.append(item.title)

    for i, raw in enumerate(nd_raw):
        item = _transform_nd(raw, i, recent_titles, regime_state, num_states)
        all_items.append(item)
        recent_titles.append(item.title)

    for i, raw in enumerate(cp_raw):
        item = _transform_cp(raw, i, recent_titles, regime_state, num_states)
        all_items.append(item)
        recent_titles.append(item.title)

    # Bayesian fusion of duplicate stories across sources
    fused = fuse_duplicate_sentiments(all_items)

    # Sort by timestamp descending, cap
    fused.sort(key=lambda n: n.timestamp, reverse=True)
    fused = fused[:max_items]

    # Compute aggregate market signal
    signal = compute_market_signal(fused)

    elapsed = round((time.time() - t0) * 1000)
    logger.info(
        "News fetch: CC=%d ND=%d CP=%d → %d items, %d fused | signal=%s %.3f | %dms",
        len(cc_raw), len(nd_raw), len(cp_raw), len(all_items), len(fused),
        signal.direction, signal.strength, elapsed,
    )

    return {
        "items": [_item_to_dict(n) for n in fused],
        "signal": _signal_to_dict(signal),
        "source_counts": {
            "cryptocompare": len(cc_raw),
            "newsdata": len(nd_raw),
            "cryptopanic": len(cp_raw),
        },
        "meta": {
            "errors": errors,
            "elapsed_ms": elapsed,
            "total": len(fused),
            "regime_state": regime_state,
        },
    }


# ── Serialization helpers ──

def _item_to_dict(item: NewsItem) -> dict:
    return {
        "id": item.id,
        "source": item.source,
        "api_source": item.api_source,
        "title": item.title,
        "body": item.body,
        "url": item.url,
        "timestamp": item.timestamp,
        "assets": item.assets,
        "sentiment": {
            "score": item.sentiment.score,
            "confidence": item.sentiment.confidence,
            "label": item.sentiment.label,
            "bull_weight": item.sentiment.bull_weight,
            "bear_weight": item.sentiment.bear_weight,
            "entropy": item.sentiment.entropy,
        },
        "impact": item.impact,
        "novelty": item.novelty,
        "source_credibility": item.source_credibility,
        "time_decay": item.time_decay,
        "regime_multiplier": item.regime_multiplier,
    }


def _signal_to_dict(sig: MarketSignal) -> dict:
    return {
        "sentiment_ewma": sig.sentiment_ewma,
        "sentiment_momentum": sig.sentiment_momentum,
        "entropy": sig.entropy,
        "confidence": sig.confidence,
        "direction": sig.direction,
        "strength": sig.strength,
        "n_sources": sig.n_sources,
        "n_items": sig.n_items,
        "bull_pct": sig.bull_pct,
        "bear_pct": sig.bear_pct,
        "neutral_pct": sig.neutral_pct,
    }