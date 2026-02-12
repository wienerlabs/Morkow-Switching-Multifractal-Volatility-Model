"""Tests for cortex/news.py — News sentiment intelligence engine."""
import math
import time
from unittest.mock import patch

import numpy as np
import pytest

from cortex.news import (
    SentimentResult,
    NewsItem,
    MarketSignal,
    compute_sentiment,
    compute_impact,
    compute_novelty,
    _jaccard_similarity,
    fuse_duplicate_sentiments,
    compute_market_signal,
    HALF_LIFE_HOURS,
    DECAY_LAMBDA,
)


# ── compute_sentiment ──

def test_sentiment_bullish():
    result = compute_sentiment("Bitcoin surge to ATH breakout rally", "massive gain bullish")
    assert result.score > 0
    assert result.label == "Bullish"
    assert result.bull_weight > result.bear_weight


def test_sentiment_bearish():
    result = compute_sentiment("Market crash liquidation panic sell", "bearish dump plunge")
    assert result.score < 0
    assert result.label == "Bearish"
    assert result.bear_weight > result.bull_weight


def test_sentiment_neutral():
    result = compute_sentiment("Bitcoin price update", "The market is stable today")
    assert abs(result.score) <= 0.15
    assert result.label == "Neutral"


def test_sentiment_negation():
    """Negation should flip signal: 'not bullish' reduces bull weight."""
    pos = compute_sentiment("Bitcoin is bullish", "")
    neg = compute_sentiment("Bitcoin is not bullish", "")
    assert neg.bull_weight < pos.bull_weight


def test_sentiment_cc_prior():
    """CryptoCompare sentiment prior boosts the signal."""
    base = compute_sentiment("Bitcoin update", "")
    boosted = compute_sentiment("Bitcoin update", "", cc_sentiment="POSITIVE")
    assert boosted.score > base.score


def test_sentiment_source_credibility():
    """High-credibility source produces higher confidence."""
    high = compute_sentiment("surge rally", "", source="CoinDesk")
    low = compute_sentiment("surge rally", "", source="unknown_blog")
    assert high.confidence > low.confidence


def test_sentiment_entropy_range():
    result = compute_sentiment("Bitcoin surge crash", "mixed signals")
    assert 0 <= result.entropy <= 1.585 + 0.01


# ── compute_impact ──

def test_impact_basic():
    sent = SentimentResult(score=0.5, confidence=0.8, label="Bullish",
                           bull_weight=3.0, bear_weight=0.5, entropy=0.8)
    now_ms = time.time() * 1000
    impact, decay, regime_mult = compute_impact(sent, now_ms, 0.9, True)
    assert 0.5 <= impact <= 10.0
    assert 0.9 < decay <= 1.0  # recent → minimal decay
    assert regime_mult >= 1.0


def test_impact_time_decay():
    """Older news should have lower impact due to exponential decay."""
    sent = SentimentResult(score=0.5, confidence=0.8, label="Bullish",
                           bull_weight=3.0, bear_weight=0.5, entropy=0.8)
    now_ms = time.time() * 1000
    old_ms = now_ms - 8 * 3_600_000  # 8 hours ago (2 half-lives)
    impact_new, decay_new, _ = compute_impact(sent, now_ms, 0.9, True)
    impact_old, decay_old, _ = compute_impact(sent, old_ms, 0.9, True)
    assert impact_new > impact_old
    assert decay_new > decay_old


def test_impact_regime_amplification():
    """Crisis regime (state 5/5) should amplify impact vs calm (state 1/5)."""
    sent = SentimentResult(score=-0.5, confidence=0.8, label="Bearish",
                           bull_weight=0.5, bear_weight=3.0, entropy=0.8)
    now_ms = time.time() * 1000
    _, _, mult_calm = compute_impact(sent, now_ms, 0.9, True, regime_state=1, num_states=5)
    _, _, mult_crisis = compute_impact(sent, now_ms, 0.9, True, regime_state=5, num_states=5)
    assert mult_crisis > mult_calm
    assert mult_calm == pytest.approx(1.0, abs=0.01)
    assert mult_crisis == pytest.approx(1.3, abs=0.01)


# ── Jaccard / novelty ──

def test_jaccard_identical():
    assert _jaccard_similarity("bitcoin price surge", "bitcoin price surge") == 1.0


def test_jaccard_disjoint():
    assert _jaccard_similarity("bitcoin surge", "ethereum crash") == 0.0


def test_jaccard_partial():
    sim = _jaccard_similarity("bitcoin price surge today", "bitcoin price drops today")
    assert 0.3 < sim < 0.8


# ── fuse_duplicate_sentiments ──

def _make_item(title, source, score, confidence, credibility, impact=5.0):
    sent = SentimentResult(score=score, confidence=confidence,
                           label="Bullish" if score > 0.15 else "Bearish" if score < -0.15 else "Neutral",
                           bull_weight=2.0, bear_weight=1.0, entropy=1.0)
    return NewsItem(
        id="test", source=source, api_source="cryptocompare",
        title=title, body="", url="", timestamp=time.time() * 1000,
        assets=["BTC"], sentiment=sent, impact=impact, novelty=1.0,
        source_credibility=credibility, time_decay=1.0, regime_multiplier=1.0,
    )


def test_fuse_single_item():
    items = [_make_item("Bitcoin surges", "CoinDesk", 0.5, 0.8, 0.92)]
    fused = fuse_duplicate_sentiments(items)
    assert len(fused) == 1


def test_fuse_duplicates():
    items = [
        _make_item("Bitcoin surges to new high", "CoinDesk", 0.6, 0.8, 0.92),
        _make_item("Bitcoin surges to new high today", "Decrypt", 0.4, 0.7, 0.85),
    ]
    fused = fuse_duplicate_sentiments(items)
    assert len(fused) == 1
    assert fused[0].source == "CoinDesk"  # highest credibility kept
    assert fused[0].impact > 5.0  # multi-source boost


def test_fuse_distinct():
    items = [
        _make_item("Bitcoin surges to new high", "CoinDesk", 0.6, 0.8, 0.92),
        _make_item("Ethereum upgrade launches successfully", "Decrypt", 0.3, 0.7, 0.85),
    ]
    fused = fuse_duplicate_sentiments(items)
    assert len(fused) == 2


# ── compute_market_signal ──

def test_market_signal_empty():
    signal = compute_market_signal([])
    assert signal.n_items == 0
    assert signal.direction == "NEUTRAL"
    assert signal.strength == 0.0


def test_market_signal_bullish():
    items = [
        _make_item("Bitcoin surge rally", "CoinDesk", 0.7, 0.9, 0.92),
        _make_item("Crypto breakout ATH", "Decrypt", 0.5, 0.8, 0.85),
        _make_item("Massive gain bullish", "BlockWorks", 0.6, 0.85, 0.88),
    ]
    signal = compute_market_signal(items)
    assert signal.sentiment_ewma > 0
    assert signal.direction == "LONG"
    assert signal.n_items == 3
    assert signal.bull_pct > signal.bear_pct


def test_market_signal_bearish():
    items = [
        _make_item("Market crash dump", "CoinDesk", -0.7, 0.9, 0.92),
        _make_item("Crypto plunge panic", "Decrypt", -0.5, 0.8, 0.85),
    ]
    signal = compute_market_signal(items)
    assert signal.sentiment_ewma < 0
    assert signal.direction == "SHORT"
    assert signal.bear_pct > signal.bull_pct


def test_market_signal_entropy():
    """Mixed sentiment should produce higher entropy."""
    items = [
        _make_item("Bitcoin surge", "CoinDesk", 0.7, 0.9, 0.92),
        _make_item("Market crash", "Decrypt", -0.7, 0.8, 0.85),
        _make_item("Stable market", "BlockWorks", 0.0, 0.5, 0.88),
    ]
    signal = compute_market_signal(items)
    assert signal.entropy > 1.0  # high disagreement

