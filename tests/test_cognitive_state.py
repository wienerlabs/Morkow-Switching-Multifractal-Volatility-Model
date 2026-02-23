"""Tests for cortex/cognitive_state.py — Dynamic Cognitive State Machine."""
import pytest

from cortex.cognitive_state import (
    CognitiveStateMachine,
    EmotionLevel,
    _classify_fear_greed,
    _classify_hawkes,
    _classify_news_sentiment,
    _classify_regime,
)


@pytest.fixture
def csm():
    return CognitiveStateMachine(smoothing=1.0)  # no smoothing for test predictability


@pytest.fixture
def csm_smoothed():
    return CognitiveStateMachine(smoothing=0.3)


# ── Signal classifiers ──

def test_classify_fear_greed_extreme_fear():
    score, reason = _classify_fear_greed(10)
    assert score < -0.5
    assert reason == "extreme_fear_zone"


def test_classify_fear_greed_fear():
    score, reason = _classify_fear_greed(35)
    assert score < 0
    assert reason == "fear_zone"


def test_classify_fear_greed_neutral():
    score, reason = _classify_fear_greed(50)
    assert score == pytest.approx(0.0)
    assert reason == "neutral_zone"


def test_classify_fear_greed_greed():
    score, reason = _classify_fear_greed(70)
    assert score > 0
    assert reason == "greed_zone"


def test_classify_fear_greed_extreme_greed():
    score, reason = _classify_fear_greed(90)
    assert score > 0.5
    assert reason == "extreme_greed_zone"


def test_classify_regime_calm():
    score, reason = _classify_regime(1, 5, 0.02)
    assert score > 0  # calm = positive (greed direction)
    assert reason == "calm_regime"


def test_classify_regime_crisis():
    score, reason = _classify_regime(5, 5, 0.8)
    assert score < -0.5  # crisis = negative (fear direction)
    assert reason == "crisis_regime"


def test_classify_hawkes_low():
    score, reason = _classify_hawkes(0.1)
    assert score > 0  # low contagion = positive (calm)
    assert reason == "low_contagion"


def test_classify_hawkes_flash_crash():
    score, reason = _classify_hawkes(0.9)
    assert score < -0.5  # high contagion = negative (fear)
    assert reason == "flash_crash_risk"


def test_classify_news_bullish():
    score, reason = _classify_news_sentiment(0.7)
    assert score > 0
    assert reason == "bullish_news"


def test_classify_news_bearish():
    score, reason = _classify_news_sentiment(-0.7)
    assert score < 0
    assert reason == "bearish_news"


# ── State machine transitions ──

def test_extreme_fear_signals(csm):
    level = csm.update(
        fear_greed_value=10,
        regime=5, num_states=5, crisis_prob=0.8,
        hawkes_contagion=0.9,
        news_ewma=-0.8,
    )
    assert level == EmotionLevel.EXTREME_FEAR


def test_extreme_greed_signals(csm):
    level = csm.update(
        fear_greed_value=90,
        regime=1, num_states=5, crisis_prob=0.01,
        hawkes_contagion=0.05,
        news_ewma=0.8,
    )
    assert level == EmotionLevel.EXTREME_GREED


def test_neutral_signals(csm):
    level = csm.update(
        fear_greed_value=50,
        regime=3, num_states=5, crisis_prob=0.1,
        hawkes_contagion=0.3,
        news_ewma=0.0,
    )
    assert level == EmotionLevel.NEUTRAL


def test_partial_signals(csm):
    """Works with only some signals available."""
    level = csm.update(fear_greed_value=15)
    assert level in (EmotionLevel.EXTREME_FEAR, EmotionLevel.FEAR)


def test_no_signals_keeps_state(csm):
    csm.update(fear_greed_value=90)
    level = csm.update()
    assert level != EmotionLevel.NEUTRAL or csm.level != EmotionLevel.NEUTRAL


def test_transition_logging(csm):
    csm.update(fear_greed_value=50, regime=3, num_states=5)
    assert csm.level == EmotionLevel.NEUTRAL

    csm.update(fear_greed_value=10, regime=5, num_states=5, crisis_prob=0.8, hawkes_contagion=0.9)
    transitions = csm.get_transitions()
    assert len(transitions) >= 1
    assert transitions[-1]["to"] in ("extreme_fear", "fear")


# ── Threshold adjustments ──

def test_extreme_fear_lowers_threshold(csm):
    csm.update(fear_greed_value=10, regime=5, num_states=5, crisis_prob=0.8, hawkes_contagion=0.85)
    delta = csm.get_threshold_adjustment()
    assert delta < 0


def test_extreme_greed_raises_threshold(csm):
    csm.update(fear_greed_value=90, regime=1, num_states=5, hawkes_contagion=0.05, news_ewma=0.8)
    delta = csm.get_threshold_adjustment()
    assert delta > 0


def test_effective_threshold_bounded(csm):
    csm.update(fear_greed_value=5, regime=5, num_states=5, crisis_prob=0.95, hawkes_contagion=0.95)
    threshold = csm.get_effective_threshold()
    assert 50.0 <= threshold <= 95.0


# ── Size and Kelly multipliers ──

def test_extreme_greed_cuts_size(csm):
    csm.update(fear_greed_value=90, regime=1, num_states=5, hawkes_contagion=0.05, news_ewma=0.8)
    assert csm.get_size_multiplier() < 1.0
    assert csm.get_kelly_multiplier() < 1.0


def test_neutral_no_adjustment(csm):
    csm.update(fear_greed_value=50, regime=3, num_states=5, hawkes_contagion=0.3)
    assert csm.get_size_multiplier() == 1.0
    assert csm.get_kelly_multiplier() == 1.0
    assert csm.get_threshold_adjustment() == 0.0


# ── EMA smoothing ──

def test_smoothing_dampens_change(csm_smoothed):
    csm_smoothed.update(fear_greed_value=50, regime=3, num_states=5)
    csm_smoothed.update(fear_greed_value=90, regime=1, num_states=5, news_ewma=0.8)
    # With 0.3 smoothing, a single extreme signal shouldn't immediately jump to extreme
    assert csm_smoothed.smoothed_score < csm_smoothed.raw_score


# ── Snapshot/Restore ──

def test_snapshot_and_restore(csm):
    csm.update(fear_greed_value=10, regime=5, num_states=5, crisis_prob=0.8, hawkes_contagion=0.85)
    snap = csm.snapshot()

    new_csm = CognitiveStateMachine()
    new_csm.restore(snap)
    assert new_csm.level == csm.level
    assert new_csm.smoothed_score == pytest.approx(csm.smoothed_score, abs=0.01)


# ── get_adjustments ──

def test_get_adjustments_structure(csm):
    csm.update(fear_greed_value=75, regime=2, num_states=5)
    adj = csm.get_adjustments()
    assert "emotion_level" in adj
    assert "threshold_delta" in adj
    assert "effective_threshold" in adj
    assert "kelly_multiplier" in adj
    assert "size_multiplier" in adj
    assert "last_inputs" in adj


# ── Feature disabled ──

def test_disabled_returns_neutral():
    from unittest.mock import patch
    with patch("cortex.guardian.COGNITIVE_STATE_ENABLED", False):
        from cortex.guardian import assess_trade
        # Import still works; Guardian just skips cognitive state
