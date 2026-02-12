"""
Guardian Integration Layer — Unified risk veto endpoint for Cortex autonomous trading.

Consolidates EVT (25%), SVJ (20%), Hawkes (20%), Regime (20%), and News (15%)
risk assessments into a single composite score with circuit breaker logic
and position sizing.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

WEIGHTS = {"evt": 0.25, "svj": 0.20, "hawkes": 0.20, "regime": 0.20, "news": 0.15}
CIRCUIT_BREAKER_THRESHOLD = 90
CACHE_TTL_SECONDS = 5.0
DECISION_VALIDITY_SECONDS = 30.0

_cache: dict[str, dict[str, Any]] = {}


def _score_evt(evt_data: dict, alpha: float = 0.005) -> dict:
    """Score EVT tail risk (0-100). Higher = more dangerous."""
    from extreme_value_theory import evt_var

    var_loss = evt_var(
        xi=evt_data["xi"], beta=evt_data["beta"],
        threshold=evt_data["threshold"],
        n_total=evt_data["n_total"],
        n_exceedances=evt_data["n_exceedances"],
        alpha=alpha,
    )
    # Score: map VaR loss to 0-100. VaR < 5% → low risk, VaR > 15% → extreme
    score = min(100.0, max(0.0, (var_loss - 2.0) / 13.0 * 100.0))
    return {
        "component": "evt",
        "score": round(score, 2),
        "details": {
            "var_995": round(var_loss, 4),
            "xi": round(evt_data["xi"], 4),
            "threshold": round(evt_data["threshold"], 4),
        },
    }


def _score_svj(svj_data: dict) -> dict:
    """Score SVJ jump risk (0-100). Veto-worthy if jump_share > 60%."""
    from svj_model import decompose_risk

    risk = decompose_risk(svj_data["returns"], svj_data["calibration"])
    jump_share = risk["jump_share_pct"]
    lambda_ = svj_data["calibration"]["lambda_"]

    # Base score from jump share: 0% → 0, 100% → 80
    base = min(80.0, jump_share * 0.8)
    # Bonus if jump intensity is high (λ > 50 jumps/year)
    intensity_bonus = min(20.0, max(0.0, (lambda_ - 20) / 80 * 20))
    score = min(100.0, base + intensity_bonus)

    return {
        "component": "svj",
        "score": round(score, 2),
        "details": {
            "jump_share_pct": round(jump_share, 2),
            "lambda": round(lambda_, 4),
            "daily_jump_vol": risk["daily_jump_vol"],
        },
    }


def _score_hawkes(hawkes_data: dict) -> dict:
    """Score Hawkes contagion risk (0-100). Block if contagion > 0.75."""
    from hawkes_process import detect_flash_crash_risk

    risk = detect_flash_crash_risk(hawkes_data["event_times"], hawkes_data)
    # contagion_risk_score is already 0-1, map to 0-100
    score = risk["contagion_risk_score"] * 100.0

    return {
        "component": "hawkes",
        "score": round(score, 2),
        "details": {
            "contagion_risk_score": round(risk["contagion_risk_score"], 4),
            "risk_level": risk["risk_level"],
            "intensity_ratio": round(risk["intensity_ratio"], 4),
            "recent_events": risk["recent_event_count"],
        },
    }


def _score_regime(model_data: dict) -> dict:
    """Score regime risk (0-100). Crisis regime → high score."""
    filter_probs = model_data["filter_probs"]
    num_states = model_data["calibration"]["num_states"]

    pi_t = np.asarray(
        filter_probs.iloc[-1] if hasattr(filter_probs, "iloc") else filter_probs[-1]
    )
    current_regime = int(np.argmax(pi_t)) + 1  # 1-based
    crisis_prob = float(pi_t[-1]) if len(pi_t) > 0 else 0.0

    # Score: weighted by regime position and crisis probability
    # Regime 1 → 0, Regime K → 80, plus crisis_prob bonus up to 20
    regime_base = (current_regime - 1) / max(num_states - 1, 1) * 80.0
    crisis_bonus = crisis_prob * 20.0
    score = min(100.0, regime_base + crisis_bonus)

    return {
        "component": "regime",
        "score": round(score, 2),
        "details": {
            "current_regime": current_regime,
            "num_states": num_states,
            "crisis_probability": round(crisis_prob, 4),
            "regime_probs": [round(float(p), 4) for p in pi_t],
        },
    }


def _score_news(news_signal: dict, direction: str) -> dict:
    """
    Score news sentiment risk (0-100) relative to trade direction.

    For LONG trades: bearish news = high risk, bullish = low risk.
    For SHORT trades: bullish news = high risk, bearish = low risk.

    Factors: sentiment_ewma, strength, confidence, entropy.
    """
    ewma = news_signal.get("sentiment_ewma", 0.0)
    strength = news_signal.get("strength", 0.0)
    confidence = news_signal.get("confidence", 0.0)
    entropy = news_signal.get("entropy", 1.585)
    signal_direction = news_signal.get("direction", "NEUTRAL")
    n_items = news_signal.get("n_items", 0)

    if n_items == 0:
        return {
            "component": "news",
            "score": round(50.0, 2),
            "details": {
                "sentiment_ewma": 0.0,
                "signal_direction": "NEUTRAL",
                "strength": 0.0,
                "confidence": 0.0,
                "entropy": entropy,
                "n_items": 0,
                "direction_conflict": False,
            },
        }

    # Determine if news opposes the trade direction
    is_long = direction.lower() == "long"
    # effective_sentiment: positive = favorable for trade, negative = against
    effective_sentiment = ewma if is_long else -ewma

    # Base score: map effective_sentiment from [-1, 1] to [0, 100]
    # -1 (strongly against) → 100, 0 (neutral) → 50, +1 (strongly for) → 0
    base_score = 50.0 - effective_sentiment * 50.0

    # Scale by confidence: low confidence → pull toward 50 (uncertain)
    score = 50.0 + (base_score - 50.0) * confidence

    # Entropy penalty: high entropy (disagreement) → pull toward 50
    # max entropy for 3 categories = log2(3) ≈ 1.585
    consensus = max(0.0, 1.0 - entropy / 1.585)
    score = 50.0 + (score - 50.0) * (0.5 + 0.5 * consensus)

    score = round(min(100.0, max(0.0, score)), 2)

    direction_conflict = (
        (is_long and signal_direction == "SHORT")
        or (not is_long and signal_direction == "LONG")
    )

    return {
        "component": "news",
        "score": score,
        "details": {
            "sentiment_ewma": round(ewma, 4),
            "signal_direction": signal_direction,
            "strength": round(strength, 4),
            "confidence": round(confidence, 4),
            "entropy": round(entropy, 4),
            "n_items": n_items,
            "direction_conflict": direction_conflict,
        },
    }


def _recommend_size(
    requested_size: float, risk_score: float, current_regime: int, num_states: int
) -> float:
    """Scale position size down based on risk score and regime."""
    # Linear scaling: risk_score 0 → 100% of requested, risk_score 100 → 0%
    scale = max(0.0, 1.0 - risk_score / 100.0)
    # Regime penalty: crisis regime gets additional 50% haircut
    if current_regime >= num_states:
        scale *= 0.5
    elif current_regime >= num_states - 1:
        scale *= 0.75
    return round(requested_size * scale, 2)


def assess_trade(
    token: str,
    trade_size_usd: float,
    direction: str,
    model_data: dict | None,
    evt_data: dict | None,
    svj_data: dict | None,
    hawkes_data: dict | None,
    news_data: dict | None = None,
) -> dict:
    """Run all risk components and return composite Guardian assessment."""
    cache_key = f"{token}:{direction}"
    now = time.time()

    if cache_key in _cache and (now - _cache[cache_key]["ts"]) < CACHE_TTL_SECONDS:
        cached = _cache[cache_key]["result"].copy()
        cached["from_cache"] = True
        return cached

    scores: list[dict] = []
    veto_reasons: list[str] = []
    available_weights = 0.0

    # EVT component (25%)
    if evt_data:
        evt_score = _score_evt(evt_data)
        scores.append(evt_score)
        available_weights += WEIGHTS["evt"]
        if evt_score["score"] > CIRCUIT_BREAKER_THRESHOLD:
            veto_reasons.append("evt_extreme_tail")

    # SVJ component (20%)
    if svj_data:
        svj_score = _score_svj(svj_data)
        scores.append(svj_score)
        available_weights += WEIGHTS["svj"]
        if svj_score["score"] > CIRCUIT_BREAKER_THRESHOLD:
            veto_reasons.append("svj_jump_crisis")
        if svj_score["details"]["jump_share_pct"] > 60:
            veto_reasons.append("svj_high_jump_share")

    # Hawkes component (20%)
    if hawkes_data:
        hawkes_score = _score_hawkes(hawkes_data)
        scores.append(hawkes_score)
        available_weights += WEIGHTS["hawkes"]
        if hawkes_score["score"] > CIRCUIT_BREAKER_THRESHOLD:
            veto_reasons.append("hawkes_critical_contagion")
        if hawkes_score["details"]["contagion_risk_score"] > 0.75:
            veto_reasons.append("hawkes_flash_crash_risk")

    # Regime component (20%)
    current_regime = 1
    num_states = 5
    if model_data:
        regime_score = _score_regime(model_data)
        scores.append(regime_score)
        available_weights += WEIGHTS["regime"]
        current_regime = regime_score["details"]["current_regime"]
        num_states = regime_score["details"]["num_states"]
        if regime_score["score"] > CIRCUIT_BREAKER_THRESHOLD:
            veto_reasons.append("regime_extreme_crisis")

    # News component (15%)
    if news_data:
        news_score = _score_news(news_data, direction)
        scores.append(news_score)
        available_weights += WEIGHTS["news"]
        if news_score["score"] > CIRCUIT_BREAKER_THRESHOLD:
            veto_reasons.append("news_extreme_negative")
        if news_score["details"]["direction_conflict"] and news_score["details"]["strength"] > 0.6:
            veto_reasons.append("news_strong_direction_conflict")

    # Composite score (re-normalize weights to available components)
    if available_weights > 0 and scores:
        weight_map = {s["component"]: WEIGHTS[s["component"]] for s in scores}
        total_w = sum(weight_map.values())
        risk_score = sum(
            s["score"] * weight_map[s["component"]] / total_w for s in scores
        )
    else:
        risk_score = 50.0

    risk_score = round(min(100.0, max(0.0, risk_score)), 2)
    approved = len(veto_reasons) == 0 and risk_score < 75.0
    confidence = round(available_weights / sum(WEIGHTS.values()), 4)
    recommended_size = _recommend_size(
        trade_size_usd, risk_score, current_regime, num_states
    )
    expires_at = datetime.fromtimestamp(
        now + DECISION_VALIDITY_SECONDS, tz=timezone.utc
    )

    result = {
        "approved": approved,
        "risk_score": risk_score,
        "veto_reasons": list(set(veto_reasons)),
        "recommended_size": recommended_size,
        "regime_state": current_regime,
        "confidence": confidence,
        "expires_at": expires_at.isoformat(),
        "component_scores": scores,
        "from_cache": False,
    }

    logger.info(
        "guardian_decision token=%s direction=%s size=%.2f approved=%s "
        "risk_score=%.2f veto_reasons=%s regime=%d confidence=%.2f",
        token, direction, trade_size_usd, approved,
        risk_score, veto_reasons, current_regime, confidence,
    )

    _cache[cache_key] = {"ts": now, "result": result}
    return result

