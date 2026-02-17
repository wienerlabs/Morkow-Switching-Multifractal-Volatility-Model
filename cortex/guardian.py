"""
Guardian Integration Layer — Unified risk veto endpoint for Cortex autonomous trading.

Consolidates EVT (25%), SVJ (20%), Hawkes (20%), Regime (20%), and News (15%)
risk assessments into a single composite score with circuit breaker logic
and position sizing.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any

import numpy as np

from cortex.config import (
    GUARDIAN_WEIGHTS,
    CIRCUIT_BREAKER_THRESHOLD,
    CACHE_TTL_SECONDS,
    DECISION_VALIDITY_SECONDS,
    APPROVAL_THRESHOLD,
    EVT_SCORE_FLOOR,
    EVT_SCORE_RANGE,
    SVJ_BASE_CAP,
    SVJ_BASE_MULT,
    SVJ_INTENSITY_FLOOR,
    SVJ_INTENSITY_RANGE,
    SVJ_INTENSITY_CAP,
    REGIME_BASE_MAX,
    REGIME_CRISIS_BONUS_MAX,
    CRISIS_REGIME_HAIRCUT,
    NEAR_CRISIS_REGIME_HAIRCUT,
    GUARDIAN_KELLY_FRACTION,
    GUARDIAN_KELLY_MIN_TRADES,
    ALAMS_SCORE_VAR_FLOOR,
    ALAMS_SCORE_VAR_CEILING,
    ALAMS_CRISIS_REGIME_BONUS,
    ALAMS_HIGH_DELTA_BONUS,
    ALAMS_HIGH_DELTA_THRESHOLD,
    # Task 1-8 feature flags
    CALIBRATION_ENABLED,
    KELLY_REGIME_AWARE,
    KELLY_REGIME_FRACTIONS,
    ADAPTIVE_WEIGHTS_ENABLED,
    HAWKES_TIMING_GATE_ENABLED,
    HAWKES_TIMING_KAPPA,
    REGIME_THRESHOLD_SCALING_ENABLED,
    REGIME_THRESHOLD_LAMBDA,
    COPULA_RISK_GATE_ENABLED,
    TAIL_DEPENDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

WEIGHTS = GUARDIAN_WEIGHTS


class _TTLCache:
    """Simple TTL cache compatible with cashews patterns.

    Provides dict-like .clear() for test backward compat while
    auto-expiring entries after CACHE_TTL_SECONDS.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        if (time.time() - entry["ts"]) >= CACHE_TTL_SECONDS:
            del self._store[key]
            return None
        return entry["result"]

    def set(self, key: str, result: dict) -> None:
        self._store[key] = {"ts": time.time(), "result": result}

    def clear(self) -> None:
        self._store.clear()

    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


_cache = _TTLCache()
_trade_history: deque[dict] = deque(maxlen=500)


def _score_evt(evt_data: dict, alpha: float = 0.005) -> dict:
    """Score EVT tail risk (0-100). Higher = more dangerous."""
    from cortex.evt import evt_var

    var_loss = evt_var(
        xi=evt_data["xi"], beta=evt_data["beta"],
        threshold=evt_data["threshold"],
        n_total=evt_data["n_total"],
        n_exceedances=evt_data["n_exceedances"],
        alpha=alpha,
    )
    # Score: map VaR loss to 0-100. VaR < 5% → low risk, VaR > 15% → extreme
    score = min(100.0, max(0.0, (var_loss - EVT_SCORE_FLOOR) / EVT_SCORE_RANGE * 100.0))
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
    from cortex.svj import decompose_risk

    risk = decompose_risk(svj_data["returns"], svj_data["calibration"])
    jump_share = risk["jump_share_pct"]
    lambda_ = svj_data["calibration"]["lambda_"]

    # Base score from jump share: 0% → 0, 100% → 80
    base = min(SVJ_BASE_CAP, jump_share * SVJ_BASE_MULT)
    intensity_bonus = min(SVJ_INTENSITY_CAP, max(0.0, (lambda_ - SVJ_INTENSITY_FLOOR) / SVJ_INTENSITY_RANGE * SVJ_INTENSITY_CAP))
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
    from cortex.hawkes import detect_flash_crash_risk

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
    regime_base = (current_regime - 1) / max(num_states - 1, 1) * REGIME_BASE_MAX
    crisis_bonus = crisis_prob * REGIME_CRISIS_BONUS_MAX
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


def _score_alams(alams_data: dict) -> dict:
    """
    Score A-LAMS VaR risk (0-100).

    Maps VaR magnitude, regime state, and asymmetry parameter δ to a
    composite risk score for Guardian integration.

    Scoring curve:
      - VaR(95%) < FLOOR (1%) → low risk (score ≈ 0)
      - VaR(95%) in [FLOOR, CEILING] → linear interpolation (0 → 80)
      - VaR(95%) > CEILING (8%) → extreme risk (score 80+)
      - Crisis regime (index >= 4) → +CRISIS_BONUS (20)
      - High asymmetry δ > THRESHOLD (0.3) → +DELTA_BONUS (10)
        (indicates strong negative-return → high-vol transition pressure)

    Expected alams_data keys:
      var_total: float      — total VaR (regime + slippage)
      current_regime: int   — most likely regime index (0-based)
      regime_probs: list     — K-length probability vector
      delta: float          — asymmetry parameter
      regime_sigmas: list   — per-regime volatilities (optional, for detail)
    """
    var_total = alams_data.get("var_total", 0.0)
    current_regime = alams_data.get("current_regime", 0)
    delta = alams_data.get("delta", 0.0)
    regime_probs = alams_data.get("regime_probs", [])

    # Base score: map VaR from [floor, ceiling] → [0, 80]
    var_range = ALAMS_SCORE_VAR_CEILING - ALAMS_SCORE_VAR_FLOOR
    if var_range > 0:
        base = (var_total - ALAMS_SCORE_VAR_FLOOR) / var_range * 80.0
    else:
        base = 0.0
    base = max(0.0, min(80.0, base))

    # Crisis regime bonus: regime index 4 (highest in 5-regime model)
    crisis_bonus = 0.0
    if current_regime >= 4:
        crisis_bonus = ALAMS_CRISIS_REGIME_BONUS
    elif current_regime >= 3:
        # Near-crisis: partial bonus scaled by high-regime probability
        high_prob = float(regime_probs[-1]) if regime_probs else 0.0
        crisis_bonus = ALAMS_CRISIS_REGIME_BONUS * 0.5 * high_prob

    # High-delta bonus: strong asymmetry implies market stress
    delta_bonus = 0.0
    if delta > ALAMS_HIGH_DELTA_THRESHOLD:
        excess = (delta - ALAMS_HIGH_DELTA_THRESHOLD) / (0.5 - ALAMS_HIGH_DELTA_THRESHOLD)
        delta_bonus = min(ALAMS_HIGH_DELTA_BONUS, ALAMS_HIGH_DELTA_BONUS * excess)

    score = min(100.0, base + crisis_bonus + delta_bonus)

    return {
        "component": "alams",
        "score": round(score, 2),
        "details": {
            "var_total": round(var_total, 6),
            "var_total_pct": round(var_total * 100, 2),
            "current_regime": current_regime,
            "delta": round(delta, 4),
            "base_score": round(base, 2),
            "crisis_bonus": round(crisis_bonus, 2),
            "delta_bonus": round(delta_bonus, 2),
            "regime_probs": [round(float(p), 4) for p in regime_probs] if regime_probs else [],
        },
    }


def record_trade_outcome(
    pnl: float,
    size: float,
    token: str = "",
    regime: int = -1,
    component_scores: list[dict] | None = None,
    risk_score: float = 0.0,
) -> None:
    """Record a completed trade for Kelly Criterion and adaptive weights."""
    _trade_history.append({
        "pnl": pnl, "size": size, "token": token,
        "regime": regime, "ts": time.time(),
    })

    # Task 3: Update adaptive weights
    if ADAPTIVE_WEIGHTS_ENABLED and component_scores:
        try:
            from cortex.adaptive_weights import record_outcome
            record_outcome(component_scores, risk_score, pnl)
        except Exception:
            logger.debug("Adaptive weight update failed", exc_info=True)


def _compute_kelly(current_regime: int = -1) -> dict[str, Any]:
    """Compute Kelly fraction from trade history.

    Task 2: When KELLY_REGIME_AWARE is enabled, computes separate Kelly
    fractions per regime and uses the regime-specific fractional multiplier.
    """
    n = len(_trade_history)
    if n < GUARDIAN_KELLY_MIN_TRADES:
        return {"active": False, "n_trades": n, "reason": f"need {GUARDIAN_KELLY_MIN_TRADES} trades, have {n}"}

    wins = [t for t in _trade_history if t["pnl"] > 0]
    losses = [t for t in _trade_history if t["pnl"] <= 0]

    if not losses:
        return {"active": False, "n_trades": n, "reason": "no losses yet"}

    p = len(wins) / n
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = abs(sum(t["pnl"] for t in losses) / len(losses))

    if avg_loss == 0:
        return {"active": False, "n_trades": n, "reason": "avg_loss is zero"}

    b = avg_win / avg_loss
    q = 1.0 - p
    kelly_full = (p * b - q) / b if b > 0 else 0.0

    # Task 2: Regime-conditional Kelly fraction
    fraction_used = GUARDIAN_KELLY_FRACTION
    regime_stats = None
    if KELLY_REGIME_AWARE and current_regime >= 0:
        regime_key = str(current_regime)
        fraction_used = KELLY_REGIME_FRACTIONS.get(regime_key, GUARDIAN_KELLY_FRACTION)

        # Compute regime-specific win rate for diagnostics
        regime_trades = [t for t in _trade_history if t.get("regime") == current_regime]
        if len(regime_trades) >= 10:
            r_wins = [t for t in regime_trades if t["pnl"] > 0]
            regime_stats = {
                "regime": current_regime,
                "n_trades": len(regime_trades),
                "win_rate": round(len(r_wins) / len(regime_trades), 4),
                "fraction_override": round(fraction_used, 4),
            }

    kelly_fraction = max(0.0, kelly_full * fraction_used)

    result: dict[str, Any] = {
        "active": True,
        "n_trades": n,
        "win_rate": round(p, 4),
        "win_loss_ratio": round(b, 4),
        "kelly_full": round(kelly_full, 4),
        "kelly_fraction": round(kelly_fraction, 4),
        "fraction_used": fraction_used,
    }
    if regime_stats:
        result["regime_stats"] = regime_stats
    return result


def get_kelly_stats() -> dict[str, Any]:
    """Public accessor for Kelly stats."""
    return _compute_kelly()


def _recommend_size(
    requested_size: float, risk_score: float, current_regime: int, num_states: int
) -> float:
    """Scale position size using Kelly fraction (if active) or linear fallback."""
    kelly = _compute_kelly(current_regime=current_regime)

    if kelly["active"]:
        kelly_f = kelly["kelly_fraction"]
        risk_scale = max(0.0, 1.0 - risk_score / 100.0)
        scale = kelly_f * risk_scale
    else:
        scale = max(0.0, 1.0 - risk_score / 100.0)

    if current_regime >= num_states:
        scale *= CRISIS_REGIME_HAIRCUT
    elif current_regime >= num_states - 1:
        scale *= NEAR_CRISIS_REGIME_HAIRCUT

    return round(requested_size * max(0.0, scale), 2)


def assess_trade(
    token: str,
    trade_size_usd: float,
    direction: str,
    model_data: dict | None,
    evt_data: dict | None,
    svj_data: dict | None,
    hawkes_data: dict | None,
    news_data: dict | None = None,
    alams_data: dict | None = None,
    strategy: str | None = None,
    run_debate: bool = False,
) -> dict:
    """Run all risk components and return composite Guardian assessment."""
    cache_key = f"{token}:{direction}"
    now = time.time()

    cached_result = _cache.get(cache_key)
    if cached_result is not None:
        cached = cached_result.copy()
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

    # News component (10%)
    if news_data:
        news_score = _score_news(news_data, direction)
        scores.append(news_score)
        available_weights += WEIGHTS["news"]
        if news_score["score"] > CIRCUIT_BREAKER_THRESHOLD:
            veto_reasons.append("news_extreme_negative")
        if news_score["details"]["direction_conflict"] and news_score["details"]["strength"] > 0.6:
            veto_reasons.append("news_strong_direction_conflict")

    # A-LAMS VaR component (25%)
    if alams_data:
        alams_score = _score_alams(alams_data)
        scores.append(alams_score)
        available_weights += WEIGHTS["alams"]
        if alams_score["score"] > CIRCUIT_BREAKER_THRESHOLD:
            veto_reasons.append("alams_extreme_var")
        if alams_data.get("current_regime", 0) >= 4:
            veto_reasons.append("alams_crisis_regime")

    # ── Task 3: Use adaptive weights if enabled ──
    active_weights = WEIGHTS
    if ADAPTIVE_WEIGHTS_ENABLED:
        try:
            from cortex.adaptive_weights import get_weights
            active_weights = get_weights()
        except Exception:
            logger.debug("Adaptive weights unavailable, using static", exc_info=True)

    # Composite score (re-normalize weights to available components)
    if available_weights > 0 and scores:
        weight_map = {s["component"]: active_weights.get(s["component"], WEIGHTS.get(s["component"], 0.0)) for s in scores}
        total_w = sum(weight_map.values())
        risk_score = sum(
            s["score"] * weight_map[s["component"]] / total_w for s in scores
        )
    else:
        risk_score = 50.0

    risk_score = round(min(100.0, max(0.0, risk_score)), 2)

    # ── Task 5: Regime-conditional entry thresholds ──
    effective_threshold = APPROVAL_THRESHOLD
    if REGIME_THRESHOLD_SCALING_ENABLED and model_data:
        try:
            regime_sigma = model_data.get("calibration", {}).get("sigma_states", [])
            if regime_sigma and current_regime >= 1:
                sigma_current = regime_sigma[min(current_regime - 1, len(regime_sigma) - 1)]
                sigma_median = float(np.median(regime_sigma))
                if sigma_median > 0:
                    ratio = sigma_current / sigma_median
                    effective_threshold = APPROVAL_THRESHOLD * (1.0 + REGIME_THRESHOLD_LAMBDA * (ratio - 1.0))
                    effective_threshold = max(50.0, min(95.0, effective_threshold))
        except Exception:
            logger.debug("Regime threshold scaling failed", exc_info=True)

    # ── Circuit breaker check ──
    from cortex.circuit_breaker import is_blocked, record_score

    cb_states = record_score(risk_score, strategy=strategy)
    blocked, blockers = is_blocked(strategy=strategy)
    if blocked:
        veto_reasons.extend(f"circuit_breaker_{b}" for b in blockers)

    # ── Portfolio risk check ──
    try:
        from cortex.portfolio_risk import check_limits
        limits = check_limits(token)
        if limits["blocked"]:
            veto_reasons.extend(f"portfolio_{b}" for b in limits["blockers"])
    except Exception:
        limits = None
        logger.debug("Portfolio risk check unavailable", exc_info=True)

    # ── Task 4: Hawkes intensity-gated trade timing ──
    hawkes_deferred = False
    if HAWKES_TIMING_GATE_ENABLED and hawkes_data:
        try:
            from cortex.hawkes import hawkes_intensity as _hawkes_intensity
            h_params = hawkes_data
            events = hawkes_data.get("event_times", np.array([]))
            if len(events) > 0 and h_params.get("mu") and h_params.get("alpha") and h_params.get("beta"):
                h_info = _hawkes_intensity(events, h_params)
                mu = h_params["mu"]
                alpha_h = h_params["alpha"]
                beta_h = h_params["beta"]
                steady_state = mu + alpha_h / beta_h if beta_h > 0 else mu
                gate_threshold = mu + HAWKES_TIMING_KAPPA * (steady_state - mu)
                if h_info["current_intensity"] > gate_threshold:
                    hawkes_deferred = True
                    veto_reasons.append("hawkes_intensity_gate_defer")
        except Exception:
            logger.debug("Hawkes timing gate check failed", exc_info=True)

    # ── Task 8: Copula tail-dependence risk gate ──
    copula_gate_triggered = False
    if COPULA_RISK_GATE_ENABLED:
        try:
            from cortex.copula import _tail_dependence
            # Check if a recent copula fit is available via model_data
            copula_info = (model_data or {}).get("copula_fit")
            if copula_info and copula_info.get("family") in ("clayton", "student_t"):
                td = _tail_dependence(copula_info["family"], copula_info.get("params", {}))
                lambda_lower = td.get("lambda_lower", 0.0)
                if lambda_lower > TAIL_DEPENDENCE_THRESHOLD:
                    copula_gate_triggered = True
                    veto_reasons.append(f"copula_tail_dependence_{lambda_lower:.3f}")
        except Exception:
            logger.debug("Copula risk gate check failed", exc_info=True)

    approved = len(veto_reasons) == 0 and risk_score < effective_threshold
    confidence = round(available_weights / sum(WEIGHTS.values()), 4)

    # ── Task 1: Calibrate approval confidence ──
    calibrated_confidence = None
    if CALIBRATION_ENABLED:
        try:
            from cortex.calibration_bridge import calibrate_probability
            raw_approval_prob = max(0.0, 1.0 - risk_score / 100.0)
            calibrated_confidence = round(calibrate_probability(raw_approval_prob), 4)
        except Exception:
            logger.debug("Calibration bridge unavailable", exc_info=True)

    recommended_size = _recommend_size(
        trade_size_usd, risk_score, current_regime, num_states
    )
    expires_at = datetime.fromtimestamp(
        now + DECISION_VALIDITY_SECONDS, tz=timezone.utc
    )

    # ── Adversarial debate (optional) ──
    debate_result = None
    if run_debate:
        try:
            from cortex.debate import run_debate as _run_debate
            debate_result = _run_debate(
                risk_score=risk_score,
                component_scores=scores,
                veto_reasons=list(set(veto_reasons)),
                direction=direction,
                trade_size_usd=trade_size_usd,
                original_approved=approved,
                strategy=strategy or "spot",
                alams_data=alams_data,
            )
            if debate_result["decision_changed"]:
                approved = debate_result["final_decision"] == "approve"
        except Exception:
            logger.debug("Debate system unavailable", exc_info=True)

    result = {
        "approved": approved,
        "risk_score": risk_score,
        "veto_reasons": list(set(veto_reasons)),
        "recommended_size": recommended_size,
        "regime_state": current_regime,
        "confidence": confidence,
        "calibrated_confidence": calibrated_confidence,
        "effective_threshold": round(effective_threshold, 2),
        "hawkes_deferred": hawkes_deferred,
        "copula_gate_triggered": copula_gate_triggered,
        "expires_at": expires_at.isoformat(),
        "component_scores": scores,
        "circuit_breaker": cb_states,
        "portfolio_limits": limits,
        "debate": debate_result,
        "from_cache": False,
    }

    logger.info(
        "guardian_decision token=%s direction=%s size=%.2f approved=%s "
        "risk_score=%.2f veto_reasons=%s regime=%d confidence=%.2f",
        token, direction, trade_size_usd, approved,
        risk_score, veto_reasons, current_regime, confidence,
    )

    _cache.set(cache_key, result)
    return result

