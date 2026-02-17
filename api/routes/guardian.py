"""Guardian (Unified Risk Veto) endpoints — assess, Kelly stats, circuit breakers, debate."""

import logging

from fastapi import APIRouter, HTTPException

from api.models import (
    CircuitBreakerItem,
    CircuitBreakersResponse,
    DebateResponse,
    GuardianAssessRequest,
    GuardianAssessResponse,
    GuardianComponentScore,
    KellyStatsResponse,
    TradeOutcomeRequest,
)
from api.stores import (
    _evt_store,
    _hawkes_store,
    _model_store,
    _svj_store,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["guardian"])


@router.post("/guardian/assess", response_model=GuardianAssessResponse)
def guardian_assess(req: GuardianAssessRequest):
    from cortex.guardian import _cache as guardian_cache
    from cortex.guardian import assess_trade

    if req.urgency:
        cache_key = f"{req.token}:{req.direction}"
        guardian_cache.pop(cache_key, None)

    model_data = _model_store.get(req.token)
    evt_data = _evt_store.get(req.token)
    svj_data = _svj_store.get(req.token)
    hawkes_data = _hawkes_store.get(req.token)

    if not any([model_data, evt_data, svj_data, hawkes_data]):
        raise HTTPException(
            status_code=404,
            detail=f"No calibrated models for '{req.token}'. "
            "Calibrate at least one model first (MSM, EVT, SVJ, or Hawkes).",
        )

    news_data = None
    try:
        from cortex.news import news_buffer
        news_data = news_buffer.get_signal()
    except Exception:
        logger.debug("News buffer unavailable for guardian assessment", exc_info=True)

    result = assess_trade(
        token=req.token,
        trade_size_usd=req.trade_size_usd,
        direction=req.direction,
        model_data=model_data,
        evt_data=evt_data,
        svj_data=svj_data,
        hawkes_data=hawkes_data,
        news_data=news_data,
        strategy=req.strategy,
        run_debate=req.run_debate,
        agent_confidence=req.agent_confidence,
    )

    # Broadcast to SSE subscribers
    try:
        from api.routes.streams import broadcast_guardian_score
        broadcast_guardian_score({
            "token": req.token,
            "direction": req.direction,
            "risk_score": result["risk_score"],
            "approved": result["approved"],
            "regime_state": result["regime_state"],
            "veto_reasons": result["veto_reasons"],
        })
    except Exception:
        logger.debug("Guardian SSE broadcast failed", exc_info=True)

    return GuardianAssessResponse(
        approved=result["approved"],
        risk_score=result["risk_score"],
        veto_reasons=result["veto_reasons"],
        recommended_size=result["recommended_size"],
        regime_state=result["regime_state"],
        confidence=result["confidence"],
        calibrated_confidence=result.get("calibrated_confidence"),
        effective_threshold=result.get("effective_threshold", 75.0),
        hawkes_deferred=result.get("hawkes_deferred", False),
        copula_gate_triggered=result.get("copula_gate_triggered", False),
        expires_at=result["expires_at"],
        component_scores=[
            GuardianComponentScore(**s) for s in result["component_scores"]
        ],
        circuit_breaker=result.get("circuit_breaker"),
        portfolio_limits=result.get("portfolio_limits"),
        debate=result.get("debate"),
        from_cache=result["from_cache"],
    )


@router.get("/guardian/kelly-stats", response_model=KellyStatsResponse)
def get_kelly_stats():
    from cortex.guardian import get_kelly_stats as _get_stats
    return KellyStatsResponse(**_get_stats())


@router.post("/guardian/trade-outcome")
def record_trade_outcome(req: TradeOutcomeRequest):
    from cortex.guardian import record_trade_outcome as _record
    _record(
        pnl=req.pnl,
        size=req.size,
        token=req.token,
        regime=req.regime,
        component_scores=req.component_scores,
        risk_score=req.risk_score,
    )

    cb_status = None
    if req.strategy:
        try:
            from cortex.debate import record_debate_outcome
            record_debate_outcome(
                strategy=req.strategy,
                approved=req.pnl > 0,
                pnl=req.pnl,
            )
        except Exception:
            logger.debug("Debate outcome recording failed", exc_info=True)

        # Cross-wire: also feed the outcome circuit breaker
        try:
            from cortex.circuit_breaker import record_trade_outcome as _cb_record
            cb_status = _cb_record(
                strategy=req.strategy,
                success=req.pnl > 0,
                pnl=req.pnl,
                loss_type="market_loss" if req.pnl <= 0 else "",
                details=f"via /trade-outcome: token={req.token}",
            )
        except Exception:
            logger.debug("CB outcome recording failed", exc_info=True)

    return {
        "status": "recorded",
        "pnl": req.pnl,
        "size": req.size,
        "token": req.token,
        "regime": req.regime,
        "strategy": req.strategy,
        "circuit_breaker": cb_status,
    }


@router.get("/guardian/circuit-breakers")
def get_circuit_breakers():
    import time
    from cortex.circuit_breaker import get_all_states
    states = get_all_states()
    return {"breakers": states, "timestamp": time.time()}


@router.post("/guardian/circuit-breakers/reset")
def reset_circuit_breakers(name: str | None = None):
    from cortex.circuit_breaker import reset_all, reset_breaker
    if name:
        ok = reset_breaker(name)
        if not ok:
            raise HTTPException(status_code=404, detail=f"Breaker '{name}' not found")
        return {"status": "reset", "breaker": name}
    reset_all()
    return {"status": "all_reset"}


@router.post("/guardian/debate")
def run_debate_endpoint(req: GuardianAssessRequest):
    """Run adversarial debate for a trade proposal (standalone, without full assess)."""
    from cortex.debate import run_debate
    from cortex.guardian import (
        WEIGHTS,
        _score_evt,
        _score_hawkes,
        _score_news,
        _score_regime,
        _score_svj,
    )

    model_data = _model_store.get(req.token)
    evt_data = _evt_store.get(req.token)
    svj_data = _svj_store.get(req.token)
    hawkes_data = _hawkes_store.get(req.token)

    scores: list[dict] = []
    veto_reasons: list[str] = []
    available_weights = 0.0

    if evt_data:
        try:
            s = _score_evt(evt_data)
            scores.append(s)
            available_weights += WEIGHTS["evt"]
            if s["score"] > 90:
                veto_reasons.append("evt_extreme_tail")
        except Exception:
            logger.debug("EVT scoring failed in debate endpoint", exc_info=True)

    if svj_data:
        try:
            s = _score_svj(svj_data)
            scores.append(s)
            available_weights += WEIGHTS["svj"]
            if s["score"] > 90:
                veto_reasons.append("svj_jump_crisis")
            if s["details"]["jump_share_pct"] > 60:
                veto_reasons.append("svj_high_jump_share")
        except Exception:
            logger.debug("SVJ scoring failed in debate endpoint", exc_info=True)

    if hawkes_data:
        try:
            s = _score_hawkes(hawkes_data)
            scores.append(s)
            available_weights += WEIGHTS["hawkes"]
            if s["score"] > 90:
                veto_reasons.append("hawkes_critical_contagion")
            if s["details"]["contagion_risk_score"] > 0.75:
                veto_reasons.append("hawkes_flash_crash_risk")
        except Exception:
            logger.debug("Hawkes scoring failed in debate endpoint", exc_info=True)

    if model_data:
        try:
            s = _score_regime(model_data)
            scores.append(s)
            available_weights += WEIGHTS["regime"]
            if s["score"] > 90:
                veto_reasons.append("regime_extreme_crisis")
        except Exception:
            logger.debug("Regime scoring failed in debate endpoint", exc_info=True)

    # Read news from background buffer (no latency)
    news_data = None
    try:
        from cortex.news import news_buffer
        news_data = news_buffer.get_signal()
    except Exception:
        logger.debug("News buffer unavailable for debate endpoint", exc_info=True)

    if news_data:
        try:
            s = _score_news(news_data, req.direction)
            scores.append(s)
            available_weights += WEIGHTS["news"]
            if s["score"] > 90:
                veto_reasons.append("news_extreme_negative")
        except Exception:
            logger.debug("News scoring failed in debate endpoint", exc_info=True)

    # Compute weighted risk score
    if available_weights > 0 and scores:
        total_w = sum(WEIGHTS.get(s["component"], 0.0) for s in scores)
        if total_w > 0:
            risk_score = sum(
                s["score"] * WEIGHTS.get(s["component"], 0.0) / total_w
                for s in scores
            )
            risk_score = round(min(100.0, max(0.0, risk_score)), 2)
        else:
            risk_score = 50.0
    else:
        risk_score = 50.0

    approved = len(veto_reasons) == 0 and risk_score < 75.0

    result = run_debate(
        risk_score=risk_score,
        component_scores=scores,
        veto_reasons=veto_reasons,
        direction=req.direction,
        trade_size_usd=req.trade_size_usd,
        original_approved=approved,
        strategy=req.strategy or "spot",
    )
    return result


@router.post("/guardian/trade-outcome/strategy")
def record_strategy_trade_outcome(
    strategy: str,
    success: bool,
    pnl: float = 0.0,
    loss_type: str = "",
    details: str = "",
):
    """Record a trade outcome for strategy-specific circuit breakers.

    This feeds the outcome-based circuit breakers:
    - LP: 3 consecutive IL → pause
    - Arb: 5 consecutive failed executions → pause
    - Perp: 2 consecutive stop-losses → pause
    """
    from cortex.circuit_breaker import record_trade_outcome
    result = record_trade_outcome(
        strategy=strategy, success=success, pnl=pnl,
        loss_type=loss_type, details=details,
    )
    return result


@router.get("/guardian/circuit-breakers/outcomes")
def get_outcome_circuit_breakers():
    """Get status of outcome-based circuit breakers only."""
    import time as _time
    from cortex.circuit_breaker import get_outcome_states
    states = get_outcome_states()
    return {"outcome_breakers": states, "timestamp": _time.time()}


# ── Debate Transcript Store Endpoints ────────────────────────────────────────

@router.get("/guardian/debates/recent")
def get_recent_debates(limit: int = 20):
    """Get most recent debate transcripts from HOT tier."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    return {"transcripts": store.get_recent(limit=limit), "count": min(limit, len(store._hot))}


@router.get("/guardian/debates/stats")
def get_debate_stats(hours: float = 24.0):
    """Get aggregate debate decision statistics over a time window."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    return store.get_decision_stats(hours=hours)


@router.get("/guardian/debates/storage/stats")
def get_debate_storage_stats():
    """Get storage statistics across all tiers (HOT/WARM/COLD)."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    return store.get_storage_stats()


@router.post("/guardian/debates/storage/rotate")
def force_debate_rotation():
    """Manually trigger cold rotation of old debate transcripts."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    return store.force_rotation()


@router.get("/guardian/debates/by-strategy/{strategy}")
def get_debates_by_strategy(strategy: str, limit: int = 50):
    """Query debate transcripts by strategy."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    transcripts = store.get_by_strategy(strategy, limit=limit)
    return {"strategy": strategy, "transcripts": transcripts, "count": len(transcripts)}


@router.get("/guardian/debates/by-token/{token}")
def get_debates_by_token(token: str, limit: int = 50):
    """Query debate transcripts by token."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    transcripts = store.get_by_token(token, limit=limit)
    return {"token": token, "transcripts": transcripts, "count": len(transcripts)}


@router.get("/guardian/debates/{transcript_id}")
def get_debate_transcript(transcript_id: str):
    """Get a specific debate transcript by ID."""
    from cortex.debate_store import get_debate_store
    store = get_debate_store()
    transcript = store.get_by_id(transcript_id)
    if not transcript:
        raise HTTPException(status_code=404, detail=f"Transcript '{transcript_id}' not found")
    return transcript

