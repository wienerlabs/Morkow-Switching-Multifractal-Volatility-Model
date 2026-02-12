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
    _current_regime_state,
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
        from cortex.news import fetch_news_intelligence
        regime_state = _current_regime_state()
        news_result = fetch_news_intelligence(
            regime_state=regime_state, max_items=30, timeout=10.0,
        )
        news_data = news_result.get("signal")
    except Exception:
        logger.debug("News intelligence unavailable for guardian assessment", exc_info=True)

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
    )

    return GuardianAssessResponse(
        approved=result["approved"],
        risk_score=result["risk_score"],
        veto_reasons=result["veto_reasons"],
        recommended_size=result["recommended_size"],
        regime_state=result["regime_state"],
        confidence=result["confidence"],
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
    _record(pnl=req.pnl, size=req.size, token=req.token)
    return {"status": "recorded", "pnl": req.pnl, "size": req.size, "token": req.token}


@router.get("/guardian/circuit-breakers", response_model=CircuitBreakersResponse)
def get_circuit_breakers():
    import time
    from cortex.circuit_breaker import get_all_states
    states = get_all_states()
    return CircuitBreakersResponse(
        breakers=[CircuitBreakerItem(**s) for s in states],
        timestamp=time.time(),
    )


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


@router.post("/guardian/debate", response_model=DebateResponse)
def run_debate_endpoint(req: GuardianAssessRequest):
    """Run adversarial debate for a trade proposal (standalone, without full assess)."""
    from cortex.debate import run_debate

    result = run_debate(
        risk_score=50.0,
        component_scores=[],
        veto_reasons=[],
        direction=req.direction,
        trade_size_usd=req.trade_size_usd,
        original_approved=True,
    )
    return DebateResponse(**result)

