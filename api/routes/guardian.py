"""Guardian (Unified Risk Veto) endpoint."""

import logging

from fastapi import APIRouter, HTTPException

from api.models import (
    GuardianAssessRequest,
    GuardianAssessResponse,
    GuardianComponentScore,
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
        from_cache=result["from_cache"],
    )

