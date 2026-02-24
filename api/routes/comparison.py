"""Model comparison and benchmark report endpoints."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.models import (
    CompareRequest,
    CompareResponse,
    ComparisonReportResponse,
    ModelMetricsRow,
)
from api.stores import _comparison_cache, _get_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["comparison"])


@router.post("/compare", response_model=CompareResponse)
def run_model_comparison(req: CompareRequest):
    from cortex.comparison import compare_models, _MODEL_REGISTRY

    m = _get_model(req.token)
    returns = m["returns"]

    if req.models:
        invalid = [k for k in req.models if k not in _MODEL_REGISTRY]
        if invalid:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model keys: {invalid}. Valid: {list(_MODEL_REGISTRY.keys())}",
            )

    try:
        df = compare_models(returns, alpha=req.alpha, models=req.models)
    except Exception as exc:
        logger.exception("Model comparison failed for token=%s", req.token)
        raise HTTPException(status_code=500, detail=f"Comparison error: {exc}")

    _comparison_cache[req.token] = (df, req.alpha)

    results = [ModelMetricsRow(**row.to_dict()) for _, row in df.iterrows()]

    return CompareResponse(
        token=req.token,
        alpha=req.alpha,
        num_observations=len(returns),
        models_compared=[r.model for r in results],
        results=results,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/compare/report/{token}", response_model=ComparisonReportResponse)
def get_comparison_report(token: str, alpha: float = Query(0.05)):
    from cortex.comparison import generate_comparison_report

    if token not in _comparison_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No comparison results for '{token}'. Call POST /compare first.",
        )

    df, cached_alpha = _comparison_cache[token]
    report = generate_comparison_report(df, alpha=cached_alpha)

    return ComparisonReportResponse(
        token=token,
        alpha=cached_alpha,
        summary_table=report["summary_table"],
        winners=report["winners"],
        pass_fail=report["pass_fail"],
        ranking=report["ranking"],
        timestamp=datetime.now(timezone.utc),
    )

