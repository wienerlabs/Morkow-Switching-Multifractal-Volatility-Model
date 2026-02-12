"""Multifractal / Hurst analysis endpoints."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query

from api.models import (
    ErrorResponse,
    FractalDiagnosticsResponse,
    HurstResponse,
    MultifractalSpectrumResponse,
    RegimeHurstItem,
    RegimeHurstResponse,
)
from api.stores import _get_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["fractal"])


@router.get(
    "/fractal/hurst",
    response_model=HurstResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Hurst exponent (R/S or DFA)",
)
def get_fractal_hurst(
    token: str = Query(...),
    method: str = Query("rs", pattern="^(rs|dfa)$"),
):
    from cortex.multifractal import hurst_dfa, hurst_rs

    m = _get_model(token)
    ret = m["returns"]

    result = hurst_rs(ret) if method == "rs" else hurst_dfa(ret)

    return HurstResponse(
        token=token,
        H=result["H"],
        H_se=result["H_se"],
        r_squared=result["r_squared"],
        interpretation=result["interpretation"],
        method=result["method"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/fractal/spectrum",
    response_model=MultifractalSpectrumResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Multifractal spectrum f(α)",
)
def get_fractal_spectrum(token: str = Query(...)):
    from cortex.multifractal import multifractal_spectrum

    m = _get_model(token)
    spec = multifractal_spectrum(m["returns"])

    return MultifractalSpectrumResponse(
        token=token,
        width=spec["width"],
        peak_alpha=spec["peak_alpha"],
        is_multifractal=spec["is_multifractal"],
        q_values=spec["q_values"],
        tau_q=spec["tau_q"],
        H_q=spec["H_q"],
        alpha=spec["alpha"],
        f_alpha=spec["f_alpha"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/fractal/regime-hurst",
    response_model=RegimeHurstResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Per-regime Hurst analysis",
)
def get_fractal_regime_hurst(token: str = Query(...)):
    from cortex.multifractal import compare_fractal_regimes

    m = _get_model(token)
    result = compare_fractal_regimes(
        m["returns"], m["filter_probs"], m["sigma_states"]
    )

    items = [RegimeHurstItem(**r) for r in result["per_regime"]]

    return RegimeHurstResponse(
        token=token,
        per_regime=items,
        n_states=result["n_states"],
        summary=result["summary"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/fractal/diagnostics",
    response_model=FractalDiagnosticsResponse,
    responses={404: {"model": ErrorResponse}},
    summary="Full fractal diagnostics",
)
def get_fractal_diagnostics(token: str = Query(...)):
    from cortex.multifractal import (
        hurst_dfa,
        hurst_rs,
        long_range_dependence_test,
        multifractal_spectrum,
    )

    m = _get_model(token)
    ret = m["returns"]

    rs = hurst_rs(ret)
    dfa = hurst_dfa(ret)
    spec = multifractal_spectrum(ret)
    lrd = long_range_dependence_test(ret)

    return FractalDiagnosticsResponse(
        token=token,
        H_rs=rs["H"],
        H_dfa=dfa["H"],
        spectrum_width=spec["width"],
        is_multifractal=spec["is_multifractal"],
        is_long_range_dependent=lrd["is_long_range_dependent"],
        confidence_z=lrd["confidence_z"],
        timestamp=datetime.now(timezone.utc),
    )

