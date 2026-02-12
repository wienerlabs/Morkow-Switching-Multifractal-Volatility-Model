"""Liquidity-Adjusted VaR (LVaR) endpoints."""

import logging
from datetime import datetime, timezone

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from api.models import (
    LVaREstimateRequest,
    LVaREstimateResponse,
    MarketImpactRequest,
    MarketImpactResponse,
    RegimeLiquidityItem,
    RegimeLiquidityProfileResponse,
    RegimeLVaRBreakdownItem,
    RegimeLVaRResponse,
    SpreadEstimate,
)
from api.stores import _get_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["lvar"])


@router.post("/lvar/estimate", response_model=LVaREstimateResponse)
def estimate_lvar(req: LVaREstimateRequest):
    from cortex import msm
    from cortex.liquidity import estimate_spread, liquidity_adjusted_var

    m = _get_model(req.token)

    alpha = 1.0 - req.confidence / 100.0
    st = m.get("use_student_t", False)
    df = m.get("nu", 5.0)

    var_t1, sigma_t1, z_alpha, pi_t1 = msm.msm_var_forecast_next_day(
        m["filter_probs"], m["sigma_states"], m["P_matrix"],
        alpha=alpha, use_student_t=st, nu=df,
        leverage_gamma=m.get("leverage_gamma", 0.0),
        last_return=float(m["returns"].iloc[-1]),
        p_stay=m["calibration"]["p_stay"],
    )

    returns = m["returns"]
    prices = np.exp(np.cumsum(np.asarray(returns) / 100.0)) * 100.0

    spread = estimate_spread(prices, method="roll", window=req.window)

    lvar_result = liquidity_adjusted_var(
        var_value=var_t1,
        spread_pct=spread["spread_pct"],
        spread_vol_pct=spread.get("spread_vol_pct", 0.0),
        position_value=req.position_value,
        alpha=alpha,
        holding_period=req.holding_period,
    )

    return LVaREstimateResponse(
        token=req.token,
        lvar=lvar_result["lvar"],
        base_var=lvar_result["base_var"],
        liquidity_cost_pct=lvar_result["liquidity_cost_pct"],
        liquidity_cost_abs=lvar_result["liquidity_cost_abs"],
        lvar_abs=lvar_result["lvar_abs"],
        lvar_ratio=lvar_result["lvar_ratio"],
        spread=SpreadEstimate(
            spread_pct=spread["spread_pct"],
            spread_abs=spread["spread_abs"],
            spread_vol_pct=spread.get("spread_vol_pct", 0.0),
            method=spread["method"],
            n_obs=spread["n_obs"],
        ),
        alpha=alpha,
        holding_period=req.holding_period,
        position_value=req.position_value,
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/lvar/regime-var", response_model=RegimeLVaRResponse)
def get_regime_lvar(
    token: str = Query(...),
    confidence: float = Query(95.0, gt=50.0, le=99.99),
    position_value: float = Query(100_000.0, gt=0),
    holding_period: int = Query(1, ge=1, le=30),
):
    from cortex import msm
    from cortex.liquidity import compute_lvar_with_regime, regime_liquidity_profile

    m = _get_model(token)

    alpha = 1.0 - confidence / 100.0
    st = m.get("use_student_t", False)
    df = m.get("nu", 5.0)

    var_t1, sigma_t1, z_alpha_val, pi_t1 = msm.msm_var_forecast_next_day(
        m["filter_probs"], m["sigma_states"], m["P_matrix"],
        alpha=alpha, use_student_t=st, nu=df,
        leverage_gamma=m.get("leverage_gamma", 0.0),
        last_return=float(m["returns"].iloc[-1]),
        p_stay=m["calibration"]["p_stay"],
    )

    returns = m["returns"]
    prices = np.exp(np.cumsum(np.asarray(returns) / 100.0)) * 100.0

    fprobs = m["filter_probs"]
    regime_labels = np.argmax(np.asarray(fprobs), axis=1) + 1
    num_states = m["calibration"]["num_states"]

    profiles = regime_liquidity_profile(
        prices=prices,
        regime_labels=regime_labels,
        num_states=num_states,
    )

    current_probs = np.asarray(fprobs.iloc[-1])

    result = compute_lvar_with_regime(
        var_value=var_t1,
        regime_profiles=profiles,
        current_regime_probs=current_probs,
        position_value=position_value,
        alpha=alpha,
        holding_period=holding_period,
    )

    return RegimeLVaRResponse(
        token=token,
        lvar=result["lvar"],
        base_var=result["base_var"],
        liquidity_cost_pct=result["liquidity_cost_pct"],
        regime_weighted_spread_pct=result["regime_weighted_spread_pct"],
        regime_breakdown=[
            RegimeLVaRBreakdownItem(**rb) for rb in result["regime_breakdown"]
        ],
        alpha=alpha,
        holding_period=holding_period,
        position_value=position_value,
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/lvar/impact", response_model=MarketImpactResponse)
def estimate_market_impact(req: MarketImpactRequest):
    from cortex import msm
    from cortex.liquidity import market_impact_cost

    m = _get_model(req.token)

    _, sigma_t1, _, _ = msm.msm_var_forecast_next_day(
        m["filter_probs"], m["sigma_states"], m["P_matrix"],
        leverage_gamma=m.get("leverage_gamma", 0.0),
        last_return=float(m["returns"].iloc[-1]),
        p_stay=m["calibration"]["p_stay"],
    )

    sigma_decimal = abs(sigma_t1) / 100.0

    adv = req.adv_usd
    if adv is None:
        adv = 1_000_000.0

    result = market_impact_cost(
        sigma=sigma_decimal,
        trade_size_usd=req.trade_size_usd,
        adv_usd=adv,
        participation_rate=req.participation_rate,
    )

    return MarketImpactResponse(
        token=req.token,
        impact_pct=result["impact_pct"],
        impact_usd=result["impact_usd"],
        participation_rate=result["participation_rate"],
        participation_warning=result["participation_warning"],
        sigma_daily=result["sigma_daily"],
        trade_size_usd=result["trade_size_usd"],
        adv_usd=result["adv_usd"],
        timestamp=datetime.now(timezone.utc),
    )


@router.get("/lvar/regime-profile", response_model=RegimeLiquidityProfileResponse)
def get_regime_liquidity_profile(token: str = Query(...)):
    from cortex.liquidity import regime_liquidity_profile

    m = _get_model(token)

    returns = m["returns"]
    prices = np.exp(np.cumsum(np.asarray(returns) / 100.0)) * 100.0

    fprobs = m["filter_probs"]
    regime_labels = np.argmax(np.asarray(fprobs), axis=1) + 1
    num_states = m["calibration"]["num_states"]

    result = regime_liquidity_profile(
        prices=prices,
        regime_labels=regime_labels,
        num_states=num_states,
    )

    return RegimeLiquidityProfileResponse(
        token=token,
        num_states=result["num_states"],
        profiles=[RegimeLiquidityItem(**p) for p in result["profiles"]],
        weighted_avg_spread_pct=result["weighted_avg_spread_pct"],
        n_total=result["n_total"],
        timestamp=datetime.now(timezone.utc),
    )

