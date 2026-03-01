"""On-chain data API routes — liquidity depth, realized spread, on-chain LVaR."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Query

from api.models import (
    DexSpreadItem,
    LaunchRiskRequest,
    LaunchRiskResponse,
    OnchainDepthRequest,
    OnchainDepthResponse,
    OnchainLVaRRequest,
    OnchainLVaRResponse,
    RealizedSpreadRequest,
    RealizedSpreadResponse,
)
from api.stores import _get_model

logger = logging.getLogger(__name__)

router = APIRouter(tags=["onchain"])


@router.post("/lvar/onchain-depth", response_model=OnchainDepthResponse)
def get_onchain_depth(req: OnchainDepthRequest):
    from cortex.data.onchain_liquidity import build_liquidity_depth_curve

    result = build_liquidity_depth_curve(req.pool_address, num_ticks=req.num_ticks)
    return OnchainDepthResponse(
        pool=result.get("pool", req.pool_address),
        current_price=result.get("current_price", 0.0),
        bid_prices=result.get("bid_prices", []),
        ask_prices=result.get("ask_prices", []),
        bid_depth=result.get("bid_depth", []),
        ask_depth=result.get("ask_depth", []),
        total_bid_liquidity=result.get("total_bid_liquidity", 0.0),
        total_ask_liquidity=result.get("total_ask_liquidity", 0.0),
        depth_imbalance=result.get("depth_imbalance", 0.0),
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/lvar/realized-spread", response_model=RealizedSpreadResponse)
def get_realized_spread(req: RealizedSpreadRequest):
    from cortex.data.onchain_liquidity import (
        compute_realized_spread,
        fetch_swap_history,
        get_volume_weighted_spread,
    )

    swaps = fetch_swap_history(req.token_address, limit=req.limit)
    spread = compute_realized_spread(swaps)
    vwas = get_volume_weighted_spread(swaps)

    by_dex = [
        DexSpreadItem(dex=dex, **info) for dex, info in spread.get("by_dex", {}).items()
    ]

    return RealizedSpreadResponse(
        token_address=req.token_address,
        realized_spread_pct=spread["realized_spread_pct"],
        realized_spread_vol_pct=spread["realized_spread_vol_pct"],
        n_swaps=spread["n_swaps"],
        by_dex=by_dex,
        vwas_pct=vwas["vwas_pct"],
        total_volume=vwas["total_volume"],
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/lvar/onchain-estimate", response_model=OnchainLVaRResponse)
def estimate_onchain_lvar(req: OnchainLVaRRequest):
    from cortex import msm
    from cortex.liquidity import liquidity_adjusted_var_with_onchain

    m = _get_model(req.token)
    alpha = 1.0 - req.confidence / 100.0
    var_result = msm.msm_var_forecast_next_day(
        m["filter_probs"],
        m["sigma_states"],
        m["P_matrix"],
        alpha=alpha,
    )
    base_var = var_result["var"]

    prices = m["returns"].values
    result = liquidity_adjusted_var_with_onchain(
        var_value=base_var,
        token_address=req.token_address,
        pair_address=req.pair_address,
        prices=prices,
        position_value=req.position_value,
        alpha=alpha,
        holding_period=req.holding_period,
    )

    by_dex = None
    if "by_dex" in result:
        by_dex = [
            DexSpreadItem(dex=dex, **info) for dex, info in result["by_dex"].items()
        ]

    return OnchainLVaRResponse(
        token=req.token,
        lvar=result["lvar"],
        base_var=result["base_var"],
        liquidity_cost_pct=result["liquidity_cost_pct"],
        spread_pct=result["spread_pct"],
        spread_source=result["spread_source"],
        by_dex=by_dex,
        confidence=req.confidence,
        holding_period=req.holding_period,
        position_value=req.position_value,
        timestamp=datetime.now(timezone.utc),
    )


@router.post("/lvar/launch-risk", response_model=LaunchRiskResponse)
def get_launch_risk(req: LaunchRiskRequest):
    from cortex.data.launch_tracker import compute_launch_risk

    result = compute_launch_risk(
        token_mint=req.token_mint,
        holder_concentration_pct=req.holder_concentration_pct,
    )
    return LaunchRiskResponse(
        score=result.score,
        cex_funded=result.cex_funded,
        bundle_detected=result.bundle_detected,
        deployer_age_days=result.deployer_age_days,
        top10_concentration_pct=result.top10_concentration_pct,
        deploy_to_first_trade_sec=result.deploy_to_first_trade_sec,
        details=result.details,
        risk_factors=result.risk_factors,
        timestamp=datetime.now(timezone.utc),
    )

