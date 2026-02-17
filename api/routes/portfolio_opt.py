"""Portfolio optimization endpoints — Mean-CVaR, HRP, Min-Variance via skfolio."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.models import (
    DataSource,
    PortfolioOptCompareResponse,
    PortfolioOptRequest,
    PortfolioOptWeights,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["portfolio-optimization"])


def _load_returns(req: PortfolioOptRequest) -> pd.DataFrame:
    """Fetch return data for the requested tokens."""
    if req.data_source == DataSource.SOLANA:
        from cortex.data.solana import get_token_ohlcv, ohlcv_to_returns

        now = datetime.now(timezone.utc)
        period = req.period
        if period.endswith("y"):
            days = int(period[:-1]) * 365
        elif period.endswith("mo"):
            days = int(period[:-2]) * 30
        else:
            days = int(period.rstrip("d"))
        start = (now - timedelta(days=days)).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        frames = {}
        for token in req.tokens:
            df = get_token_ohlcv(token, start, end, "1D")
            rets = ohlcv_to_returns(df)
            if len(rets) < 30:
                raise HTTPException(400, f"Insufficient data for {token}")
            frames[token] = rets.rename(token)
        returns_df = pd.DataFrame(frames).dropna()
        if len(returns_df) < 30:
            raise HTTPException(400, f"Only {len(returns_df)} observations after alignment")
        return returns_df

    import yfinance as yf

    frames = {}
    for token in req.tokens:
        df = yf.download(token, period=req.period, auto_adjust=True, progress=False)
        if df.empty or len(df) < 30:
            raise HTTPException(400, f"Insufficient data for {token}")
        close = df["Close"].squeeze()
        rets = 100.0 * np.diff(np.log(close.values))
        frames[token] = pd.Series(rets, index=close.index[1:], name=token)
    returns_df = pd.DataFrame(frames).dropna()
    if len(returns_df) < 30:
        raise HTTPException(400, f"Only {len(returns_df)} observations after alignment")
    return returns_df


def _to_response(result: dict) -> PortfolioOptWeights:
    return PortfolioOptWeights(
        method=result["method"],
        engine=result["engine"],
        weights=result["weights"],
        expected_return=result.get("expected_return", 0.0),
        cvar=result.get("cvar"),
        cvar_beta=result.get("cvar_beta"),
        variance=result.get("variance"),
        n_assets=result["n_assets"],
    )


@router.post("/portfolio/optimize/mean-cvar", response_model=PortfolioOptWeights)
async def optimize_mean_cvar_endpoint(req: PortfolioOptRequest):
    """Optimize portfolio weights to maximize return/CVaR ratio."""
    from cortex.portfolio_opt import optimize_mean_cvar

    returns_df = await asyncio.to_thread(_load_returns, req)
    try:
        result = await asyncio.to_thread(
            optimize_mean_cvar, returns_df, req.cvar_beta, req.max_weight,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    return _to_response(result)


@router.post("/portfolio/optimize/hrp", response_model=PortfolioOptWeights)
async def optimize_hrp_endpoint(req: PortfolioOptRequest):
    """Hierarchical Risk Parity — distribution-free portfolio optimization."""
    from cortex.portfolio_opt import optimize_hrp

    returns_df = await asyncio.to_thread(_load_returns, req)
    try:
        result = await asyncio.to_thread(optimize_hrp, returns_df)
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    return _to_response(result)


@router.post("/portfolio/optimize/min-variance", response_model=PortfolioOptWeights)
async def optimize_min_variance_endpoint(req: PortfolioOptRequest):
    """Minimum variance portfolio."""
    from cortex.portfolio_opt import optimize_min_variance

    returns_df = await asyncio.to_thread(_load_returns, req)
    try:
        result = await asyncio.to_thread(
            optimize_min_variance, returns_df, req.max_weight,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    return _to_response(result)


@router.post("/portfolio/optimize/compare", response_model=PortfolioOptCompareResponse)
async def compare_strategies_endpoint(req: PortfolioOptRequest):
    """Run all optimization strategies and compare results."""
    from cortex.portfolio_opt import compare_strategies

    returns_df = await asyncio.to_thread(_load_returns, req)
    try:
        results = await asyncio.to_thread(
            compare_strategies, returns_df, req.cvar_beta, req.max_weight,
        )
    except RuntimeError as e:
        raise HTTPException(503, str(e))

    return PortfolioOptCompareResponse(
        strategies=[_to_response(r) for r in results],
        n_assets=len(req.tokens),
        n_observations=len(returns_df),
        timestamp=datetime.now(timezone.utc),
    )
