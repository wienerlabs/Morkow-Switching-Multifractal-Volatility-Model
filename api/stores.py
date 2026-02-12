"""Shared in-memory stores and helpers used across route modules."""

import logging

import numpy as np
import pandas as pd
from fastapi import HTTPException

from api.models import CalibrateRequest

logger = logging.getLogger(__name__)

# In-memory model state (per-token)
_model_store: dict[str, dict] = {}
_portfolio_store: dict[str, dict] = {}
_evt_store: dict[str, dict] = {}
_copula_store: dict[str, dict] = {}
_hawkes_store: dict[str, dict] = {}
_rough_store: dict[str, dict] = {}
_svj_store: dict[str, dict] = {}

# Cache comparison results per token for the report endpoint
_comparison_cache: dict[str, tuple[pd.DataFrame, float]] = {}

_PORTFOLIO_KEY = "default"


def _get_model(token: str) -> dict:
    if token not in _model_store:
        raise HTTPException(
            status_code=404,
            detail=f"No calibrated model for '{token}'. Call POST /calibrate first.",
        )
    return _model_store[token]


def _load_returns(req: CalibrateRequest) -> pd.Series:
    """Fetch data and convert to log-returns in %."""
    if req.data_source.value == "solana":
        from cortex.data.solana import get_token_ohlcv, ohlcv_to_returns

        df = get_token_ohlcv(req.token, req.start_date, req.end_date, req.interval)
        return ohlcv_to_returns(df)

    import yfinance as yf

    df = yf.download(req.token, start=req.start_date, end=req.end_date, progress=False)
    if df.empty:
        raise HTTPException(status_code=400, detail=f"No yfinance data for '{req.token}'")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    close = close.dropna()
    vals = close.values.flatten()
    rets = 100 * np.diff(np.log(vals))
    return pd.Series(rets, index=close.index[1:], name="r")


def _current_regime_state() -> int:
    """Get regime state from any calibrated model, default 3 (Normal)."""
    for m in _model_store.values():
        probs = np.asarray(m["filter_probs"].iloc[-1])
        return int(np.argmax(probs)) + 1
    return 3


def _get_portfolio_model() -> dict:
    if _PORTFOLIO_KEY not in _portfolio_store:
        raise HTTPException(404, "No calibrated portfolio. Call POST /portfolio/calibrate first.")
    return _portfolio_store[_PORTFOLIO_KEY]


def _get_copula_fit() -> dict:
    if _PORTFOLIO_KEY not in _copula_store:
        raise HTTPException(
            404,
            "No copula fit. Call POST /portfolio/calibrate with copula_family "
            "or POST /portfolio/copula/compare first.",
        )
    return _copula_store[_PORTFOLIO_KEY]

