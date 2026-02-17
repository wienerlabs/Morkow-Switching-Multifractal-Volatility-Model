"""Shared persistent stores and helpers used across route modules."""

import logging

import numpy as np
import pandas as pd
from fastapi import HTTPException

from api.models import CalibrateRequest
from cortex.persistence import PersistentStore, VersionedPersistentStore

logger = logging.getLogger(__name__)

# Persistent model state (per-token) — backed by Redis when available
# Uses VersionedPersistentStore for rollback capability (keeps last N versions)
_model_store: VersionedPersistentStore = VersionedPersistentStore("model")
_portfolio_store: PersistentStore = PersistentStore("portfolio")
_evt_store: PersistentStore = PersistentStore("evt")
_copula_store: PersistentStore = PersistentStore("copula")
_hawkes_store: PersistentStore = PersistentStore("hawkes")
_rough_store: PersistentStore = PersistentStore("rough")
_svj_store: PersistentStore = PersistentStore("svj")

# Circuit breaker store — snapshots score-based and outcome-based CB state
from cortex.circuit_breaker import _get_cb_store
_circuit_breaker_store: PersistentStore = _get_cb_store()

# Kelly trade history + debate outcome priors
from cortex.guardian import _get_kelly_store
_kelly_store: PersistentStore = _get_kelly_store()

from cortex.debate import _get_debate_outcome_store
_debate_outcome_store: PersistentStore = _get_debate_outcome_store()

# Liquidity snapshot time series (per pool address)
_liquidity_store: PersistentStore = PersistentStore("liquidity")

# Comparison cache is ephemeral — no need to persist
_comparison_cache: dict[str, tuple[pd.DataFrame, float]] = {}

ALL_STORES: list[PersistentStore] = [
    _model_store, _portfolio_store, _evt_store, _copula_store,
    _hawkes_store, _rough_store, _svj_store, _circuit_breaker_store,
    _kelly_store, _debate_outcome_store, _liquidity_store,
]

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

