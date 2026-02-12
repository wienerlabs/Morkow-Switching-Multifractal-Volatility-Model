"""Tests for cortex/data/solana.py — Solana DeFi data adapter."""
import os
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from cortex.data.solana import (
    _resolve_token_address,
    _birdeye_headers,
    _to_unix,
    get_token_ohlcv,
    get_funding_rates,
    ohlcv_to_returns,
    TOKEN_REGISTRY,
    DRIFT_PERP_MARKETS,
)


# ── _resolve_token_address ──

def test_resolve_known_symbol():
    assert _resolve_token_address("SOL") == TOKEN_REGISTRY["SOL"]
    assert _resolve_token_address("sol") == TOKEN_REGISTRY["SOL"]


def test_resolve_mint_address():
    mint = "So11111111111111111111111111111111111111112"
    assert _resolve_token_address(mint) == mint


def test_resolve_unknown_symbol():
    with pytest.raises(ValueError, match="Unknown token symbol"):
        _resolve_token_address("FAKE")


# ── _birdeye_headers ──

def test_birdeye_headers_missing_key():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("BIRDEYE_API_KEY", None)
        with pytest.raises(EnvironmentError, match="BIRDEYE_API_KEY"):
            _birdeye_headers()


def test_birdeye_headers_with_key():
    with patch.dict(os.environ, {"BIRDEYE_API_KEY": "test-key-123"}):
        headers = _birdeye_headers()
        assert headers["X-API-KEY"] == "test-key-123"
        assert headers["x-chain"] == "solana"


# ── _to_unix ──

def test_to_unix_datetime():
    dt = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    result = _to_unix(dt)
    assert result == int(dt.timestamp())


def test_to_unix_string():
    result = _to_unix("2024-01-15T12:00:00Z")
    expected = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert result == int(expected.timestamp())


def test_to_unix_naive_datetime():
    dt = datetime(2024, 1, 15, 12, 0, 0)
    result = _to_unix(dt)
    expected = dt.replace(tzinfo=timezone.utc)
    assert result == int(expected.timestamp())


# ── get_token_ohlcv (mocked HTTP) ──

def _mock_ohlcv_response():
    return {
        "data": {
            "items": [
                {"unixTime": 1705276800, "o": 100.0, "h": 105.0, "l": 98.0, "c": 103.0, "v": 1e6},
                {"unixTime": 1705363200, "o": 103.0, "h": 108.0, "l": 101.0, "c": 106.0, "v": 1.2e6},
                {"unixTime": 1705449600, "o": 106.0, "h": 110.0, "l": 104.0, "c": 107.0, "v": 0.9e6},
            ]
        }
    }


@patch.dict(os.environ, {"BIRDEYE_API_KEY": "test-key"})
@patch("cortex.data.solana.httpx.Client")
def test_get_token_ohlcv(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_ohlcv_response()
    mock_resp.raise_for_status = MagicMock()
    mock_client.get.return_value = mock_resp

    df = get_token_ohlcv("SOL", "2024-01-15", "2024-01-17")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(df) == 3
    assert df["Close"].iloc[0] == 103.0


# ── get_funding_rates (mocked HTTP) ──

@patch("cortex.data.solana.httpx.Client")
def test_get_funding_rates(mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value.__enter__ = MagicMock(return_value=mock_client)
    mock_client_cls.return_value.__exit__ = MagicMock(return_value=False)
    mock_resp = MagicMock()
    mock_resp.json.return_value = [
        {"ts": 1705276800, "fundingRate": 1000000, "oraclePriceTwap": 100000000, "markPriceTwap": 100500000},
        {"ts": 1705363200, "fundingRate": -500000, "oraclePriceTwap": 101000000, "markPriceTwap": 100800000},
    ]
    mock_resp.raise_for_status = MagicMock()
    mock_client.get.return_value = mock_resp

    df = get_funding_rates("SOL-PERP")
    assert isinstance(df, pd.DataFrame)
    assert "fundingRate" in df.columns
    assert len(df) == 2


def test_get_funding_rates_unknown_market():
    with pytest.raises(ValueError, match="Unknown perp market"):
        get_funding_rates("FAKE-PERP")


# ── ohlcv_to_returns ──

def test_ohlcv_to_returns():
    df = pd.DataFrame({
        "Close": [100.0, 105.0, 103.0, 110.0],
        "Open": [99.0, 100.0, 105.0, 103.0],
        "High": [106.0, 108.0, 107.0, 112.0],
        "Low": [98.0, 99.0, 101.0, 102.0],
        "Volume": [1e6, 1.1e6, 0.9e6, 1.2e6],
    }, index=pd.date_range("2024-01-01", periods=4, freq="D"))
    rets = ohlcv_to_returns(df)
    assert isinstance(rets, pd.Series)
    assert len(rets) == 3
    expected_first = 100 * np.log(105.0 / 100.0)
    assert rets.iloc[0] == pytest.approx(expected_first, rel=1e-6)

