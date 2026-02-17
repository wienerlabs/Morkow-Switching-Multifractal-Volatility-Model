"""Tests for Wave 11B route integrations: vine copula, ccxt, portfolio optimization."""

from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ═══════════════════════════════════════════════════════════════════
# ccxt endpoints
# ═══════════════════════════════════════════════════════════════════


class TestCcxtEndpoints:
    """Test ccxt data feed endpoints with mocked exchange calls."""

    @patch("cortex.data.ccxt_feed._CCXT_AVAILABLE", True)
    @patch("cortex.data.ccxt_feed._get_exchange")
    def test_ohlcv(self, mock_exchange):
        mock_ex = MagicMock()
        mock_ex.fetch_ohlcv.return_value = [
            [1700000000000, 100.0, 105.0, 99.0, 103.0, 1000.0],
            [1700086400000, 103.0, 108.0, 101.0, 106.0, 1200.0],
        ]
        mock_exchange.return_value = mock_ex

        r = client.post("/api/v1/ccxt/ohlcv", json={
            "symbol": "BTC/USDT",
            "timeframe": "1d",
            "limit": 2,
        })
        assert r.status_code == 200
        data = r.json()
        assert data["symbol"] == "BTC/USDT"
        assert data["n_candles"] == 2
        assert data["last_close"] == 106.0

    @patch("cortex.data.ccxt_feed._CCXT_AVAILABLE", True)
    @patch("cortex.data.ccxt_feed._get_exchange")
    def test_order_book(self, mock_exchange):
        mock_ex = MagicMock()
        mock_ex.fetch_order_book.return_value = {
            "bids": [[100.0, 10.0], [99.5, 20.0]],
            "asks": [[100.5, 15.0], [101.0, 25.0]],
        }
        mock_exchange.return_value = mock_ex

        r = client.get("/api/v1/ccxt/orderbook", params={"symbol": "BTC/USDT"})
        assert r.status_code == 200
        data = r.json()
        assert data["best_bid"] == 100.0
        assert data["best_ask"] == 100.5
        assert data["mid_price"] == pytest.approx(100.25)
        assert data["spread"] == pytest.approx(0.5)

    @patch("cortex.data.ccxt_feed._CCXT_AVAILABLE", True)
    @patch("cortex.data.ccxt_feed._get_exchange")
    def test_ticker(self, mock_exchange):
        mock_ex = MagicMock()
        mock_ex.fetch_ticker.return_value = {
            "last": 42000.0,
            "high": 43000.0,
            "low": 41000.0,
            "baseVolume": 100.0,
            "quoteVolume": 4200000.0,
            "percentage": 2.5,
            "vwap": 42100.0,
            "bid": 41999.0,
            "ask": 42001.0,
        }
        mock_exchange.return_value = mock_ex

        r = client.get("/api/v1/ccxt/ticker", params={"symbol": "BTC/USDT"})
        assert r.status_code == 200
        data = r.json()
        assert data["last"] == 42000.0
        assert data["change_pct"] == 2.5

    @patch("cortex.data.ccxt_feed._CCXT_AVAILABLE", True)
    @patch("cortex.data.ccxt_feed._ccxt")
    def test_list_exchanges(self, mock_ccxt_mod):
        mock_ccxt_mod.exchanges = ["binance", "kraken", "coinbasepro"]

        r = client.get("/api/v1/ccxt/exchanges")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] == 3
        assert "binance" in data["exchanges"]

    def test_ohlcv_without_ccxt(self):
        """Verify graceful 503 when ccxt is not installed."""
        with patch("cortex.data.ccxt_feed._CCXT_AVAILABLE", False):
            r = client.post("/api/v1/ccxt/ohlcv", json={
                "symbol": "BTC/USDT",
                "timeframe": "1d",
                "limit": 10,
            })
            assert r.status_code == 503


# ═══════════════════════════════════════════════════════════════════
# Vine copula endpoints
# ═══════════════════════════════════════════════════════════════════


class TestVineCopulaEndpoints:
    """Test vine copula endpoints — requires calibrated portfolio."""

    def test_fit_without_portfolio(self):
        """Should 404 when no portfolio is calibrated."""
        r = client.post("/api/v1/portfolio/vine-copula/fit", json={
            "structure": "rvine",
        })
        assert r.status_code == 404

    def test_simulate_without_fit(self):
        """Should 404 when no vine fit exists."""
        r = client.post("/api/v1/portfolio/vine-copula/simulate")
        assert r.status_code == 404

    def test_var_without_fit(self):
        """Should 404 when no vine fit exists."""
        r = client.post("/api/v1/portfolio/vine-copula/var")
        assert r.status_code == 404

    @patch("api.routes.vine_copula._vine_store", {"default": {"_vine_object": MagicMock()}})
    @patch("api.stores._portfolio_store")
    def test_vine_var_with_mock(self, mock_portfolio):
        """Test vine VaR with mocked portfolio and vine fit."""
        mock_model = {
            "assets": ["A", "B"],
            "num_states": 3,
            "current_probs": np.array([0.2, 0.5, 0.3]),
            "per_asset": {
                "A": {"sigma_states": [1.0, 2.0, 4.0]},
                "B": {"sigma_states": [1.5, 2.5, 5.0]},
            },
        }
        mock_portfolio.__contains__ = lambda self, k: True
        mock_portfolio.__getitem__ = lambda self, k: mock_model

        mock_vine_result = {
            "vine_var": -3.5,
            "gaussian_var": -3.0,
            "var_ratio": 1.17,
            "engine": "pyvinecopulib",
            "structure": "rvine",
            "n_params": 5,
            "n_simulations": 10000,
            "alpha": 0.05,
        }

        with patch("cortex.copula.vine_copula_portfolio_var", return_value=mock_vine_result):
            r = client.post("/api/v1/portfolio/vine-copula/var", json={
                "A": 0.5, "B": 0.5,
            }, params={"alpha": 0.05, "n_simulations": 1000, "seed": 42})
            assert r.status_code == 200
            data = r.json()
            assert data["vine_var"] == -3.5
            assert data["engine"] == "pyvinecopulib"


# ═══════════════════════════════════════════════════════════════════
# Portfolio optimization endpoints
# ═══════════════════════════════════════════════════════════════════


class TestPortfolioOptEndpoints:
    """Test portfolio optimization endpoints."""

    def _mock_returns(self):
        np.random.seed(42)
        n = 252
        data = {
            "AAPL": np.random.normal(0.05, 1.5, n),
            "GOOGL": np.random.normal(0.03, 1.8, n),
            "MSFT": np.random.normal(0.04, 1.4, n),
        }
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        return pd.DataFrame(data, index=idx)

    @patch("api.routes.portfolio_opt._load_returns")
    def test_mean_cvar(self, mock_load):
        mock_load.return_value = self._mock_returns()

        mock_result = {
            "method": "mean_cvar",
            "engine": "skfolio",
            "weights": {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3},
            "expected_return": 0.04,
            "cvar": -0.025,
            "cvar_beta": 0.95,
            "n_assets": 3,
        }

        with patch("cortex.portfolio_opt.optimize_mean_cvar", return_value=mock_result):
            r = client.post("/api/v1/portfolio/optimize/mean-cvar", json={
                "tokens": ["AAPL", "GOOGL", "MSFT"],
                "period": "2y",
            })
            assert r.status_code == 200
            data = r.json()
            assert data["method"] == "mean_cvar"
            assert "weights" in data
            assert len(data["weights"]) == 3

    @patch("api.routes.portfolio_opt._load_returns")
    def test_hrp(self, mock_load):
        mock_load.return_value = self._mock_returns()

        mock_result = {
            "method": "hrp",
            "engine": "skfolio",
            "weights": {"AAPL": 0.35, "GOOGL": 0.30, "MSFT": 0.35},
            "expected_return": 0.04,
            "n_assets": 3,
        }

        with patch("cortex.portfolio_opt.optimize_hrp", return_value=mock_result):
            r = client.post("/api/v1/portfolio/optimize/hrp", json={
                "tokens": ["AAPL", "GOOGL", "MSFT"],
            })
            assert r.status_code == 200
            data = r.json()
            assert data["method"] == "hrp"

    @patch("api.routes.portfolio_opt._load_returns")
    def test_min_variance(self, mock_load):
        mock_load.return_value = self._mock_returns()

        mock_result = {
            "method": "min_variance",
            "engine": "skfolio",
            "weights": {"AAPL": 0.4, "GOOGL": 0.2, "MSFT": 0.4},
            "expected_return": 0.035,
            "variance": 0.0012,
            "n_assets": 3,
        }

        with patch("cortex.portfolio_opt.optimize_min_variance", return_value=mock_result):
            r = client.post("/api/v1/portfolio/optimize/min-variance", json={
                "tokens": ["AAPL", "GOOGL", "MSFT"],
            })
            assert r.status_code == 200
            data = r.json()
            assert data["method"] == "min_variance"
            assert "variance" in data

    @patch("api.routes.portfolio_opt._load_returns")
    def test_compare_strategies(self, mock_load):
        mock_load.return_value = self._mock_returns()

        mock_results = [
            {
                "method": "mean_cvar",
                "engine": "skfolio",
                "weights": {"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3},
                "expected_return": 0.04,
                "cvar": -0.025,
                "cvar_beta": 0.95,
                "n_assets": 3,
            },
            {
                "method": "equal_weight",
                "engine": "native",
                "weights": {"AAPL": 0.333, "GOOGL": 0.333, "MSFT": 0.334},
                "expected_return": 0.038,
                "n_assets": 3,
            },
        ]

        with patch("cortex.portfolio_opt.compare_strategies", return_value=mock_results):
            r = client.post("/api/v1/portfolio/optimize/compare", json={
                "tokens": ["AAPL", "GOOGL", "MSFT"],
            })
            assert r.status_code == 200
            data = r.json()
            assert len(data["strategies"]) == 2
            assert data["n_assets"] == 3

    def test_mean_cvar_too_few_tokens(self):
        """Should 422 with fewer than 2 tokens."""
        r = client.post("/api/v1/portfolio/optimize/mean-cvar", json={
            "tokens": ["AAPL"],
        })
        assert r.status_code == 422


# ═══════════════════════════════════════════════════════════════════
# OpenAPI schema validation
# ═══════════════════════════════════════════════════════════════════


class TestOpenAPIRoutes:
    """Verify new endpoints appear in OpenAPI schema."""

    def test_new_routes_in_schema(self):
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"]
        expected = [
            "/api/v1/ccxt/ohlcv",
            "/api/v1/ccxt/orderbook",
            "/api/v1/ccxt/ticker",
            "/api/v1/ccxt/exchanges",
            "/api/v1/portfolio/vine-copula/fit",
            "/api/v1/portfolio/vine-copula/simulate",
            "/api/v1/portfolio/vine-copula/var",
            "/api/v1/portfolio/optimize/mean-cvar",
            "/api/v1/portfolio/optimize/hrp",
            "/api/v1/portfolio/optimize/min-variance",
            "/api/v1/portfolio/optimize/compare",
        ]
        for route in expected:
            assert route in paths, f"Missing route: {route}"
