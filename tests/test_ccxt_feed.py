"""Tests for ccxt multi-exchange data feed — Wave 11B."""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pytest

from cortex.data import ccxt_feed


# ── Availability flag ──

class TestCcxtAvailability:
    def test_available_flag_is_bool(self):
        assert isinstance(ccxt_feed._CCXT_AVAILABLE, bool)


# ── Tests when ccxt is NOT installed ──

class TestCcxtFallbackWhenUnavailable:
    def test_get_exchange_raises(self):
        with patch.object(ccxt_feed, "_CCXT_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="ccxt not installed"):
                ccxt_feed._get_exchange()

    def test_fetch_ohlcv_raises(self):
        with patch.object(ccxt_feed, "_CCXT_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="ccxt not installed"):
                ccxt_feed.fetch_ohlcv("BTC/USDT")

    def test_list_exchanges_raises(self):
        with patch.object(ccxt_feed, "_CCXT_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="ccxt not installed"):
                ccxt_feed.list_exchanges()


# ── Exchange caching ──

class TestExchangeCache:
    def setup_method(self):
        ccxt_feed._exchange_cache.clear()

    def test_get_exchange_caches_instance(self):
        mock_cls = MagicMock(return_value=MagicMock())
        mock_ccxt = MagicMock()
        mock_ccxt.binance = mock_cls

        with patch.object(ccxt_feed, "_CCXT_AVAILABLE", True), \
             patch.object(ccxt_feed, "_ccxt", mock_ccxt):
            ex1 = ccxt_feed._get_exchange("binance")
            ex2 = ccxt_feed._get_exchange("binance")

        assert ex1 is ex2
        mock_cls.assert_called_once_with({"enableRateLimit": True})

    def test_get_exchange_unknown_raises(self):
        mock_ccxt = MagicMock()
        mock_ccxt.binance = None  # getattr returns None
        mock_ccxt.exchanges = ["kraken", "coinbase"]

        with patch.object(ccxt_feed, "_CCXT_AVAILABLE", True), \
             patch.object(ccxt_feed, "_ccxt", mock_ccxt):
            with pytest.raises(ValueError, match="Unknown exchange"):
                ccxt_feed._get_exchange("binance")

    def teardown_method(self):
        ccxt_feed._exchange_cache.clear()


# ── OHLCV fetch ──

class TestFetchOHLCV:
    def setup_method(self):
        ccxt_feed._exchange_cache.clear()

    def test_returns_dataframe(self):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1700000000000, 100.0, 105.0, 99.0, 103.0, 1000.0],
            [1700086400000, 103.0, 108.0, 102.0, 106.0, 1200.0],
            [1700172800000, 106.0, 110.0, 104.0, 109.0, 1100.0],
        ]

        with patch.object(ccxt_feed, "_get_exchange", return_value=mock_exchange):
            df = ccxt_feed.fetch_ohlcv("BTC/USDT", timeframe="1d", limit=3)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.index.name == "timestamp"
        assert len(df) == 3

    def test_timestamp_is_utc(self):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1700000000000, 100, 105, 99, 103, 1000],
        ]

        with patch.object(ccxt_feed, "_get_exchange", return_value=mock_exchange):
            df = ccxt_feed.fetch_ohlcv("BTC/USDT")

        assert str(df.index.tz) == "UTC"

    def test_passes_parameters_to_exchange(self):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []

        with patch.object(ccxt_feed, "_get_exchange", return_value=mock_exchange):
            ccxt_feed.fetch_ohlcv("SOL/USDT", timeframe="4h", limit=100, exchange_id="kraken")

        mock_exchange.fetch_ohlcv.assert_called_once_with("SOL/USDT", timeframe="4h", limit=100)

    def teardown_method(self):
        ccxt_feed._exchange_cache.clear()


# ── Order book ──

class TestFetchOrderBook:
    def test_returns_spread_info(self):
        mock_exchange = MagicMock()
        mock_exchange.fetch_order_book.return_value = {
            "bids": [[100.0, 10.0], [99.5, 20.0]],
            "asks": [[100.5, 8.0], [101.0, 15.0]],
        }

        with patch.object(ccxt_feed, "_get_exchange", return_value=mock_exchange):
            ob = ccxt_feed.fetch_order_book("BTC/USDT", limit=2)

        assert ob["best_bid"] == 100.0
        assert ob["best_ask"] == 100.5
        assert ob["mid_price"] == 100.25
        assert ob["spread"] == 0.5
        assert ob["spread_bps"] > 0
        assert ob["bid_depth"] == 30.0
        assert ob["ask_depth"] == 23.0

    def test_empty_order_book(self):
        mock_exchange = MagicMock()
        mock_exchange.fetch_order_book.return_value = {"bids": [], "asks": []}

        with patch.object(ccxt_feed, "_get_exchange", return_value=mock_exchange):
            ob = ccxt_feed.fetch_order_book("BTC/USDT")

        assert ob["best_bid"] == 0.0
        assert ob["best_ask"] == 0.0
        assert ob["mid_price"] == 0.0
        assert ob["spread"] == 0.0


# ── Ticker ──

class TestFetchTicker:
    def test_returns_expected_keys(self):
        mock_exchange = MagicMock()
        mock_exchange.fetch_ticker.return_value = {
            "last": 105.0,
            "high": 110.0,
            "low": 98.0,
            "baseVolume": 50000.0,
            "quoteVolume": 5250000.0,
            "percentage": 2.5,
            "vwap": 104.5,
            "bid": 104.9,
            "ask": 105.1,
        }

        with patch.object(ccxt_feed, "_get_exchange", return_value=mock_exchange):
            t = ccxt_feed.fetch_ticker("BTC/USDT")

        assert t["last"] == 105.0
        assert t["high"] == 110.0
        assert t["volume"] == 50000.0
        assert t["change_pct"] == 2.5
        assert t["symbol"] == "BTC/USDT"


# ── Multi-exchange OHLCV ──

class TestFetchMultiExchangeOHLCV:
    def test_returns_dict_of_dataframes(self):
        mock_df = pd.DataFrame({
            "open": [100.0], "high": [105.0], "low": [99.0],
            "close": [103.0], "volume": [1000.0],
        })

        with patch.object(ccxt_feed, "fetch_ohlcv", return_value=mock_df):
            results = ccxt_feed.fetch_multi_exchange_ohlcv(
                "BTC/USDT", exchanges=["binance", "kraken"]
            )

        assert "binance" in results
        assert "kraken" in results
        assert isinstance(results["binance"], pd.DataFrame)

    def test_handles_exchange_failure_gracefully(self):
        def side_effect(symbol, timeframe, limit, exchange_id=None):
            if exchange_id == "kraken":
                raise Exception("kraken down")
            return pd.DataFrame({"close": [100.0]})

        with patch.object(ccxt_feed, "fetch_ohlcv", side_effect=side_effect):
            results = ccxt_feed.fetch_multi_exchange_ohlcv(
                "BTC/USDT", exchanges=["binance", "kraken"]
            )

        assert "binance" in results
        assert "kraken" not in results

    def test_default_exchanges(self):
        with patch.object(ccxt_feed, "fetch_ohlcv", return_value=pd.DataFrame()) as mock:
            ccxt_feed.fetch_multi_exchange_ohlcv("BTC/USDT")

        # Default exchanges: binance, kraken, coinbasepro
        assert mock.call_count == 3


# ── Returns conversion ──

class TestOHLCVToReturns:
    def test_log_returns_pct(self):
        df = pd.DataFrame({"close": [100.0, 105.0, 103.0, 110.0]})
        ret = ccxt_feed.ohlcv_to_returns(df, pct=True)

        assert isinstance(ret, pd.Series)
        assert len(ret) == 3  # dropna removes first
        # Returns should be in % (multiplied by 100)
        expected_first = np.log(105.0 / 100.0) * 100
        assert abs(ret.iloc[0] - expected_first) < 1e-10

    def test_log_returns_decimal(self):
        df = pd.DataFrame({"close": [100.0, 105.0, 103.0]})
        ret = ccxt_feed.ohlcv_to_returns(df, pct=False)

        expected_first = np.log(105.0 / 100.0)
        assert abs(ret.iloc[0] - expected_first) < 1e-10


# ── List exchanges ──

class TestListExchanges:
    def test_returns_list(self):
        mock_ccxt = MagicMock()
        mock_ccxt.exchanges = ["binance", "kraken", "coinbase"]

        with patch.object(ccxt_feed, "_CCXT_AVAILABLE", True), \
             patch.object(ccxt_feed, "_ccxt", mock_ccxt):
            result = ccxt_feed.list_exchanges()

        assert result == ["binance", "kraken", "coinbase"]
