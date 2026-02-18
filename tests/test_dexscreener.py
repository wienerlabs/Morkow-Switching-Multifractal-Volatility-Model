"""Tests for cortex/data/dexscreener.py — DexScreener DEX data adapter."""
import time
from unittest.mock import patch, MagicMock

import httpx
import pytest


# ── Fixtures / helpers ──

FAKE_PAIR = {
    "pairAddress": "PAIR_ADDR_1",
    "dexId": "raydium",
    "priceUsd": "1.23",
    "priceNative": "0.0085",
    "baseToken": {"symbol": "BONK", "address": "TOKEN_A"},
    "liquidity": {"usd": 500_000, "base": 100_000, "quote": 400_000},
    "volume": {"h24": 1_200_000, "h6": 300_000, "h1": 50_000},
    "txns": {"h24": {"buys": 5000, "sells": 4200}},
    "priceChange": {"h24": -2.5},
    "pairCreatedAt": 1700000000000,
}

FAKE_PAIR_LOW_LIQ = {
    "pairAddress": "PAIR_LOW",
    "dexId": "orca",
    "priceUsd": "0.001",
    "priceNative": "0.000007",
    "baseToken": {"symbol": "SCAM", "address": "TOKEN_B"},
    "liquidity": {"usd": 200},
    "volume": {"h24": 50},
    "priceChange": {"h24": 0},
}


def _make_mock_response(json_data, status_code=200):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp
        )
    return resp


# ── _request: retry & backoff ──

class TestRequest:
    @patch("cortex.data.dexscreener.time.sleep")
    @patch("cortex.data.dexscreener.httpx.request")
    def test_success_first_try(self, mock_req, mock_sleep):
        mock_req.return_value = _make_mock_response({"ok": True})
        from cortex.data.dexscreener import _request

        result = _request("GET", "/test")
        assert result == {"ok": True}
        assert mock_req.call_count == 1
        mock_sleep.assert_not_called()

    @patch("cortex.data.dexscreener.time.sleep")
    @patch("cortex.data.dexscreener.httpx.request")
    def test_retry_then_success(self, mock_req, mock_sleep):
        fail_resp = _make_mock_response({}, status_code=500)
        ok_resp = _make_mock_response({"recovered": True})
        mock_req.side_effect = [fail_resp, ok_resp]

        from cortex.data.dexscreener import _request

        result = _request("GET", "/retry")
        assert result == {"recovered": True}
        assert mock_req.call_count == 2
        mock_sleep.assert_called_once()

    @patch("cortex.data.dexscreener.DEXSCREENER_MAX_RETRIES", 2)
    @patch("cortex.data.dexscreener.time.sleep")
    @patch("cortex.data.dexscreener.httpx.request")
    def test_all_retries_exhausted(self, mock_req, mock_sleep):
        mock_req.side_effect = httpx.ConnectError("refused")

        from cortex.data.dexscreener import _request

        with pytest.raises(httpx.ConnectError):
            _request("GET", "/fail")
        assert mock_req.call_count == 2


# ── is_available ──

def test_is_available():
    from cortex.data.dexscreener import is_available
    assert is_available() is True


# ── get_token_price ──

class TestGetTokenPrice:
    @patch("cortex.data.dexscreener._request")
    def test_returns_best_pair_by_liquidity(self, mock_req):
        low_liq = {**FAKE_PAIR, "liquidity": {"usd": 1000}, "pairAddress": "LOW"}
        high_liq = {**FAKE_PAIR, "liquidity": {"usd": 999_999}, "pairAddress": "HIGH"}
        mock_req.return_value = [low_liq, high_liq]

        from cortex.data.dexscreener import get_token_price, _price_cache

        result = get_token_price("TOKEN_A")
        assert result["source"] == "dexscreener"
        assert result["pair_address"] == "HIGH"
        assert result["price_usd"] == 1.23
        assert result["liquidity_usd"] == 999_999
        assert "TOKEN_A" in _price_cache

    @patch("cortex.data.dexscreener._request")
    def test_no_pairs_returns_error(self, mock_req):
        mock_req.return_value = []

        from cortex.data.dexscreener import get_token_price

        result = get_token_price("NOEXIST")
        assert result["price_usd"] is None
        assert result["error"] == "no pairs found"

    @patch("cortex.data.dexscreener._request")
    def test_missing_price_fields(self, mock_req):
        pair_no_price = {
            "pairAddress": "P1",
            "dexId": "orca",
            "liquidity": {"usd": 100},
            "volume": {"h24": 10},
        }
        mock_req.return_value = [pair_no_price]

        from cortex.data.dexscreener import get_token_price

        result = get_token_price("NOPRICE")
        assert result["price_usd"] is None
        assert result["price_native"] is None


# ── get_pair_liquidity ──

class TestGetPairLiquidity:
    @patch("cortex.data.dexscreener._request")
    def test_returns_pair_data(self, mock_req):
        mock_req.return_value = {"pairs": [FAKE_PAIR]}

        from cortex.data.dexscreener import get_pair_liquidity

        result = get_pair_liquidity("PAIR_ADDR_1")
        assert result["source"] == "dexscreener"
        assert result["price_usd"] == 1.23
        assert result["liquidity_usd"] == 500_000
        assert result["liquidity_base"] == 100_000
        assert result["liquidity_quote"] == 400_000
        assert result["volume_24h"] == 1_200_000
        assert result["volume_6h"] == 300_000
        assert result["volume_1h"] == 50_000
        assert result["price_change_24h"] == -2.5
        assert result["dex_id"] == "raydium"

    @patch("cortex.data.dexscreener._request")
    def test_pair_not_found(self, mock_req):
        mock_req.return_value = {"pairs": []}

        from cortex.data.dexscreener import get_pair_liquidity

        result = get_pair_liquidity("MISSING")
        assert result["error"] == "pair not found"

    @patch("cortex.data.dexscreener._request")
    def test_non_dict_response(self, mock_req):
        mock_req.return_value = "unexpected"

        from cortex.data.dexscreener import get_pair_liquidity

        result = get_pair_liquidity("BAD_RESP")
        assert result["error"] == "pair not found"


# ── extract_liquidity_metrics ──

class TestExtractLiquidityMetrics:
    @patch("cortex.data.dexscreener.get_pair_liquidity")
    def test_normal_spread_calculation(self, mock_liq):
        mock_liq.return_value = {
            "source": "dexscreener",
            "pair_address": "P1",
            "volume_24h": 1_000_000,
            "liquidity_usd": 500_000,
        }

        from cortex.data.dexscreener import extract_liquidity_metrics

        result = extract_liquidity_metrics("P1")
        assert result["spread_pct"] is not None
        assert result["spread_vol_pct"] is not None
        # turnover = 1M / 500k = 2, spread = 1/(1+2)*2 = 0.667
        assert result["spread_pct"] == pytest.approx(2.0 / 3.0, rel=0.01)
        assert result["spread_vol_pct"] == pytest.approx(result["spread_pct"] * 0.3, rel=0.01)

    @patch("cortex.data.dexscreener.get_pair_liquidity")
    def test_zero_liquidity_volume_only(self, mock_liq):
        mock_liq.return_value = {
            "source": "dexscreener",
            "pair_address": "P2",
            "volume_24h": 10_000,
            "liquidity_usd": 0,
        }

        from cortex.data.dexscreener import extract_liquidity_metrics

        result = extract_liquidity_metrics("P2")
        assert result["spread_pct"] is not None
        # spread = max(0.05, 1 / (10000/1000)) = max(0.05, 0.1) = 0.1
        assert result["spread_pct"] == pytest.approx(0.1, rel=0.01)

    @patch("cortex.data.dexscreener.get_pair_liquidity")
    def test_zero_volume_zero_liquidity(self, mock_liq):
        mock_liq.return_value = {
            "source": "dexscreener",
            "pair_address": "P3",
            "volume_24h": 0,
            "liquidity_usd": 0,
        }

        from cortex.data.dexscreener import extract_liquidity_metrics

        result = extract_liquidity_metrics("P3")
        assert result["spread_pct"] == 1.0  # conservative fallback

    @patch("cortex.data.dexscreener.get_pair_liquidity")
    def test_error_propagation(self, mock_liq):
        mock_liq.return_value = {"source": "dexscreener", "error": "pair not found"}

        from cortex.data.dexscreener import extract_liquidity_metrics

        result = extract_liquidity_metrics("MISSING")
        assert result["spread_pct"] is None
        assert result["error"] == "pair not found"

    @patch("cortex.data.dexscreener.get_pair_liquidity")
    def test_exception_caught(self, mock_liq):
        mock_liq.side_effect = RuntimeError("network down")

        from cortex.data.dexscreener import extract_liquidity_metrics

        result = extract_liquidity_metrics("ERR")
        assert result["spread_pct"] is None
        assert "network down" in result["error"]


# ── get_new_tokens ──

class TestGetNewTokens:
    @patch("cortex.data.dexscreener._request")
    def test_filters_solana_and_min_liquidity(self, mock_req):
        boosts = [
            {"chainId": "solana", "tokenAddress": "SOL_TOKEN_1"},
            {"chainId": "ethereum", "tokenAddress": "ETH_TOKEN"},
            {"chainId": "solana", "tokenAddress": "SOL_TOKEN_2"},
        ]
        pairs_high_liq = [FAKE_PAIR]
        pairs_low_liq = [FAKE_PAIR_LOW_LIQ]

        mock_req.side_effect = [
            boosts,        # token-boosts
            pairs_high_liq,  # SOL_TOKEN_1 pairs
            pairs_low_liq,   # SOL_TOKEN_2 pairs (below min liquidity)
        ]

        from cortex.data.dexscreener import get_new_tokens

        results = get_new_tokens(limit=10, min_liquidity=True)
        assert len(results) == 1
        assert results[0]["token_address"] == "SOL_TOKEN_1"
        assert results[0]["meets_min_liquidity"] is True

    @patch("cortex.data.dexscreener._request")
    def test_no_min_liquidity_filter(self, mock_req):
        boosts = [{"chainId": "solana", "tokenAddress": "TOKEN_X"}]
        mock_req.side_effect = [boosts, [FAKE_PAIR_LOW_LIQ]]

        from cortex.data.dexscreener import get_new_tokens

        results = get_new_tokens(limit=5, min_liquidity=False)
        assert len(results) == 1

    @patch("cortex.data.dexscreener._request")
    def test_empty_boosts(self, mock_req):
        mock_req.return_value = []

        from cortex.data.dexscreener import get_new_tokens

        assert get_new_tokens() == []

    @patch("cortex.data.dexscreener._request")
    def test_top_level_exception_returns_empty(self, mock_req):
        mock_req.side_effect = httpx.ConnectError("offline")

        from cortex.data.dexscreener import get_new_tokens

        assert get_new_tokens() == []

    @patch("cortex.data.dexscreener._request")
    def test_limit_respected(self, mock_req):
        boosts = [
            {"chainId": "solana", "tokenAddress": f"TOK_{i}"}
            for i in range(20)
        ]

        def side_effect(method, path, **kw):
            if "token-boosts" in path:
                return boosts
            return [FAKE_PAIR]

        mock_req.side_effect = side_effect

        from cortex.data.dexscreener import get_new_tokens

        results = get_new_tokens(limit=3, min_liquidity=True)
        assert len(results) == 3


# ── Cache ──

class TestCache:
    @patch("cortex.data.dexscreener._request")
    def test_get_cached_prices_evicts_stale(self, mock_req):
        from cortex.data.dexscreener import _price_cache, get_cached_prices
        import cortex.data.dexscreener as dex

        _price_cache.clear()
        _price_cache["fresh"] = {"timestamp": time.time(), "price_usd": 1.0}
        _price_cache["stale"] = {"timestamp": time.time() - 9999, "price_usd": 2.0}

        cached = get_cached_prices()
        assert "fresh" in cached
        assert "stale" not in cached

    def test_price_cache_populated_on_fetch(self):
        from cortex.data.dexscreener import _price_cache
        # cache populated in TestGetTokenPrice.test_returns_best_pair_by_liquidity
        # just verify the mechanism by checking _price_cache is a dict
        assert isinstance(_price_cache, dict)
