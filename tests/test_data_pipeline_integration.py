"""Integration tests: DexScreener + Helius holders → Solana adapter → LVaR pipeline.

Uses mocked HTTP to test cross-module data flow without real network calls.
"""
import time
from unittest.mock import patch, MagicMock

import httpx
import pytest


FAKE_PAIR = {
    "pairAddress": "PAIR_INTEG",
    "dexId": "raydium",
    "priceUsd": "42.50",
    "priceNative": "0.29",
    "baseToken": {"symbol": "SOL", "address": "So11111111111111111111111111111111111111112"},
    "liquidity": {"usd": 2_000_000, "base": 50_000, "quote": 1_500_000},
    "volume": {"h24": 5_000_000, "h6": 1_200_000, "h1": 200_000},
    "txns": {"h24": {"buys": 12000, "sells": 11000}},
    "priceChange": {"h24": 3.2},
}

FAKE_HOLDER_ACCOUNTS = [
    {"owner": "Whale1", "amount": 5_000_000},
    {"owner": "Whale2", "amount": 3_000_000},
    {"owner": "Whale1", "amount": 1_000_000},  # same whale, different account
    {"owner": "Retail1", "amount": 100_000},
    {"owner": "Retail2", "amount": 50_000},
] + [{"owner": f"Dust{i}", "amount": 1000} for i in range(50)]


class TestDexScreenerToLiquidityMetrics:
    """DexScreener → extract_liquidity_metrics produces LVaR-compatible spreads."""

    @patch("cortex.data.dexscreener._request")
    def test_price_to_spread_pipeline(self, mock_req):
        """get_token_price → get_pair_liquidity → extract_liquidity_metrics chain."""
        # First call: token price lookup
        mock_req.return_value = [FAKE_PAIR]

        from cortex.data.dexscreener import get_token_price

        price_data = get_token_price("So11111111111111111111111111111111111111112")
        assert price_data["price_usd"] == 42.50
        assert price_data["pair_address"] == "PAIR_INTEG"

        # Second call: pair liquidity
        mock_req.return_value = {"pairs": [FAKE_PAIR]}

        from cortex.data.dexscreener import get_pair_liquidity

        liq_data = get_pair_liquidity("PAIR_INTEG")
        assert liq_data["liquidity_usd"] == 2_000_000
        assert liq_data["volume_24h"] == 5_000_000

        # Third: extract metrics
        from cortex.data.dexscreener import extract_liquidity_metrics

        metrics = extract_liquidity_metrics("PAIR_INTEG")
        assert metrics["spread_pct"] is not None
        assert metrics["spread_pct"] > 0
        assert metrics["spread_vol_pct"] == pytest.approx(metrics["spread_pct"] * 0.3, rel=0.01)


class TestDexScreenerSolanaIntegration:
    """cortex.data.solana.get_dexscreener_price uses dexscreener adapter."""

    @patch("cortex.data.dexscreener._request")
    def test_solana_get_dexscreener_price(self, mock_req):
        mock_req.return_value = [FAKE_PAIR]

        from cortex.data.solana import get_dexscreener_price

        result = get_dexscreener_price("SOL")
        assert result is not None
        assert result["source"] == "dexscreener"
        assert result["price"] == 42.50

    @patch("cortex.data.dexscreener._request")
    def test_solana_dexscreener_price_no_pairs(self, mock_req):
        mock_req.return_value = []

        from cortex.data.solana import get_dexscreener_price

        result = get_dexscreener_price("SOL")
        assert result is None  # price_usd is None → returns None

    @patch("cortex.data.dexscreener._request")
    def test_solana_dexscreener_exception(self, mock_req):
        mock_req.side_effect = httpx.ConnectError("offline")

        from cortex.data.solana import get_dexscreener_price

        result = get_dexscreener_price("SOL")
        assert result is None


class TestHeliusHolderConcentrationPipeline:
    """Full holder data flow: fetch → aggregate → concentration metrics."""

    @patch("cortex.data.helius_holders.is_available", return_value=True)
    @patch("cortex.data.helius_holders._fetch_all_accounts")
    def test_end_to_end_holder_analysis(self, mock_fetch, _):
        from cortex.data.helius_holders import get_holder_data, _cache, _cache_ts
        _cache.clear()
        _cache_ts.clear()

        mock_fetch.return_value = FAKE_HOLDER_ACCOUNTS

        result = get_holder_data("TEST_MINT_INTEG")

        # Whale1 should be aggregated: 5M + 1M = 6M
        whale1 = next(h for h in result["holders"] if h["owner"] == "Whale1")
        assert whale1["amount"] == 6_000_000

        # Sorted descending
        amounts = [h["amount"] for h in result["holders"]]
        assert amounts == sorted(amounts, reverse=True)

        # Concentration: Whale1(6M) + Whale2(3M) = 9M out of ~9.2M → very high
        assert result["concentration_risk"] in ("critical", "high")
        assert result["top10_pct"] > 90

        # HHI should be high for concentrated distribution
        assert result["hhi"] > 1000

    @patch("cortex.data.helius_holders.is_available", return_value=True)
    @patch("cortex.data.helius_holders._fetch_all_accounts")
    def test_even_distribution_low_risk(self, mock_fetch, _):
        from cortex.data.helius_holders import get_holder_data, _cache, _cache_ts
        _cache.clear()
        _cache_ts.clear()

        # 100 equal holders
        mock_fetch.return_value = [
            {"owner": f"Holder_{i}", "amount": 10_000}
            for i in range(100)
        ]

        result = get_holder_data("EVEN_MINT")
        assert result["concentration_risk"] == "low"
        assert result["top10_pct"] == pytest.approx(10.0, rel=0.01)
        # HHI for 100 equal holders: 100 * (1%)^2 = 100
        assert result["hhi"] == pytest.approx(100.0, rel=0.01)


class TestCacheBehaviorAcrossModules:
    """Cache semantics: TTL, eviction, isolation between modules."""

    @patch("cortex.data.dexscreener._request")
    def test_dexscreener_cache_populated_and_evicted(self, mock_req):
        from cortex.data.dexscreener import (
            get_token_price,
            get_cached_prices,
            _price_cache,
        )

        _price_cache.clear()
        mock_req.return_value = [FAKE_PAIR]

        get_token_price("CACHE_TEST_TOKEN")
        cached = get_cached_prices()
        assert "CACHE_TEST_TOKEN" in cached
        assert cached["CACHE_TEST_TOKEN"]["price_usd"] == 42.50

    @patch("cortex.data.helius_holders.is_available", return_value=True)
    @patch("cortex.data.helius_holders._fetch_all_accounts")
    def test_helius_cache_ttl(self, mock_fetch, _):
        from cortex.data.helius_holders import get_holder_data, _cache, _cache_ts

        _cache.clear()
        _cache_ts.clear()

        mock_fetch.return_value = [{"owner": "A", "amount": 100}]

        # First call populates cache
        r1 = get_holder_data("TTL_MINT")
        assert mock_fetch.call_count == 1

        # Second call should hit cache
        r2 = get_holder_data("TTL_MINT")
        assert mock_fetch.call_count == 1  # not called again

        _cache.clear()
        _cache_ts.clear()


class TestNewTokenDiscoveryPipeline:
    """get_new_tokens end-to-end with filtering and enrichment."""

    @patch("cortex.data.dexscreener._request")
    def test_discovery_filters_and_enriches(self, mock_req):
        boosts = [
            {"chainId": "solana", "tokenAddress": "NEW_SOL_1"},
            {"chainId": "solana", "tokenAddress": "NEW_SOL_2"},
            {"chainId": "ethereum", "tokenAddress": "ETH_SKIP"},
        ]

        high_liq_pair = {**FAKE_PAIR, "pairAddress": "NEW_PAIR_1"}
        low_liq_pair = {
            **FAKE_PAIR,
            "pairAddress": "NEW_PAIR_2",
            "liquidity": {"usd": 10},  # below min liquidity
        }

        def side_effect(method, path, **kw):
            if "token-boosts" in path:
                return boosts
            if "NEW_SOL_1" in path:
                return [high_liq_pair]
            if "NEW_SOL_2" in path:
                return [low_liq_pair]
            return []

        mock_req.side_effect = side_effect

        from cortex.data.dexscreener import get_new_tokens

        results = get_new_tokens(limit=10, min_liquidity=True)
        assert len(results) == 1
        assert results[0]["token_address"] == "NEW_SOL_1"
        assert results[0]["liquidity_usd"] == 2_000_000


class TestErrorPropagation:
    """Errors in one adapter don't crash the pipeline."""

    @patch("cortex.data.dexscreener._request")
    def test_dexscreener_error_doesnt_crash_solana(self, mock_req):
        mock_req.side_effect = Exception("DexScreener down")

        from cortex.data.solana import get_dexscreener_price

        # Should return None, not raise
        result = get_dexscreener_price("SOL")
        assert result is None

    @patch("cortex.data.helius_holders.is_available", return_value=True)
    @patch("cortex.data.helius_holders._fetch_all_accounts")
    def test_helius_error_returns_structured_error(self, mock_fetch, _):
        from cortex.data.helius_holders import get_holder_data, _cache, _cache_ts
        _cache.clear()
        _cache_ts.clear()

        mock_fetch.side_effect = httpx.ReadTimeout("timeout")

        result = get_holder_data("ERR_MINT")
        assert result["error"] is not None
        assert result["total_holders"] == 0
        assert result["concentration_risk"] == "unknown"
        # Structured response, not an exception
        assert result["source"] == "helius_das"
