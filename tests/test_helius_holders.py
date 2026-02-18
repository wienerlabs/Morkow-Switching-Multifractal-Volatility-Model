"""Tests for cortex/data/helius_holders.py — Helius DAS holder concentration."""
import time
from unittest.mock import patch, MagicMock

import httpx
import pytest


# ── _compute_concentration (pure logic, no mocks needed) ──

class TestComputeConcentration:
    def _call(self, holders):
        from cortex.data.helius_holders import _compute_concentration
        return _compute_concentration(holders)

    def test_empty_holders(self):
        result = self._call([])
        assert result["concentration_risk"] == "unknown"
        assert result["hhi"] == 0.0

    def test_zero_total(self):
        result = self._call([{"amount": 0}, {"amount": 0}])
        assert result["concentration_risk"] == "unknown"

    def test_single_holder_critical(self):
        result = self._call([{"amount": 1_000_000}])
        assert result["top10_pct"] == 100.0
        assert result["concentration_risk"] == "critical"
        assert result["hhi"] == 10_000.0  # 100^2

    def test_two_equal_holders(self):
        result = self._call([{"amount": 500}, {"amount": 500}])
        assert result["top10_pct"] == 100.0
        assert result["concentration_risk"] == "critical"
        # HHI = 50^2 + 50^2 = 5000
        assert result["hhi"] == pytest.approx(5000.0, rel=0.01)

    def test_many_small_holders_low_risk(self):
        holders = [{"amount": 10} for _ in range(100)]
        result = self._call(holders)
        assert result["top10_pct"] == pytest.approx(10.0, rel=0.01)
        assert result["concentration_risk"] == "low"

    def test_medium_concentration(self):
        # 5 holders with 8% each (40%) + 60 holders with 1% each
        holders = [{"amount": 80} for _ in range(5)]
        holders += [{"amount": 10} for _ in range(60)]
        result = self._call(holders)
        assert result["concentration_risk"] == "medium"

    def test_high_concentration(self):
        # 3 holders dominate 60%, rest are small
        holders = [{"amount": 200} for _ in range(3)]
        holders += [{"amount": 4} for _ in range(100)]
        result = self._call(holders)
        assert result["concentration_risk"] == "high"

    def test_sorting_is_independent_of_input_order(self):
        holders_asc = [{"amount": 1}, {"amount": 10}, {"amount": 100}]
        holders_desc = [{"amount": 100}, {"amount": 10}, {"amount": 1}]
        assert self._call(holders_asc) == self._call(holders_desc)

    def test_top50_includes_all_when_fewer(self):
        holders = [{"amount": 100} for _ in range(5)]
        result = self._call(holders)
        assert result["top50_pct"] == 100.0
        assert result["top10_pct"] == 100.0


# ── is_available ──

class TestIsAvailable:
    @patch("cortex.data.helius_holders.HELIUS_API_KEY", "test-key-123")
    def test_available_with_key(self):
        from cortex.data.helius_holders import is_available
        assert is_available() is True

    @patch("cortex.data.helius_holders.HELIUS_API_KEY", "")
    def test_unavailable_without_key(self):
        from cortex.data.helius_holders import is_available
        assert is_available() is False


# ── _post_with_retry ──

class TestPostWithRetry:
    @patch("cortex.data.helius_holders.time.sleep")
    @patch("cortex.data.helius_holders.httpx.post")
    def test_success_first_try(self, mock_post, mock_sleep):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"result": {"token_accounts": []}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        from cortex.data.helius_holders import _post_with_retry

        result = _post_with_retry({"test": True})
        assert result == {"result": {"token_accounts": []}}
        mock_sleep.assert_not_called()

    @patch("cortex.data.helius_holders.time.sleep")
    @patch("cortex.data.helius_holders.httpx.post")
    def test_retry_on_failure(self, mock_post, mock_sleep):
        mock_post.side_effect = [
            httpx.ConnectError("refused"),
            MagicMock(
                json=MagicMock(return_value={"ok": True}),
                raise_for_status=MagicMock(),
            ),
        ]

        from cortex.data.helius_holders import _post_with_retry

        result = _post_with_retry({})
        assert result == {"ok": True}
        assert mock_post.call_count == 2

    @patch("cortex.data.helius_holders._MAX_RETRIES", 2)
    @patch("cortex.data.helius_holders.time.sleep")
    @patch("cortex.data.helius_holders.httpx.post")
    def test_exhausted_retries(self, mock_post, mock_sleep):
        mock_post.side_effect = httpx.ConnectError("down")

        from cortex.data.helius_holders import _post_with_retry

        with pytest.raises(httpx.ConnectError):
            _post_with_retry({})


# ── _fetch_all_accounts ──

class TestFetchAllAccounts:
    @patch("cortex.data.helius_holders.HELIUS_RPC_URL", "")
    def test_raises_without_rpc_url(self):
        from cortex.data.helius_holders import _fetch_all_accounts

        with pytest.raises(RuntimeError, match="HELIUS_RPC_URL not configured"):
            _fetch_all_accounts("MINT_ADDR")

    @patch("cortex.data.helius_holders.HELIUS_RPC_URL", "https://rpc.example.com")
    @patch("cortex.data.helius_holders._post_with_retry")
    def test_single_page(self, mock_post):
        page1 = {"result": {"token_accounts": [
            {"owner": "W1", "amount": 1000},
            {"owner": "W2", "amount": 500},
        ]}}
        page2 = {"result": {"token_accounts": []}}  # terminates pagination
        mock_post.side_effect = [page1, page2]

        from cortex.data.helius_holders import _fetch_all_accounts

        accounts = _fetch_all_accounts("MINT")
        assert len(accounts) == 2

    @patch("cortex.data.helius_holders.HELIUS_RPC_URL", "https://rpc.example.com")
    @patch("cortex.data.helius_holders._post_with_retry")
    def test_pagination(self, mock_post):
        page1 = {"result": {"token_accounts": [{"owner": f"W{i}", "amount": 100} for i in range(1000)]}}
        page2 = {"result": {"token_accounts": [{"owner": f"X{i}", "amount": 50} for i in range(200)]}}
        page3 = {"result": {"token_accounts": []}}
        mock_post.side_effect = [page1, page2, page3]

        from cortex.data.helius_holders import _fetch_all_accounts

        accounts = _fetch_all_accounts("MINT")
        assert len(accounts) == 1200


# ── get_holder_data ──

class TestGetHolderData:
    @patch("cortex.data.helius_holders.HELIUS_API_KEY", "")
    def test_unavailable_returns_error(self):
        from cortex.data.helius_holders import get_holder_data

        result = get_holder_data("MINT")
        assert result["error"] == "HELIUS_API_KEY not configured"
        assert result["concentration_risk"] == "unknown"
        assert result["total_holders"] == 0

    @patch("cortex.data.helius_holders.is_available", return_value=True)
    @patch("cortex.data.helius_holders._fetch_all_accounts")
    def test_successful_fetch(self, mock_fetch, _):
        from cortex.data.helius_holders import get_holder_data, _cache, _cache_ts
        _cache.clear()
        _cache_ts.clear()

        mock_fetch.return_value = [
            {"owner": "Whale", "amount": 800_000},
            {"owner": "Whale", "amount": 200_000},  # same owner, should aggregate
            {"owner": "Small1", "amount": 50_000},
            {"owner": "Small2", "amount": 50_000},
        ]

        result = get_holder_data("TEST_MINT")
        assert result["source"] == "helius_das"
        assert result["total_holders"] == 3  # Whale aggregated
        assert result["holders"][0]["owner"] == "Whale"
        assert result["holders"][0]["amount"] == 1_000_000
        assert result["holders"][0]["pct"] > 0
        assert result["concentration_risk"] == "critical"  # Whale has ~91%

    @patch("cortex.data.helius_holders.is_available", return_value=True)
    @patch("cortex.data.helius_holders._fetch_all_accounts")
    def test_cache_hit(self, mock_fetch, _):
        from cortex.data.helius_holders import get_holder_data, _cache, _cache_ts

        cached_result = {"source": "helius_das", "cached": True}
        _cache["CACHED_MINT"] = cached_result
        _cache_ts["CACHED_MINT"] = time.time()

        result = get_holder_data("CACHED_MINT")
        assert result["cached"] is True
        mock_fetch.assert_not_called()

        _cache.clear()
        _cache_ts.clear()

    @patch("cortex.data.helius_holders.is_available", return_value=True)
    @patch("cortex.data.helius_holders._fetch_all_accounts")
    def test_cache_expired(self, mock_fetch, _):
        from cortex.data.helius_holders import get_holder_data, _cache, _cache_ts

        _cache["OLD_MINT"] = {"source": "helius_das", "old": True}
        _cache_ts["OLD_MINT"] = time.time() - 99999

        mock_fetch.return_value = [{"owner": "A", "amount": 100}]

        result = get_holder_data("OLD_MINT")
        assert result.get("old") is not True  # not the cached value
        mock_fetch.assert_called_once()

        _cache.clear()
        _cache_ts.clear()

    @patch("cortex.data.helius_holders.is_available", return_value=True)
    @patch("cortex.data.helius_holders._fetch_all_accounts")
    def test_fetch_exception_returns_error(self, mock_fetch, _):
        from cortex.data.helius_holders import get_holder_data, _cache, _cache_ts
        _cache.clear()
        _cache_ts.clear()

        mock_fetch.side_effect = RuntimeError("RPC timeout")

        result = get_holder_data("ERR_MINT")
        assert "RPC timeout" in result["error"]
        assert result["concentration_risk"] == "unknown"

    @patch("cortex.data.helius_holders.is_available", return_value=True)
    @patch("cortex.data.helius_holders._fetch_all_accounts")
    def test_holder_pct_sums_to_100(self, mock_fetch, _):
        from cortex.data.helius_holders import get_holder_data, _cache, _cache_ts
        _cache.clear()
        _cache_ts.clear()

        mock_fetch.return_value = [
            {"owner": f"W{i}", "amount": (i + 1) * 100}
            for i in range(20)
        ]

        result = get_holder_data("PCT_MINT")
        total_pct = sum(h["pct"] for h in result["holders"])
        assert total_pct == pytest.approx(100.0, abs=0.1)
