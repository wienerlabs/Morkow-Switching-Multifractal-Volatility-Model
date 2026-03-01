"""Tests for cortex.data.launch_tracker — CEX-funded launch detection."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

# Ensure test mode
os.environ.setdefault("TESTING", "1")

from cortex.data.launch_tracker import (
    LaunchInfo,
    FundingSource,
    EarlyBuyer,
    BundleCluster,
    LaunchRiskResult,
    CEX_HOT_WALLETS,
    detect_bundle_cluster,
    get_token_creation_info,
    get_deployer_funding_sources,
    get_early_buyers,
    compute_launch_risk,
    _cache,
)


# ── Fixtures ──

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear module cache before each test."""
    _cache.clear()
    yield
    _cache.clear()


def _make_buyer(wallet: str, offset_sec: float, amount: float = 1.0) -> EarlyBuyer:
    """Helper to create an EarlyBuyer with a time offset from a base time."""
    base = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return EarlyBuyer(
        wallet=wallet,
        buy_time=base + timedelta(seconds=offset_sec),
        amount_sol=amount,
        tx_signature=f"sig_{wallet}_{offset_sec}",
        latency_from_deploy_sec=offset_sec,
    )


# ── Dataclass Tests ──

class TestDataclasses:
    def test_launch_info_creation(self):
        now = datetime.now(timezone.utc)
        info = LaunchInfo(
            token_mint="mint123",
            deployer_wallet="deployer456",
            created_at=now,
            token_age_hours=2.5,
        )
        assert info.token_mint == "mint123"
        assert info.deployer_wallet == "deployer456"
        assert info.token_age_hours == 2.5

    def test_funding_source_cex(self):
        fs = FundingSource(
            source_address="addr1",
            source_type="cex",
            amount_sol=10.0,
            exchange_name="Binance",
            tx_signature="sig1",
        )
        assert fs.source_type == "cex"
        assert fs.exchange_name == "Binance"

    def test_funding_source_unknown(self):
        fs = FundingSource(
            source_address="addr2",
            source_type="unknown",
            amount_sol=5.0,
            exchange_name=None,
            tx_signature="sig2",
        )
        assert fs.source_type == "unknown"
        assert fs.exchange_name is None

    def test_early_buyer(self):
        now = datetime.now(timezone.utc)
        buyer = EarlyBuyer(
            wallet="wallet1",
            buy_time=now,
            amount_sol=2.5,
            tx_signature="sig3",
            latency_from_deploy_sec=15.0,
        )
        assert buyer.latency_from_deploy_sec == 15.0

    def test_bundle_cluster(self):
        cluster = BundleCluster(
            wallets=["w1", "w2", "w3"],
            buy_window_sec=10.0,
            total_amount_sol=30.0,
            is_bundle=True,
        )
        assert cluster.is_bundle
        assert len(cluster.wallets) == 3

    def test_launch_risk_result_defaults(self):
        result = LaunchRiskResult(
            score=50,
            cex_funded=True,
            bundle_detected=False,
            deployer_age_days=3.0,
            top10_concentration_pct=45.0,
            deploy_to_first_trade_sec=30.0,
        )
        assert result.details == {}
        assert result.risk_factors == []


# ── CEX Wallet Tests ──

class TestCEXWallets:
    def test_known_wallets_not_empty(self):
        assert len(CEX_HOT_WALLETS) > 0

    def test_binance_in_wallets(self):
        binance_wallets = [k for k, v in CEX_HOT_WALLETS.items() if v == "Binance"]
        assert len(binance_wallets) > 0

    def test_multiple_exchanges(self):
        exchanges = set(CEX_HOT_WALLETS.values())
        assert len(exchanges) >= 4  # At least Binance, Coinbase, Kraken, Bybit


# ── Bundle Detection Tests (Pure Function) ──

class TestBundleDetection:
    def test_no_buyers(self):
        cluster = detect_bundle_cluster([])
        assert not cluster.is_bundle
        assert cluster.wallets == []
        assert cluster.total_amount_sol == 0.0

    def test_one_buyer(self):
        buyers = [_make_buyer("w1", 0.0)]
        cluster = detect_bundle_cluster(buyers)
        assert not cluster.is_bundle

    def test_two_buyers_not_enough(self):
        buyers = [_make_buyer("w1", 0.0), _make_buyer("w2", 5.0)]
        cluster = detect_bundle_cluster(buyers)
        assert not cluster.is_bundle
        assert len(cluster.wallets) == 2

    def test_exactly_three_in_window_is_bundle(self):
        buyers = [
            _make_buyer("w1", 0.0),
            _make_buyer("w2", 10.0),
            _make_buyer("w3", 25.0),
        ]
        cluster = detect_bundle_cluster(buyers, window_sec=30.0)
        assert cluster.is_bundle
        assert len(cluster.wallets) == 3

    def test_three_outside_window_not_bundle(self):
        buyers = [
            _make_buyer("w1", 0.0),
            _make_buyer("w2", 20.0),
            _make_buyer("w3", 60.0),  # outside 30s window from w1
        ]
        cluster = detect_bundle_cluster(buyers, window_sec=30.0)
        assert not cluster.is_bundle

    def test_boundary_exactly_at_window(self):
        buyers = [
            _make_buyer("w1", 0.0),
            _make_buyer("w2", 15.0),
            _make_buyer("w3", 30.0),  # exactly at 30s boundary
        ]
        cluster = detect_bundle_cluster(buyers, window_sec=30.0)
        assert cluster.is_bundle

    def test_five_buyers_four_in_cluster(self):
        buyers = [
            _make_buyer("w1", 0.0, 2.0),
            _make_buyer("w2", 5.0, 3.0),
            _make_buyer("w3", 10.0, 1.5),
            _make_buyer("w4", 20.0, 2.5),
            _make_buyer("w5", 120.0, 5.0),  # outlier
        ]
        cluster = detect_bundle_cluster(buyers, window_sec=30.0)
        assert cluster.is_bundle
        assert len(cluster.wallets) == 4
        assert cluster.total_amount_sol == pytest.approx(9.0)

    def test_custom_min_wallets(self):
        buyers = [
            _make_buyer("w1", 0.0),
            _make_buyer("w2", 5.0),
            _make_buyer("w3", 10.0),
        ]
        # min_wallets=4 → 3 is not enough
        cluster = detect_bundle_cluster(buyers, window_sec=30.0, min_wallets=4)
        assert not cluster.is_bundle

    def test_unsorted_input(self):
        buyers = [
            _make_buyer("w3", 25.0),
            _make_buyer("w1", 0.0),
            _make_buyer("w2", 10.0),
        ]
        cluster = detect_bundle_cluster(buyers, window_sec=30.0)
        assert cluster.is_bundle

    def test_duplicate_wallets_counted_once(self):
        buyers = [
            _make_buyer("w1", 0.0),
            _make_buyer("w1", 5.0),  # same wallet
            _make_buyer("w2", 10.0),
        ]
        cluster = detect_bundle_cluster(buyers, window_sec=30.0)
        # Only 2 unique wallets
        assert len(cluster.wallets) == 2
        assert not cluster.is_bundle


# ── Mocked API Tests ──

class TestGetTokenCreationInfo:
    @patch("cortex.data.launch_tracker._post_rpc")
    def test_returns_launch_info(self, mock_rpc):
        mock_rpc.return_value = {
            "result": {
                "authorities": [{"address": "deployer1", "scopes": ["mint"]}],
                "created_at": "2025-01-01T12:00:00Z",
                "token_info": {},
                "content": {"metadata": {}},
            }
        }
        info = get_token_creation_info("mint_abc")
        assert info is not None
        assert info.deployer_wallet == "deployer1"
        assert info.token_mint == "mint_abc"

    @patch("cortex.data.launch_tracker._post_rpc")
    def test_returns_none_on_rpc_failure(self, mock_rpc):
        mock_rpc.return_value = None
        info = get_token_creation_info("mint_bad")
        assert info is None

    @patch("cortex.data.launch_tracker._post_rpc")
    def test_returns_none_on_no_authority(self, mock_rpc):
        mock_rpc.return_value = {
            "result": {
                "authorities": [],
                "content": {},
            }
        }
        info = get_token_creation_info("mint_no_auth")
        assert info is None

    @patch("cortex.data.launch_tracker._post_rpc")
    def test_caches_result(self, mock_rpc):
        mock_rpc.return_value = {
            "result": {
                "authorities": [{"address": "dep2", "scopes": ["mint"]}],
                "created_at": "2025-06-01T00:00:00Z",
                "token_info": {},
                "content": {},
            }
        }
        info1 = get_token_creation_info("mint_cache_test")
        info2 = get_token_creation_info("mint_cache_test")
        assert info1 is info2
        assert mock_rpc.call_count == 1  # second call used cache


class TestGetDeployerFundingSources:
    @patch("cortex.data.launch_tracker._post_rpc")
    def test_returns_empty_on_failure(self, mock_rpc):
        mock_rpc.return_value = None
        sources = get_deployer_funding_sources("deployer_bad")
        assert sources == []

    @patch("cortex.data.launch_tracker._post_rpc")
    def test_detects_cex_funding(self, mock_rpc):
        binance_addr = list(CEX_HOT_WALLETS.keys())[0]
        binance_name = CEX_HOT_WALLETS[binance_addr]

        def side_effect(payload, url=None):
            method = payload.get("method", "")
            if method == "getSignaturesForAddress":
                return {"result": [{"signature": "sig_fund_1"}]}
            if method == "getTransaction":
                return {
                    "result": {
                        "transaction": {
                            "message": {
                                "accountKeys": [binance_addr, "deployer_x"]
                            }
                        },
                        "meta": {
                            "err": None,
                            "preBalances": [10_000_000_000, 0],
                            "postBalances": [5_000_000_000, 5_000_000_000],
                        },
                    }
                }
            return None

        mock_rpc.side_effect = side_effect
        sources = get_deployer_funding_sources("deployer_x")
        assert len(sources) == 1
        assert sources[0].source_type == "cex"
        assert sources[0].exchange_name == binance_name


class TestComputeLaunchRisk:
    @patch("cortex.data.launch_tracker.get_token_creation_info")
    def test_returns_zero_when_info_unavailable(self, mock_info):
        mock_info.return_value = None
        result = compute_launch_risk("mint_unknown")
        assert result.score == 0
        assert "error" in result.details

    @patch("cortex.data.launch_tracker.get_early_buyers")
    @patch("cortex.data.launch_tracker.get_deployer_funding_sources")
    @patch("cortex.data.launch_tracker.get_token_creation_info")
    def test_short_circuit_old_token(self, mock_info, mock_funding, mock_buyers):
        mock_info.return_value = LaunchInfo(
            token_mint="old_mint",
            deployer_wallet="dep",
            created_at=datetime.now(timezone.utc) - timedelta(hours=48),
            token_age_hours=48.0,
        )
        result = compute_launch_risk("old_mint")
        assert result.score == 0
        assert result.details.get("short_circuit") == "token_too_old"
        mock_funding.assert_not_called()
        mock_buyers.assert_not_called()

    @patch("cortex.data.launch_tracker.detect_bundle_cluster")
    @patch("cortex.data.launch_tracker.get_early_buyers")
    @patch("cortex.data.launch_tracker.get_deployer_funding_sources")
    @patch("cortex.data.launch_tracker.get_token_creation_info")
    def test_max_score_all_risk_factors(self, mock_info, mock_funding, mock_buyers, mock_bundle):
        now = datetime.now(timezone.utc)
        mock_info.return_value = LaunchInfo(
            token_mint="risky_mint",
            deployer_wallet="dep_risky",
            created_at=now - timedelta(hours=2),
            token_age_hours=2.0,
        )
        mock_funding.return_value = [
            FundingSource("cex_addr", "cex", 50.0, "Binance", "sig_f1")
        ]
        mock_buyers.return_value = [
            EarlyBuyer("b1", now - timedelta(hours=1, minutes=59), 5.0, "sig_b1", 10.0),
            EarlyBuyer("b2", now - timedelta(hours=1, minutes=59), 3.0, "sig_b2", 15.0),
            EarlyBuyer("b3", now - timedelta(hours=1, minutes=58), 4.0, "sig_b3", 20.0),
        ]
        mock_bundle.return_value = BundleCluster(
            wallets=["b1", "b2", "b3"],
            buy_window_sec=10.0,
            total_amount_sol=12.0,
            is_bundle=True,
        )

        result = compute_launch_risk("risky_mint", holder_concentration_pct=90.0)

        # CEX(25) + Bundle(30) + FastDeploy(15) + Concentration(20) + YoungDeployer(10) = 100
        assert result.score == 100
        assert result.cex_funded is True
        assert result.bundle_detected is True
        assert len(result.risk_factors) == 5

    @patch("cortex.data.launch_tracker.detect_bundle_cluster")
    @patch("cortex.data.launch_tracker.get_early_buyers")
    @patch("cortex.data.launch_tracker.get_deployer_funding_sources")
    @patch("cortex.data.launch_tracker.get_token_creation_info")
    def test_partial_score(self, mock_info, mock_funding, mock_buyers, mock_bundle):
        now = datetime.now(timezone.utc)
        mock_info.return_value = LaunchInfo(
            token_mint="partial_mint",
            deployer_wallet="dep_partial",
            created_at=now - timedelta(hours=2),
            token_age_hours=2.0,
        )
        mock_funding.return_value = [
            FundingSource("unknown_addr", "unknown", 10.0, None, "sig_f2")
        ]
        mock_buyers.return_value = [
            EarlyBuyer("b1", now - timedelta(minutes=110), 5.0, "sig_b1", 120.0),
        ]
        mock_bundle.return_value = BundleCluster(
            wallets=["b1"],
            buy_window_sec=0.0,
            total_amount_sol=5.0,
            is_bundle=False,
        )

        result = compute_launch_risk("partial_mint", holder_concentration_pct=30.0)

        # No CEX, no bundle, deploy_to_trade > 60s, concentration < 80%, young deployer(10)
        assert result.score == 10
        assert result.cex_funded is False
        assert result.bundle_detected is False

    @patch("cortex.data.launch_tracker.detect_bundle_cluster")
    @patch("cortex.data.launch_tracker.get_early_buyers")
    @patch("cortex.data.launch_tracker.get_deployer_funding_sources")
    @patch("cortex.data.launch_tracker.get_token_creation_info")
    def test_score_clamped_to_100(self, mock_info, mock_funding, mock_buyers, mock_bundle):
        now = datetime.now(timezone.utc)
        mock_info.return_value = LaunchInfo(
            token_mint="max_mint",
            deployer_wallet="dep_max",
            created_at=now - timedelta(minutes=30),
            token_age_hours=0.5,
        )
        mock_funding.return_value = [
            FundingSource("cex1", "cex", 100.0, "Binance", "sig1"),
        ]
        mock_buyers.return_value = [
            _make_buyer("w1", 5.0, 10.0),
            _make_buyer("w2", 10.0, 10.0),
            _make_buyer("w3", 15.0, 10.0),
            _make_buyer("w4", 20.0, 10.0),
        ]
        mock_bundle.return_value = BundleCluster(
            wallets=["w1", "w2", "w3", "w4"],
            buy_window_sec=15.0,
            total_amount_sol=40.0,
            is_bundle=True,
        )

        result = compute_launch_risk("max_mint", holder_concentration_pct=95.0)
        assert result.score <= 100
