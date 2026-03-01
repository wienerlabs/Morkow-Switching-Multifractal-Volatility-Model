"""Tests for cortex.data.ghost_watcher — dormant whale reactivation detection."""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

import pytest

# Ensure test mode
os.environ.setdefault("TESTING", "1")

from cortex.data.ghost_watcher import (
    WalletActivity,
    DormantWallet,
    Reactivation,
    ReactivationCluster,
    GhostWatcherResult,
    get_wallet_last_activity,
    classify_dormant_wallets,
    detect_reactivations,
    detect_reactivation_cluster,
    compute_ghost_risk,
    _cache,
)


# ── Fixtures ──

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear module cache before each test."""
    _cache.clear()
    yield
    _cache.clear()


# ── Helpers ──

def _make_wallet_activity(
    wallet: str,
    last_tx_age_hours: float = 1.0,
    prev_tx_age_days: float | None = 120.0,
    signature: str | None = None,
) -> WalletActivity:
    """Helper to create a WalletActivity with timestamps relative to now."""
    now = time.time()
    last_ts = now - (last_tx_age_hours * 3600)
    prev_ts = (now - (prev_tx_age_days * 86400)) if prev_tx_age_days is not None else None
    dormancy = prev_tx_age_days if prev_tx_age_days is not None else last_tx_age_hours / 24.0
    return WalletActivity(
        wallet=wallet,
        last_tx_timestamp=last_ts,
        last_tx_signature=signature or f"sig_{wallet}",
        dormancy_days=dormancy,
        prev_tx_timestamp=prev_ts,
    )


def _make_dormant_wallet(
    wallet: str,
    balance_pct: float = 5.0,
    dormancy_days: float = 180.0,
    last_activity_offset_days: float = 180.0,
) -> DormantWallet:
    """Helper to create a DormantWallet."""
    last_activity = datetime.now(timezone.utc) - timedelta(days=last_activity_offset_days)
    return DormantWallet(
        wallet=wallet,
        balance_pct=balance_pct,
        dormancy_days=dormancy_days,
        last_activity=last_activity,
    )


def _make_reactivation(
    wallet: str,
    balance_pct: float = 5.0,
    dormancy_days: float = 180.0,
    sent_to_cex: bool = False,
    cex_name: str | None = None,
) -> Reactivation:
    """Helper to create a Reactivation."""
    return Reactivation(
        wallet=wallet,
        balance_pct=balance_pct,
        dormancy_days=dormancy_days,
        reactivation_tx=f"reactivation_sig_{wallet}",
        sent_to_cex=sent_to_cex,
        cex_name=cex_name,
    )


def _make_holder(owner: str, pct: float) -> dict:
    """Helper to create a holder dict in helius_holders format."""
    return {"owner": owner, "amount": int(pct * 1_000_000), "pct": pct}


# ── Dataclass Tests ──

class TestDataclasses:
    def test_wallet_activity_creation(self):
        now = time.time()
        activity = WalletActivity(
            wallet="wallet_abc",
            last_tx_timestamp=now - 3600,
            last_tx_signature="sig_abc",
            dormancy_days=90.5,
            prev_tx_timestamp=now - (90 * 86400),
        )
        assert activity.wallet == "wallet_abc"
        assert activity.last_tx_signature == "sig_abc"
        assert activity.dormancy_days == 90.5
        assert activity.prev_tx_timestamp is not None

    def test_wallet_activity_no_prev_tx(self):
        now = time.time()
        activity = WalletActivity(
            wallet="wallet_single",
            last_tx_timestamp=now - 7200,
            last_tx_signature="sig_single",
            dormancy_days=30.0,
            prev_tx_timestamp=None,
        )
        assert activity.prev_tx_timestamp is None

    def test_dormant_wallet_creation(self):
        now = datetime.now(timezone.utc)
        dormant = DormantWallet(
            wallet="dormant_wallet_1",
            balance_pct=8.5,
            dormancy_days=200.0,
            last_activity=now - timedelta(days=200),
        )
        assert dormant.wallet == "dormant_wallet_1"
        assert dormant.balance_pct == 8.5
        assert dormant.dormancy_days == 200.0

    def test_reactivation_creation(self):
        reactivation = Reactivation(
            wallet="reactivated_wallet",
            balance_pct=3.2,
            dormancy_days=150.0,
            reactivation_tx="sig_reactivated",
            sent_to_cex=True,
            cex_name="Binance",
        )
        assert reactivation.wallet == "reactivated_wallet"
        assert reactivation.sent_to_cex is True
        assert reactivation.cex_name == "Binance"

    def test_reactivation_no_cex(self):
        reactivation = Reactivation(
            wallet="reactivated_wallet_2",
            balance_pct=2.0,
            dormancy_days=100.0,
            reactivation_tx="sig_react_2",
            sent_to_cex=False,
            cex_name=None,
        )
        assert reactivation.sent_to_cex is False
        assert reactivation.cex_name is None

    def test_reactivation_cluster_creation(self):
        cluster = ReactivationCluster(
            wallets=["w1", "w2", "w3"],
            count=3,
            total_balance_pct=15.0,
            is_cluster=True,
        )
        assert cluster.is_cluster is True
        assert cluster.count == 3
        assert len(cluster.wallets) == 3
        assert cluster.total_balance_pct == 15.0

    def test_ghost_watcher_result_creation(self):
        result = GhostWatcherResult(
            token_mint="mint_test",
            risk_score=65,
            dormant_whales_detected=5,
            wallets_reactivating=2,
            aggregate_dormant_balance_pct=12.5,
            cluster_detected=True,
            details={},
            risk_factors=[],
        )
        assert result.token_mint == "mint_test"
        assert result.risk_score == 65
        assert result.dormant_whales_detected == 5
        assert result.wallets_reactivating == 2
        assert result.cluster_detected is True

    def test_ghost_watcher_result_defaults(self):
        result = GhostWatcherResult(
            token_mint="mint_default",
            risk_score=0,
            dormant_whales_detected=0,
            wallets_reactivating=0,
            aggregate_dormant_balance_pct=0.0,
            cluster_detected=False,
            details={},
            risk_factors=[],
        )
        assert result.details == {}
        assert result.risk_factors == []


# ── detect_reactivation_cluster Tests (Pure Function) ──

class TestDetectReactivationCluster:
    def test_empty_list(self):
        cluster = detect_reactivation_cluster([])
        assert cluster.is_cluster is False
        assert cluster.count == 0
        assert cluster.wallets == []
        assert cluster.total_balance_pct == 0.0

    def test_zero_reactivations(self):
        cluster = detect_reactivation_cluster([])
        assert cluster.is_cluster is False

    def test_one_reactivation(self):
        reactivations = [_make_reactivation("w1", balance_pct=5.0)]
        cluster = detect_reactivation_cluster(reactivations)
        assert cluster.is_cluster is False
        assert cluster.count == 1
        assert cluster.total_balance_pct == pytest.approx(5.0)

    def test_two_reactivations(self):
        reactivations = [
            _make_reactivation("w1", balance_pct=5.0),
            _make_reactivation("w2", balance_pct=3.0),
        ]
        cluster = detect_reactivation_cluster(reactivations)
        assert cluster.is_cluster is False
        assert cluster.count == 2
        assert cluster.total_balance_pct == pytest.approx(8.0)

    def test_exactly_three_is_cluster(self):
        reactivations = [
            _make_reactivation("w1", balance_pct=4.0),
            _make_reactivation("w2", balance_pct=3.0),
            _make_reactivation("w3", balance_pct=2.0),
        ]
        cluster = detect_reactivation_cluster(reactivations, min_wallets=3)
        assert cluster.is_cluster is True
        assert cluster.count == 3
        assert cluster.total_balance_pct == pytest.approx(9.0)

    def test_five_reactivations_is_cluster(self):
        reactivations = [
            _make_reactivation("w1", balance_pct=4.0),
            _make_reactivation("w2", balance_pct=3.0),
            _make_reactivation("w3", balance_pct=2.5),
            _make_reactivation("w4", balance_pct=1.5),
            _make_reactivation("w5", balance_pct=1.0),
        ]
        cluster = detect_reactivation_cluster(reactivations)
        assert cluster.is_cluster is True
        assert cluster.count == 5
        assert cluster.total_balance_pct == pytest.approx(12.0)
        assert set(cluster.wallets) == {"w1", "w2", "w3", "w4", "w5"}

    def test_custom_min_wallets(self):
        reactivations = [
            _make_reactivation("w1", balance_pct=4.0),
            _make_reactivation("w2", balance_pct=3.0),
            _make_reactivation("w3", balance_pct=2.0),
        ]
        # min_wallets=4 means 3 is not enough
        cluster = detect_reactivation_cluster(reactivations, min_wallets=4)
        assert cluster.is_cluster is False
        assert cluster.count == 3

    def test_wallets_list_populated(self):
        reactivations = [
            _make_reactivation("alpha", balance_pct=5.0),
            _make_reactivation("beta", balance_pct=3.0),
            _make_reactivation("gamma", balance_pct=2.0),
        ]
        cluster = detect_reactivation_cluster(reactivations)
        assert "alpha" in cluster.wallets
        assert "beta" in cluster.wallets
        assert "gamma" in cluster.wallets


# ── detect_reactivations Tests (Pure Function) ──

class TestDetectReactivations:
    def test_dormant_wallet_no_recent_activity(self):
        """Wallet dormant for 180d with no recent tx -> no reactivation."""
        dormant = [_make_dormant_wallet("w1", dormancy_days=180.0)]
        # Activity shows last_tx was also old (100 days ago), not recent
        activities = {
            "w1": _make_wallet_activity("w1", last_tx_age_hours=2400, prev_tx_age_days=300.0),
        }
        result = detect_reactivations(dormant, activities)
        assert len(result) == 0

    def test_dormant_wallet_with_recent_reactivation(self):
        """Wallet dormant for 180d, now has a tx within last 24h with old prev_tx -> reactivation."""
        dormant = [_make_dormant_wallet("w1", balance_pct=5.0, dormancy_days=180.0)]
        activities = {
            "w1": _make_wallet_activity(
                "w1", last_tx_age_hours=2.0, prev_tx_age_days=180.0
            ),
        }
        result = detect_reactivations(dormant, activities)
        assert len(result) == 1
        assert result[0].wallet == "w1"
        assert result[0].dormancy_days >= 90  # was dormant a long time

    def test_active_wallet_not_reactivation(self):
        """Wallet with recent last_tx AND recent prev_tx -> not a reactivation (was never dormant)."""
        dormant = [_make_dormant_wallet("w1", dormancy_days=180.0)]
        activities = {
            "w1": _make_wallet_activity(
                "w1", last_tx_age_hours=1.0, prev_tx_age_days=2.0  # prev_tx was only 2 days ago
            ),
        }
        result = detect_reactivations(dormant, activities)
        assert len(result) == 0

    def test_sent_to_cex_detection(self):
        """Reactivation that sends tokens to a CEX should be flagged."""
        dormant = [_make_dormant_wallet("w1", balance_pct=8.0, dormancy_days=200.0)]
        activity = _make_wallet_activity(
            "w1", last_tx_age_hours=1.0, prev_tx_age_days=200.0
        )
        activities = {"w1": activity}

        result = detect_reactivations(dormant, activities)
        assert len(result) == 1
        # The reactivation object should be created; CEX detection depends on implementation
        assert result[0].wallet == "w1"
        assert result[0].balance_pct == 8.0

    def test_multiple_wallets_mixed(self):
        """Mix of reactivated and non-reactivated wallets."""
        dormant = [
            _make_dormant_wallet("w1", balance_pct=5.0, dormancy_days=180.0),
            _make_dormant_wallet("w2", balance_pct=3.0, dormancy_days=150.0),
            _make_dormant_wallet("w3", balance_pct=7.0, dormancy_days=200.0),
        ]
        activities = {
            # w1: reactivated (recent tx, old prev_tx)
            "w1": _make_wallet_activity("w1", last_tx_age_hours=2.0, prev_tx_age_days=180.0),
            # w2: not reactivated (no recent activity)
            "w2": _make_wallet_activity("w2", last_tx_age_hours=3600, prev_tx_age_days=300.0),
            # w3: reactivated
            "w3": _make_wallet_activity("w3", last_tx_age_hours=5.0, prev_tx_age_days=200.0),
        }
        result = detect_reactivations(dormant, activities)
        reactivated_wallets = {r.wallet for r in result}
        assert "w1" in reactivated_wallets
        assert "w2" not in reactivated_wallets
        assert "w3" in reactivated_wallets
        assert len(result) == 2

    def test_wallet_missing_from_activities(self):
        """Dormant wallet with no entry in activities dict -> no reactivation."""
        dormant = [_make_dormant_wallet("w1", dormancy_days=180.0)]
        activities = {}  # empty: no activity data for w1
        result = detect_reactivations(dormant, activities)
        assert len(result) == 0

    def test_wallet_activity_with_no_prev_tx(self):
        """Wallet activity with prev_tx_timestamp=None (only 1 tx ever) -> edge case."""
        dormant = [_make_dormant_wallet("w1", balance_pct=4.0, dormancy_days=100.0)]
        activities = {
            "w1": _make_wallet_activity("w1", last_tx_age_hours=1.0, prev_tx_age_days=None),
        }
        result = detect_reactivations(dormant, activities)
        # With no prev_tx, we cannot confirm dormancy from tx history;
        # behavior depends on implementation but should not crash
        assert isinstance(result, list)


# ── get_wallet_last_activity Tests (Mocked RPC) ──

class TestGetWalletLastActivity:
    @patch("cortex.data.ghost_watcher._post_rpc")
    def test_two_signatures_returns_both_timestamps(self, mock_rpc):
        """RPC returns 2 signatures -> both timestamps populated."""
        now = time.time()
        old_time = now - (120 * 86400)  # 120 days ago
        mock_rpc.return_value = {
            "result": [
                {"signature": "sig_latest", "blockTime": int(now - 3600)},
                {"signature": "sig_prev", "blockTime": int(old_time)},
            ]
        }
        activity = get_wallet_last_activity("wallet_two_sigs")
        assert activity is not None
        assert activity.wallet == "wallet_two_sigs"
        assert activity.last_tx_signature == "sig_latest"
        assert activity.last_tx_timestamp == pytest.approx(now - 3600, abs=1)
        assert activity.prev_tx_timestamp == pytest.approx(old_time, abs=1)

    @patch("cortex.data.ghost_watcher._post_rpc")
    def test_one_signature_prev_is_none(self, mock_rpc):
        """RPC returns 1 signature -> prev_tx_timestamp is None."""
        now = time.time()
        mock_rpc.return_value = {
            "result": [
                {"signature": "sig_only", "blockTime": int(now - 7200)},
            ]
        }
        activity = get_wallet_last_activity("wallet_one_sig")
        assert activity is not None
        assert activity.wallet == "wallet_one_sig"
        assert activity.last_tx_signature == "sig_only"
        assert activity.prev_tx_timestamp is None

    @patch("cortex.data.ghost_watcher._post_rpc")
    def test_zero_signatures_returns_none(self, mock_rpc):
        """RPC returns empty result -> returns None."""
        mock_rpc.return_value = {"result": []}
        activity = get_wallet_last_activity("wallet_no_sigs")
        assert activity is None

    @patch("cortex.data.ghost_watcher._post_rpc")
    def test_rpc_failure_returns_none(self, mock_rpc):
        """RPC returns None (failure) -> returns None."""
        mock_rpc.return_value = None
        activity = get_wallet_last_activity("wallet_rpc_fail")
        assert activity is None

    @patch("cortex.data.ghost_watcher._post_rpc")
    def test_rpc_no_result_key_returns_none(self, mock_rpc):
        """RPC returns dict without 'result' key -> returns None."""
        mock_rpc.return_value = {"error": "some error"}
        activity = get_wallet_last_activity("wallet_error")
        assert activity is None

    @patch("cortex.data.ghost_watcher._post_rpc")
    def test_cache_hit(self, mock_rpc):
        """Second call for same wallet uses cache, RPC called only once."""
        now = time.time()
        mock_rpc.return_value = {
            "result": [
                {"signature": "sig_cached", "blockTime": int(now - 1800)},
                {"signature": "sig_old", "blockTime": int(now - (60 * 86400))},
            ]
        }
        activity1 = get_wallet_last_activity("wallet_cache_test")
        activity2 = get_wallet_last_activity("wallet_cache_test")
        assert activity1 is activity2  # same object from cache
        assert mock_rpc.call_count == 1  # only one RPC call

    @patch("cortex.data.ghost_watcher._post_rpc")
    def test_dormancy_days_calculated(self, mock_rpc):
        """Dormancy days should be calculated based on latest tx blockTime."""
        now = time.time()
        latest_time = now - (90 * 86400)  # 90 days ago
        prev_time = now - (180 * 86400)  # 180 days ago
        mock_rpc.return_value = {
            "result": [
                {"signature": "sig_recent", "blockTime": int(latest_time)},
                {"signature": "sig_old", "blockTime": int(prev_time)},
            ]
        }
        activity = get_wallet_last_activity("wallet_dormancy_calc")
        assert activity is not None
        assert activity.dormancy_days == pytest.approx(90.0, abs=1.0)

    @patch("cortex.data.ghost_watcher._post_rpc")
    def test_signature_without_blocktime_skipped(self, mock_rpc):
        """Signatures missing blockTime should be handled gracefully."""
        now = time.time()
        mock_rpc.return_value = {
            "result": [
                {"signature": "sig_no_time"},  # no blockTime
                {"signature": "sig_with_time", "blockTime": int(now - 3600)},
            ]
        }
        # Implementation may skip entries without blockTime or handle differently
        # The key is it should not crash
        activity = get_wallet_last_activity("wallet_missing_blocktime")
        assert activity is None or isinstance(activity, WalletActivity)


# ── classify_dormant_wallets Tests (Mocked) ──

class TestClassifyDormantWallets:
    @patch("cortex.data.ghost_watcher.get_wallet_last_activity")
    def test_some_dormant_some_active(self, mock_activity):
        """Holders with mix of dormant and active wallets -> correct filtering."""
        now = time.time()
        holders = [
            _make_holder("dormant_1", 5.0),
            _make_holder("active_1", 3.0),
            _make_holder("dormant_2", 4.0),
        ]

        def side_effect(wallet):
            if wallet == "dormant_1":
                return WalletActivity(
                    wallet="dormant_1",
                    last_tx_timestamp=now - (200 * 86400),
                    last_tx_signature="sig_d1",
                    dormancy_days=200.0,
                    prev_tx_timestamp=now - (300 * 86400),
                )
            elif wallet == "active_1":
                return WalletActivity(
                    wallet="active_1",
                    last_tx_timestamp=now - 3600,
                    last_tx_signature="sig_a1",
                    dormancy_days=0.04,  # ~1 hour
                    prev_tx_timestamp=now - 7200,
                )
            elif wallet == "dormant_2":
                return WalletActivity(
                    wallet="dormant_2",
                    last_tx_timestamp=now - (150 * 86400),
                    last_tx_signature="sig_d2",
                    dormancy_days=150.0,
                    prev_tx_timestamp=now - (200 * 86400),
                )
            return None

        mock_activity.side_effect = side_effect

        result = classify_dormant_wallets(holders)
        dormant_wallets = {d.wallet for d in result}
        assert "dormant_1" in dormant_wallets
        assert "dormant_2" in dormant_wallets
        assert "active_1" not in dormant_wallets

    @patch("cortex.data.ghost_watcher.get_wallet_last_activity")
    def test_min_holder_pct_filtering(self, mock_activity):
        """Wallets below min_holder_pct threshold should be excluded."""
        now = time.time()
        holders = [
            _make_holder("big_whale", 10.0),
            _make_holder("small_fish", 0.1),  # below typical threshold
        ]

        def side_effect(wallet):
            return WalletActivity(
                wallet=wallet,
                last_tx_timestamp=now - (200 * 86400),
                last_tx_signature=f"sig_{wallet}",
                dormancy_days=200.0,
                prev_tx_timestamp=now - (300 * 86400),
            )

        mock_activity.side_effect = side_effect

        # Use a min_holder_pct that filters out small_fish
        result = classify_dormant_wallets(holders, min_holder_pct=1.0)
        dormant_wallets = {d.wallet for d in result}
        assert "big_whale" in dormant_wallets
        assert "small_fish" not in dormant_wallets

    @patch("cortex.data.ghost_watcher.get_wallet_last_activity")
    def test_max_wallets_cap(self, mock_activity):
        """Results should be capped at max_wallets."""
        now = time.time()
        holders = [_make_holder(f"whale_{i}", 5.0) for i in range(20)]

        mock_activity.side_effect = lambda wallet: WalletActivity(
            wallet=wallet,
            last_tx_timestamp=now - (200 * 86400),
            last_tx_signature=f"sig_{wallet}",
            dormancy_days=200.0,
            prev_tx_timestamp=now - (300 * 86400),
        )

        result = classify_dormant_wallets(holders, max_wallets=5)
        assert len(result) <= 5

    @patch("cortex.data.ghost_watcher.get_wallet_last_activity")
    def test_all_active_returns_empty(self, mock_activity):
        """All wallets active -> empty dormant list."""
        now = time.time()
        holders = [
            _make_holder("active_a", 5.0),
            _make_holder("active_b", 4.0),
            _make_holder("active_c", 3.0),
        ]

        mock_activity.side_effect = lambda wallet: WalletActivity(
            wallet=wallet,
            last_tx_timestamp=now - 3600,  # 1 hour ago
            last_tx_signature=f"sig_{wallet}",
            dormancy_days=0.04,
            prev_tx_timestamp=now - 7200,
        )

        result = classify_dormant_wallets(holders)
        assert len(result) == 0

    @patch("cortex.data.ghost_watcher.get_wallet_last_activity")
    def test_all_dormant_all_returned(self, mock_activity):
        """All wallets dormant -> all returned."""
        now = time.time()
        holders = [
            _make_holder("dormant_a", 5.0),
            _make_holder("dormant_b", 4.0),
            _make_holder("dormant_c", 3.0),
        ]

        mock_activity.side_effect = lambda wallet: WalletActivity(
            wallet=wallet,
            last_tx_timestamp=now - (180 * 86400),
            last_tx_signature=f"sig_{wallet}",
            dormancy_days=180.0,
            prev_tx_timestamp=now - (250 * 86400),
        )

        result = classify_dormant_wallets(holders)
        assert len(result) == 3

    @patch("cortex.data.ghost_watcher.get_wallet_last_activity")
    def test_wallet_activity_returns_none_skipped(self, mock_activity):
        """If get_wallet_last_activity returns None for a wallet, skip it."""
        holders = [
            _make_holder("ok_wallet", 5.0),
            _make_holder("broken_wallet", 4.0),
        ]
        now = time.time()

        def side_effect(wallet):
            if wallet == "ok_wallet":
                return WalletActivity(
                    wallet="ok_wallet",
                    last_tx_timestamp=now - (200 * 86400),
                    last_tx_signature="sig_ok",
                    dormancy_days=200.0,
                    prev_tx_timestamp=now - (300 * 86400),
                )
            return None  # broken_wallet

        mock_activity.side_effect = side_effect
        result = classify_dormant_wallets(holders)
        dormant_wallets = {d.wallet for d in result}
        assert "broken_wallet" not in dormant_wallets

    @patch("cortex.data.ghost_watcher.get_wallet_last_activity")
    def test_empty_holders_list(self, mock_activity):
        """Empty holders list -> empty result."""
        result = classify_dormant_wallets([])
        assert result == []
        mock_activity.assert_not_called()


# ── compute_ghost_risk Tests (Mocked) ──

class TestComputeGhostRisk:
    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_no_dormant_wallets_score_zero(self, mock_classify, mock_reactivations, mock_cluster):
        """No dormant wallets found -> risk score 0."""
        mock_classify.return_value = []
        mock_reactivations.return_value = []
        mock_cluster.return_value = ReactivationCluster(
            wallets=[], count=0, total_balance_pct=0.0, is_cluster=False
        )

        result = compute_ghost_risk(
            "mint_safe",
            holders=[_make_holder("w1", 5.0), _make_holder("w2", 3.0)],
        )
        assert result.risk_score == 0
        assert result.dormant_whales_detected == 0
        assert result.wallets_reactivating == 0
        assert result.cluster_detected is False

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_single_reactivation_score_30(self, mock_classify, mock_reactivations, mock_cluster):
        """Single wallet reactivation -> score=30 (base reactivation score)."""
        dormant = [_make_dormant_wallet("w1", balance_pct=5.0, dormancy_days=180.0)]
        reactivation = [_make_reactivation("w1", balance_pct=5.0, dormancy_days=180.0)]

        mock_classify.return_value = dormant
        mock_reactivations.return_value = reactivation
        mock_cluster.return_value = ReactivationCluster(
            wallets=["w1"], count=1, total_balance_pct=5.0, is_cluster=False
        )

        result = compute_ghost_risk(
            "mint_one_react",
            holders=[_make_holder("w1", 5.0)],
        )
        assert result.risk_score == 30
        assert result.wallets_reactivating == 1
        assert result.cluster_detected is False

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_cluster_reactivation_score_55(self, mock_classify, mock_reactivations, mock_cluster):
        """Cluster (3+ wallets) reactivation -> score=30+25=55."""
        dormant = [
            _make_dormant_wallet(f"w{i}", balance_pct=3.0, dormancy_days=180.0)
            for i in range(3)
        ]
        reactivations = [
            _make_reactivation(f"w{i}", balance_pct=3.0, dormancy_days=180.0)
            for i in range(3)
        ]

        mock_classify.return_value = dormant
        mock_reactivations.return_value = reactivations
        mock_cluster.return_value = ReactivationCluster(
            wallets=["w0", "w1", "w2"],
            count=3,
            total_balance_pct=9.0,
            is_cluster=True,
        )

        result = compute_ghost_risk(
            "mint_cluster",
            holders=[_make_holder(f"w{i}", 3.0) for i in range(3)],
        )
        assert result.risk_score == 55
        assert result.cluster_detected is True
        assert result.wallets_reactivating == 3

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_high_concentration_adds_20(self, mock_classify, mock_reactivations, mock_cluster):
        """Aggregate dormant balance > 10% -> +20 score."""
        dormant = [_make_dormant_wallet("w1", balance_pct=12.0, dormancy_days=180.0)]
        reactivation = [_make_reactivation("w1", balance_pct=12.0, dormancy_days=180.0)]

        mock_classify.return_value = dormant
        mock_reactivations.return_value = reactivation
        mock_cluster.return_value = ReactivationCluster(
            wallets=["w1"], count=1, total_balance_pct=12.0, is_cluster=False
        )

        result = compute_ghost_risk(
            "mint_high_conc",
            holders=[_make_holder("w1", 12.0)],
        )
        # 30 (reactivation) + 20 (high concentration >10%) = 50
        assert result.risk_score == 50
        assert result.aggregate_dormant_balance_pct >= 10.0

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_extreme_dormancy_adds_15(self, mock_classify, mock_reactivations, mock_cluster):
        """Dormancy > 365 days -> +15 score."""
        dormant = [_make_dormant_wallet("w1", balance_pct=5.0, dormancy_days=400.0)]
        reactivation = [_make_reactivation("w1", balance_pct=5.0, dormancy_days=400.0)]

        mock_classify.return_value = dormant
        mock_reactivations.return_value = reactivation
        mock_cluster.return_value = ReactivationCluster(
            wallets=["w1"], count=1, total_balance_pct=5.0, is_cluster=False
        )

        result = compute_ghost_risk(
            "mint_extreme_dormancy",
            holders=[_make_holder("w1", 5.0)],
        )
        # 30 (reactivation) + 15 (extreme dormancy > 365d) = 45
        assert result.risk_score == 45

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_cex_destination_adds_10(self, mock_classify, mock_reactivations, mock_cluster):
        """Tokens sent to CEX -> +10 score."""
        dormant = [_make_dormant_wallet("w1", balance_pct=5.0, dormancy_days=180.0)]
        reactivation = [
            _make_reactivation(
                "w1", balance_pct=5.0, dormancy_days=180.0,
                sent_to_cex=True, cex_name="Binance"
            ),
        ]

        mock_classify.return_value = dormant
        mock_reactivations.return_value = reactivation
        mock_cluster.return_value = ReactivationCluster(
            wallets=["w1"], count=1, total_balance_pct=5.0, is_cluster=False
        )

        result = compute_ghost_risk(
            "mint_cex_dest",
            holders=[_make_holder("w1", 5.0)],
        )
        # 30 (reactivation) + 10 (CEX destination) = 40
        assert result.risk_score == 40

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_maximum_score_clamped_to_100(self, mock_classify, mock_reactivations, mock_cluster):
        """All risk factors combined should be clamped to max 100."""
        dormant = [
            _make_dormant_wallet(f"w{i}", balance_pct=5.0, dormancy_days=400.0)
            for i in range(5)
        ]
        reactivations = [
            _make_reactivation(
                f"w{i}", balance_pct=5.0, dormancy_days=400.0,
                sent_to_cex=True, cex_name="Binance"
            )
            for i in range(5)
        ]

        mock_classify.return_value = dormant
        mock_reactivations.return_value = reactivations
        mock_cluster.return_value = ReactivationCluster(
            wallets=[f"w{i}" for i in range(5)],
            count=5,
            total_balance_pct=25.0,
            is_cluster=True,
        )

        result = compute_ghost_risk(
            "mint_max_risk",
            holders=[_make_holder(f"w{i}", 5.0) for i in range(5)],
        )
        # 30 (reactivation) + 25 (cluster) + 20 (high conc >10%) + 15 (extreme dormancy) + 10 (CEX) = 100
        assert result.risk_score <= 100
        assert result.risk_score == 100

    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    @patch("cortex.data.ghost_watcher.get_wallet_last_activity")
    def test_holders_none_calls_helius_holders(self, mock_activity, mock_classify):
        """When holders=None, compute_ghost_risk should fetch from helius_holders."""
        mock_classify.return_value = []

        with patch("cortex.data.ghost_watcher.detect_reactivations") as mock_react, \
             patch("cortex.data.ghost_watcher.detect_reactivation_cluster") as mock_cluster:
            mock_react.return_value = []
            mock_cluster.return_value = ReactivationCluster(
                wallets=[], count=0, total_balance_pct=0.0, is_cluster=False
            )

            with patch("cortex.data.helius_holders.get_holder_data") as mock_ghd:
                mock_ghd.return_value = {
                    "holders": [
                        {"owner": "w1", "amount": 5000000, "pct": 5.0},
                        {"owner": "w2", "amount": 3000000, "pct": 3.0},
                    ],
                    "total_holders": 2,
                }
                result = compute_ghost_risk("mint_fetch_holders", holders=None)
                mock_ghd.assert_called_once_with("mint_fetch_holders")

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_result_has_correct_token_mint(self, mock_classify, mock_reactivations, mock_cluster):
        """Result should always carry the correct token_mint."""
        mock_classify.return_value = []
        mock_reactivations.return_value = []
        mock_cluster.return_value = ReactivationCluster(
            wallets=[], count=0, total_balance_pct=0.0, is_cluster=False
        )

        result = compute_ghost_risk("mint_xyz", holders=[])
        assert result.token_mint == "mint_xyz"

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_risk_factors_populated(self, mock_classify, mock_reactivations, mock_cluster):
        """Risk factors list should describe each contributing factor."""
        dormant = [_make_dormant_wallet("w1", balance_pct=12.0, dormancy_days=400.0)]
        reactivation = [
            _make_reactivation(
                "w1", balance_pct=12.0, dormancy_days=400.0,
                sent_to_cex=True, cex_name="Coinbase"
            ),
        ]

        mock_classify.return_value = dormant
        mock_reactivations.return_value = reactivation
        mock_cluster.return_value = ReactivationCluster(
            wallets=["w1"], count=1, total_balance_pct=12.0, is_cluster=False
        )

        result = compute_ghost_risk(
            "mint_risk_factors",
            holders=[_make_holder("w1", 12.0)],
        )
        assert len(result.risk_factors) > 0
        assert isinstance(result.risk_factors, list)
        # All risk factors should be strings
        for rf in result.risk_factors:
            assert isinstance(rf, str)

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_details_dict_populated(self, mock_classify, mock_reactivations, mock_cluster):
        """Details dict should contain analysis metadata."""
        dormant = [_make_dormant_wallet("w1", balance_pct=5.0, dormancy_days=180.0)]
        mock_classify.return_value = dormant
        mock_reactivations.return_value = [
            _make_reactivation("w1", balance_pct=5.0, dormancy_days=180.0)
        ]
        mock_cluster.return_value = ReactivationCluster(
            wallets=["w1"], count=1, total_balance_pct=5.0, is_cluster=False
        )

        result = compute_ghost_risk(
            "mint_details",
            holders=[_make_holder("w1", 5.0)],
        )
        assert isinstance(result.details, dict)

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_dormant_whales_count(self, mock_classify, mock_reactivations, mock_cluster):
        """dormant_whales_detected should match number of classified dormant wallets."""
        dormant = [
            _make_dormant_wallet("w1", balance_pct=5.0, dormancy_days=180.0),
            _make_dormant_wallet("w2", balance_pct=4.0, dormancy_days=150.0),
            _make_dormant_wallet("w3", balance_pct=3.0, dormancy_days=200.0),
        ]

        mock_classify.return_value = dormant
        mock_reactivations.return_value = []
        mock_cluster.return_value = ReactivationCluster(
            wallets=[], count=0, total_balance_pct=0.0, is_cluster=False
        )

        result = compute_ghost_risk(
            "mint_count",
            holders=[_make_holder(f"w{i}", 4.0) for i in range(3)],
        )
        assert result.dormant_whales_detected == 3

    def test_cache_clearing_fixture_works(self):
        """Verify cache is cleared between tests via the autouse fixture."""
        # The clear_cache fixture runs before/after each test
        assert len(_cache) == 0

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_score_never_negative(self, mock_classify, mock_reactivations, mock_cluster):
        """Risk score should never be negative."""
        mock_classify.return_value = []
        mock_reactivations.return_value = []
        mock_cluster.return_value = ReactivationCluster(
            wallets=[], count=0, total_balance_pct=0.0, is_cluster=False
        )

        result = compute_ghost_risk("mint_safe_score", holders=[])
        assert result.risk_score >= 0

    @patch("cortex.data.ghost_watcher.detect_reactivation_cluster")
    @patch("cortex.data.ghost_watcher.detect_reactivations")
    @patch("cortex.data.ghost_watcher.classify_dormant_wallets")
    def test_aggregate_dormant_balance_pct(self, mock_classify, mock_reactivations, mock_cluster):
        """aggregate_dormant_balance_pct should sum dormant wallet balances."""
        dormant = [
            _make_dormant_wallet("w1", balance_pct=5.0, dormancy_days=180.0),
            _make_dormant_wallet("w2", balance_pct=3.0, dormancy_days=150.0),
        ]

        mock_classify.return_value = dormant
        mock_reactivations.return_value = []
        mock_cluster.return_value = ReactivationCluster(
            wallets=[], count=0, total_balance_pct=0.0, is_cluster=False
        )

        result = compute_ghost_risk(
            "mint_agg_pct",
            holders=[_make_holder("w1", 5.0), _make_holder("w2", 3.0)],
        )
        assert result.aggregate_dormant_balance_pct == pytest.approx(8.0)
