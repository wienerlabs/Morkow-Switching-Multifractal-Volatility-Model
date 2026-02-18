"""Tests for cortex/vault_delta.py — DX-Research Task 7: Vault State Delta."""
import time
from unittest.mock import patch

import pytest

from cortex.vault_delta import (
    VaultDelta,
    VaultDeltaTracker,
    VaultSnapshot,
    _pct_change,
    get_tracker,
    get_vault_features,
    ingest_snapshot,
    score_vault_delta,
)
import cortex.vault_delta as vault_delta_module


@pytest.fixture(autouse=True)
def reset_tracker():
    old = vault_delta_module._tracker
    vault_delta_module._tracker = None
    yield
    vault_delta_module._tracker = old


class TestPctChange:
    def test_positive_change(self):
        assert _pct_change(110, 100) == pytest.approx(10.0)

    def test_negative_change(self):
        assert _pct_change(90, 100) == pytest.approx(-10.0)

    def test_zero_previous(self):
        assert _pct_change(100, 0) == 0.0

    def test_no_change(self):
        assert _pct_change(100, 100) == 0.0


class TestVaultSnapshot:
    def test_auto_timestamp(self):
        snap = VaultSnapshot(vault_id="v1", total_assets=1_000_000, share_price=1.05)
        assert snap.ts > 0

    def test_explicit_timestamp(self):
        snap = VaultSnapshot(vault_id="v1", total_assets=1_000_000, share_price=1.05, ts=12345.0)
        assert snap.ts == 12345.0


class TestVaultDelta:
    def test_to_dict(self):
        d = VaultDelta(vault_id="v1", tvl_24h_pct=-5.5, events=["large_withdrawal"])
        result = d.to_dict()
        assert result["vault_id"] == "v1"
        assert result["tvl_24h_pct"] == -5.5
        assert "large_withdrawal" in result["events"]

    def test_to_feature_vector(self):
        d = VaultDelta(
            vault_id="v1",
            tvl_1h_pct=-2.0,
            tvl_24h_pct=-8.0,
            share_price_1h_pct=1.5,
            events=["large_withdrawal"],
        )
        fv = d.to_feature_vector()
        assert fv["vault_tvl_1h_pct"] == -2.0
        assert fv["vault_tvl_24h_pct"] == -8.0
        assert fv["vault_has_large_withdrawal"] == 1.0
        assert fv["vault_has_large_deposit"] == 0.0

    def test_feature_vector_no_events(self):
        d = VaultDelta(vault_id="v1")
        fv = d.to_feature_vector()
        assert fv["vault_has_large_withdrawal"] == 0.0
        assert fv["vault_has_large_deposit"] == 0.0


class TestVaultDeltaTracker:
    def _make_tracker_with_history(self) -> VaultDeltaTracker:
        tracker = VaultDeltaTracker()
        now = time.time()
        # 7 days of snapshots: stable vault at 1M, then drops to 850K
        for i in range(8):
            t = now - (7 - i) * 86_400
            assets = 1_000_000 if i < 6 else 900_000 if i == 6 else 850_000
            price = 1.0 if i < 6 else 0.97 if i == 6 else 0.95
            tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=assets, share_price=price, ts=t))
        return tracker

    def test_empty_vault_returns_zero_delta(self):
        tracker = VaultDeltaTracker()
        delta = tracker.compute_delta("nonexistent")
        assert delta.tvl_24h_pct == 0.0
        assert delta.num_snapshots == 0

    def test_single_snapshot_returns_zero_delta(self):
        tracker = VaultDeltaTracker()
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=1_000_000, share_price=1.0))
        delta = tracker.compute_delta("v1")
        assert delta.tvl_24h_pct == 0.0
        assert delta.num_snapshots == 1

    def test_24h_delta_computed(self):
        tracker = self._make_tracker_with_history()
        delta = tracker.compute_delta("v1")
        # Last snapshot (850K) vs 24h ago (900K) = -5.56%
        assert delta.tvl_24h_pct < 0
        assert delta.num_snapshots == 8

    def test_7d_delta_computed(self):
        tracker = self._make_tracker_with_history()
        delta = tracker.compute_delta("v1")
        # Last snapshot (850K) vs 7d ago (1M) = -15%
        assert delta.tvl_7d_pct == pytest.approx(-15.0)

    def test_share_price_delta(self):
        tracker = self._make_tracker_with_history()
        delta = tracker.compute_delta("v1")
        # Share price: 0.95 vs 1.0 (7d ago) = -5%
        assert delta.share_price_7d_pct == pytest.approx(-5.0)

    def test_large_withdrawal_event(self):
        tracker = VaultDeltaTracker()
        now = time.time()
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=1_000_000, share_price=1.0, ts=now - 86_400))
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=800_000, share_price=0.9, ts=now))
        delta = tracker.compute_delta("v1")
        assert "large_withdrawal" in delta.events
        assert delta.tvl_24h_pct == pytest.approx(-20.0)

    def test_large_deposit_event(self):
        tracker = VaultDeltaTracker()
        now = time.time()
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=1_000_000, share_price=1.0, ts=now - 86_400))
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=1_200_000, share_price=1.1, ts=now))
        delta = tracker.compute_delta("v1")
        assert "large_deposit" in delta.events

    def test_share_price_jump_event(self):
        tracker = VaultDeltaTracker()
        now = time.time()
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=1_000_000, share_price=1.0, ts=now - 3600))
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=1_000_000, share_price=1.08, ts=now))
        delta = tracker.compute_delta("v1")
        assert "share_price_jump" in delta.events

    def test_tvl_crash_event(self):
        tracker = VaultDeltaTracker()
        now = time.time()
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=1_000_000, share_price=1.0, ts=now - 86_400))
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=700_000, share_price=0.7, ts=now))
        delta = tracker.compute_delta("v1")
        assert "tvl_crash" in delta.events
        assert "large_withdrawal" in delta.events

    def test_deposit_flow_positive(self):
        tracker = VaultDeltaTracker()
        now = time.time()
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=1_000_000, share_price=1.0, ts=now - 86_400))
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=1_050_000, share_price=1.01, ts=now))
        delta = tracker.compute_delta("v1")
        assert delta.deposit_flow_24h == pytest.approx(50_000.0)

    def test_multiple_vaults_independent(self):
        tracker = VaultDeltaTracker()
        now = time.time()
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=1_000_000, share_price=1.0, ts=now - 86_400))
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=900_000, share_price=0.95, ts=now))
        tracker.ingest(VaultSnapshot(vault_id="v2", total_assets=500_000, share_price=2.0, ts=now - 86_400))
        tracker.ingest(VaultSnapshot(vault_id="v2", total_assets=600_000, share_price=2.1, ts=now))

        d1 = tracker.compute_delta("v1")
        d2 = tracker.compute_delta("v2")
        assert d1.tvl_24h_pct < 0  # v1 dropped
        assert d2.tvl_24h_pct > 0  # v2 grew

    def test_get_all_deltas(self):
        tracker = VaultDeltaTracker()
        now = time.time()
        for vid in ("v1", "v2", "v3"):
            tracker.ingest(VaultSnapshot(vault_id=vid, total_assets=1_000_000, share_price=1.0, ts=now - 3600))
            tracker.ingest(VaultSnapshot(vault_id=vid, total_assets=1_000_000, share_price=1.0, ts=now))
        all_deltas = tracker.get_all_deltas()
        assert len(all_deltas) == 3

    def test_clear(self):
        tracker = VaultDeltaTracker()
        tracker.ingest(VaultSnapshot(vault_id="v1", total_assets=1_000_000, share_price=1.0))
        tracker.clear()
        assert tracker.compute_delta("v1").num_snapshots == 0


class TestScoreVaultDelta:
    def test_stable_vault_low_score(self):
        delta = VaultDelta(vault_id="v1", tvl_24h_pct=0.5, share_price_24h_pct=0.1)
        result = score_vault_delta(delta)
        assert result["component"] == "vault_delta"
        assert result["score"] < 30  # stable = low risk

    def test_volatile_vault_high_score(self):
        delta = VaultDelta(
            vault_id="v1",
            tvl_1h_pct=-15.0,
            tvl_24h_pct=-25.0,
            share_price_1h_pct=-8.0,
            deposit_flow_24h=-200_000,
            events=["large_withdrawal", "tvl_crash"],
        )
        result = score_vault_delta(delta)
        assert result["score"] > 35  # volatile = elevated risk

    def test_outflow_increases_score(self):
        delta_inflow = VaultDelta(vault_id="v1", deposit_flow_24h=100_000)
        delta_outflow = VaultDelta(vault_id="v1", deposit_flow_24h=-100_000)
        score_in = score_vault_delta(delta_inflow)["score"]
        score_out = score_vault_delta(delta_outflow)["score"]
        assert score_out > score_in

    def test_events_add_penalty(self):
        delta_no_events = VaultDelta(vault_id="v1", tvl_24h_pct=-5.0)
        delta_with_events = VaultDelta(vault_id="v1", tvl_24h_pct=-5.0, events=["large_withdrawal", "share_price_jump"])
        score_no = score_vault_delta(delta_no_events)["score"]
        score_with = score_vault_delta(delta_with_events)["score"]
        assert score_with > score_no

    def test_score_bounded_0_100(self):
        delta = VaultDelta(
            vault_id="v1",
            tvl_1h_pct=-50.0,
            tvl_24h_pct=-80.0,
            share_price_1h_pct=-30.0,
            deposit_flow_24h=-1_000_000,
            events=["large_withdrawal", "tvl_crash", "share_price_jump"],
        )
        result = score_vault_delta(delta)
        assert 0 <= result["score"] <= 100

    def test_disabled_returns_zero(self):
        delta = VaultDelta(vault_id="v1", tvl_24h_pct=-50.0)
        with patch("cortex.vault_delta.VAULT_DELTA_ENABLED", False):
            result = score_vault_delta(delta)
        assert result["score"] == 0.0

    def test_details_include_features(self):
        delta = VaultDelta(vault_id="v1", tvl_24h_pct=-5.0)
        result = score_vault_delta(delta)
        assert "features" in result["details"]
        assert "vault_tvl_24h_pct" in result["details"]["features"]


class TestModuleLevelAPI:
    def test_ingest_and_features(self):
        now = time.time()
        ingest_snapshot("v1", 1_000_000, 1.0, ts=now - 86_400)
        ingest_snapshot("v1", 950_000, 0.98, ts=now)
        result = get_vault_features("v1")
        assert result["component"] == "vault_delta"
        assert result["score"] > 0
        assert "delta" in result["details"]

    def test_ingest_disabled(self):
        with patch("cortex.vault_delta.VAULT_DELTA_ENABLED", False):
            ingest_snapshot("v1", 1_000_000, 1.0)
        tracker = get_tracker()
        assert tracker.compute_delta("v1").num_snapshots == 0

    def test_get_tracker_singleton(self):
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2
