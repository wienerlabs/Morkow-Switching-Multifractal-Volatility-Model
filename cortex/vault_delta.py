"""DX-Research Task 7: Vault State Delta — On-Chain Memory as ML Feature.

DX01 finding: market state IS collective memory. Vault TVL changes, share price
movements, and deposit/withdraw flow rates encode crowd behavior that individual
agents cannot observe from price data alone.

This module:
  - Ingests vault snapshots (totalAssets, sharePrice, timestamp)
  - Computes rolling deltas across time windows (1h, 24h, 7d)
  - Classifies significant events (large withdrawal, TVL crash, share price jump)
  - Exposes ML-ready features for Guardian scoring and debate evidence
"""
from __future__ import annotations

__all__ = [
    "VaultSnapshot",
    "VaultDelta",
    "VaultDeltaTracker",
    "get_tracker",
    "ingest_snapshot",
    "get_vault_features",
    "score_vault_delta",
]

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

from cortex.config import (
    VAULT_DELTA_ENABLED,
    VAULT_DELTA_LARGE_DEPOSIT_PCT,
    VAULT_DELTA_LARGE_WITHDRAWAL_PCT,
)

logger = logging.getLogger(__name__)

# Time windows in seconds
_1H = 3600
_24H = 86_400
_7D = 604_800


@dataclass
class VaultSnapshot:
    """Point-in-time vault state."""
    vault_id: str
    total_assets: float
    share_price: float
    ts: float = 0.0

    def __post_init__(self) -> None:
        if self.ts == 0.0:
            self.ts = time.time()


@dataclass
class VaultDelta:
    """Computed deltas for a vault across time windows."""
    vault_id: str
    tvl_1h_pct: float = 0.0
    tvl_24h_pct: float = 0.0
    tvl_7d_pct: float = 0.0
    share_price_1h_pct: float = 0.0
    share_price_24h_pct: float = 0.0
    share_price_7d_pct: float = 0.0
    deposit_flow_24h: float = 0.0  # net deposits - withdrawals in USD
    events: list[str] = field(default_factory=list)
    num_snapshots: int = 0
    ts: float = 0.0

    def __post_init__(self) -> None:
        if self.ts == 0.0:
            self.ts = time.time()

    def to_dict(self) -> dict[str, Any]:
        return {
            "vault_id": self.vault_id,
            "tvl_1h_pct": round(self.tvl_1h_pct, 4),
            "tvl_24h_pct": round(self.tvl_24h_pct, 4),
            "tvl_7d_pct": round(self.tvl_7d_pct, 4),
            "share_price_1h_pct": round(self.share_price_1h_pct, 4),
            "share_price_24h_pct": round(self.share_price_24h_pct, 4),
            "share_price_7d_pct": round(self.share_price_7d_pct, 4),
            "deposit_flow_24h": round(self.deposit_flow_24h, 2),
            "events": self.events,
            "num_snapshots": self.num_snapshots,
        }

    def to_feature_vector(self) -> dict[str, float]:
        """ML-ready feature dict for model consumption."""
        return {
            "vault_tvl_1h_pct": self.tvl_1h_pct,
            "vault_tvl_24h_pct": self.tvl_24h_pct,
            "vault_tvl_7d_pct": self.tvl_7d_pct,
            "vault_sp_1h_pct": self.share_price_1h_pct,
            "vault_sp_24h_pct": self.share_price_24h_pct,
            "vault_sp_7d_pct": self.share_price_7d_pct,
            "vault_flow_24h": self.deposit_flow_24h,
            "vault_has_large_withdrawal": 1.0 if "large_withdrawal" in self.events else 0.0,
            "vault_has_large_deposit": 1.0 if "large_deposit" in self.events else 0.0,
        }


def _pct_change(current: float, previous: float) -> float:
    """Compute percentage change, safe against zero division."""
    if previous == 0.0:
        return 0.0
    return ((current - previous) / abs(previous)) * 100.0


def _find_snapshot_at(snapshots: deque[VaultSnapshot], target_ts: float) -> VaultSnapshot | None:
    """Find the snapshot closest to target_ts (but not after it)."""
    best: VaultSnapshot | None = None
    for snap in snapshots:
        if snap.ts <= target_ts:
            if best is None or snap.ts > best.ts:
                best = snap
    return best


class VaultDeltaTracker:
    """Tracks vault snapshots and computes rolling deltas."""

    def __init__(self, max_history: int = 2000) -> None:
        self._snapshots: dict[str, deque[VaultSnapshot]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    def ingest(self, snapshot: VaultSnapshot) -> None:
        self._snapshots[snapshot.vault_id].append(snapshot)

    def compute_delta(self, vault_id: str) -> VaultDelta:
        snaps = self._snapshots.get(vault_id)
        if not snaps or len(snaps) < 2:
            return VaultDelta(vault_id=vault_id, num_snapshots=len(snaps) if snaps else 0)

        current = snaps[-1]
        now = current.ts
        events: list[str] = []

        # Find reference snapshots for each time window
        snap_1h = _find_snapshot_at(snaps, now - _1H)
        snap_24h = _find_snapshot_at(snaps, now - _24H)
        snap_7d = _find_snapshot_at(snaps, now - _7D)

        tvl_1h = _pct_change(current.total_assets, snap_1h.total_assets) if snap_1h else 0.0
        tvl_24h = _pct_change(current.total_assets, snap_24h.total_assets) if snap_24h else 0.0
        tvl_7d = _pct_change(current.total_assets, snap_7d.total_assets) if snap_7d else 0.0

        sp_1h = _pct_change(current.share_price, snap_1h.share_price) if snap_1h else 0.0
        sp_24h = _pct_change(current.share_price, snap_24h.share_price) if snap_24h else 0.0
        sp_7d = _pct_change(current.share_price, snap_7d.share_price) if snap_7d else 0.0

        # Compute deposit flow as TVL delta in absolute terms (24h)
        deposit_flow = (current.total_assets - snap_24h.total_assets) if snap_24h else 0.0

        # Event classification
        if tvl_24h < -VAULT_DELTA_LARGE_WITHDRAWAL_PCT:
            events.append("large_withdrawal")
        if tvl_24h > VAULT_DELTA_LARGE_DEPOSIT_PCT:
            events.append("large_deposit")
        if abs(sp_1h) > 5.0:
            events.append("share_price_jump")
        if tvl_24h < -25.0:
            events.append("tvl_crash")

        return VaultDelta(
            vault_id=vault_id,
            tvl_1h_pct=tvl_1h,
            tvl_24h_pct=tvl_24h,
            tvl_7d_pct=tvl_7d,
            share_price_1h_pct=sp_1h,
            share_price_24h_pct=sp_24h,
            share_price_7d_pct=sp_7d,
            deposit_flow_24h=deposit_flow,
            events=events,
            num_snapshots=len(snaps),
        )

    def get_all_deltas(self) -> dict[str, VaultDelta]:
        return {vid: self.compute_delta(vid) for vid in self._snapshots}

    def clear(self) -> None:
        self._snapshots.clear()


def score_vault_delta(delta: VaultDelta) -> dict[str, Any]:
    """Score vault state health as a Guardian-compatible component (0-100).

    Higher score = more risk (consistent with other Guardian components).
    """
    if not VAULT_DELTA_ENABLED:
        return {"component": "vault_delta", "score": 0.0, "details": {}}

    # TVL volatility: large swings in either direction = risk
    tvl_vol = (abs(delta.tvl_1h_pct) * 2 + abs(delta.tvl_24h_pct)) / 3
    tvl_score = min(100.0, tvl_vol * 3.0)

    # Share price volatility
    sp_vol = (abs(delta.share_price_1h_pct) * 2 + abs(delta.share_price_24h_pct)) / 3
    sp_score = min(100.0, sp_vol * 4.0)

    # Outflow pressure: negative deposit flow = higher risk
    flow_score = 50.0
    if delta.deposit_flow_24h < 0:
        flow_score = min(100.0, 50.0 + abs(delta.deposit_flow_24h) / 10_000 * 5)
    elif delta.deposit_flow_24h > 0:
        flow_score = max(0.0, 50.0 - delta.deposit_flow_24h / 10_000 * 3)

    # Event penalty
    event_penalty = len(delta.events) * 10.0

    score = min(100.0, tvl_score * 0.35 + sp_score * 0.25 + flow_score * 0.25 + event_penalty * 0.15)

    return {
        "component": "vault_delta",
        "score": round(max(0.0, min(100.0, score)), 2),
        "details": {
            "tvl_volatility_score": round(tvl_score, 2),
            "share_price_volatility_score": round(sp_score, 2),
            "flow_score": round(flow_score, 2),
            "event_count": len(delta.events),
            "events": delta.events,
            "features": delta.to_feature_vector(),
        },
    }


# ── Module-level singleton ──

_tracker: VaultDeltaTracker | None = None


def get_tracker() -> VaultDeltaTracker:
    global _tracker
    if _tracker is None:
        _tracker = VaultDeltaTracker()
    return _tracker


def ingest_snapshot(
    vault_id: str,
    total_assets: float,
    share_price: float,
    ts: float | None = None,
) -> None:
    if not VAULT_DELTA_ENABLED:
        return
    snap = VaultSnapshot(
        vault_id=vault_id,
        total_assets=total_assets,
        share_price=share_price,
        ts=ts or time.time(),
    )
    get_tracker().ingest(snap)


def get_vault_features(vault_id: str) -> dict[str, Any]:
    """Get ML features + score for a vault."""
    delta = get_tracker().compute_delta(vault_id)
    scored = score_vault_delta(delta)
    scored["details"]["delta"] = delta.to_dict()
    return scored
