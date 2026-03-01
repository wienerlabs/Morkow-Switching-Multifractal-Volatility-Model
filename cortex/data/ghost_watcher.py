"""Ghost watcher — detect dormant wallet reactivations for Solana tokens.

Analyzes token holder wallets for suspicious reactivation patterns:
  - Wallets dormant 90+ days that suddenly become active
  - Coordinated reactivation clusters (3+ dormant wallets waking together)
  - High concentration of supply held by reactivating wallets
  - Extremely long dormancy periods (365+ days)
  - Reactivated wallets sending tokens to known CEX hot wallets

Feeds into Guardian composite risk scoring as an optional component.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from cortex.config import (
    GHOST_WATCHER_DORMANCY_THRESHOLD_DAYS,
    GHOST_WATCHER_MIN_HOLDER_PCT,
    GHOST_WATCHER_MAX_WALLETS_TO_CHECK,
    GHOST_WATCHER_REACTIVATION_WINDOW_HOURS,
    GHOST_WATCHER_SINGLE_REACTIVATION_SCORE,
    GHOST_WATCHER_CLUSTER_SCORE,
    GHOST_WATCHER_HIGH_CONCENTRATION_SCORE,
    GHOST_WATCHER_EXTREME_DORMANCY_SCORE,
    GHOST_WATCHER_CEX_DESTINATION_SCORE,
    GHOST_WATCHER_CLUSTER_MIN_WALLETS,
)
from cortex.data.launch_tracker import (
    _post_rpc,
    CEX_HOT_WALLETS,
    _cache,
    _cached,
    _set_cache,
)

logger = logging.getLogger(__name__)


# ── Dataclasses ──


@dataclass
class WalletActivity:
    wallet: str
    last_tx_timestamp: float  # UNIX timestamp
    last_tx_signature: str
    dormancy_days: float
    prev_tx_timestamp: float | None  # second-to-last tx UNIX timestamp
    prev_tx_signature: str | None = None


@dataclass
class DormantWallet:
    wallet: str
    balance_pct: float  # percentage of token supply held
    dormancy_days: float
    last_activity: datetime


@dataclass
class Reactivation:
    wallet: str
    balance_pct: float
    dormancy_days: float
    reactivation_tx: str
    sent_to_cex: bool
    cex_name: str | None


@dataclass
class ReactivationCluster:
    wallets: list[str]
    count: int
    total_balance_pct: float
    is_cluster: bool  # True if count >= GHOST_WATCHER_CLUSTER_MIN_WALLETS


@dataclass
class GhostWatcherResult:
    token_mint: str
    risk_score: int  # 0-100
    dormant_whales_detected: int
    wallets_reactivating: int
    aggregate_dormant_balance_pct: float
    cluster_detected: bool
    details: dict = field(default_factory=dict)
    risk_factors: list[str] = field(default_factory=list)


# ── Core Functions ──


def get_wallet_last_activity(wallet: str, limit: int = 2) -> WalletActivity | None:
    """Get the most recent (and optionally second-to-last) transaction for a wallet.

    Uses ``getSignaturesForAddress`` with the given *limit* to retrieve
    recent transaction signatures.  Calculates dormancy as the elapsed
    time since the latest ``blockTime``.

    Parameters
    ----------
    wallet:
        Solana wallet address to inspect.
    limit:
        Number of recent signatures to fetch (default 2 so we can detect
        reactivation by comparing the latest vs. previous tx).

    Returns ``None`` when the RPC call fails or yields no results.
    """
    cached = _cached(f"ghost_activity:{wallet}")
    if cached is not None:
        return cached

    payload = {
        "jsonrpc": "2.0",
        "id": "ghost-watcher-sigs",
        "method": "getSignaturesForAddress",
        "params": [wallet, {"limit": limit}],
    }

    data = _post_rpc(payload)
    if not data or "result" not in data:
        return None

    sigs = data["result"]
    if not sigs:
        return None

    # Latest transaction
    latest = sigs[0]
    block_time = latest.get("blockTime")
    if block_time is None:
        return None

    now = time.time()
    dormancy_days = (now - block_time) / 86400.0

    # Previous transaction (if available)
    prev_ts: float | None = None
    prev_sig: str | None = None
    if len(sigs) >= 2:
        prev = sigs[1]
        prev_ts = prev.get("blockTime")
        prev_sig = prev.get("signature")

    activity = WalletActivity(
        wallet=wallet,
        last_tx_timestamp=float(block_time),
        last_tx_signature=latest.get("signature", ""),
        dormancy_days=dormancy_days,
        prev_tx_timestamp=prev_ts,
        prev_tx_signature=prev_sig,
    )
    _set_cache(f"ghost_activity:{wallet}", activity)
    return activity


def classify_dormant_wallets(
    holders: list[dict],
    dormancy_threshold_days: float = GHOST_WATCHER_DORMANCY_THRESHOLD_DAYS,
    min_holder_pct: float = GHOST_WATCHER_MIN_HOLDER_PCT,
    max_wallets: int = GHOST_WATCHER_MAX_WALLETS_TO_CHECK,
) -> list[DormantWallet]:
    """Classify holder wallets as dormant based on their on-chain activity.

    Filters *holders* to those controlling at least *min_holder_pct* of the
    token supply, then checks their last on-chain activity via RPC.  A wallet
    is considered **dormant** when its last transaction is older than
    *dormancy_threshold_days*.

    Parameters
    ----------
    holders:
        List of holder dicts, each expected to have ``"address"`` (or
        ``"owner"``) and ``"pct"`` (percentage of supply) keys.
    dormancy_threshold_days:
        Minimum days since last activity to classify a wallet as dormant.
    min_holder_pct:
        Minimum percentage of token supply a wallet must hold to be
        considered (filters out dust holders).
    max_wallets:
        Cap on the number of wallets to inspect (saves RPC quota).

    Returns a list of :class:`DormantWallet` sorted by *dormancy_days*
    descending (longest dormancy first).
    """
    # Filter holders by minimum percentage
    significant = [
        h for h in holders
        if h.get("pct", 0.0) >= min_holder_pct
    ]

    dormant: list[DormantWallet] = []

    for holder in significant[:max_wallets]:
        wallet = holder.get("address") or holder.get("owner", "")
        if not wallet:
            continue

        activity = get_wallet_last_activity(wallet)
        if activity is None:
            continue

        if activity.dormancy_days > dormancy_threshold_days:
            dormant.append(DormantWallet(
                wallet=wallet,
                balance_pct=holder.get("pct", 0.0),
                dormancy_days=activity.dormancy_days,
                last_activity=datetime.fromtimestamp(
                    activity.last_tx_timestamp, tz=timezone.utc,
                ),
            ))

    # Sort by dormancy descending (longest dormancy first)
    dormant.sort(key=lambda d: d.dormancy_days, reverse=True)
    return dormant


def detect_reactivations(
    dormant_wallets: list[DormantWallet],
    current_activities: dict[str, WalletActivity],
    dormancy_threshold_days: float = GHOST_WATCHER_DORMANCY_THRESHOLD_DAYS,
    reactivation_window_hours: float = GHOST_WATCHER_REACTIVATION_WINDOW_HOURS,
) -> list[Reactivation]:
    """Identify wallets that were dormant but have recently reactivated.

    This is a **pure function** (no I/O) — the caller must supply
    *current_activities* so the function can be tested deterministically.

    A wallet is classified as *reactivated* when:
    1. Its latest transaction is within *reactivation_window_hours* of now, AND
    2. Its second-to-last transaction is older than *dormancy_threshold_days*,
       proving it was previously dormant and only just woke up.

    If the wallet lacks a second-to-last transaction the original dormancy
    classification from *dormant_wallets* is trusted instead.

    Parameters
    ----------
    dormant_wallets:
        Wallets previously classified as dormant.
    current_activities:
        Mapping of wallet address -> :class:`WalletActivity` with
        *prev_tx_timestamp* populated so we can verify the two-transaction
        gap pattern.
    dormancy_threshold_days:
        The threshold used during the initial dormancy classification.
    reactivation_window_hours:
        How recently the latest tx must have occurred to qualify as a
        reactivation (default 24 hours).
    """
    now = time.time()
    window_sec = reactivation_window_hours * 3600.0
    threshold_sec = dormancy_threshold_days * 86400.0
    reactivations: list[Reactivation] = []

    for dw in dormant_wallets:
        activity = current_activities.get(dw.wallet)
        if activity is None:
            continue

        # Latest tx must be recent (within the reactivation window)
        latest_age_sec = now - activity.last_tx_timestamp
        if latest_age_sec > window_sec:
            # Still dormant — no reactivation
            continue

        # Verify the wallet was truly dormant before this latest tx:
        # the second-to-last tx must be older than the dormancy threshold.
        if activity.prev_tx_timestamp is not None:
            prev_age_sec = now - activity.prev_tx_timestamp
            if prev_age_sec < threshold_sec:
                # Previous tx is also recent — wallet was continuously
                # active, not a dormant reactivation.
                continue
            dormancy_days_actual = prev_age_sec / 86400.0
        else:
            # No prior tx available — trust the original dormancy figure
            dormancy_days_actual = dw.dormancy_days

        # Check if the recent tx sent tokens to a known CEX
        sent_to_cex = False
        cex_name: str | None = None

        # Look up the recent tx to see if any destination is a CEX wallet.
        # We use the signature from the activity to fetch tx details.
        tx_sig = activity.last_tx_signature
        if tx_sig:
            tx_payload = {
                "jsonrpc": "2.0",
                "id": "ghost-reactivation-tx",
                "method": "getTransaction",
                "params": [
                    tx_sig,
                    {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
                ],
            }
            tx_data = _post_rpc(tx_payload)
            if tx_data and "result" in tx_data and tx_data["result"] is not None:
                tx = tx_data["result"]
                account_keys = (
                    tx.get("transaction", {})
                    .get("message", {})
                    .get("accountKeys", [])
                )
                for key in account_keys:
                    addr = key if isinstance(key, str) else key.get("pubkey", "")
                    if addr in CEX_HOT_WALLETS and addr != dw.wallet:
                        sent_to_cex = True
                        cex_name = CEX_HOT_WALLETS[addr]
                        break

        reactivations.append(Reactivation(
            wallet=dw.wallet,
            balance_pct=dw.balance_pct,
            dormancy_days=dormancy_days_actual,
            reactivation_tx=tx_sig or "",
            sent_to_cex=sent_to_cex,
            cex_name=cex_name,
        ))

    return reactivations


def detect_reactivation_cluster(
    reactivations: list[Reactivation],
    min_wallets: int = GHOST_WATCHER_CLUSTER_MIN_WALLETS,
) -> ReactivationCluster:
    """Determine whether reactivations form a coordinated cluster.

    Pure function — no I/O.  A cluster is flagged when the number of
    reactivating wallets meets or exceeds *min_wallets*.
    """
    wallets = [r.wallet for r in reactivations]
    total_pct = sum(r.balance_pct for r in reactivations)

    return ReactivationCluster(
        wallets=wallets,
        count=len(wallets),
        total_balance_pct=total_pct,
        is_cluster=len(wallets) >= min_wallets,
    )


def compute_ghost_risk(
    token_mint: str,
    holders: list[dict] | None = None,
) -> GhostWatcherResult:
    """Compute composite ghost-watcher risk score for a token.

    Orchestrates the full pipeline:
    1. Fetch/filter holders
    2. Classify dormant wallets (I/O — RPC calls)
    3. Re-fetch latest activity for dormant wallets (I/O)
    4. Detect reactivations (pure, using fresh activity data)
    5. Detect reactivation cluster (pure)
    6. Score and return result

    Parameters
    ----------
    token_mint:
        The SPL token mint address.
    holders:
        Pre-fetched holder list.  When ``None`` the function will import
        and call ``cortex.data.helius_holders.get_holder_data`` to obtain
        holders automatically.
    """
    # Step 1: Obtain holders if not provided
    if holders is None:
        try:
            from cortex.data.helius_holders import get_holder_data
            holder_data = get_holder_data(token_mint)
            holders = holder_data.get("holders", []) if isinstance(holder_data, dict) else []
        except Exception as exc:
            logger.error("Failed to fetch holder data for %s: %s", token_mint, exc)
            return GhostWatcherResult(
                token_mint=token_mint,
                risk_score=0,
                dormant_whales_detected=0,
                wallets_reactivating=0,
                aggregate_dormant_balance_pct=0.0,
                cluster_detected=False,
                details={"error": "could_not_fetch_holders"},
                risk_factors=[],
            )

    # Step 2: Classify dormant wallets (performs RPC calls internally)
    dormant = classify_dormant_wallets(holders)

    if not dormant:
        return GhostWatcherResult(
            token_mint=token_mint,
            risk_score=0,
            dormant_whales_detected=0,
            wallets_reactivating=0,
            aggregate_dormant_balance_pct=0.0,
            cluster_detected=False,
            details={"dormant_wallets": 0},
            risk_factors=[],
        )

    # Step 3: Re-fetch current activity for all dormant wallets
    # (bypass cache to get the freshest data for reactivation detection)
    current_activities: dict[str, WalletActivity] = {}
    for dw in dormant:
        activity = get_wallet_last_activity(dw.wallet)
        if activity is not None:
            current_activities[dw.wallet] = activity

    # Step 4: Detect reactivations (pure function with CEX tx lookup)
    reactivations = detect_reactivations(dormant, current_activities)

    # Step 5: Detect cluster
    cluster = detect_reactivation_cluster(reactivations)

    # Step 6: Scoring
    score = 0
    risk_factors: list[str] = []

    # Any reactivation detected
    if reactivations:
        score += GHOST_WATCHER_SINGLE_REACTIVATION_SCORE
        risk_factors.append(
            f"dormant_reactivation:{len(reactivations)}_wallets"
        )

    # Cluster of 3+ coordinated reactivations
    if cluster.is_cluster:
        score += GHOST_WATCHER_CLUSTER_SCORE
        risk_factors.append(
            f"reactivation_cluster:{cluster.count}_wallets"
        )

    # Reactivating wallets hold >10% of supply
    if cluster.total_balance_pct > 10.0:
        score += GHOST_WATCHER_HIGH_CONCENTRATION_SCORE
        risk_factors.append(
            f"high_concentration:{cluster.total_balance_pct:.1f}%_supply"
        )

    # Any wallet dormant >365 days
    extreme_dormancy = any(dw.dormancy_days > 365.0 for dw in dormant)
    if extreme_dormancy:
        score += GHOST_WATCHER_EXTREME_DORMANCY_SCORE
        max_dormancy = max(dw.dormancy_days for dw in dormant)
        risk_factors.append(
            f"extreme_dormancy:{max_dormancy:.0f}_days"
        )

    # Any reactivation sent to CEX
    cex_reactivations = [r for r in reactivations if r.sent_to_cex]
    if cex_reactivations:
        score += GHOST_WATCHER_CEX_DESTINATION_SCORE
        cex_names = list({r.cex_name for r in cex_reactivations if r.cex_name})
        risk_factors.append(
            f"cex_destination:{','.join(cex_names)}"
        )

    # Clamp 0-100
    score = min(100, max(0, score))

    aggregate_dormant_pct = sum(dw.balance_pct for dw in dormant)

    result = GhostWatcherResult(
        token_mint=token_mint,
        risk_score=score,
        dormant_whales_detected=len(dormant),
        wallets_reactivating=len(reactivations),
        aggregate_dormant_balance_pct=aggregate_dormant_pct,
        cluster_detected=cluster.is_cluster,
        details={
            "token_mint": token_mint,
            "dormant_wallets": len(dormant),
            "reactivations": len(reactivations),
            "cluster_size": cluster.count,
            "cluster_balance_pct": cluster.total_balance_pct,
            "cex_sends": len(cex_reactivations),
            "max_dormancy_days": max(dw.dormancy_days for dw in dormant),
            "dormant_wallet_addresses": [dw.wallet for dw in dormant],
            "reactivated_wallet_addresses": [r.wallet for r in reactivations],
        },
        risk_factors=risk_factors,
    )

    return result
