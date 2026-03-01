"""Launch tracker — detect CEX-funded token launches and bundle-buy patterns.

Analyzes recently-deployed Solana tokens for suspicious launch patterns:
  - Deployer wallet funded from known CEX hot wallets
  - Bundle buying (3+ wallets buying within 30s window)
  - Fast deploy-to-first-trade latency
  - High holder concentration

Feeds into Guardian composite risk scoring as an optional component.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import httpx

from cortex.config import (
    HELIUS_RPC_URL,
    ONCHAIN_HTTP_TIMEOUT,
    ONCHAIN_CACHE_TTL,
    LAUNCH_TRACKER_BUNDLE_WINDOW_SEC,
    LAUNCH_TRACKER_BUNDLE_MIN_WALLETS,
    LAUNCH_TRACKER_MAX_AGE_HOURS,
    LAUNCH_TRACKER_CEX_SCORE,
    LAUNCH_TRACKER_BUNDLE_SCORE,
    LAUNCH_TRACKER_FAST_DEPLOY_SCORE,
    LAUNCH_TRACKER_CONCENTRATION_SCORE,
    LAUNCH_TRACKER_YOUNG_DEPLOYER_SCORE,
)

logger = logging.getLogger(__name__)

# ── Known CEX Hot Wallets (Solana mainnet) ──
# Sources: on-chain labeling services (Arkham, Solscan labels)
CEX_HOT_WALLETS: dict[str, str] = {
    # Binance
    "2ojv9BAiHUrvsQVrjxMPJH3QoJw3Fwrw2vfjqXVpLLoH": "Binance",
    "5tzFkiKscjHsFKR9VpSFEMwTyLFjLxPgYHpJyoKpCGap": "Binance",
    "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM": "Binance",
    # Coinbase
    "GJRs4FwHtemZ5ZE9Q3sEoHTNHo3R2vNaLrgtABXmSdQQ": "Coinbase",
    "H8sMJSCQxfKiFTCfDR3DUMLPwcRbM61LGFJ8N4dK3WjS": "Coinbase",
    # Kraken
    "FWznbcNXWQuHTawe9RxvQ2LdCENssh12dsznf4RiouN5": "Kraken",
    "4wBqpZM9k7AKhEDtJVcRnKFLqfobLkRByqoqGS5yBVvS": "Kraken",
    # Bybit
    "AC5RDfQFmDS1deWZos921JfqscXdByf4BKMt8MUyA6CV": "Bybit",
    # OKX
    "5VCwKtCXgCJ6kit5FybXjvFocNUNL1dFo6PGaBv5M5Hj": "OKX",
    # KuCoin
    "BmFdpraQhkiDQE6SnfG5PVddRftYjP3sBb6fPKPruf8H": "KuCoin",
}


# ── Dataclasses ──

@dataclass
class LaunchInfo:
    token_mint: str
    deployer_wallet: str
    created_at: datetime
    token_age_hours: float


@dataclass
class FundingSource:
    source_address: str
    source_type: Literal["cex", "dex", "unknown", "contract"]
    amount_sol: float
    exchange_name: str | None
    tx_signature: str


@dataclass
class EarlyBuyer:
    wallet: str
    buy_time: datetime
    amount_sol: float
    tx_signature: str
    latency_from_deploy_sec: float


@dataclass
class BundleCluster:
    wallets: list[str]
    buy_window_sec: float
    total_amount_sol: float
    is_bundle: bool


@dataclass
class LaunchRiskResult:
    score: int  # 0-100
    cex_funded: bool
    bundle_detected: bool
    deployer_age_days: float
    top10_concentration_pct: float
    deploy_to_first_trade_sec: float
    details: dict = field(default_factory=dict)
    risk_factors: list[str] = field(default_factory=list)


# ── Cache ──

_cache: dict[str, tuple[float, Any]] = {}


def _cached(key: str, ttl: float = ONCHAIN_CACHE_TTL) -> Any | None:
    entry = _cache.get(key)
    if entry and (time.time() - entry[0]) < ttl:
        return entry[1]
    return None


def _set_cache(key: str, value: Any) -> None:
    _cache[key] = (time.time(), value)


# ── Helius RPC helpers ──

_MAX_RETRIES = 3


def _post_rpc(payload: dict, url: str | None = None) -> dict | None:
    """Post a JSON-RPC request to Helius with retries."""
    target = url or HELIUS_RPC_URL
    if not target:
        logger.warning("HELIUS_RPC_URL not configured — launch tracker disabled")
        return None

    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = httpx.post(target, json=payload, timeout=ONCHAIN_HTTP_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                logger.warning("RPC error: %s", data["error"])
                return None
            return data
        except Exception as exc:
            last_err = exc
            if attempt < _MAX_RETRIES:
                time.sleep(0.5 * attempt)
    logger.error("Helius RPC failed after %d attempts: %s", _MAX_RETRIES, last_err)
    return None


# ── Core Functions ──

def get_token_creation_info(token_mint: str) -> LaunchInfo | None:
    """Get token creation metadata via Helius DAS getAsset."""
    cached = _cached(f"launch_info:{token_mint}")
    if cached is not None:
        return cached

    payload = {
        "jsonrpc": "2.0",
        "id": "launch-tracker-asset",
        "method": "getAsset",
        "params": {"id": token_mint},
    }

    data = _post_rpc(payload)
    if not data or "result" not in data:
        return None

    result = data["result"]

    # Extract authority (deployer)
    authorities = result.get("authorities", [])
    deployer = None
    for auth in authorities:
        if "mint" in auth.get("scopes", []):
            deployer = auth.get("address")
            break
    if not deployer:
        deployer = authorities[0]["address"] if authorities else None

    if not deployer:
        logger.warning("No authority found for token %s", token_mint)
        return None

    # Extract creation time from content metadata or supply
    created_ts = None

    # Try token_info.supply first (mint timestamp from on-chain)
    token_info = result.get("token_info", {})
    if token_info.get("mint_authority"):
        deployer = token_info["mint_authority"]

    # Try content metadata
    content = result.get("content", {})
    metadata = content.get("metadata", {})

    # Helius often includes created_at in the response
    if "created_at" in result:
        try:
            created_ts = datetime.fromisoformat(result["created_at"].replace("Z", "+00:00"))
        except (ValueError, TypeError):
            pass

    # Fallback: use the slot from the response to estimate time
    if created_ts is None:
        slot = result.get("slot", 0)
        if slot > 0:
            # Approximate: Solana epoch 0 started ~March 2020, ~2.5 slots/sec
            # Better approach: use getBlockTime for the slot
            block_time_payload = {
                "jsonrpc": "2.0",
                "id": "block-time",
                "method": "getBlockTime",
                "params": [slot],
            }
            bt_data = _post_rpc(block_time_payload)
            if bt_data and "result" in bt_data and bt_data["result"]:
                created_ts = datetime.fromtimestamp(bt_data["result"], tz=timezone.utc)

    if created_ts is None:
        created_ts = datetime.now(timezone.utc)
        logger.warning("Could not determine creation time for %s, using now", token_mint)

    now = datetime.now(timezone.utc)
    age_hours = (now - created_ts).total_seconds() / 3600.0

    info = LaunchInfo(
        token_mint=token_mint,
        deployer_wallet=deployer,
        created_at=created_ts,
        token_age_hours=age_hours,
    )
    _set_cache(f"launch_info:{token_mint}", info)
    return info


def get_deployer_funding_sources(
    deployer: str, lookback_txns: int = 50
) -> list[FundingSource]:
    """Analyze where the deployer wallet received SOL funding."""
    cached = _cached(f"funding:{deployer}")
    if cached is not None:
        return cached

    # Use Helius enhanced transaction history
    payload = {
        "jsonrpc": "2.0",
        "id": "deployer-history",
        "method": "getSignaturesForAddress",
        "params": [deployer, {"limit": lookback_txns}],
    }

    data = _post_rpc(payload)
    if not data or "result" not in data:
        return []

    signatures = [tx["signature"] for tx in data["result"] if "signature" in tx]
    if not signatures:
        return []

    # Fetch transaction details in batches
    sources: list[FundingSource] = []

    for sig in signatures[:lookback_txns]:
        tx_payload = {
            "jsonrpc": "2.0",
            "id": "tx-detail",
            "method": "getTransaction",
            "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
        }

        tx_data = _post_rpc(tx_payload)
        if not tx_data or "result" not in tx_data or tx_data["result"] is None:
            continue

        tx = tx_data["result"]
        meta = tx.get("meta", {})
        if meta.get("err") is not None:
            continue

        # Check for SOL transfers TO the deployer
        pre_balances = meta.get("preBalances", [])
        post_balances = meta.get("postBalances", [])
        account_keys = (
            tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
        )

        if not account_keys or not pre_balances or not post_balances:
            continue

        deployer_idx = None
        for i, key in enumerate(account_keys):
            addr = key if isinstance(key, str) else key.get("pubkey", "")
            if addr == deployer:
                deployer_idx = i
                break

        if deployer_idx is None:
            continue

        if deployer_idx < len(pre_balances) and deployer_idx < len(post_balances):
            delta_lamports = post_balances[deployer_idx] - pre_balances[deployer_idx]
            if delta_lamports > 0:
                amount_sol = delta_lamports / 1e9

                # Find the sender (who lost SOL)
                sender_addr = None
                for i, key in enumerate(account_keys):
                    if i == deployer_idx:
                        continue
                    addr = key if isinstance(key, str) else key.get("pubkey", "")
                    if i < len(pre_balances) and i < len(post_balances):
                        if post_balances[i] - pre_balances[i] < 0:
                            sender_addr = addr
                            break

                if sender_addr:
                    exchange_name = CEX_HOT_WALLETS.get(sender_addr)
                    source_type: Literal["cex", "dex", "unknown", "contract"] = (
                        "cex" if exchange_name else "unknown"
                    )

                    sources.append(FundingSource(
                        source_address=sender_addr,
                        source_type=source_type,
                        amount_sol=amount_sol,
                        exchange_name=exchange_name,
                        tx_signature=sig,
                    ))

    _set_cache(f"funding:{deployer}", sources)
    return sources


def get_early_buyers(
    token_mint: str, deploy_time: datetime, max_buyers: int = 20
) -> list[EarlyBuyer]:
    """Get the first N buy transactions for a token after deployment."""
    cached = _cached(f"early_buyers:{token_mint}")
    if cached is not None:
        return cached

    # Use Helius getSignaturesForAddress on the token mint
    payload = {
        "jsonrpc": "2.0",
        "id": "early-buyers",
        "method": "getSignaturesForAddress",
        "params": [token_mint, {"limit": max_buyers * 3}],  # fetch more to filter
    }

    data = _post_rpc(payload)
    if not data or "result" not in data:
        return []

    signatures = data["result"]
    buyers: list[EarlyBuyer] = []
    seen_wallets: set[str] = set()

    for sig_info in signatures:
        if len(buyers) >= max_buyers:
            break

        sig = sig_info.get("signature", "")
        block_time = sig_info.get("blockTime")

        if not block_time:
            continue

        buy_time = datetime.fromtimestamp(block_time, tz=timezone.utc)
        latency = (buy_time - deploy_time).total_seconds()

        if latency < 0:
            continue  # transaction before deploy — skip

        # Get transaction to find the buyer wallet
        tx_payload = {
            "jsonrpc": "2.0",
            "id": "buyer-tx",
            "method": "getTransaction",
            "params": [sig, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
        }

        tx_data = _post_rpc(tx_payload)
        if not tx_data or "result" not in tx_data or tx_data["result"] is None:
            continue

        tx = tx_data["result"]
        meta = tx.get("meta", {})
        if meta.get("err") is not None:
            continue

        # Find the fee payer (buyer)
        account_keys = (
            tx.get("transaction", {}).get("message", {}).get("accountKeys", [])
        )
        if not account_keys:
            continue

        fee_payer = account_keys[0]
        if isinstance(fee_payer, dict):
            fee_payer = fee_payer.get("pubkey", "")

        if fee_payer in seen_wallets:
            continue
        seen_wallets.add(fee_payer)

        # Estimate SOL spent from balance change
        pre_balances = meta.get("preBalances", [])
        post_balances = meta.get("postBalances", [])
        amount_sol = 0.0
        if pre_balances and post_balances:
            delta = pre_balances[0] - post_balances[0]
            amount_sol = max(0, delta) / 1e9

        buyers.append(EarlyBuyer(
            wallet=fee_payer,
            buy_time=buy_time,
            amount_sol=amount_sol,
            tx_signature=sig,
            latency_from_deploy_sec=latency,
        ))

    # Sort by buy time
    buyers.sort(key=lambda b: b.buy_time)
    _set_cache(f"early_buyers:{token_mint}", buyers)
    return buyers


def detect_bundle_cluster(
    buyers: list[EarlyBuyer],
    window_sec: float = LAUNCH_TRACKER_BUNDLE_WINDOW_SEC,
    min_wallets: int = LAUNCH_TRACKER_BUNDLE_MIN_WALLETS,
) -> BundleCluster:
    """Detect if early buyers form a bundle cluster (coordinated buying).

    Pure function — no I/O. Looks for N+ unique wallets buying within
    a time window, suggesting coordinated/automated purchases.
    """
    if len(buyers) < min_wallets:
        return BundleCluster(
            wallets=[b.wallet for b in buyers],
            buy_window_sec=0.0,
            total_amount_sol=sum(b.amount_sol for b in buyers),
            is_bundle=False,
        )

    sorted_buyers = sorted(buyers, key=lambda b: b.buy_time)

    best_cluster: list[EarlyBuyer] = []
    best_window = 0.0

    for i in range(len(sorted_buyers)):
        cluster = [sorted_buyers[i]]
        for j in range(i + 1, len(sorted_buyers)):
            delta = (sorted_buyers[j].buy_time - sorted_buyers[i].buy_time).total_seconds()
            if delta <= window_sec:
                cluster.append(sorted_buyers[j])
            else:
                break

        if len(cluster) > len(best_cluster):
            best_cluster = cluster
            if len(cluster) >= 2:
                best_window = (cluster[-1].buy_time - cluster[0].buy_time).total_seconds()

    unique_wallets = list({b.wallet for b in best_cluster})

    return BundleCluster(
        wallets=unique_wallets,
        buy_window_sec=best_window,
        total_amount_sol=sum(b.amount_sol for b in best_cluster),
        is_bundle=len(unique_wallets) >= min_wallets,
    )


def compute_launch_risk(
    token_mint: str,
    holder_concentration_pct: float | None = None,
) -> LaunchRiskResult:
    """Compute composite launch risk score for a token.

    Orchestrates all sub-analyses and produces a 0-100 risk score.
    Tokens older than LAUNCH_TRACKER_MAX_AGE_HOURS get score=0 (short-circuit).
    """
    # Step 1: Get token creation info
    info = get_token_creation_info(token_mint)
    if info is None:
        return LaunchRiskResult(
            score=0,
            cex_funded=False,
            bundle_detected=False,
            deployer_age_days=0.0,
            top10_concentration_pct=0.0,
            deploy_to_first_trade_sec=0.0,
            details={"error": "could_not_fetch_token_info"},
            risk_factors=[],
        )

    # Short-circuit for established tokens
    if info.token_age_hours > LAUNCH_TRACKER_MAX_AGE_HOURS:
        return LaunchRiskResult(
            score=0,
            cex_funded=False,
            bundle_detected=False,
            deployer_age_days=info.token_age_hours / 24.0,
            top10_concentration_pct=holder_concentration_pct or 0.0,
            deploy_to_first_trade_sec=0.0,
            details={"short_circuit": "token_too_old", "age_hours": info.token_age_hours},
            risk_factors=[],
        )

    score = 0
    risk_factors: list[str] = []

    # Step 2: Check deployer funding sources
    funding = get_deployer_funding_sources(info.deployer_wallet)
    cex_funded = any(f.source_type == "cex" for f in funding)
    cex_names = [f.exchange_name for f in funding if f.exchange_name]

    if cex_funded:
        score += LAUNCH_TRACKER_CEX_SCORE
        risk_factors.append(f"deployer_cex_funded:{','.join(set(cex_names))}")

    # Step 3: Get early buyers and detect bundles
    buyers = get_early_buyers(token_mint, info.created_at)
    cluster = detect_bundle_cluster(buyers)

    if cluster.is_bundle:
        score += LAUNCH_TRACKER_BUNDLE_SCORE
        risk_factors.append(f"bundle_detected:{len(cluster.wallets)}_wallets_in_{cluster.buy_window_sec:.1f}s")

    # Step 4: Deploy-to-first-trade latency
    deploy_to_trade = 0.0
    if buyers:
        deploy_to_trade = buyers[0].latency_from_deploy_sec
        if deploy_to_trade < 60.0:
            score += LAUNCH_TRACKER_FAST_DEPLOY_SCORE
            risk_factors.append(f"fast_deploy_to_trade:{deploy_to_trade:.1f}s")

    # Step 5: Holder concentration
    concentration = holder_concentration_pct or 0.0
    if concentration > 80.0:
        score += LAUNCH_TRACKER_CONCENTRATION_SCORE
        risk_factors.append(f"high_concentration:top10={concentration:.1f}%")

    # Step 6: Deployer wallet age
    deployer_age_days = info.token_age_hours / 24.0  # proxy: token age ≈ deployer activity age
    if deployer_age_days < 7.0:
        score += LAUNCH_TRACKER_YOUNG_DEPLOYER_SCORE
        risk_factors.append(f"young_deployer:{deployer_age_days:.1f}d")

    # Clamp score
    score = min(100, max(0, score))

    result = LaunchRiskResult(
        score=score,
        cex_funded=cex_funded,
        bundle_detected=cluster.is_bundle,
        deployer_age_days=deployer_age_days,
        top10_concentration_pct=concentration,
        deploy_to_first_trade_sec=deploy_to_trade,
        details={
            "token_mint": token_mint,
            "deployer": info.deployer_wallet,
            "created_at": info.created_at.isoformat(),
            "token_age_hours": info.token_age_hours,
            "funding_sources": len(funding),
            "cex_sources": cex_names,
            "early_buyers_count": len(buyers),
            "bundle_wallets": len(cluster.wallets),
            "bundle_window_sec": cluster.buy_window_sec,
        },
        risk_factors=risk_factors,
    )

    return result
