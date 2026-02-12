"""Axiom Trade DEX aggregator adapter for the Cortex Risk Engine.

Provides:
  1. Token info & cross-DEX price feeds (complementary to Birdeye/Pyth)
  2. Pair stats & liquidity depth for LVaR spread estimation
  3. New token detection via WebSocket (AGGRESSIVE mode)
  4. Wallet balance queries
  5. Trade execution helpers (buy/sell with MEV protection)

Auth: Axiom does NOT use API keys. It uses email/password + OTP login
to obtain access_token/refresh_token. Three ways to authenticate:
  A) Set AXIOM_AUTH_TOKEN + AXIOM_REFRESH_TOKEN in .env (recommended)
  B) Set AXIOM_EMAIL + AXIOM_PASSWORD, then call login_interactive()
  C) Let SDK use saved tokens from disk (use_saved_tokens=True)
"""

import asyncio
import logging
import time
from collections import deque
from typing import Any, Callable

from cortex.config import (
    AXIOM_AUTH_TOKEN,
    AXIOM_EMAIL,
    AXIOM_MAX_RETRIES,
    AXIOM_MIN_LIQUIDITY_SOL,
    AXIOM_NEW_TOKEN_BUFFER,
    AXIOM_PASSWORD,
    AXIOM_REFRESH_TOKEN,
    AXIOM_WS_RECONNECT_DELAY,
)

logger = logging.getLogger(__name__)

_client = None
_new_token_buffer: deque[dict[str, Any]] = deque(maxlen=AXIOM_NEW_TOKEN_BUFFER)
_price_cache: dict[str, dict[str, Any]] = {}
_ws_status: dict[str, Any] = {
    "connected": False,
    "last_event_time": None,
    "tokens_detected": 0,
    "started_at": None,
}


def _get_client():
    """Lazy-init AxiomTradeClient with best available auth method."""
    global _client
    if _client is not None:
        return _client
    try:
        from axiomtradeapi import AxiomTradeClient
    except ImportError:
        raise ImportError("axiomtradeapi not installed. Run: pip install axiomtradeapi>=1.1.0")

    # Priority: stored tokens > email/password > saved tokens from disk
    if AXIOM_AUTH_TOKEN and AXIOM_REFRESH_TOKEN:
        _client = AxiomTradeClient(
            auth_token=AXIOM_AUTH_TOKEN,
            refresh_token=AXIOM_REFRESH_TOKEN,
        )
        logger.info("Axiom client initialized with stored tokens")
    elif AXIOM_EMAIL and AXIOM_PASSWORD:
        _client = AxiomTradeClient(
            username=AXIOM_EMAIL,
            password=AXIOM_PASSWORD,
            use_saved_tokens=True,
        )
        logger.info("Axiom client initialized with email/password (login required)")
    else:
        _client = AxiomTradeClient(use_saved_tokens=True)
        logger.warning("Axiom: no credentials set — using saved tokens or read-only mode")
    return _client


def login_interactive(email: str = None, password: str = None) -> dict:
    """Interactive login — prompts for OTP. Run once to get tokens.

    Usage:
        python -c "from cortex.data.axiom import login_interactive; login_interactive()"

    After login, copy the printed tokens to your .env file.
    """
    client = _get_client()
    result = client.login(email=email or AXIOM_EMAIL, password=password or AXIOM_PASSWORD)
    if result.get("success"):
        print(f"\n✅ Login successful! Add these to your .env:\n")
        print(f"AXIOM_AUTH_TOKEN={result['access_token']}")
        print(f"AXIOM_REFRESH_TOKEN={result['refresh_token']}")
    else:
        print(f"\n❌ Login failed: {result.get('message')}")
    return result


def is_available() -> bool:
    """Check if Axiom client has valid authentication."""
    try:
        client = _get_client()
        return client.is_authenticated()
    except Exception:
        return False


def _retry(func: Callable, *args, retries: int = AXIOM_MAX_RETRIES, **kwargs) -> Any:
    last_err = None
    for attempt in range(retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_err = e
            logger.warning("Axiom API attempt %d/%d failed: %s", attempt + 1, retries, e)
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
    raise last_err


# ── 1. Token Info & Price Feed ──


def get_token_price(token_address: str) -> dict[str, Any]:
    """Fetch token info including cross-DEX aggregated price."""
    client = _get_client()
    data = _retry(client.get_token_info, token_address)
    result = {
        "source": "axiom",
        "token_address": token_address,
        "raw": data,
        "timestamp": time.time(),
    }
    _price_cache[token_address] = result
    return result


def get_token_detailed() -> dict[str, Any]:
    """Fetch detailed token info (requires auth)."""
    client = _get_client()
    return _retry(client.get_token_info_detailed)


# ── 2. Pair Stats & Liquidity Depth ──


def get_pair_liquidity(pair_address: str) -> dict[str, Any]:
    """Fetch pair stats for LVaR liquidity depth estimation."""
    client = _get_client()
    info = _retry(client.get_pair_info, pair_address)
    stats = _retry(client.get_pair_stats, pair_address)
    return {
        "source": "axiom",
        "pair_address": pair_address,
        "pair_info": info,
        "pair_stats": stats,
        "timestamp": time.time(),
    }


def extract_liquidity_metrics(pair_address: str) -> dict[str, Any]:
    """Extract LVaR-compatible spread and depth metrics from Axiom pair data.

    Returns spread_pct, spread_vol_pct suitable for liquidity_adjusted_var().
    Uses Amihud illiquidity proxy: |return| / volume as spread estimate.
    """
    try:
        data = get_pair_liquidity(pair_address)
        stats = data.get("pair_stats") or {}
        info = data.get("pair_info") or {}

        volume_24h = float(stats.get("volume_24h", 0) or info.get("volume", 0) or 0)
        liquidity = float(info.get("liquidity", 0) or stats.get("liquidity", 0) or 0)
        price = float(info.get("price", 0) or stats.get("price", 0) or 0)

        if liquidity > 0 and volume_24h > 0:
            turnover = volume_24h / liquidity
            spread_pct = max(0.01, (1.0 / (1.0 + turnover)) * 2.0)
        elif volume_24h > 0 and price > 0:
            spread_pct = max(0.05, 1.0 / (volume_24h / 1000.0))
        else:
            spread_pct = 1.0  # conservative fallback

        spread_vol_pct = spread_pct * 0.3  # heuristic: 30% of spread as vol

        return {
            "source": "axiom",
            "pair_address": pair_address,
            "spread_pct": spread_pct,
            "spread_vol_pct": spread_vol_pct,
            "volume_24h": volume_24h,
            "liquidity": liquidity,
            "timestamp": time.time(),
        }
    except Exception as e:
        logger.warning("Axiom liquidity metrics failed for %s: %s", pair_address, e)
        return {"source": "axiom", "error": str(e), "spread_pct": None, "spread_vol_pct": None}


# ── 3. Holder Data & Token Analysis (rug-pull risk) ──


def get_holder_data(pair_address: str) -> dict[str, Any]:
    """Fetch holder distribution for rug-pull risk assessment."""
    client = _get_client()
    data = _retry(client.get_holder_data, pair_address)
    return {"source": "axiom", "pair_address": pair_address, "holders": data, "timestamp": time.time()}


def get_token_analysis(dev_address: str, token_ticker: str) -> dict[str, Any]:
    """Analyze token developer history for scam detection."""
    client = _get_client()
    data = _retry(client.get_token_analysis, dev_address, token_ticker)
    return {"source": "axiom", "dev_address": dev_address, "ticker": token_ticker, "analysis": data, "timestamp": time.time()}


# ── 4. WebSocket New Token Detection ──


def _on_new_token(token_data: dict) -> None:
    """Callback for WebSocket new token events."""
    event = {
        "token_data": token_data,
        "timestamp": time.time(),
        "meets_min_liquidity": False,
    }
    try:
        liq = float(token_data.get("liquidity", 0) or 0)
        event["meets_min_liquidity"] = liq >= AXIOM_MIN_LIQUIDITY_SOL
    except (ValueError, TypeError):
        pass

    _new_token_buffer.append(event)
    _ws_status["tokens_detected"] += 1
    _ws_status["last_event_time"] = time.time()
    logger.debug("New token detected: %s", token_data.get("name", "unknown"))


async def start_new_token_stream() -> None:
    """Start WebSocket stream for new token detection (AGGRESSIVE mode)."""
    client = _get_client()
    _ws_status["started_at"] = time.time()

    while True:
        try:
            ws_client = client.get_websocket_client()
            ws_client.subscribe_new_tokens(_on_new_token)
            _ws_status["connected"] = True
            logger.info("Axiom new-token WebSocket connected")
            await asyncio.get_event_loop().run_in_executor(None, ws_client.start)
        except Exception:
            _ws_status["connected"] = False
            logger.exception("Axiom WebSocket error, reconnecting in %ds", AXIOM_WS_RECONNECT_DELAY)
            await asyncio.sleep(AXIOM_WS_RECONNECT_DELAY)


def get_new_tokens(limit: int = 20, min_liquidity: bool = True) -> list[dict[str, Any]]:
    """Return recently detected new tokens."""
    tokens = list(_new_token_buffer)
    if min_liquidity:
        tokens = [t for t in tokens if t.get("meets_min_liquidity")]
    return tokens[-limit:]


def get_ws_status() -> dict[str, Any]:
    """Return WebSocket connection status."""
    return dict(_ws_status)


# ── 5. Wallet Balance ──


def get_wallet_balance(wallet_address: str) -> dict[str, Any]:
    """Get SOL balance for a wallet."""
    client = _get_client()
    balance = _retry(client.get_sol_balance, wallet_address)
    return {"source": "axiom", "wallet": wallet_address, "sol_balance": balance, "timestamp": time.time()}


def get_token_balance(wallet_address: str, token_mint: str) -> dict[str, Any]:
    """Get SPL token balance for a wallet."""
    client = _get_client()
    balance = _retry(client.get_token_balance, wallet_address, token_mint)
    return {"source": "axiom", "wallet": wallet_address, "token_mint": token_mint, "balance": balance, "timestamp": time.time()}


# ── 6. Trade Execution (Wave 9) ──


def execute_buy(
    private_key: str,
    token_mint: str,
    amount_sol: float,
    slippage_bps: int = 100,
    mev_protection: bool = True,
) -> dict[str, Any]:
    """Execute a buy order via Axiom DEX aggregator."""
    client = _get_client()
    result = _retry(
        client.buy_token,
        private_key,
        token_mint,
        amount_sol,
        slippage=slippage_bps,
    )
    return {"source": "axiom", "action": "buy", "token_mint": token_mint, "amount_sol": amount_sol, "result": result, "timestamp": time.time()}


def execute_sell(
    private_key: str,
    token_mint: str,
    amount_pct: float = 100.0,
    slippage_bps: int = 100,
    mev_protection: bool = True,
) -> dict[str, Any]:
    """Execute a sell order via Axiom DEX aggregator."""
    client = _get_client()
    result = _retry(
        client.sell_token,
        private_key,
        token_mint,
        amount_pct,
        slippage=slippage_bps,
    )
    return {"source": "axiom", "action": "sell", "token_mint": token_mint, "amount_pct": amount_pct, "result": result, "timestamp": time.time()}


# ── 7. Cache Accessors ──


def get_cached_prices() -> dict[str, dict[str, Any]]:
    """Return all cached token prices."""
    return dict(_price_cache)

