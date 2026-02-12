"""Jupiter Swap API adapter — official Solana DEX aggregator for trade execution.

Uses the Metis Swap API (https://api.jup.ag/swap/v1/) for:
  - GET /quote — best route for a token pair
  - POST /swap — build a swap transaction
  - Local signing with solders.Keypair → send to Solana RPC

No third-party SDKs. Private keys never leave the local process.
"""

import base64
import logging
import time
from typing import Any

import httpx

from cortex.config import (
    JUPITER_API_URL,
    JUPITER_RPC_URL,
    JUPITER_TIMEOUT,
    JUPITER_MAX_RETRIES,
    JUPITER_SLIPPAGE_BPS,
    JUPITER_API_KEY,
)

logger = logging.getLogger(__name__)

SOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
LAMPORTS_PER_SOL = 1_000_000_000


def _headers() -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json"}
    if JUPITER_API_KEY:
        h["x-api-key"] = JUPITER_API_KEY
    return h


def _get_quote(input_mint: str, output_mint: str, amount_lamports: int, slippage_bps: int) -> dict[str, Any]:
    params = {
        "inputMint": input_mint, "outputMint": output_mint,
        "amount": str(amount_lamports), "slippageBps": slippage_bps,
        "restrictIntermediateTokens": "true",
    }
    for attempt in range(1, JUPITER_MAX_RETRIES + 1):
        try:
            resp = httpx.get(f"{JUPITER_API_URL}/quote", params=params, headers=_headers(), timeout=JUPITER_TIMEOUT)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.warning("Jupiter quote attempt %d/%d failed: %s", attempt, JUPITER_MAX_RETRIES, e)
            if attempt == JUPITER_MAX_RETRIES:
                raise
            time.sleep(1 * attempt)
    raise RuntimeError("Jupiter quote failed after retries")


def _build_swap_tx(quote: dict[str, Any], user_pubkey: str) -> bytes:
    body = {"quoteResponse": quote, "userPublicKey": user_pubkey, "wrapAndUnwrapSol": True}
    for attempt in range(1, JUPITER_MAX_RETRIES + 1):
        try:
            resp = httpx.post(f"{JUPITER_API_URL}/swap", json=body, headers=_headers(), timeout=JUPITER_TIMEOUT)
            resp.raise_for_status()
            return base64.b64decode(resp.json()["swapTransaction"])
        except Exception as e:
            logger.warning("Jupiter swap build attempt %d/%d failed: %s", attempt, JUPITER_MAX_RETRIES, e)
            if attempt == JUPITER_MAX_RETRIES:
                raise
            time.sleep(1 * attempt)
    raise RuntimeError("Jupiter swap build failed after retries")


def _sign_and_send(tx_bytes: bytes, private_key: str) -> str:
    from solders.keypair import Keypair  # type: ignore[import-untyped]
    from solders.transaction import VersionedTransaction  # type: ignore[import-untyped]
    import base58 as b58

    kp = Keypair.from_bytes(b58.b58decode(private_key))
    tx = VersionedTransaction.from_bytes(tx_bytes)
    signed = VersionedTransaction(tx.message, [kp])
    encoded = base64.b64encode(bytes(signed)).decode("ascii")

    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
        "params": [encoded, {"encoding": "base64", "skipPreflight": False, "maxRetries": 3}],
    }
    resp = httpx.post(JUPITER_RPC_URL, json=payload, timeout=JUPITER_TIMEOUT)
    resp.raise_for_status()
    result = resp.json()
    if "error" in result:
        raise RuntimeError(f"RPC error: {result['error']}")
    return result["result"]


def _get_token_balance_lamports(owner: str, token_mint: str) -> int:
    payload = {
        "jsonrpc": "2.0", "id": 1, "method": "getTokenAccountsByOwner",
        "params": [owner, {"mint": token_mint}, {"encoding": "jsonParsed"}],
    }
    resp = httpx.post(JUPITER_RPC_URL, json=payload, timeout=JUPITER_TIMEOUT)
    resp.raise_for_status()
    result = resp.json()
    if "error" in result:
        raise RuntimeError(f"RPC error fetching balance: {result['error']}")
    accounts = result.get("result", {}).get("value", [])
    if not accounts:
        return 0
    return int(accounts[0]["account"]["data"]["parsed"]["info"]["tokenAmount"]["amount"])




def execute_buy(
    private_key: str, token_mint: str, amount_sol: float,
    slippage_bps: int = JUPITER_SLIPPAGE_BPS, mev_protection: bool = True,
) -> dict[str, Any]:
    """Buy a token with SOL via Jupiter Swap API. Private key stays local."""
    from solders.keypair import Keypair  # type: ignore[import-untyped]
    import base58 as b58

    kp = Keypair.from_bytes(b58.b58decode(private_key))
    user_pubkey = str(kp.pubkey())
    amount_lamports = int(amount_sol * LAMPORTS_PER_SOL)

    logger.info("Jupiter BUY: %s SOL → %s (slippage=%dbps)", amount_sol, token_mint[:8], slippage_bps)
    quote = _get_quote(SOL_MINT, token_mint, amount_lamports, slippage_bps)
    tx_bytes = _build_swap_tx(quote, user_pubkey)
    tx_sig = _sign_and_send(tx_bytes, private_key)

    return {
        "source": "jupiter", "action": "buy", "token_mint": token_mint,
        "amount_sol": amount_sol, "out_amount": quote.get("outAmount"),
        "price_impact_pct": quote.get("priceImpactPct"),
        "tx_signature": tx_sig, "route_plan": quote.get("routePlan"),
        "timestamp": time.time(),
    }


def execute_sell(
    private_key: str, token_mint: str, amount_pct: float = 100.0,
    slippage_bps: int = JUPITER_SLIPPAGE_BPS, mev_protection: bool = True,
) -> dict[str, Any]:
    """Sell a token for SOL via Jupiter Swap API. Private key stays local."""
    from solders.keypair import Keypair  # type: ignore[import-untyped]
    import base58 as b58

    kp = Keypair.from_bytes(b58.b58decode(private_key))
    user_pubkey = str(kp.pubkey())
    token_balance = _get_token_balance_lamports(user_pubkey, token_mint)
    sell_amount = int(token_balance * (amount_pct / 100.0))
    if sell_amount <= 0:
        raise ValueError(f"No token balance to sell for {token_mint}")

    logger.info("Jupiter SELL: %s%% of %s (amount=%d, slippage=%dbps)", amount_pct, token_mint[:8], sell_amount, slippage_bps)
    quote = _get_quote(token_mint, SOL_MINT, sell_amount, slippage_bps)
    tx_bytes = _build_swap_tx(quote, user_pubkey)
    tx_sig = _sign_and_send(tx_bytes, private_key)

    return {
        "source": "jupiter", "action": "sell", "token_mint": token_mint,
        "amount_pct": amount_pct, "sell_amount_lamports": sell_amount,
        "out_amount": quote.get("outAmount"),
        "price_impact_pct": quote.get("priceImpactPct"),
        "tx_signature": tx_sig, "route_plan": quote.get("routePlan"),
        "timestamp": time.time(),
    }
