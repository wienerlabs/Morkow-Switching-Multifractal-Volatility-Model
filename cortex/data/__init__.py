"""Data adapters for the Cortex Risk Engine.

Modules:
    solana          — Birdeye/Drift/Raydium OHLCV + funding + liquidity
    oracle          — Pyth Network price feeds (discovery, live, historical, SSE)
    dexscreener     — DexScreener token prices, pair liquidity, new token discovery
    helius_holders  — Helius DAS holder concentration risk
    streams         — Helius WebSocket transaction monitor
    social          — Twitter/social sentiment aggregation
    macro           — CoinGecko + Fear & Greed macro indicators
    onchain_events  — On-chain event detection (large swaps, oracle jumps)
    onchain_liquidity — CLMM tick data + liquidity depth curves
    tick_data       — Fine-grained OHLCV tick bars
    ccxt_feed       — CCXT exchange data adapter
    jupiter         — Jupiter swap execution
    rpc_failover    — Resilient RPC connection pooling
"""

__all__ = [
    "solana",
    "oracle",
    "dexscreener",
    "helius_holders",
    "streams",
    "social",
    "macro",
]
