from __future__ import annotations
"""
Environment settings and API configurations.
Solana-focused DeFi agent configuration.
"""
import os
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Config directory path
CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


def load_yaml_config(filename: str) -> dict[str, Any]:
    """Load a YAML config file."""
    config_path = CONFIG_DIR / filename
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


class Settings(BaseModel):
    """Environment configuration for Cortex AI Agent."""

    # ==========================================================================
    # SOLANA CONFIGURATION (PRIMARY)
    # ==========================================================================

    # Solana RPC
    solana_rpc_url: str = Field(
        default_factory=lambda: os.getenv("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com")
    )
    solana_chain_id: str = Field(default="mainnet-beta")

    # Helius API (Solana transactions, DAS API, webhooks)
    helius_api_key: str = Field(default_factory=lambda: os.getenv("HELIUS_API_KEY", ""))
    helius_rpc_url: str = Field(
        default="https://mainnet.helius-rpc.com/?api-key={api_key}"
    )
    helius_api_url: str = Field(
        default="https://api.helius.xyz/v0"
    )

    # Birdeye API (Solana DEX data, token prices, OHLCV)
    birdeye_api_key: str = Field(default_factory=lambda: os.getenv("BIRDEYE_API_KEY", ""))
    birdeye_api_url: str = Field(default="https://public-api.birdeye.so")

    # Jupiter API (Solana DEX aggregator, swap routes)
    jupiter_api_key: str = Field(default_factory=lambda: os.getenv("JUPITER_API_KEY", ""))
    jupiter_api_url: str = Field(default="https://api.jup.ag/ultra/v1")  # Jupiter Ultra endpoint
    jupiter_quote_url: str = Field(default="https://api.jup.ag/quote/v6")
    jupiter_price_api_url: str = Field(default="https://price.jup.ag/v6")

    # Solscan API (Solana block explorer - optional, requires paid tier)
    solscan_api_key: str = Field(default_factory=lambda: os.getenv("SOLSCAN_API_KEY", ""))
    solscan_api_url: str = Field(default="https://pro-api.solscan.io/v2.0")

    # Solana DEX Pool Addresses
    # Raydium AMM V4 (SOL/USDC)
    raydium_sol_usdc_pool: str = Field(
        default="58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2"
    )
    # Orca Whirlpool (SOL/USDC)
    orca_sol_usdc_pool: str = Field(
        default="HJPjoWUrhoZzkNfRpHuieeFk9WcZWjwy6PBjZ81ngndJ"
    )
    # Phoenix (SOL/USDC) - Order book DEX
    phoenix_sol_usdc_market: str = Field(
        default="4DoNfFBfF7UokCC2FQzriy7yHK6DY6NVdYpuekQ5pRgg"
    )

    # Solana Token Addresses
    sol_mint: str = Field(default="So11111111111111111111111111111111111111112")  # Wrapped SOL
    usdc_mint: str = Field(default="EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v")  # USDC
    usdt_mint: str = Field(default="Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB")  # USDT
    ray_mint: str = Field(default="4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R")  # RAY
    orca_mint: str = Field(default="orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE")  # ORCA



    # ==========================================================================
    # SHARED CONFIGURATION
    # ==========================================================================

    # Database Configuration (Supabase PostgreSQL)
    database_url: str = Field(default_factory=lambda: os.getenv("DATABASE_URL", ""))

    # Redis Configuration (Upstash)
    redis_url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", ""))

    # HuggingFace Hub (Model Storage)
    hf_token: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))
    hf_repo_id: str = Field(default_factory=lambda: os.getenv("HF_REPO_ID", "cortex-ai/models"))

    # API Server Configuration
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default_factory=lambda: int(os.getenv("AGENT_API_PORT", "8001")))
    cors_origins: list[str] = Field(
        default_factory=lambda: os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:5173"
        ).split(",")
    )

    # Vault Configuration
    vault_address: str = Field(default_factory=lambda: os.getenv("VAULT_ADDRESS", ""))

    # Private key for transactions
    solana_private_key: str = Field(default_factory=lambda: os.getenv("SOLANA_PRIVATE_KEY", ""))  # Solana base58

    # ==========================================================================
    # SOLANA API HELPERS
    # ==========================================================================

    def get_helius_rpc_url(self) -> str:
        """Get Helius RPC URL with API key."""
        return self.helius_rpc_url.format(api_key=self.helius_api_key)

    def get_helius_api_url(self, endpoint: str = "") -> str:
        """Get Helius API URL with API key."""
        base = f"{self.helius_api_url}/{endpoint}?api-key={self.helius_api_key}"
        return base.rstrip("?api-key=") if not self.helius_api_key else base

    def get_birdeye_headers(self) -> dict[str, str]:
        """Get Birdeye API headers."""
        return {
            "X-API-KEY": self.birdeye_api_key,
            "x-chain": "solana"
        }

    def get_solscan_headers(self) -> dict[str, str]:
        """Get Solscan API headers."""
        return {"token": self.solscan_api_key}


settings = Settings()


# Load YAML configs
domain_params = load_yaml_config("domain_params.yaml")
risk_params = load_yaml_config("risk_params.yaml")
model_config = load_yaml_config("model_config.yaml")
