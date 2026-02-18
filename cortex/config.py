"""Centralized configuration — single source of truth for all tuneable constants.

Every value is backed by an environment variable with a sensible default so the
system works out-of-the-box while remaining fully configurable in production.
"""

__all__ = [
    "validate_config",
    # Guardian
    "GUARDIAN_WEIGHTS",
    "CIRCUIT_BREAKER_THRESHOLD",
    "APPROVAL_THRESHOLD",
    "GUARDIAN_KELLY_FRACTION",
    "GUARDIAN_KELLY_MIN_TRADES",
    "GUARDIAN_AUTO_DEBATE_THRESHOLD",
    # Scoring
    "EVT_SCORE_FLOOR",
    "EVT_SCORE_RANGE",
    "SVJ_BASE_CAP",
    "REGIME_BASE_MAX",
    "ALAMS_SCORE_VAR_FLOOR",
    "ALAMS_SCORE_VAR_CEILING",
    "CRISIS_REGIME_HAIRCUT",
    "NEAR_CRISIS_REGIME_HAIRCUT",
    # News & sentiment
    "SOURCE_CREDIBILITY",
    "DEFAULT_CREDIBILITY",
    "HALF_LIFE_HOURS",
    "DECAY_LAMBDA",
    # Data sources
    "BIRDEYE_BASE",
    "DRIFT_DATA_API",
    "RAYDIUM_API",
    "PYTH_HERMES_URL",
    "PYTH_BUFFER_DEPTH",
    "PYTH_DEFAULT_WATCHLIST",
    "DEXSCREENER_BASE_URL",
    "DEXSCREENER_TIMEOUT",
    "DEXSCREENER_MAX_RETRIES",
    "DEXSCREENER_CACHE_TTL",
    "DEXSCREENER_MIN_LIQUIDITY_USD",
    # Helius / on-chain
    "HELIUS_API_KEY",
    "HELIUS_RPC_URL",
    "ONCHAIN_HTTP_TIMEOUT",
    "ONCHAIN_CACHE_TTL",
    # Execution
    "EXECUTION_ENABLED",
    "SIMULATION_MODE",
    "TRADING_MODE",
    "JUPITER_API_URL",
    "JUPITER_SLIPPAGE_BPS",
    # Circuit breaker
    "CB_THRESHOLD",
    "CB_CONSECUTIVE_CHECKS",
    "CB_COOLDOWN_SECONDS",
    "CB_STRATEGIES",
    # Debate
    "DEBATE_MAX_ROUNDS",
    "DEBATE_TIMEOUT_MS",
    # Engine selection
    "EVT_ENGINE",
    "FBM_ENGINE",
    "HAWKES_ENGINE",
    "COPULA_ENGINE",
    "PORTFOLIO_OPT_ENGINE",
    # Observability
    "LOG_FORMAT",
    "LOG_LEVEL",
    "METRICS_ENABLED",
    "API_VERSION",
]

import json
import math
import os

# ── Guardian weights & thresholds ──

GUARDIAN_WEIGHTS: dict[str, float] = json.loads(
    os.environ.get(
        "GUARDIAN_WEIGHTS",
        json.dumps({"evt": 0.20, "svj": 0.15, "hawkes": 0.15, "regime": 0.15, "news": 0.10, "alams": 0.25, "agent_confidence": 0.10}),
    )
)
CIRCUIT_BREAKER_THRESHOLD = float(os.environ.get("CIRCUIT_BREAKER_THRESHOLD", "90"))
CACHE_TTL_SECONDS = float(os.environ.get("CACHE_TTL_SECONDS", "5.0"))
DECISION_VALIDITY_SECONDS = float(os.environ.get("DECISION_VALIDITY_SECONDS", "30.0"))
APPROVAL_THRESHOLD = float(os.environ.get("APPROVAL_THRESHOLD", "75.0"))

# ── Guardian scoring constants ──

EVT_SCORE_FLOOR = float(os.environ.get("EVT_SCORE_FLOOR", "2.0"))
EVT_SCORE_RANGE = float(os.environ.get("EVT_SCORE_RANGE", "13.0"))

SVJ_BASE_CAP = float(os.environ.get("SVJ_BASE_CAP", "80.0"))
SVJ_BASE_MULT = float(os.environ.get("SVJ_BASE_MULT", "0.8"))
SVJ_INTENSITY_FLOOR = float(os.environ.get("SVJ_INTENSITY_FLOOR", "20"))
SVJ_INTENSITY_RANGE = float(os.environ.get("SVJ_INTENSITY_RANGE", "80"))
SVJ_INTENSITY_CAP = float(os.environ.get("SVJ_INTENSITY_CAP", "20.0"))

REGIME_BASE_MAX = float(os.environ.get("REGIME_BASE_MAX", "80.0"))
REGIME_CRISIS_BONUS_MAX = float(os.environ.get("REGIME_CRISIS_BONUS_MAX", "20.0"))

# ── A-LAMS VaR scoring (Guardian) ──

ALAMS_SCORE_VAR_FLOOR = float(os.environ.get("ALAMS_SCORE_VAR_FLOOR", "0.01"))
ALAMS_SCORE_VAR_CEILING = float(os.environ.get("ALAMS_SCORE_VAR_CEILING", "0.08"))
ALAMS_CRISIS_REGIME_BONUS = float(os.environ.get("ALAMS_CRISIS_REGIME_BONUS", "20.0"))
ALAMS_HIGH_DELTA_BONUS = float(os.environ.get("ALAMS_HIGH_DELTA_BONUS", "10.0"))
ALAMS_HIGH_DELTA_THRESHOLD = float(os.environ.get("ALAMS_HIGH_DELTA_THRESHOLD", "0.3"))

CRISIS_REGIME_HAIRCUT = float(os.environ.get("CRISIS_REGIME_HAIRCUT", "0.5"))
NEAR_CRISIS_REGIME_HAIRCUT = float(os.environ.get("NEAR_CRISIS_REGIME_HAIRCUT", "0.75"))

# ── News intelligence ──

NEWSDATA_BASE = os.environ.get("NEWSDATA_BASE", "https://newsdata.io/api/1/crypto")
CC_BASE = os.environ.get("CC_BASE", "https://data-api.cryptocompare.com/news/v1/article/list")
CP_BASE = os.environ.get("CP_BASE", "https://cryptopanic.com/api/developer/v2/posts/")

SOURCE_CREDIBILITY: dict[str, float] = json.loads(
    os.environ.get(
        "SOURCE_CREDIBILITY",
        json.dumps({
            "coindesk": 0.92, "theblock": 0.90, "blockworks": 0.88,
            "decrypt": 0.85, "bitcoinmagazine": 0.84, "cryptoslate": 0.80,
            "cryptobriefing": 0.78, "coinpedia": 0.72, "cryptopotato": 0.70,
            "dailyhodl": 0.68, "newsbtc": 0.65, "utoday": 0.65,
            "bitcoinist": 0.64, "coingape": 0.62, "cryptonomist": 0.60,
            "bitcoin.com": 0.75, "investing_comcryptonews": 0.70,
        }),
    )
)
DEFAULT_CREDIBILITY = float(os.environ.get("DEFAULT_CREDIBILITY", "0.55"))

HALF_LIFE_HOURS = float(os.environ.get("HALF_LIFE_HOURS", "4.0"))
DECAY_LAMBDA = math.log(2) / HALF_LIFE_HOURS

# ── Solana data adapter ──

BIRDEYE_BASE = os.environ.get("BIRDEYE_BASE", "https://public-api.birdeye.so")
DRIFT_DATA_API = os.environ.get("DRIFT_DATA_API", "https://data.api.drift.trade")
RAYDIUM_API = os.environ.get("RAYDIUM_API", "https://api-v3.raydium.io")

SOLANA_HTTP_TIMEOUT = float(os.environ.get("SOLANA_HTTP_TIMEOUT", "30"))
SOLANA_MAX_CONNECTIONS = int(os.environ.get("SOLANA_MAX_CONNECTIONS", "20"))
SOLANA_MAX_KEEPALIVE = int(os.environ.get("SOLANA_MAX_KEEPALIVE", "10"))

# ── Pyth Oracle ──

PYTH_HERMES_URL = os.environ.get("PYTH_HERMES_URL", "https://hermes.pyth.network")
PYTH_BUFFER_DEPTH = int(os.environ.get("PYTH_BUFFER_DEPTH", "100"))
PYTH_SSE_TIMEOUT = float(os.environ.get("PYTH_SSE_TIMEOUT", "60"))
PYTH_FEED_CACHE_TTL = float(os.environ.get("PYTH_FEED_CACHE_TTL", "600"))
PYTH_DEFAULT_WATCHLIST: list[str] = json.loads(
    os.environ.get("PYTH_DEFAULT_WATCHLIST", json.dumps(["SOL", "BTC", "ETH"]))
)

# ── Helius Transaction Streams ──

HELIUS_API_KEY = os.environ.get("HELIUS_API_KEY", "")
HELIUS_WS_URL = os.environ.get("HELIUS_WS_URL", "wss://atlas-mainnet.helius-rpc.com")
HELIUS_ACCOUNTS: list[str] = json.loads(
    os.environ.get("HELIUS_ACCOUNTS", "[]")
)
HELIUS_PING_INTERVAL = int(os.environ.get("HELIUS_PING_INTERVAL", "20"))
HELIUS_EVENT_BUFFER = int(os.environ.get("HELIUS_EVENT_BUFFER", "200"))
LARGE_SWAP_THRESHOLD_USD = float(os.environ.get("LARGE_SWAP_THRESHOLD_USD", "100000"))

# ── Social Feed ──

TWITTER_BEARER_TOKEN = os.environ.get("TWITTER_BEARER_TOKEN", "")
SOCIAL_CACHE_TTL = float(os.environ.get("SOCIAL_CACHE_TTL", "120"))

# ── Macro Data ──

COINGECKO_BASE = os.environ.get("COINGECKO_BASE", "https://api.coingecko.com/api/v3")
FEAR_GREED_URL = os.environ.get("FEAR_GREED_URL", "https://api.alternative.me/fng/")
MACRO_CACHE_TTL = float(os.environ.get("MACRO_CACHE_TTL", "300"))

# ── Kelly Criterion ──

GUARDIAN_KELLY_FRACTION = float(os.environ.get("GUARDIAN_KELLY_FRACTION", "0.25"))
GUARDIAN_KELLY_MIN_TRADES = int(os.environ.get("GUARDIAN_KELLY_MIN_TRADES", "50"))

# ── Portfolio Risk ──

MAX_DAILY_DRAWDOWN = float(os.environ.get("MAX_DAILY_DRAWDOWN", "0.05"))
MAX_WEEKLY_DRAWDOWN = float(os.environ.get("MAX_WEEKLY_DRAWDOWN", "0.10"))
MAX_CORRELATED_EXPOSURE = float(os.environ.get("MAX_CORRELATED_EXPOSURE", "0.15"))
PORTFOLIO_HISTORY_DAYS = int(os.environ.get("PORTFOLIO_HISTORY_DAYS", "30"))

# ── Circuit Breaker ──

CB_THRESHOLD = float(os.environ.get("CB_THRESHOLD", "90"))
CB_CONSECUTIVE_CHECKS = int(os.environ.get("CB_CONSECUTIVE_CHECKS", "3"))
CB_COOLDOWN_SECONDS = float(os.environ.get("CB_COOLDOWN_SECONDS", "300"))
CB_STRATEGIES = json.loads(os.environ.get("CB_STRATEGIES", '["lp", "arb", "perp"]'))

# ── Adversarial Debate ──

DEBATE_MAX_ROUNDS = int(os.environ.get("DEBATE_MAX_ROUNDS", "3"))
DEBATE_TIMEOUT_MS = int(os.environ.get("DEBATE_TIMEOUT_MS", "5000"))
GUARDIAN_AUTO_DEBATE_THRESHOLD = float(os.environ.get("GUARDIAN_AUTO_DEBATE_THRESHOLD", "65.0"))

# ── DexScreener DEX Data (free, keyless REST API) ──

DEXSCREENER_BASE_URL = os.environ.get("DEXSCREENER_BASE_URL", "https://api.dexscreener.com")
DEXSCREENER_TIMEOUT = float(os.environ.get("DEXSCREENER_TIMEOUT", "30"))
DEXSCREENER_MAX_RETRIES = int(os.environ.get("DEXSCREENER_MAX_RETRIES", "3"))
DEXSCREENER_CACHE_TTL = float(os.environ.get("DEXSCREENER_CACHE_TTL", "30"))
DEXSCREENER_MIN_LIQUIDITY_USD = float(os.environ.get("DEXSCREENER_MIN_LIQUIDITY_USD", "5000"))

# ── Jupiter Swap API (Trade Execution) ──

JUPITER_API_URL = os.environ.get("JUPITER_API_URL", "https://api.jup.ag/swap/v1")
JUPITER_API_KEY = os.environ.get("JUPITER_API_KEY", "")
JUPITER_RPC_URL = os.environ.get("JUPITER_RPC_URL", os.environ.get("SOLANA_RPC_URL", "https://api.mainnet-beta.solana.com"))
JUPITER_TIMEOUT = float(os.environ.get("JUPITER_TIMEOUT", "30"))
JUPITER_MAX_RETRIES = int(os.environ.get("JUPITER_MAX_RETRIES", "3"))
JUPITER_SLIPPAGE_BPS = int(os.environ.get("JUPITER_SLIPPAGE_BPS", "100"))

# ── Unified Slippage (per-strategy, basis points) ──

SLIPPAGE_ARBITRAGE_BPS = int(os.environ.get("SLIPPAGE_ARBITRAGE_BPS", str(JUPITER_SLIPPAGE_BPS)))
SLIPPAGE_SPOT_BPS = int(os.environ.get("SLIPPAGE_SPOT_BPS", "100"))
SLIPPAGE_LENDING_BPS = int(os.environ.get("SLIPPAGE_LENDING_BPS", "50"))
SLIPPAGE_LP_DEPOSIT_BPS = int(os.environ.get("SLIPPAGE_LP_DEPOSIT_BPS", "100"))
SLIPPAGE_LP_WITHDRAW_BPS = int(os.environ.get("SLIPPAGE_LP_WITHDRAW_BPS", "50"))

# ── Execution Layer (Wave 9) ──

EXECUTION_ENABLED = os.environ.get("EXECUTION_ENABLED", "false").lower() == "true"
EXECUTION_MAX_SLIPPAGE_BPS = int(os.environ.get("EXECUTION_MAX_SLIPPAGE_BPS", "100"))
EXECUTION_MEV_PROTECTION = os.environ.get("EXECUTION_MEV_PROTECTION", "true").lower() == "true"
SIMULATION_MODE = os.environ.get("SIMULATION_MODE", "true").lower() != "false"
TRADING_MODE = os.environ.get("TRADING_MODE", "NORMAL")

# ── On-Chain Data Pipeline (Wave 10) ──

HELIUS_RPC_URL = os.environ.get(
    "HELIUS_RPC_URL",
    f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}" if HELIUS_API_KEY else "",
)
ONCHAIN_HTTP_TIMEOUT = float(os.environ.get("ONCHAIN_HTTP_TIMEOUT", "30"))
ONCHAIN_MAX_CONNECTIONS = int(os.environ.get("ONCHAIN_MAX_CONNECTIONS", "10"))
ONCHAIN_CACHE_TTL = float(os.environ.get("ONCHAIN_CACHE_TTL", "60"))

# DEX program IDs (Solana mainnet)
RAYDIUM_AMM_V4 = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
RAYDIUM_CLMM = "CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK"
ORCA_WHIRLPOOL = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
METEORA_DLMM = "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo"

# On-chain event thresholds
ONCHAIN_LARGE_SWAP_USD = float(os.environ.get("ONCHAIN_LARGE_SWAP_USD", "50000"))
ONCHAIN_ORACLE_JUMP_PCT = float(os.environ.get("ONCHAIN_ORACLE_JUMP_PCT", "2.0"))
ONCHAIN_FUNDING_SPIKE_BPS = float(os.environ.get("ONCHAIN_FUNDING_SPIKE_BPS", "50"))
ONCHAIN_EVENT_LOOKBACK_SLOTS = int(os.environ.get("ONCHAIN_EVENT_LOOKBACK_SLOTS", "216000"))

# Tick data
TICK_DEFAULT_LOOKBACK_DAYS = int(os.environ.get("TICK_DEFAULT_LOOKBACK_DAYS", "7"))
TICK_MAX_BARS = int(os.environ.get("TICK_MAX_BARS", "10000"))

# ── Engine Selection (pyextremes / fbm) ──

EVT_ENGINE = os.environ.get("EVT_ENGINE", "native")   # "native" or "pyextremes"
FBM_ENGINE = os.environ.get("FBM_ENGINE", "native")   # "native" or "fbm"

# ── Wave 11B: Model Enhancement Engines ──

COPULA_ENGINE = os.environ.get("COPULA_ENGINE", "native")  # "native" or "vine"
PORTFOLIO_OPT_ENGINE = os.environ.get("PORTFOLIO_OPT_ENGINE", "native")  # "native" or "skfolio"
CCXT_DEFAULT_EXCHANGE = os.environ.get("CCXT_DEFAULT_EXCHANGE", "binance")

# ── Metrics ──

METRICS_ENABLED = os.environ.get("METRICS_ENABLED", "true").lower() == "true"

# ── Cache (cashews) ──

REDIS_URL = os.environ.get("REDIS_URL", "")
CACHE_ENABLED = os.environ.get("CACHE_ENABLED", "true").lower() == "true"
CACHE_DEFAULT_TTL = int(os.environ.get("CACHE_DEFAULT_TTL", "300"))

# ── Model Persistence (Redis) ──

PERSISTENCE_ENABLED = os.environ.get("PERSISTENCE_ENABLED", "true").lower() == "true"
PERSISTENCE_REDIS_URL = os.environ.get("PERSISTENCE_REDIS_URL", os.environ.get("REDIS_URL", ""))
PERSISTENCE_KEY_PREFIX = os.environ.get("PERSISTENCE_KEY_PREFIX", "cortex:store:")

# ── Hawkes Engine ──

HAWKES_ENGINE = os.environ.get("HAWKES_ENGINE", "numba")  # "native", "numba", "tick"

# ── Logging ──

LOG_FORMAT = os.environ.get("LOG_FORMAT", "console")  # "json" or "console"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# ── Win-Rate Enhancement Features (Tasks 1-8) ──

# Task 1: Platt Calibration Bridge
CALIBRATION_ENABLED = os.environ.get("CALIBRATION_ENABLED", "false").lower() == "true"
CALIBRATION_PARAMS_PATH = os.environ.get("CALIBRATION_PARAMS_PATH", "")

# Task 2: Regime-Conditional Kelly Criterion
KELLY_REGIME_AWARE = os.environ.get("KELLY_REGIME_AWARE", "false").lower() == "true"
KELLY_REGIME_FRACTIONS: dict[str, float] = json.loads(
    os.environ.get(
        "KELLY_REGIME_FRACTIONS",
        json.dumps({"0": 0.30, "1": 0.25, "2": 0.20, "3": 0.10, "4": 0.05}),
    )
)

# Task 3: Adaptive Guardian Weights (Exponential Gradient)
ADAPTIVE_WEIGHTS_ENABLED = os.environ.get("ADAPTIVE_WEIGHTS_ENABLED", "false").lower() == "true"
ADAPTIVE_WEIGHTS_LR = float(os.environ.get("ADAPTIVE_WEIGHTS_LR", "0.1"))
ADAPTIVE_WEIGHTS_MIN_SAMPLES = int(os.environ.get("ADAPTIVE_WEIGHTS_MIN_SAMPLES", "30"))

# Task 4: Hawkes Intensity-Gated Trade Timing
HAWKES_TIMING_GATE_ENABLED = os.environ.get("HAWKES_TIMING_GATE_ENABLED", "false").lower() == "true"
HAWKES_TIMING_KAPPA = float(os.environ.get("HAWKES_TIMING_KAPPA", "2.0"))

# Task 5: Regime-Conditional Entry Thresholds
REGIME_THRESHOLD_SCALING_ENABLED = os.environ.get("REGIME_THRESHOLD_SCALING_ENABLED", "false").lower() == "true"
REGIME_THRESHOLD_LAMBDA = float(os.environ.get("REGIME_THRESHOLD_LAMBDA", "0.5"))

# Task 6: Multi-Timeframe Regime Confirmation
MULTI_TF_CONFIRMATION_ENABLED = os.environ.get("MULTI_TF_CONFIRMATION_ENABLED", "false").lower() == "true"
MULTI_TF_MIN_AGREEMENT = int(os.environ.get("MULTI_TF_MIN_AGREEMENT", "2"))
MULTI_TF_WINDOWS: list[int] = json.loads(os.environ.get("MULTI_TF_WINDOWS", "[30, 90, 180]"))

# Task 7: Empirical Bayes Debate Prior
DEBATE_EMPIRICAL_PRIOR_ENABLED = os.environ.get("DEBATE_EMPIRICAL_PRIOR_ENABLED", "false").lower() == "true"
DEBATE_PRIOR_MIN = float(os.environ.get("DEBATE_PRIOR_MIN", "0.3"))
DEBATE_PRIOR_MAX = float(os.environ.get("DEBATE_PRIOR_MAX", "0.7"))

# Task 8: Copula Tail-Dependence Risk Gate
COPULA_RISK_GATE_ENABLED = os.environ.get("COPULA_RISK_GATE_ENABLED", "false").lower() == "true"
TAIL_DEPENDENCE_THRESHOLD = float(os.environ.get("TAIL_DEPENDENCE_THRESHOLD", "0.5"))

# Task 9: Agent ONNX Confidence → Guardian Composite
AGENT_CONFIDENCE_ENABLED = os.environ.get("AGENT_CONFIDENCE_ENABLED", "true").lower() == "true"
AGENT_CONFIDENCE_VETO_THRESHOLD = float(os.environ.get("AGENT_CONFIDENCE_VETO_THRESHOLD", "0.3"))

# ── Background News Collector ──
NEWS_COLLECTOR_INTERVAL_SECONDS = int(os.environ.get("NEWS_COLLECTOR_INTERVAL_SECONDS", "30"))
NEWS_BUFFER_MAX_ITEMS = int(os.environ.get("NEWS_BUFFER_MAX_ITEMS", "100"))
NEWS_COLLECTOR_ENABLED = os.environ.get("NEWS_COLLECTOR_ENABLED", "true").lower() == "true"

# ── Recalibration ──
RECALIBRATION_INTERVAL_HOURS = float(os.environ.get("RECALIBRATION_INTERVAL_HOURS", "24"))
RECALIBRATION_ENABLED = os.environ.get("RECALIBRATION_ENABLED", "true").lower() == "true"

# ── Liquidity Snapshot Collector ──
LIQUIDITY_SNAPSHOT_INTERVAL_SECONDS = int(os.environ.get("LIQUIDITY_SNAPSHOT_INTERVAL_SECONDS", "300"))
LIQUIDITY_SNAPSHOT_ENABLED = os.environ.get("LIQUIDITY_SNAPSHOT_ENABLED", "true").lower() == "true"
LIQUIDITY_SNAPSHOT_RETENTION_HOURS = int(os.environ.get("LIQUIDITY_SNAPSHOT_RETENTION_HOURS", "168"))
LIQUIDITY_WATCHLIST: list[str] = json.loads(
    os.environ.get("LIQUIDITY_WATCHLIST", "[]")
)

# ── Model Versioning ──
MODEL_VERSION_HISTORY_SIZE = int(os.environ.get("MODEL_VERSION_HISTORY_SIZE", "3"))

# ── API ──

API_VERSION = os.environ.get("API_VERSION", "1.2.0")


# ── Startup Validation ──

_POSITIVE_FLOATS = [
    ("CACHE_TTL_SECONDS", CACHE_TTL_SECONDS),
    ("DECISION_VALIDITY_SECONDS", DECISION_VALIDITY_SECONDS),
    ("HALF_LIFE_HOURS", HALF_LIFE_HOURS),
    ("SOLANA_HTTP_TIMEOUT", SOLANA_HTTP_TIMEOUT),
    ("DEXSCREENER_TIMEOUT", DEXSCREENER_TIMEOUT),
    ("DEXSCREENER_CACHE_TTL", DEXSCREENER_CACHE_TTL),
    ("JUPITER_TIMEOUT", JUPITER_TIMEOUT),
    ("PYTH_SSE_TIMEOUT", PYTH_SSE_TIMEOUT),
    ("ONCHAIN_HTTP_TIMEOUT", ONCHAIN_HTTP_TIMEOUT),
    ("CB_COOLDOWN_SECONDS", CB_COOLDOWN_SECONDS),
]

_POSITIVE_INTS = [
    ("DEXSCREENER_MAX_RETRIES", DEXSCREENER_MAX_RETRIES),
    ("JUPITER_MAX_RETRIES", JUPITER_MAX_RETRIES),
    ("PYTH_BUFFER_DEPTH", PYTH_BUFFER_DEPTH),
    ("DEBATE_MAX_ROUNDS", DEBATE_MAX_ROUNDS),
    ("CB_CONSECUTIVE_CHECKS", CB_CONSECUTIVE_CHECKS),
    ("NEWS_COLLECTOR_INTERVAL_SECONDS", NEWS_COLLECTOR_INTERVAL_SECONDS),
    ("NEWS_BUFFER_MAX_ITEMS", NEWS_BUFFER_MAX_ITEMS),
    ("MODEL_VERSION_HISTORY_SIZE", MODEL_VERSION_HISTORY_SIZE),
]

_BOUNDED_0_100 = [
    ("CIRCUIT_BREAKER_THRESHOLD", CIRCUIT_BREAKER_THRESHOLD),
    ("APPROVAL_THRESHOLD", APPROVAL_THRESHOLD),
    ("CB_THRESHOLD", CB_THRESHOLD),
    ("SVJ_BASE_CAP", SVJ_BASE_CAP),
    ("REGIME_BASE_MAX", REGIME_BASE_MAX),
]

_VALID_ENGINES = {
    "EVT_ENGINE": (EVT_ENGINE, {"native", "pyextremes"}),
    "FBM_ENGINE": (FBM_ENGINE, {"native", "fbm"}),
    "HAWKES_ENGINE": (HAWKES_ENGINE, {"native", "numba", "tick"}),
    "COPULA_ENGINE": (COPULA_ENGINE, {"native", "vine"}),
    "PORTFOLIO_OPT_ENGINE": (PORTFOLIO_OPT_ENGINE, {"native", "skfolio"}),
    "TRADING_MODE": (TRADING_MODE, {"NORMAL", "PAPER", "LIVE"}),
    "LOG_FORMAT": (LOG_FORMAT, {"json", "console"}),
}


def validate_config() -> list[str]:
    """Validate all config values at startup. Returns list of warnings."""
    warnings: list[str] = []

    for name, val in _POSITIVE_FLOATS:
        if val <= 0:
            warnings.append(f"{name} must be > 0, got {val}")

    for name, val in _POSITIVE_INTS:
        if val <= 0:
            warnings.append(f"{name} must be > 0, got {val}")

    for name, val in _BOUNDED_0_100:
        if not 0 <= val <= 100:
            warnings.append(f"{name} must be in [0, 100], got {val}")

    for name, (val, choices) in _VALID_ENGINES.items():
        if val not in choices:
            warnings.append(f"{name} must be one of {choices}, got '{val}'")

    weight_sum = sum(GUARDIAN_WEIGHTS.values())
    if abs(weight_sum - 1.0) > 0.05:
        warnings.append(f"GUARDIAN_WEIGHTS sum to {weight_sum:.3f}, expected ~1.0")

    if warnings:
        import logging
        _log = logging.getLogger(__name__)
        for w in warnings:
            _log.warning("Config validation: %s", w)

    return warnings

