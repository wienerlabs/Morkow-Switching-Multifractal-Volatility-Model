"""Centralized configuration — single source of truth for all tuneable constants.

Every value is backed by an environment variable with a sensible default so the
system works out-of-the-box while remaining fully configurable in production.
"""

import json
import math
import os

# ── Guardian weights & thresholds ──

GUARDIAN_WEIGHTS: dict[str, float] = json.loads(
    os.environ.get(
        "GUARDIAN_WEIGHTS",
        json.dumps({"evt": 0.25, "svj": 0.20, "hawkes": 0.20, "regime": 0.20, "news": 0.15}),
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

# ── API ──

API_VERSION = os.environ.get("API_VERSION", "1.1.0")

