from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import structlog

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cashews.contrib.fastapi import CacheEtagMiddleware, CacheDeleteMiddleware

from api.middleware import RateLimitMiddleware, RequestIDMiddleware, get_allowed_origins
from api.routes import router
from api.stores import ALL_STORES
from cortex.cache import cache, setup_cache
from cortex.config import API_VERSION, METRICS_ENABLED
from cortex.logging import setup_logging
from cortex.persistence import close_persistence, init_persistence
from cortex.data.rpc_failover import init_rpc_health_redis, close_rpc_health_redis

setup_logging()
logger = structlog.get_logger("cortex_risk_engine")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CortexAgent Risk Engine starting up")
    await setup_cache()

    await init_persistence()
    await init_rpc_health_redis()
    total = 0
    for store in ALL_STORES:
        total += await store.restore()
    if total:
        logger.info("Restored %d model state(s) from Redis", total)

    # Restore singleton state from Redis snapshots
    from cortex.circuit_breaker import restore_cb_state
    from cortex.guardian import restore_kelly_state
    from cortex.debate import restore_debate_state
    restore_cb_state()
    restore_kelly_state()
    restore_debate_state()

    # Start periodic recalibration scheduler
    from api.tasks import start_recalibration_scheduler, stop_recalibration_scheduler
    start_recalibration_scheduler()

    # Start background news collector
    from api.tasks import start_news_collector, stop_news_collector
    start_news_collector()

    # Start liquidity snapshot collector
    from api.tasks import start_liquidity_snapshot_collector, stop_liquidity_snapshot_collector
    start_liquidity_snapshot_collector()

    yield

    stop_liquidity_snapshot_collector()
    stop_news_collector()
    stop_recalibration_scheduler()

    # Persist singleton state before shutdown
    from cortex.circuit_breaker import persist_cb_state
    from cortex.guardian import persist_kelly_state
    from cortex.debate import persist_debate_state
    persist_cb_state()
    persist_kelly_state()
    persist_debate_state()

    persisted = 0
    for store in ALL_STORES:
        persisted += await store.persist_all()
    if persisted:
        logger.info("Persisted %d model state(s) to Redis", persisted)

    await close_rpc_health_redis()
    await close_persistence()
    await cache.close()
    logger.info("CortexAgent Risk Engine shutting down")


OPENAPI_TAGS = [
    {"name": "guardian", "description": "Unified risk veto — trade assessment, Kelly sizing, circuit breakers, adversarial debate"},
    {"name": "msm", "description": "Markov-Switching Multifractal — calibration, regime detection, VaR, volatility forecasts"},
    {"name": "evt", "description": "Extreme Value Theory — GPD tail fitting, tail VaR/CVaR, backtesting"},
    {"name": "hawkes", "description": "Hawkes process — event clustering, contagion intensity, flash-crash risk"},
    {"name": "svj", "description": "Stochastic Volatility with Jumps — jump detection, decomposition, SVJ-VaR"},
    {"name": "rough", "description": "Rough volatility — rBergomi/rHeston calibration, Hurst exponent, forecasting"},
    {"name": "fractal", "description": "Multifractal analysis — Hurst exponent, DFA, spectrum width"},
    {"name": "lvar", "description": "Liquidity-adjusted VaR — spread estimation, market impact, regime-conditional liquidity"},
    {"name": "portfolio", "description": "Portfolio VaR — multi-asset calibration, copula fitting, stress testing"},
    {"name": "portfolio-risk", "description": "Portfolio risk management — positions, drawdown limits, correlation exposure"},
    {"name": "portfolio-optimization", "description": "Portfolio optimization — min-CVaR, max-Sharpe, risk parity via skfolio"},
    {"name": "comparison", "description": "Model comparison — MSM vs GARCH vs EGARCH benchmark metrics"},
    {"name": "regime", "description": "Regime analytics — durations, history, transition alerts, per-regime statistics"},
    {"name": "news", "description": "News intelligence — sentiment feed, market signal, background buffer"},
    {"name": "streams", "description": "Real-time streams — Helius transaction events, Guardian SSE alerts"},
    {"name": "oracle", "description": "Oracle price feeds — Pyth price data, historical lookups, streaming"},
    {"name": "dex", "description": "DexScreener — token prices, pair liquidity, new token discovery"},
    {"name": "onchain", "description": "On-chain liquidity — CLMM depth, realized spreads, on-chain LVaR"},
    {"name": "Tick Data & Backtesting", "description": "Tick-level data — custom bar types, intraday VaR backtesting"},
    {"name": "hawkes-onchain", "description": "On-chain Hawkes — multivariate event clustering, cross-excitation, flash-crash scoring"},
    {"name": "token", "description": "Token info — metadata, price, market cap via DexScreener/Birdeye"},
    {"name": "vine-copula", "description": "Vine copula — R-vine/C-vine/D-vine fitting, tail dependence, simulation"},
    {"name": "ccxt", "description": "CCXT data feeds — OHLCV candles, order books, tickers from CEXs"},
    {"name": "social", "description": "Social sentiment — aggregated sentiment from Twitter, Reddit, Telegram"},
    {"name": "macro", "description": "Macro indicators — Fear & Greed index, BTC dominance, market overview"},
    {"name": "execution", "description": "Trade execution — preflight checks, swap execution, execution log"},
    {"name": "calibration-tasks", "description": "Async calibration — task status polling for long-running calibrations"},
    {"name": "models", "description": "Model versioning — list versions, rollback to previous calibrations"},
    {"name": "narrator", "description": "LLM narrator — trade explanations, news interpretation, market briefings, Q&A"},
    {"name": "agents", "description": "Agent status and performance — live signals, accuracy heatmaps"},
    {"name": "strategies", "description": "Strategy configuration — list, toggle, circuit breaker state"},
    {"name": "system", "description": "System health — data source availability, latency, connection status"},
]

app = FastAPI(
    title="CortexAgent Risk Engine",
    description=(
        "Multi-model volatility and risk management API for autonomous DeFi agents on Solana.\n\n"
        "## Core Models\n"
        "- **MSM** — Markov-Switching Multifractal regime detection with 5-state Hamilton filter\n"
        "- **EVT** — Extreme Value Theory tail risk via Generalized Pareto Distribution\n"
        "- **SVJ** — Stochastic Volatility with Jumps (Bates model + Hawkes clustering)\n"
        "- **Hawkes** — Self-exciting point process for flash-crash contagion\n"
        "- **Rough Vol** — Rough Bergomi / Rough Heston with fractional kernels\n"
        "- **Copula** — Vine copula portfolio dependence modeling\n\n"
        "## Risk Engine\n"
        "- **Guardian** — Unified risk veto combining all models into a composite score 0–100\n"
        "- **Adversarial Debate** — Multi-agent debate system for trade approval\n"
        "- **Circuit Breakers** — Strategy-specific outcome-based circuit breakers\n"
        "- **Kelly Criterion** — Adaptive position sizing from trade history\n\n"
        "## Authentication\n"
        "All endpoints require `X-API-Key` header. Configure via `CORTEX_API_KEYS` env var."
    ),
    version=API_VERSION,
    lifespan=lifespan,
    openapi_tags=OPENAPI_TAGS,
)

allowed_origins = get_allowed_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-API-Key", "Content-Type", "Authorization"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "Retry-After"],
)

app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(CacheEtagMiddleware)
app.add_middleware(CacheDeleteMiddleware)

app.include_router(router, prefix="/api/v1", tags=["msm"])

if METRICS_ENABLED:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)

_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
_ui_dir = _frontend_dir / "ui"


@app.get("/health")
def health():
    return {"status": "ok", "service": "cortex-risk-engine", "version": API_VERSION}


@app.get("/", include_in_schema=False)
def serve_api_explorer():
    return FileResponse(_frontend_dir / "index.html", media_type="text/html")


@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
def serve_ui_dashboard():
    return FileResponse(_ui_dir / "index.html", media_type="text/html")


@app.get("/ui/{page}.html", include_in_schema=False)
def serve_ui_page(page: str):
    target = _ui_dir / f"{page}.html"
    if target.is_file():
        return FileResponse(target, media_type="text/html")
    return FileResponse(_ui_dir / "index.html", media_type="text/html")


app.mount("/ui", StaticFiles(directory=str(_ui_dir)), name="ui-static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

