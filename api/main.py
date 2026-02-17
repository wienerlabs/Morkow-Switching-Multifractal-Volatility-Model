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


app = FastAPI(
    title="CortexAgent Risk Engine",
    description="Multi-model volatility and risk management API — MSM regime detection, EVT, SVJ, Hawkes, copula VaR, rough volatility, and Guardian risk veto for autonomous DeFi agents on Solana.",
    version=API_VERSION,
    lifespan=lifespan,
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


@app.get("/health")
def health():
    return {"status": "ok", "service": "cortex-risk-engine", "version": API_VERSION}


@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(_frontend_dir / "index.html", media_type="text/html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

