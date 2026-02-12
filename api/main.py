import logging
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.middleware import RateLimitMiddleware, get_allowed_origins
from api.routes import router
from cortex.config import API_VERSION

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("cortex_risk_engine")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("CortexAgent Risk Engine starting up")
    yield
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

app.include_router(router, prefix="/api/v1", tags=["msm"])

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

