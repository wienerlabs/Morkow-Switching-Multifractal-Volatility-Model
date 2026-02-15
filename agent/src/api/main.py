from __future__ import annotations
"""
FastAPI Application for Cortex DeFi Agent.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .routes import predictions, execution, health, solana, risk
from ..config import settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Cortex DeFi Agent API")
    
    # Initialize components
    # Models, database connections, etc. would be initialized here
    
    yield
    
    # Shutdown
    logger.info("Shutting down Cortex DeFi Agent API")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Cortex DeFi Agent",
        description="XGBoost-based DeFi trading agent with explainable predictions",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(execution.router, prefix="/api/v1", tags=["Execution"])
    app.include_router(solana.router, prefix="/api/v1", tags=["Solana"])
    app.include_router(risk.router, prefix="/api/v1", tags=["Risk"])

    return app


app = create_app()

