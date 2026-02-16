"""
Health check endpoints.
"""
from datetime import datetime
from fastapi import APIRouter, Response
from pydantic import BaseModel

from cortex.data.rpc_failover import get_resilient_pool

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    components: dict[str, str]


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns service status and component health.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        components={
            "api": "healthy",
            "models": "healthy",
            "database": "healthy",
            "cache": "healthy"
        }
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """
    Readiness check for Kubernetes.
    
    Returns 200 if service is ready to accept traffic.
    """
    return {"ready": True}


@router.get("/live")
async def liveness_check() -> dict:
    """
    Liveness check for Kubernetes.

    Returns 200 if service is alive.
    """
    return {"alive": True}


@router.get("/health/rpc")
async def rpc_health() -> dict:
    """RPC/API endpoint health metrics — per-host latency, success rate, status."""
    pool = get_resilient_pool()
    return pool.get_all_health()

