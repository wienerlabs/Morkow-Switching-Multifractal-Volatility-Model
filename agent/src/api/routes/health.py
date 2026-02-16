"""
Health check endpoints — real system checks, not cosmetic.
"""
from datetime import datetime
from fastapi import APIRouter, Response
from pydantic import BaseModel

from cortex.data.rpc_failover import get_resilient_pool

router = APIRouter()

_started_at = datetime.utcnow()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    components: dict[str, str]
    uptime_seconds: float


@router.get("/health", response_model=HealthResponse)
async def health_check(response: Response) -> HealthResponse:
    """
    Health check endpoint — returns real component status.

    Returns 503 if critical components (RPC) are down.
    """
    pool = get_resilient_pool()
    rpc_health = pool.get_all_health()
    rpc_status = rpc_health.get("status", "unknown")

    # Derive component statuses from real state
    rpc_ok = rpc_status in ("healthy", "degraded")

    # Check Redis connectivity
    redis_status = "healthy"
    try:
        from cortex.data.rpc_failover import _redis_client
        if _redis_client is not None:
            await _redis_client.ping()
        else:
            redis_status = "unavailable"
    except Exception:
        redis_status = "down"

    # Overall status
    if not rpc_ok:
        overall = "unhealthy"
        response.status_code = 503
    elif rpc_status == "degraded" or redis_status != "healthy":
        overall = "degraded"
    else:
        overall = "healthy"

    uptime = (datetime.utcnow() - _started_at).total_seconds()

    return HealthResponse(
        status=overall,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0",
        components={
            "rpc": rpc_status,
            "redis": redis_status,
            "api": "healthy",
        },
        uptime_seconds=round(uptime, 1),
    )


@router.get("/ready")
async def readiness_check(response: Response) -> dict:
    """
    Readiness check for Kubernetes.

    Returns 503 if RPC pool has all endpoints down.
    """
    pool = get_resilient_pool()
    rpc_health = pool.get_all_health()
    all_down = rpc_health.get("status") == "down"

    if all_down:
        response.status_code = 503
        return {"ready": False, "reason": "all RPC endpoints down"}
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
