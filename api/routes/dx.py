"""DX-Research API routes — expose DX module state for OpenServ agent.

Endpoints:
  GET  /dx/stigmergy            — full board snapshot
  GET  /dx/stigmergy/{token}    — single token consensus
  GET  /dx/cascade/{token}      — Ising cascade risk
  GET  /dx/overrides            — active human overrides
  POST /dx/overrides            — create a new override
  DELETE /dx/overrides/{id}     — revoke an override
  GET  /dx/vault/{vault_id}     — vault delta features
  GET  /dx/status               — all DX module status
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(prefix="/dx", tags=["dx-research"])


# ── Response Models ──────────────────────────────────────────────────────

class StigmergyConsensusResponse(BaseModel):
    token: str
    direction: str
    conviction: float
    num_sources: int
    swarm_active: bool
    bullish_weight: float
    bearish_weight: float


class StigmergyBoardResponse(BaseModel):
    enabled: bool
    total_tokens: int = 0
    tokens: dict[str, StigmergyConsensusResponse] = {}


class CascadeResponse(BaseModel):
    enabled: bool
    token: str = ""
    cascade_risk: str = "low"
    cascade_score: float = 0.0
    magnetization: float = 0.0
    effective_temperature: float = 1.0
    susceptibility: float = 0.0
    herding_direction: str = "neutral"


class OverrideCreateRequest(BaseModel):
    action: str = Field(..., description="force_approve, force_reject, size_cap, or cooldown")
    token: str = Field("*", description="Token symbol or * for global")
    reason: str = Field("", description="Why this override")
    created_by: str = Field("operator", description="Who created it")
    ttl: float | None = Field(None, description="TTL in seconds (default: 1 hour)")
    size_cap_usd: float | None = Field(None, description="Size cap for size_cap action")


class OverrideResponse(BaseModel):
    id: str
    action: str
    token: str
    reason: str
    created_by: str
    created_at: float
    expires_at: float
    ttl_remaining: float
    size_cap_usd: float | None = None
    applied_count: int = 0


class OverrideListResponse(BaseModel):
    enabled: bool
    overrides: list[dict] = []
    audit_log: list[dict] = []


class VaultDeltaResponse(BaseModel):
    enabled: bool
    component: str = "vault_delta"
    score: float = 0.0
    details: dict = {}


class DXStatusResponse(BaseModel):
    stigmergy_enabled: bool
    ising_cascade_enabled: bool
    vault_delta_enabled: bool
    human_override_enabled: bool
    persona_diversity_enabled: bool


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get("/stigmergy", response_model=StigmergyBoardResponse)
def get_stigmergy_board():
    from cortex.config import STIGMERGY_ENABLED
    if not STIGMERGY_ENABLED:
        return StigmergyBoardResponse(enabled=False)

    from cortex.stigmergy import get_board_snapshot
    snap = get_board_snapshot()

    tokens = {}
    for tk, data in snap.get("tokens", {}).items():
        tokens[tk] = StigmergyConsensusResponse(
            token=tk,
            direction=data["direction"],
            conviction=data["conviction"],
            num_sources=data["num_sources"],
            swarm_active=data["swarm_active"],
            bullish_weight=data["bullish_weight"],
            bearish_weight=data["bearish_weight"],
        )

    return StigmergyBoardResponse(
        enabled=True,
        total_tokens=snap["total_tokens"],
        tokens=tokens,
    )


@router.get("/stigmergy/{token}", response_model=StigmergyConsensusResponse)
def get_stigmergy_token(token: str):
    from cortex.config import STIGMERGY_ENABLED
    if not STIGMERGY_ENABLED:
        raise HTTPException(503, "Stigmergy is disabled.")

    from cortex.stigmergy import get_consensus
    c = get_consensus(token)
    return StigmergyConsensusResponse(
        token=token,
        direction=c.direction,
        conviction=c.conviction,
        num_sources=c.num_sources,
        swarm_active=c.swarm_active,
        bullish_weight=c.bullish_weight,
        bearish_weight=c.bearish_weight,
    )


@router.get("/cascade/{token}", response_model=CascadeResponse)
def get_cascade_risk(token: str):
    from cortex.config import ISING_CASCADE_ENABLED
    if not ISING_CASCADE_ENABLED:
        return CascadeResponse(enabled=False, token=token)

    from cortex.ising_cascade import get_cascade_score
    result = get_cascade_score(token)
    return CascadeResponse(
        enabled=True,
        token=token,
        cascade_risk=result.cascade_risk,
        cascade_score=result.cascade_score,
        magnetization=result.magnetization,
        effective_temperature=result.effective_temperature,
        susceptibility=result.susceptibility,
        herding_direction=result.herding_direction,
    )


@router.get("/overrides", response_model=OverrideListResponse)
def get_overrides():
    from cortex.config import HUMAN_OVERRIDE_ENABLED
    if not HUMAN_OVERRIDE_ENABLED:
        return OverrideListResponse(enabled=False)

    from cortex.human_override import list_active_overrides, get_registry
    return OverrideListResponse(
        enabled=True,
        overrides=list_active_overrides(),
        audit_log=get_registry().get_audit_log(n=20),
    )


@router.post("/overrides", response_model=dict)
def create_override_endpoint(req: OverrideCreateRequest):
    from cortex.config import HUMAN_OVERRIDE_ENABLED
    if not HUMAN_OVERRIDE_ENABLED:
        raise HTTPException(503, "Human override is disabled.")

    from cortex.human_override import create_override
    result = create_override(
        action=req.action,
        token=req.token,
        reason=req.reason,
        created_by=req.created_by,
        ttl=req.ttl,
        size_cap_usd=req.size_cap_usd,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.delete("/overrides/{override_id}")
def revoke_override_endpoint(override_id: str, revoked_by: str = "operator"):
    from cortex.config import HUMAN_OVERRIDE_ENABLED
    if not HUMAN_OVERRIDE_ENABLED:
        raise HTTPException(503, "Human override is disabled.")

    from cortex.human_override import revoke_override
    success = revoke_override(override_id, revoked_by)
    if not success:
        raise HTTPException(404, f"Override {override_id} not found or already revoked.")
    return {"revoked": True, "id": override_id}


@router.get("/vault/{vault_id}", response_model=VaultDeltaResponse)
def get_vault_delta(vault_id: str):
    from cortex.config import VAULT_DELTA_ENABLED
    if not VAULT_DELTA_ENABLED:
        return VaultDeltaResponse(enabled=False)

    from cortex.vault_delta import get_vault_features
    feat = get_vault_features(vault_id)
    return VaultDeltaResponse(
        enabled=True,
        component=feat["component"],
        score=feat["score"],
        details=feat["details"],
    )


@router.get("/status", response_model=DXStatusResponse)
def dx_status():
    from cortex.config import (
        STIGMERGY_ENABLED,
        ISING_CASCADE_ENABLED,
        VAULT_DELTA_ENABLED,
        HUMAN_OVERRIDE_ENABLED,
        PERSONA_DIVERSITY_ENABLED,
    )
    return DXStatusResponse(
        stigmergy_enabled=STIGMERGY_ENABLED,
        ising_cascade_enabled=ISING_CASCADE_ENABLED,
        vault_delta_enabled=VAULT_DELTA_ENABLED,
        human_override_enabled=HUMAN_OVERRIDE_ENABLED,
        persona_diversity_enabled=PERSONA_DIVERSITY_ENABLED,
    )
