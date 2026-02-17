"""Model versioning endpoints: list versions, rollback."""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.stores import _model_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])


@router.get("/models/versions")
async def list_all_versions():
    """List available model versions for all calibrated tokens."""
    versions = await _model_store.get_all_versions()
    result: dict[str, list[dict]] = {}
    for token, meta in versions.items():
        result[token] = [
            {
                "version": entry["version"],
                "timestamp": entry["timestamp"],
                "calibrated_at": entry.get("calibrated_at", ""),
            }
            for entry in meta
        ]
    return {"tokens": result}


@router.get("/models/versions/{token}")
async def list_token_versions(token: str):
    """List available model versions for a specific token."""
    if token not in _model_store:
        raise HTTPException(404, f"No calibrated model for '{token}'")
    versions = await _model_store.get_versions(token)
    return {
        "token": token,
        "versions": [
            {
                "version": entry["version"],
                "timestamp": entry["timestamp"],
                "calibrated_at": entry.get("calibrated_at", ""),
            }
            for entry in versions
        ],
        "current_calibrated_at": (
            _model_store[token]["calibrated_at"].isoformat()
            if hasattr(_model_store[token].get("calibrated_at"), "isoformat")
            else str(_model_store[token].get("calibrated_at", ""))
        ),
    }


@router.post("/models/rollback/{token}")
async def rollback_model(token: str, version: int = Query(..., description="Version number to rollback to")):
    """Rollback a token's model to a specific version."""
    if token not in _model_store:
        raise HTTPException(404, f"No calibrated model for '{token}'")

    versions = await _model_store.get_versions(token)
    valid_versions = [v["version"] for v in versions]
    if version not in valid_versions:
        raise HTTPException(
            400,
            f"Version {version} not available for '{token}'. Available: {valid_versions}",
        )

    success = await _model_store.restore_version(token, version)
    if not success:
        raise HTTPException(500, f"Failed to restore version {version} for '{token}'")

    return {
        "token": token,
        "restored_version": version,
        "restored_at": datetime.now(timezone.utc).isoformat(),
    }
