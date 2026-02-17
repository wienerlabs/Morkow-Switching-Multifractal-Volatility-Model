"""Platt Calibration Bridge — maps raw model confidence to calibrated probabilities.

Loads pre-fitted Platt parameters (A, B) and applies:
    P_cal = sigmoid(A * P_raw + B)

This is a lightweight runtime module — training happens in agent/training/calibration.py.
"""
from __future__ import annotations

import json
import logging
import math
from typing import Any

from cortex.config import CALIBRATION_ENABLED, CALIBRATION_PARAMS_PATH

logger = logging.getLogger(__name__)

_platt_params: dict[str, Any] | None = None
_loaded = False


def _load_params() -> dict[str, Any] | None:
    """Load Platt parameters from JSON file (once)."""
    global _platt_params, _loaded
    if _loaded:
        return _platt_params
    _loaded = True

    if not CALIBRATION_PARAMS_PATH:
        logger.debug("CALIBRATION_PARAMS_PATH not set — calibration passthrough")
        return None

    try:
        with open(CALIBRATION_PARAMS_PATH) as f:
            _platt_params = json.load(f)
        logger.info(
            "Loaded Platt calibration: A=%.4f B=%.4f (model=%s)",
            _platt_params["A"], _platt_params["B"], _platt_params.get("model_name", "?"),
        )
    except Exception as exc:
        logger.warning("Failed to load calibration params: %s", exc)
        _platt_params = None
    return _platt_params


def calibrate_probability(raw_prob: float) -> float:
    """Apply Platt scaling to a raw probability.

    Returns calibrated probability if enabled and params loaded,
    otherwise returns the raw probability unchanged.
    """
    if not CALIBRATION_ENABLED:
        return raw_prob

    params = _load_params()
    if params is None:
        return raw_prob

    A = params["A"]
    B = params["B"]
    logit = A * raw_prob + B
    logit = max(-10.0, min(10.0, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def get_calibration_info() -> dict[str, Any]:
    """Return current calibration state for diagnostics."""
    params = _load_params() if CALIBRATION_ENABLED else None
    return {
        "enabled": CALIBRATION_ENABLED,
        "loaded": params is not None,
        "params_path": CALIBRATION_PARAMS_PATH or None,
        "A": params["A"] if params else None,
        "B": params["B"] if params else None,
        "model_name": params.get("model_name") if params else None,
    }


def reset() -> None:
    """Reset loaded state (for testing)."""
    global _platt_params, _loaded
    _platt_params = None
    _loaded = False
