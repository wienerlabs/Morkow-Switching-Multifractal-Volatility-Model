"""Tests for cortex.config.validate_config()."""

from unittest.mock import patch

import pytest


def test_validate_config_defaults_pass():
    """Default config values should produce no warnings."""
    from cortex.config import validate_config

    warnings = validate_config()
    assert warnings == [], f"Unexpected warnings with defaults: {warnings}"


def test_validate_config_negative_float():
    """Negative float config values should be flagged."""
    with patch("cortex.config.CACHE_TTL_SECONDS", -1.0):
        from cortex.config import validate_config, _POSITIVE_FLOATS

        original = _POSITIVE_FLOATS[0]
        _POSITIVE_FLOATS[0] = ("CACHE_TTL_SECONDS", -1.0)
        try:
            warnings = validate_config()
            assert any("CACHE_TTL_SECONDS" in w for w in warnings)
        finally:
            _POSITIVE_FLOATS[0] = original


def test_validate_config_negative_int():
    """Negative int config values should be flagged."""
    from cortex.config import _POSITIVE_INTS, validate_config

    original = _POSITIVE_INTS[0]
    _POSITIVE_INTS[0] = ("DEXSCREENER_MAX_RETRIES", -5)
    try:
        warnings = validate_config()
        assert any("DEXSCREENER_MAX_RETRIES" in w for w in warnings)
    finally:
        _POSITIVE_INTS[0] = original


def test_validate_config_bounded_out_of_range():
    """Values outside [0, 100] should be flagged."""
    from cortex.config import _BOUNDED_0_100, validate_config

    original = _BOUNDED_0_100[0]
    _BOUNDED_0_100[0] = ("CIRCUIT_BREAKER_THRESHOLD", 150.0)
    try:
        warnings = validate_config()
        assert any("CIRCUIT_BREAKER_THRESHOLD" in w for w in warnings)
    finally:
        _BOUNDED_0_100[0] = original


def test_validate_config_invalid_engine():
    """Invalid engine choice should be flagged."""
    from cortex.config import _VALID_ENGINES, validate_config

    original = _VALID_ENGINES["TRADING_MODE"]
    _VALID_ENGINES["TRADING_MODE"] = ("INVALID_MODE", {"NORMAL", "PAPER", "LIVE"})
    try:
        warnings = validate_config()
        assert any("TRADING_MODE" in w for w in warnings)
    finally:
        _VALID_ENGINES["TRADING_MODE"] = original


def test_validate_config_guardian_weights_sum():
    """Guardian weights that don't sum to ~1.0 should be flagged."""
    from cortex.config import validate_config

    with patch("cortex.config.GUARDIAN_WEIGHTS", {"a": 0.1, "b": 0.1}):
        warnings = validate_config()
        assert any("GUARDIAN_WEIGHTS" in w for w in warnings)
