"""Tests for cortex/hot_config.py — Runtime Config Hot-Reload."""
import json
import os
import tempfile

import pytest

import cortex.config as cfg
from cortex.hot_config import (
    HotConfigReloader,
    RELOADABLE_KEYS,
    _coerce,
)


@pytest.fixture
def reloader():
    return HotConfigReloader(poll_interval=0.1)


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary JSON config file."""
    path = tmp_path / "hot_config.json"
    path.write_text("{}")
    return path


def test_apply_reloadable_key(reloader):
    original = cfg.APPROVAL_THRESHOLD
    try:
        applied = reloader.apply({"APPROVAL_THRESHOLD": 80.0})
        assert "APPROVAL_THRESHOLD" in applied
        assert cfg.APPROVAL_THRESHOLD == 80.0
    finally:
        cfg.APPROVAL_THRESHOLD = original


def test_apply_ignores_non_reloadable(reloader):
    applied = reloader.apply({"REDIS_URL": "redis://evil:6379"})
    assert len(applied) == 0


def test_apply_ignores_unknown_key(reloader):
    applied = reloader.apply({"TOTALLY_FAKE_KEY_XYZ": 42})
    assert len(applied) == 0


def test_apply_no_change_when_same_value(reloader):
    original = cfg.CB_THRESHOLD
    try:
        applied = reloader.apply({"CB_THRESHOLD": original})
        assert len(applied) == 0
    finally:
        cfg.CB_THRESHOLD = original


def test_apply_bool_coercion(reloader):
    original = cfg.HEARTBEAT_ENABLED
    try:
        applied = reloader.apply({"HEARTBEAT_ENABLED": "false"})
        assert cfg.HEARTBEAT_ENABLED is False
        assert "HEARTBEAT_ENABLED" in applied
    finally:
        cfg.HEARTBEAT_ENABLED = original


def test_apply_int_coercion(reloader):
    original = cfg.JUPITER_SLIPPAGE_BPS
    try:
        applied = reloader.apply({"JUPITER_SLIPPAGE_BPS": "200"})
        assert cfg.JUPITER_SLIPPAGE_BPS == 200
        assert isinstance(cfg.JUPITER_SLIPPAGE_BPS, int)
    finally:
        cfg.JUPITER_SLIPPAGE_BPS = original


def test_apply_float_coercion(reloader):
    original = cfg.MAX_DAILY_DRAWDOWN
    try:
        applied = reloader.apply({"MAX_DAILY_DRAWDOWN": "0.08"})
        assert cfg.MAX_DAILY_DRAWDOWN == pytest.approx(0.08)
    finally:
        cfg.MAX_DAILY_DRAWDOWN = original


def test_apply_multiple_keys(reloader):
    originals = {
        "CB_THRESHOLD": cfg.CB_THRESHOLD,
        "CB_COOLDOWN_SECONDS": cfg.CB_COOLDOWN_SECONDS,
    }
    try:
        applied = reloader.apply({
            "CB_THRESHOLD": 85.0,
            "CB_COOLDOWN_SECONDS": 600.0,
        })
        assert len(applied) == 2
        assert cfg.CB_THRESHOLD == 85.0
        assert cfg.CB_COOLDOWN_SECONDS == 600.0
    finally:
        for k, v in originals.items():
            setattr(cfg, k, v)


def test_get_status(reloader):
    reloader.apply({"APPROVAL_THRESHOLD": 77.0})
    status = reloader.get_status()
    assert "enabled" in status
    assert "total_changes" in status
    assert status["total_changes"] >= 1
    assert "active_overrides" in status
    # Restore
    cfg.APPROVAL_THRESHOLD = float(os.environ.get("APPROVAL_THRESHOLD", "75.0"))


def test_get_active_overrides(reloader):
    original = cfg.APPROVAL_THRESHOLD
    try:
        reloader.apply({"APPROVAL_THRESHOLD": 66.0})
        overrides = reloader.get_active_overrides()
        assert overrides["APPROVAL_THRESHOLD"] == 66.0
    finally:
        cfg.APPROVAL_THRESHOLD = original


def test_change_count_increments(reloader):
    original = cfg.APPROVAL_THRESHOLD
    try:
        assert reloader._change_count == 0
        reloader.apply({"APPROVAL_THRESHOLD": 60.0})
        assert reloader._change_count == 1
        reloader.apply({"APPROVAL_THRESHOLD": 65.0})
        assert reloader._change_count == 2
    finally:
        cfg.APPROVAL_THRESHOLD = original


def test_read_config_file(config_file):
    reloader = HotConfigReloader(config_path=str(config_file))
    config_file.write_text(json.dumps({"APPROVAL_THRESHOLD": 82.0}))
    data = reloader._read_config_file()
    assert data is not None
    assert data["APPROVAL_THRESHOLD"] == 82.0


def test_read_config_file_no_change_on_same_mtime(config_file):
    reloader = HotConfigReloader(config_path=str(config_file))
    config_file.write_text(json.dumps({"APPROVAL_THRESHOLD": 82.0}))
    data1 = reloader._read_config_file()
    assert data1 is not None
    # Second read without file change returns None (same mtime)
    data2 = reloader._read_config_file()
    assert data2 is None


def test_read_config_file_missing():
    reloader = HotConfigReloader(config_path="/tmp/nonexistent_hot_config_xyz.json")
    data = reloader._read_config_file()
    assert data is None


def test_read_config_file_invalid_json(config_file):
    reloader = HotConfigReloader(config_path=str(config_file))
    config_file.write_text("not valid json {{{")
    data = reloader._read_config_file()
    assert data is None


def test_read_config_file_not_object(config_file):
    reloader = HotConfigReloader(config_path=str(config_file))
    config_file.write_text(json.dumps([1, 2, 3]))
    data = reloader._read_config_file()
    assert data is None


def test_coerce_bool_from_string():
    assert _coerce("HEARTBEAT_ENABLED", "true") is True
    assert _coerce("HEARTBEAT_ENABLED", "false") is False
    assert _coerce("HEARTBEAT_ENABLED", "1") is True


def test_coerce_int():
    assert _coerce("JUPITER_SLIPPAGE_BPS", 200) == 200
    assert isinstance(_coerce("JUPITER_SLIPPAGE_BPS", 200), int)


def test_coerce_float():
    assert _coerce("MAX_DAILY_DRAWDOWN", "0.1") == pytest.approx(0.1)


def test_reloadable_keys_all_exist():
    """Every key in RELOADABLE_KEYS must exist in cortex.config."""
    for key in RELOADABLE_KEYS:
        assert hasattr(cfg, key), f"{key} not found in cortex.config"


def test_file_poll_applies_changes(config_file):
    """Integration: write config file, call _read + apply manually."""
    original = cfg.APPROVAL_THRESHOLD
    try:
        reloader = HotConfigReloader(config_path=str(config_file))
        config_file.write_text(json.dumps({
            "APPROVAL_THRESHOLD": 91.0,
            "REDIS_URL": "redis://should-be-ignored",
        }))
        data = reloader._read_config_file()
        assert data is not None
        applied = reloader.apply(data)
        assert "APPROVAL_THRESHOLD" in applied
        assert cfg.APPROVAL_THRESHOLD == 91.0
        assert "REDIS_URL" not in applied
    finally:
        cfg.APPROVAL_THRESHOLD = original


def test_reset_restores_env_value(reloader):
    """Reset should restore values from environment variables."""
    env_key = "APPROVAL_THRESHOLD"
    original = cfg.APPROVAL_THRESHOLD
    os.environ[env_key] = "75.0"
    try:
        reloader.apply({env_key: 99.0})
        assert cfg.APPROVAL_THRESHOLD == 99.0
        reset_result = reloader.reset()
        assert env_key in reset_result
        assert cfg.APPROVAL_THRESHOLD == 75.0
        assert len(reloader.get_active_overrides()) == 0
    finally:
        cfg.APPROVAL_THRESHOLD = original
        os.environ.pop(env_key, None)
