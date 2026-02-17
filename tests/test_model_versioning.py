"""Tests for model versioning — VersionedPersistentStore and /models/versions endpoints."""

import asyncio
import json
import pickle  # noqa: S403 — testing our own persistence layer (trusted internal data only)
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from cortex import persistence
from cortex.persistence import VersionedPersistentStore


class TestVersionedStoreDictCompat:
    """VersionedPersistentStore must behave like a dict when Redis is off."""

    def setup_method(self):
        persistence._redis_available = False
        persistence._redis_client = None
        self.store = VersionedPersistentStore("test_versioned", max_versions=3)

    def test_setitem_getitem(self):
        self.store["SOL"] = {"sigma": 0.05}
        assert self.store["SOL"]["sigma"] == 0.05

    def test_contains(self):
        assert "SOL" not in self.store
        self.store["SOL"] = {}
        assert "SOL" in self.store

    def test_delitem(self):
        self.store["SOL"] = {"a": 1}
        del self.store["SOL"]
        assert "SOL" not in self.store

    def test_len(self):
        assert len(self.store) == 0
        self.store["A"] = {}
        self.store["B"] = {}
        assert len(self.store) == 2

    def test_iter(self):
        self.store["X"] = {}
        self.store["Y"] = {}
        assert set(self.store) == {"X", "Y"}

    def test_is_persistent_store_subclass(self):
        from cortex.persistence import PersistentStore
        assert isinstance(self.store, PersistentStore)


class TestVersionedStoreVersioning:
    """Test version snapshot creation, pruning, and listing."""

    def setup_method(self):
        self.mock_redis = AsyncMock()
        persistence._redis_client = self.mock_redis
        persistence._redis_available = True
        self.store = VersionedPersistentStore("ver_test", max_versions=3)

    def teardown_method(self):
        persistence._redis_available = False
        persistence._redis_client = None

    @pytest.mark.asyncio
    async def test_setitem_creates_version_snapshot(self):
        now = datetime.now(timezone.utc)
        self.store["SOL"] = {"calibrated_at": now, "sigma": 0.05}
        await asyncio.sleep(0.1)

        # Should persist the current value + version data + version meta
        assert self.mock_redis.set.call_count >= 2

    @pytest.mark.asyncio
    async def test_version_meta_increments(self):
        self.mock_redis.get = AsyncMock(return_value=None)

        now = datetime.now(timezone.utc)
        self.store["SOL"] = {"calibrated_at": now, "v": 1}
        await asyncio.sleep(0.1)

        # Check that meta was written with version 1
        meta_calls = [
            c for c in self.mock_redis.set.call_args_list
            if b"meta" in (c[0][0] if isinstance(c[0][0], bytes) else c[0][0].encode())
        ]
        assert len(meta_calls) >= 1
        meta_data = json.loads(meta_calls[-1][0][1])
        assert meta_data[0]["version"] == 1

    @pytest.mark.asyncio
    async def test_get_versions_empty(self):
        self.mock_redis.get = AsyncMock(return_value=None)
        versions = await self.store.get_versions("NONEXISTENT")
        assert versions == []

    @pytest.mark.asyncio
    async def test_get_versions_returns_meta(self):
        meta = [
            {"version": 1, "timestamp": 1000.0, "calibrated_at": "2026-01-01T00:00:00+00:00"},
            {"version": 2, "timestamp": 2000.0, "calibrated_at": "2026-01-02T00:00:00+00:00"},
        ]
        self.mock_redis.get = AsyncMock(return_value=json.dumps(meta).encode())
        versions = await self.store.get_versions("SOL")
        assert len(versions) == 2
        assert versions[0]["version"] == 1
        assert versions[1]["version"] == 2

    @pytest.mark.asyncio
    async def test_get_all_versions(self):
        self.store._data["SOL"] = {"v": 1}
        self.store._data["ETH"] = {"v": 2}
        self.store._version_meta["SOL"] = [{"version": 1, "timestamp": 1000.0}]
        self.store._version_meta["ETH"] = [{"version": 1, "timestamp": 1000.0}]

        result = await self.store.get_all_versions()
        assert "SOL" in result
        assert "ETH" in result

    @pytest.mark.asyncio
    async def test_pruning_keeps_max_versions(self):
        self.mock_redis.get = AsyncMock(return_value=None)
        self.mock_redis.delete = AsyncMock()

        now = datetime.now(timezone.utc)
        for i in range(5):
            self.store["SOL"] = {"calibrated_at": now, "v": i}
            await asyncio.sleep(0.05)

        # After 5 writes with max_versions=3, we should have pruned 2 old versions
        delete_calls = [
            c for c in self.mock_redis.delete.call_args_list
            if "versions:" in (c[0][0] if isinstance(c[0][0], str) else c[0][0].decode())
        ]
        assert len(delete_calls) >= 2

    @pytest.mark.asyncio
    async def test_get_version_data(self):
        # pickle is safe here — we created the test data ourselves
        original = {"sigma": 0.05, "calibrated_at": datetime.now(timezone.utc)}
        pickled = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)
        self.mock_redis.get = AsyncMock(return_value=pickled)

        data = await self.store.get_version_data("SOL", 1)
        assert data is not None
        assert data["sigma"] == 0.05

    @pytest.mark.asyncio
    async def test_get_version_data_missing(self):
        self.mock_redis.get = AsyncMock(return_value=None)
        data = await self.store.get_version_data("SOL", 999)
        assert data is None

    @pytest.mark.asyncio
    async def test_restore_version(self):
        # pickle is safe here — we created the test data ourselves
        original = {"sigma": 0.05, "calibrated_at": datetime.now(timezone.utc)}
        pickled = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)
        self.mock_redis.get = AsyncMock(return_value=pickled)

        self.store._data["SOL"] = {"sigma": 0.10}
        success = await self.store.restore_version("SOL", 1)
        assert success
        assert self.store["SOL"]["sigma"] == 0.05

    @pytest.mark.asyncio
    async def test_restore_version_missing(self):
        self.mock_redis.get = AsyncMock(return_value=None)
        success = await self.store.restore_version("SOL", 999)
        assert not success


class TestVersionedStoreNoRedis:
    """Versioning degrades gracefully without Redis."""

    def setup_method(self):
        persistence._redis_available = False
        persistence._redis_client = None
        self.store = VersionedPersistentStore("no_redis_test", max_versions=3)

    @pytest.mark.asyncio
    async def test_get_versions_empty_without_redis(self):
        versions = await self.store.get_versions("SOL")
        assert versions == []

    @pytest.mark.asyncio
    async def test_get_version_data_none_without_redis(self):
        data = await self.store.get_version_data("SOL", 1)
        assert data is None

    @pytest.mark.asyncio
    async def test_restore_version_fails_without_redis(self):
        success = await self.store.restore_version("SOL", 1)
        assert not success

    def test_setitem_works_without_redis(self):
        self.store["SOL"] = {"sigma": 0.05}
        assert self.store["SOL"]["sigma"] == 0.05


class TestModelsEndpoints:
    """Test the /models/versions API endpoints."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from api.main import app
        with TestClient(app) as c:
            yield c

    def test_list_all_versions_empty(self, client):
        resp = client.get("/api/v1/models/versions", headers={"X-API-Key": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert "tokens" in data

    def test_list_token_versions_404(self, client):
        resp = client.get("/api/v1/models/versions/NONEXISTENT", headers={"X-API-Key": "test"})
        assert resp.status_code == 404

    def test_rollback_404_no_model(self, client):
        resp = client.post(
            "/api/v1/models/rollback/NONEXISTENT?version=1",
            headers={"X-API-Key": "test"},
        )
        assert resp.status_code == 404
