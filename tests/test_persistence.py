"""Tests for cortex.persistence — PersistentStore and lifecycle."""

import asyncio
import pickle  # noqa: S403 — testing our own persistence layer (trusted data only)
from unittest.mock import AsyncMock, patch

import numpy as np
import pandas as pd
import pytest

from cortex import persistence
from cortex.persistence import PersistentStore, init_persistence, close_persistence


class TestPersistentStoreDictInterface:
    """PersistentStore must behave exactly like a dict when Redis is off."""

    def setup_method(self):
        persistence._redis_available = False
        persistence._redis_client = None
        self.store = PersistentStore("test")

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

    def test_get_default(self):
        assert self.store.get("missing") is None
        assert self.store.get("missing", 42) == 42

    def test_values(self):
        self.store["A"] = {"v": 1}
        self.store["B"] = {"v": 2}
        vals = list(self.store.values())
        assert len(vals) == 2

    def test_items(self):
        self.store["A"] = {"v": 1}
        items = list(self.store.items())
        assert items == [("A", {"v": 1})]

    def test_keys(self):
        self.store["A"] = {}
        self.store["B"] = {}
        assert set(self.store.keys()) == {"A", "B"}

    def test_pop(self):
        self.store["A"] = {"v": 1}
        val = self.store.pop("A")
        assert val == {"v": 1}
        assert "A" not in self.store

    def test_pop_missing_default(self):
        assert self.store.pop("missing", None) is None

    def test_getitem_missing_raises(self):
        with pytest.raises(KeyError):
            _ = self.store["nonexistent"]


class TestPersistentStoreWithNumpyPandas:
    """Verify stores handle numpy/pandas objects (the real data types)."""

    def setup_method(self):
        persistence._redis_available = False
        persistence._redis_client = None
        self.store = PersistentStore("numpy_test")

    def test_numpy_array_roundtrip(self):
        arr = np.array([1.0, 2.0, 3.0])
        self.store["SOL"] = {"sigma_states": arr}
        np.testing.assert_array_equal(self.store["SOL"]["sigma_states"], arr)

    def test_pandas_series_roundtrip(self):
        s = pd.Series([0.01, 0.02, 0.03], name="returns")
        self.store["SOL"] = {"returns": s}
        pd.testing.assert_series_equal(self.store["SOL"]["returns"], s)

    def test_pandas_dataframe_roundtrip(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        self.store["SOL"] = {"filter_probs": df}
        pd.testing.assert_frame_equal(self.store["SOL"]["filter_probs"], df)


class TestPersistentStoreRedisWrite:
    """Verify that writes trigger Redis persistence when available."""

    def setup_method(self):
        self.mock_redis = AsyncMock()
        persistence._redis_client = self.mock_redis
        persistence._redis_available = True
        self.store = PersistentStore("write_test")

    def teardown_method(self):
        persistence._redis_available = False
        persistence._redis_client = None

    @pytest.mark.asyncio
    async def test_setitem_triggers_persist(self):
        self.store["SOL"] = {"sigma": 0.05}
        await asyncio.sleep(0.05)
        self.mock_redis.set.assert_called_once()
        key_arg = self.mock_redis.set.call_args[0][0]
        assert "write_test" in (key_arg if isinstance(key_arg, str) else key_arg.decode())

    @pytest.mark.asyncio
    async def test_delitem_triggers_delete(self):
        self.store._data["SOL"] = {"sigma": 0.05}
        del self.store["SOL"]
        await asyncio.sleep(0.05)
        self.mock_redis.delete.assert_called_once()


class TestPersistentStoreBulkOps:
    """Test restore() and persist_all()."""

    def setup_method(self):
        self.mock_redis = AsyncMock()
        persistence._redis_client = self.mock_redis
        persistence._redis_available = True
        self.store = PersistentStore("bulk_test")

    def teardown_method(self):
        persistence._redis_available = False
        persistence._redis_client = None

    @pytest.mark.asyncio
    async def test_persist_all(self):
        self.store._data["A"] = {"v": 1}
        self.store._data["B"] = {"v": 2}
        count = await self.store.persist_all()
        assert count == 2
        assert self.mock_redis.set.call_count == 2

    @pytest.mark.asyncio
    async def test_restore_from_redis(self):
        # pickle is safe here — we created the data ourselves for testing
        original = {"sigma": np.array([1.0, 2.0]), "name": "SOL"}
        pickled = pickle.dumps(original, protocol=pickle.HIGHEST_PROTOCOL)

        key = b"cortex:store:bulk_test:SOL"

        async def fake_scan_iter(match=None, count=None):
            yield key

        self.mock_redis.scan_iter = fake_scan_iter
        self.mock_redis.get = AsyncMock(return_value=pickled)

        count = await self.store.restore()
        assert count == 1
        assert "SOL" in self.store
        np.testing.assert_array_equal(self.store["SOL"]["sigma"], original["sigma"])

    @pytest.mark.asyncio
    async def test_restore_no_redis(self):
        persistence._redis_available = False
        count = await self.store.restore()
        assert count == 0

    @pytest.mark.asyncio
    async def test_persist_all_no_redis(self):
        persistence._redis_available = False
        self.store._data["A"] = {"v": 1}
        count = await self.store.persist_all()
        assert count == 0


class TestInitClosePersistence:
    """Test init_persistence / close_persistence lifecycle."""

    @pytest.mark.asyncio
    async def test_init_disabled(self):
        with patch.object(persistence, "PERSISTENCE_ENABLED", False):
            await init_persistence()
            assert not persistence._redis_available

    @pytest.mark.asyncio
    async def test_init_no_url(self):
        with patch.object(persistence, "PERSISTENCE_ENABLED", True), \
             patch.object(persistence, "PERSISTENCE_REDIS_URL", ""):
            await init_persistence()
            assert not persistence._redis_available

    @pytest.mark.asyncio
    async def test_close_cleans_up(self):
        mock_client = AsyncMock()
        persistence._redis_client = mock_client
        persistence._redis_available = True
        await close_persistence()
        assert persistence._redis_client is None
        assert not persistence._redis_available
        mock_client.aclose.assert_called_once()


class TestStoresImportCompat:
    """Verify api.stores still exports the same names after refactor."""

    def test_all_stores_importable(self):
        from api.stores import (
            _model_store, _portfolio_store, _evt_store,
            _copula_store, _hawkes_store, _rough_store, _svj_store,
            _comparison_cache, ALL_STORES,
        )
        from cortex.persistence import VersionedPersistentStore
        assert isinstance(_model_store, VersionedPersistentStore)
        assert isinstance(_model_store, PersistentStore)
        assert isinstance(_comparison_cache, dict)
        assert len(ALL_STORES) >= 7

    def test_store_dict_compat(self):
        from api.stores import _model_store
        _model_store["TEST_TOKEN"] = {"test": True}
        assert "TEST_TOKEN" in _model_store
        assert _model_store["TEST_TOKEN"]["test"] is True
        del _model_store["TEST_TOKEN"]
        assert "TEST_TOKEN" not in _model_store
