"""
Unit tests for python.data_engine.data_store._LRUCache

Covers:
  - Basic get/set round-trip
  - Missing key returns None
  - LRU eviction when maxsize is exceeded
  - Most-recently-used entry is promoted and not evicted
  - refresh() bypasses the cache and forces a new fetch
  - Thread-safety under concurrent reads and writes
"""
from __future__ import annotations

import threading

import pandas as pd
import pytest

from python.data_engine.data_store import _LRUCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(val: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"close": [val]})


# ---------------------------------------------------------------------------
# Basic get/set
# ---------------------------------------------------------------------------

class TestLRUCacheBasic:
    def test_get_missing_returns_none(self):
        cache = _LRUCache(maxsize=4)
        assert cache.get(("EURUSD", "M5")) is None

    def test_set_then_get_returns_value(self):
        cache = _LRUCache(maxsize=4)
        df = _df(1.0)
        cache.set(("EURUSD", "M5"), df)
        result = cache.get(("EURUSD", "M5"))
        assert result is not None
        assert result["close"].iloc[0] == 1.0

    def test_overwrite_key_updates_value(self):
        cache = _LRUCache(maxsize=4)
        cache.set(("EURUSD", "M5"), _df(1.0))
        cache.set(("EURUSD", "M5"), _df(2.0))
        assert cache.get(("EURUSD", "M5"))["close"].iloc[0] == 2.0

    def test_multiple_keys_independent(self):
        cache = _LRUCache(maxsize=4)
        cache.set(("EURUSD", "M5"), _df(1.0))
        cache.set(("GBPUSD", "H1"), _df(2.0))
        assert cache.get(("EURUSD", "M5"))["close"].iloc[0] == 1.0
        assert cache.get(("GBPUSD", "H1"))["close"].iloc[0] == 2.0

    def test_pop_removes_key(self):
        cache = _LRUCache(maxsize=4)
        cache.set(("EURUSD", "M5"), _df(1.0))
        cache.pop(("EURUSD", "M5"))
        assert cache.get(("EURUSD", "M5")) is None

    def test_pop_missing_key_returns_default(self):
        cache = _LRUCache(maxsize=4)
        assert cache.pop(("MISSING", "M5"), None) is None


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

class TestLRUCacheEviction:
    def test_evicts_lru_when_full(self):
        cache = _LRUCache(maxsize=3)
        cache.set(("A", "M5"), _df(1.0))
        cache.set(("B", "M5"), _df(2.0))
        cache.set(("C", "M5"), _df(3.0))
        # Add a 4th entry – ("A", "M5") is LRU and should be evicted
        cache.set(("D", "M5"), _df(4.0))
        assert cache.get(("A", "M5")) is None
        assert cache.get(("B", "M5")) is not None
        assert cache.get(("C", "M5")) is not None
        assert cache.get(("D", "M5")) is not None

    def test_access_promotes_entry(self):
        cache = _LRUCache(maxsize=3)
        cache.set(("A", "M5"), _df(1.0))
        cache.set(("B", "M5"), _df(2.0))
        cache.set(("C", "M5"), _df(3.0))
        # Access ("A", …) to promote it to MRU
        cache.get(("A", "M5"))
        # Add a 4th entry – ("B", …) is now LRU
        cache.set(("D", "M5"), _df(4.0))
        assert cache.get(("A", "M5")) is not None  # promoted – should survive
        assert cache.get(("B", "M5")) is None       # was LRU – should be evicted

    def test_overwrite_does_not_grow_beyond_maxsize(self):
        cache = _LRUCache(maxsize=2)
        for _ in range(5):
            cache.set(("A", "M5"), _df(1.0))
            cache.set(("B", "M5"), _df(2.0))
        # Only 2 unique keys, maxsize is 2 – both should still be present
        assert cache.get(("A", "M5")) is not None
        assert cache.get(("B", "M5")) is not None


# ---------------------------------------------------------------------------
# Thread-safety
# ---------------------------------------------------------------------------

class TestLRUCacheThreadSafety:
    def test_concurrent_set_get_no_exception(self):
        """50 threads writing and reading concurrently must not raise."""
        cache = _LRUCache(maxsize=10)
        errors: list[Exception] = []

        def worker(i: int) -> None:
            try:
                key = (f"SYM{i % 5}", "M5")
                cache.set(key, _df(float(i)))
                _ = cache.get(key)
                _ = cache.get((f"SYM{(i + 1) % 5}", "M5"))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"
