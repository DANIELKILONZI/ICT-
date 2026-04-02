"""
Data Engine – Data Store
Caches OHLCV data in memory (Pandas) and optionally persists to SQLite.
Provides a unified interface used by the rest of the system.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import pandas as pd

from python.config import CONFIG
from python.data_engine.csv_loader import load_csv
from python.data_engine.mt5_connector import fetch_ohlcv
from python.exceptions import DataSourceError

logger = logging.getLogger(__name__)

_SOURCE = CONFIG["data"]["source"]  # "mt5" | "csv"
_SQLITE_PATH = Path(__file__).resolve().parent.parent.parent / CONFIG["data"]["sqlite_path"]
_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Bounded LRU cache – evicts the least-recently-used entry once the cap is
# reached, preventing unbounded memory growth over long-running sessions.
# Cap is generous (128 slots = 4 symbols × 6 TFs × several variants with
# room to spare) while still bounding worst-case memory.
# ---------------------------------------------------------------------------
_CACHE_MAXSIZE: int = 128


class _LRUCache:
    """Thread-safe, bounded LRU mapping."""

    def __init__(self, maxsize: int) -> None:
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()

    def get(self, key: tuple) -> Optional[pd.DataFrame]:
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def set(self, key: tuple, value: pd.DataFrame) -> None:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = value
            while len(self._data) > self._maxsize:
                self._data.popitem(last=False)  # evict LRU entry

    def pop(self, key: tuple, default=None) -> Optional[pd.DataFrame]:
        with self._lock:
            return self._data.pop(key, default)


_cache = _LRUCache(_CACHE_MAXSIZE)


def get_ohlcv(
    symbol: str,
    timeframe: str,
    count: int = 500,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Retrieve OHLCV data for *symbol* and *timeframe*.

    Priority:  memory cache → configured source (MT5 or CSV) → SQLite snapshot
    """
    key = (symbol.upper(), timeframe.upper())

    if use_cache:
        cached = _cache.get(key)
        if cached is not None:
            return cached

    if _SOURCE == "mt5":
        try:
            df = fetch_ohlcv(symbol, timeframe, count)
        except DataSourceError as exc:
            logger.warning("MT5 fetch failed: %s", exc)
            df = pd.DataFrame()
    else:
        df = load_csv(symbol, timeframe, count=count)

    if df.empty:
        df = _load_from_sqlite(symbol, timeframe, count)

    if not df.empty:
        _cache.set(key, df)
        _persist_to_sqlite(df, symbol, timeframe)

    return df


def refresh(symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
    """Force-refresh the cache for a symbol/timeframe pair."""
    key = (symbol.upper(), timeframe.upper())
    _cache.pop(key)
    return get_ohlcv(symbol, timeframe, count, use_cache=False)


def _validate_identifier(name: str) -> str:
    """Ensure the string contains only alphanumeric characters and underscores."""
    import re
    if not re.match(r"^[A-Z0-9_]+$", name):
        raise ValueError(f"Invalid identifier for SQL table name: {name!r}")
    return name


def _table_name(symbol: str, timeframe: str) -> str:
    return _validate_identifier(f"{symbol.upper()}_{timeframe.upper()}")


def _persist_to_sqlite(df: pd.DataFrame, symbol: str, timeframe: str) -> None:
    try:
        conn = sqlite3.connect(_SQLITE_PATH)
        tbl = _table_name(symbol, timeframe)
        df_copy = df.copy()
        df_copy["time"] = df_copy["time"].astype(str)
        df_copy.to_sql(tbl, conn, if_exists="replace", index=False)
        conn.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("SQLite persist failed: %s", exc)


def _load_from_sqlite(symbol: str, timeframe: str, count: int) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(_SQLITE_PATH)
        tbl = _table_name(symbol, timeframe)
        df = pd.read_sql(
            f"SELECT * FROM {tbl} ORDER BY time DESC LIMIT {count}", conn  # noqa: S608
        )
        conn.close()
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)
        logger.debug("Loaded %d rows from SQLite for %s %s", len(df), symbol, timeframe)
        return df
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
