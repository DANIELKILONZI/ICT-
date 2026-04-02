"""
Data Engine – Data Store
Caches OHLCV data in memory (Pandas) and optionally persists to SQLite.
Provides a unified interface used by the rest of the system.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd

from python.config import CONFIG
from python.data_engine.csv_loader import load_csv
from python.data_engine.mt5_connector import fetch_ohlcv

logger = logging.getLogger(__name__)

_SOURCE = CONFIG["data"]["source"]  # "mt5" | "csv"
_SQLITE_PATH = Path(__file__).resolve().parent.parent.parent / CONFIG["data"]["sqlite_path"]
_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)

# In-memory cache: { (symbol, timeframe): DataFrame }
_cache: dict[tuple[str, str], pd.DataFrame] = {}


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

    if use_cache and key in _cache:
        return _cache[key]

    if _SOURCE == "mt5":
        df = fetch_ohlcv(symbol, timeframe, count)
    else:
        df = load_csv(symbol, timeframe, count=count)

    if df.empty:
        df = _load_from_sqlite(symbol, timeframe, count)

    if not df.empty:
        _cache[key] = df
        _persist_to_sqlite(df, symbol, timeframe)

    return df


def refresh(symbol: str, timeframe: str, count: int = 500) -> pd.DataFrame:
    """Force-refresh the cache for a symbol/timeframe pair."""
    key = (symbol.upper(), timeframe.upper())
    _cache.pop(key, None)
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
