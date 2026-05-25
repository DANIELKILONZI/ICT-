"""
Data Engine – MT5 Connector
Fetches OHLC data via the MetaTrader5 Python package.
Falls back gracefully when MT5 is not available (e.g., on Linux without Wine/MT5).

Connection pooling
------------------
A single MT5 connection is maintained for the lifetime of the process.
``_ensure_mt5()`` only (re-)initialises the terminal when no healthy
connection exists; a lightweight health check (``account_info()``) is used
to detect a stale connection before retrying once.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Optional

import pandas as pd

from python.config import CONFIG
from python.exceptions import DataSourceError

logger = logging.getLogger(__name__)
_MAX_FETCH_RETRIES = 3
_FETCH_RETRY_DELAY_SEC = 1.0

# Map string timeframe names to MT5 constants (populated lazily)
_TF_MAP: dict[str, int] = {}

try:
    import MetaTrader5 as mt5  # type: ignore

    _TF_MAP = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    _MT5_AVAILABLE = True
except ImportError:
    _MT5_AVAILABLE = False
    logger.warning("MetaTrader5 package not found – MT5 data source unavailable.")

# ---------------------------------------------------------------------------
# Persistent connection state
# ---------------------------------------------------------------------------
_conn_lock = threading.Lock()
_connected: bool = False       # True once mt5.initialize() has succeeded


def _ensure_mt5() -> bool:
    """
    Ensure a healthy MT5 connection exists.

    On first call (or after a detected disconnect) this calls
    ``mt5.initialize()``.  On subsequent calls it performs a cheap
    ``mt5.account_info()`` health-check and re-initialises only when the
    check fails, so the vast majority of calls return quickly without
    touching the MT5 IPC layer.
    """
    global _connected

    if not _MT5_AVAILABLE:
        return False

    with _conn_lock:
        if _connected:
            # Fast health-check – account_info() returns None on a dead connection
            if mt5.account_info() is not None:
                return True
            # Connection is stale; fall through to re-initialise
            logger.warning("MT5 connection lost – reconnecting.")
            _connected = False

        cfg = CONFIG.get("mt5", {})
        ok = mt5.initialize(
            login=cfg.get("login", 0),
            password=cfg.get("password", ""),
            server=cfg.get("server", ""),
        )
        if not ok:
            logger.error("MT5 initialize() failed: %s", mt5.last_error())
            return False

        _connected = True
        logger.debug("MT5 connection established.")
        return True


def fetch_ohlcv(
    symbol: str,
    timeframe: str,
    count: int = 500,
    utc_from: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from MetaTrader5.

    Returns a DataFrame with columns: time, open, high, low, close, volume
    Index is reset integer.  Returns empty DataFrame on failure.

    Raises
    ------
    DataSourceError
        When the MT5 terminal is not available (package not installed or
        terminal not running).  When the timeframe is unrecognised a plain
        ``ValueError`` is raised so the caller can surface a meaningful message.
    """
    if not _ensure_mt5():
        raise DataSourceError(
            "MT5 not connected – cannot fetch data.",
            symbol=symbol,
            timeframe=timeframe,
            source="mt5",
        )

    tf_const = _TF_MAP.get(timeframe.upper())
    if tf_const is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    rates = None
    for attempt in range(1, _MAX_FETCH_RETRIES + 1):
        try:
            if utc_from:
                rates = mt5.copy_rates_from(symbol, tf_const, utc_from, count)
            else:
                rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "MT5 copy_rates failed for %s %s (attempt %d/%d): %s",
                symbol, timeframe, attempt, _MAX_FETCH_RETRIES, exc
            )
            rates = None

        if rates is not None and len(rates) > 0:
            break
        if attempt < _MAX_FETCH_RETRIES:
            time.sleep(_FETCH_RETRY_DELAY_SEC)

    if rates is None or len(rates) == 0:
        logger.warning("No data returned for %s %s", symbol, timeframe)
        return pd.DataFrame()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").reset_index(drop=True)
    logger.debug("Fetched %d candles for %s %s", len(df), symbol, timeframe)
    return df


def get_account_info() -> dict:
    """Return account balance, equity, currency."""
    if not _ensure_mt5():
        return {}
    info = mt5.account_info()
    if info is None:
        return {}
    return {
        "balance": info.balance,
        "equity": info.equity,
        "currency": info.currency,
        "leverage": info.leverage,
    }


def get_symbol_info(symbol: str) -> dict:
    """Return symbol point, digits, trade_contract_size, spread."""
    if not _ensure_mt5():
        return {}
    info = mt5.symbol_info(symbol)
    if info is None:
        return {}
    return {
        "point": info.point,
        "digits": info.digits,
        "contract_size": info.trade_contract_size,
        "spread": info.spread,
        "volume_min": info.volume_min,
        "volume_step": info.volume_step,
    }


def shutdown_mt5() -> None:
    global _connected
    if _MT5_AVAILABLE:
        with _conn_lock:
            mt5.shutdown()
            _connected = False
