"""
Data Engine – MT5 Connector
Fetches OHLC data via the MetaTrader5 Python package.
Falls back gracefully when MT5 is not available (e.g., on Linux without Wine/MT5).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from python.config import CONFIG

logger = logging.getLogger(__name__)

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


def _ensure_mt5() -> bool:
    if not _MT5_AVAILABLE:
        return False
    cfg = CONFIG.get("mt5", {})
    if not mt5.initialize(
        login=cfg.get("login", 0),
        password=cfg.get("password", ""),
        server=cfg.get("server", ""),
    ):
        logger.error("MT5 initialize() failed: %s", mt5.last_error())
        return False
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
    """
    if not _ensure_mt5():
        logger.error("MT5 not connected – cannot fetch %s %s.", symbol, timeframe)
        return pd.DataFrame()

    tf_const = _TF_MAP.get(timeframe.upper())
    if tf_const is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    if utc_from:
        rates = mt5.copy_rates_from(symbol, tf_const, utc_from, count)
    else:
        rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)

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
    if _MT5_AVAILABLE:
        mt5.shutdown()
