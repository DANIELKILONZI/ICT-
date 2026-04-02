"""
Data Engine – CSV Loader
Loads OHLCV data from local CSV files when MT5 is unavailable.

Expected CSV columns (case-insensitive):
    time | datetime, open, high, low, close, volume
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from python.config import CONFIG

logger = logging.getLogger(__name__)

_CSV_ROOT = Path(__file__).resolve().parent.parent.parent / CONFIG["data"]["csv_path"]


def load_csv(
    symbol: str,
    timeframe: str,
    csv_path: Optional[str] = None,
    count: int = 500,
) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    File naming convention:  <SYMBOL>_<TIMEFRAME>.csv
    e.g.  EURUSD_H1.csv

    Returns DataFrame with columns: time, open, high, low, close, volume
    """
    if csv_path:
        fpath = Path(csv_path)
    else:
        fpath = _CSV_ROOT / f"{symbol.upper()}_{timeframe.upper()}.csv"

    if not fpath.exists():
        logger.error("CSV file not found: %s", fpath)
        return pd.DataFrame()

    df = pd.read_csv(fpath)
    df.columns = [c.lower() for c in df.columns]

    # Normalise time column
    time_col = next((c for c in df.columns if c in ("time", "datetime", "date")), None)
    if time_col is None:
        logger.error("No time column found in %s", fpath)
        return pd.DataFrame()

    df = df.rename(columns={time_col: "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Ensure required columns exist
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            logger.error("Missing column '%s' in %s", col, fpath)
            return pd.DataFrame()

    if "volume" not in df.columns:
        df["volume"] = 0

    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").reset_index(drop=True)

    if count and len(df) > count:
        df = df.iloc[-count:].reset_index(drop=True)

    logger.debug("Loaded %d candles from %s", len(df), fpath)
    return df
