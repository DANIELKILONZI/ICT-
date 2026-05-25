"""
Strategy Engine – Shared Utilities

Provides helper functions used by multiple modules in the strategy engine so
that common calculations are not copy-pasted.
"""
from __future__ import annotations

import pandas as pd


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Compute the Average True Range for *df* over *period* candles.

    True Range = max(High - Low, |High - PrevClose|, |Low - PrevClose|)
    ATR        = rolling mean of True Range over *period* bars.

    Returns a Series aligned to *df*'s index.  The first ``period - 1``
    values will be NaN (insufficient history).
    """
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def atr_value(df: pd.DataFrame, period: int = 14) -> float:
    """
    Return the most recent ATR value for *df* as a plain float.

    Returns ``float("nan")`` when insufficient history is available
    (fewer than *period* rows).
    """
    series = atr(df, period)
    val = series.iloc[-1]
    return float(val)
