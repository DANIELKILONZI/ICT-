"""
Strategy Engine – Fair Value Gap (FVG) Detector

Bullish FVG:   High[i]   < Low[i+2]   (gap up between candle i and candle i+2)
Bearish FVG:   Low[i]    > High[i+2]  (gap down between candle i and candle i+2)

The displacement candle (i+1) must exceed the ATR threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from python.config import STRATEGY


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


@dataclass
class FVGZone:
    index: int              # index of the displacement candle (middle candle)
    time: pd.Timestamp
    direction: str          # "BULLISH" | "BEARISH"
    gap_high: float         # upper boundary of the gap
    gap_low: float          # lower boundary of the gap
    midpoint: float
    filled: bool = False    # updated externally once price enters the zone


def detect_fvg(
    df: pd.DataFrame,
    atr_period: int = None,
    atr_multiplier: float = None,
) -> list[FVGZone]:
    """
    Scan *df* for Fair Value Gaps.

    A displacement candle must have a body/range greater than
    atr_multiplier * ATR to qualify.
    """
    period = atr_period or STRATEGY.get("atr_period", 14)
    mult = atr_multiplier or STRATEGY.get("atr_multiplier", 1.5)

    atr_series = _atr(df, period)
    zones: list[FVGZone] = []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    opens = df["open"].to_numpy()
    times = df["time"].tolist()
    atr_vals = atr_series.to_numpy()

    for i in range(len(df) - 2):
        i1 = i       # candle 1
        i2 = i + 1   # displacement candle
        i3 = i + 2   # candle 3

        atr_val = atr_vals[i2]
        if np.isnan(atr_val):
            continue

        candle2_range = abs(closes[i2] - opens[i2])
        if candle2_range < mult * atr_val:
            continue

        # Bullish FVG: gap between candle1 high and candle3 low
        if highs[i1] < lows[i3]:
            zones.append(
                FVGZone(
                    index=i2,
                    time=times[i2],
                    direction="BULLISH",
                    gap_low=highs[i1],
                    gap_high=lows[i3],
                    midpoint=(highs[i1] + lows[i3]) / 2,
                )
            )

        # Bearish FVG: gap between candle1 low and candle3 high
        elif lows[i1] > highs[i3]:
            zones.append(
                FVGZone(
                    index=i2,
                    time=times[i2],
                    direction="BEARISH",
                    gap_low=highs[i3],
                    gap_high=lows[i1],
                    midpoint=(lows[i1] + highs[i3]) / 2,
                )
            )

    return zones


def mark_filled_fvg(zones: list[FVGZone], df: pd.DataFrame) -> None:
    """
    Update the *filled* flag in-place for FVG zones where price has entered the zone.
    """
    for zone in zones:
        future = df[df.index > zone.index]
        if zone.direction == "BULLISH":
            if (future["low"] <= zone.gap_high).any():
                zone.filled = True
        else:
            if (future["high"] >= zone.gap_low).any():
                zone.filled = True


def active_fvg(
    zones: list[FVGZone], direction: Optional[str] = None
) -> list[FVGZone]:
    """Return unfilled FVG zones, optionally filtered by direction."""
    return [
        z
        for z in zones
        if not z.filled and (direction is None or z.direction == direction)
    ]


def nearest_fvg(
    zones: list[FVGZone], current_price: float, direction: Optional[str] = None
) -> Optional[FVGZone]:
    """Return the active FVG zone closest to *current_price*."""
    candidates = active_fvg(zones, direction)
    if not candidates:
        return None
    return min(candidates, key=lambda z: abs(z.midpoint - current_price))
