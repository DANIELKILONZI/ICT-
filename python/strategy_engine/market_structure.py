"""
Strategy Engine – Market Structure Module

Detects swing highs/lows and classifies trend as:
  - BULLISH  (Higher Highs + Higher Lows sequence)
  - BEARISH  (Lower Lows + Lower Highs sequence)
  - RANGING  (no clear sequence)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import pandas as pd

from python.config import STRATEGY


class TrendDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    RANGING = "RANGING"


@dataclass
class SwingPoint:
    index: int
    price: float
    time: pd.Timestamp
    kind: str  # "HIGH" | "LOW"


def detect_swing_highs(df: pd.DataFrame, lookback: int = None) -> list[SwingPoint]:
    """
    A swing high at index *i* satisfies:
        High[i] > High[i-n]  AND  High[i] > High[i+n]  for all n in 1..lookback
    """
    lb = lookback or STRATEGY.get("swing_lookback", 2)
    swings: list[SwingPoint] = []
    highs = df["high"].to_numpy()
    for i in range(lb, len(df) - lb):
        window = highs[i - lb : i + lb + 1]
        if highs[i] == window.max():
            swings.append(
                SwingPoint(
                    index=i,
                    price=highs[i],
                    time=df["time"].iloc[i],
                    kind="HIGH",
                )
            )
    return swings


def detect_swing_lows(df: pd.DataFrame, lookback: int = None) -> list[SwingPoint]:
    """
    A swing low at index *i* satisfies:
        Low[i] < Low[i-n]  AND  Low[i] < Low[i+n]  for all n in 1..lookback
    """
    lb = lookback or STRATEGY.get("swing_lookback", 2)
    swings: list[SwingPoint] = []
    lows = df["low"].to_numpy()
    for i in range(lb, len(df) - lb):
        window = lows[i - lb : i + lb + 1]
        if lows[i] == window.min():
            swings.append(
                SwingPoint(
                    index=i,
                    price=lows[i],
                    time=df["time"].iloc[i],
                    kind="LOW",
                )
            )
    return swings


def classify_trend(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    lookback_swings: int = 3,
) -> TrendDirection:
    """
    Classify market trend using the last N swing highs and lows.

    Bullish:  HH + HL  (each new swing high > previous, each new swing low > previous)
    Bearish:  LL + LH  (each new swing low < previous, each new swing high < previous)
    """
    highs = sorted(swing_highs, key=lambda s: s.index)[-lookback_swings:]
    lows = sorted(swing_lows, key=lambda s: s.index)[-lookback_swings:]

    if len(highs) < 2 or len(lows) < 2:
        return TrendDirection.RANGING

    hh = all(highs[i].price > highs[i - 1].price for i in range(1, len(highs)))
    hl = all(lows[i].price > lows[i - 1].price for i in range(1, len(lows)))

    ll = all(lows[i].price < lows[i - 1].price for i in range(1, len(lows)))
    lh = all(highs[i].price < highs[i - 1].price for i in range(1, len(highs)))

    if hh and hl:
        return TrendDirection.BULLISH
    if ll and lh:
        return TrendDirection.BEARISH
    return TrendDirection.RANGING


@dataclass
class MarketStructureResult:
    trend: TrendDirection
    swing_highs: list[SwingPoint]
    swing_lows: list[SwingPoint]
    last_swing_high: Optional[SwingPoint]
    last_swing_low: Optional[SwingPoint]


def analyse(df: pd.DataFrame, lookback: int = None) -> MarketStructureResult:
    """Full market structure analysis on a single-timeframe DataFrame."""
    swing_highs = detect_swing_highs(df, lookback)
    swing_lows = detect_swing_lows(df, lookback)
    trend = classify_trend(swing_highs, swing_lows)

    last_high = max(swing_highs, key=lambda s: s.index) if swing_highs else None
    last_low = max(swing_lows, key=lambda s: s.index) if swing_lows else None

    return MarketStructureResult(
        trend=trend,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
        last_swing_high=last_high,
        last_swing_low=last_low,
    )
