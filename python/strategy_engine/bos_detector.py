"""
Strategy Engine – Break of Structure (BOS) Detector

Rules:
  Bullish BOS: candle CLOSE > last confirmed swing high
  Bearish BOS: candle CLOSE < last confirmed swing low

Every BOS event is stored with timestamp and price for downstream use.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from python.strategy_engine.market_structure import SwingPoint


@dataclass
class BOSEvent:
    index: int
    time: pd.Timestamp
    direction: str          # "BULLISH" | "BEARISH"
    break_price: float      # the swing level that was broken
    close_price: float      # closing price that confirmed the break


def detect_bos(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
) -> list[BOSEvent]:
    """
    Scan *df* for BOS events using the provided swing points.

    A swing level is only eligible for a BOS if the candle that breaks it
    occurs *after* the swing was formed.
    """
    events: list[BOSEvent] = []

    closes = df["close"].to_numpy()
    times = df["time"].tolist()

    # Sort once
    highs_sorted = sorted(swing_highs, key=lambda s: s.index)
    lows_sorted = sorted(swing_lows, key=lambda s: s.index)

    # Pointer to the last swing high/low that is "available" (formed before current candle)
    high_ptr = 0
    low_ptr = 0

    active_high: Optional[SwingPoint] = None
    active_low: Optional[SwingPoint] = None
    last_bullish_bos_idx = -1
    last_bearish_bos_idx = -1

    for i in range(len(df)):
        # Advance swing pointers – only use swings formed before this candle
        while high_ptr < len(highs_sorted) and highs_sorted[high_ptr].index < i:
            active_high = highs_sorted[high_ptr]
            high_ptr += 1

        while low_ptr < len(lows_sorted) and lows_sorted[low_ptr].index < i:
            active_low = lows_sorted[low_ptr]
            low_ptr += 1

        close = closes[i]

        # Bullish BOS
        if (
            active_high is not None
            and close > active_high.price
            and i > last_bullish_bos_idx
        ):
            events.append(
                BOSEvent(
                    index=i,
                    time=times[i],
                    direction="BULLISH",
                    break_price=active_high.price,
                    close_price=close,
                )
            )
            last_bullish_bos_idx = i
            active_high = None  # reset – wait for a new swing high

        # Bearish BOS
        if (
            active_low is not None
            and close < active_low.price
            and i > last_bearish_bos_idx
        ):
            events.append(
                BOSEvent(
                    index=i,
                    time=times[i],
                    direction="BEARISH",
                    break_price=active_low.price,
                    close_price=close,
                )
            )
            last_bearish_bos_idx = i
            active_low = None  # reset – wait for a new swing low

    return events


def latest_bos(events: list[BOSEvent]) -> Optional[BOSEvent]:
    """Return the most recent BOS event."""
    return max(events, key=lambda e: e.index) if events else None
