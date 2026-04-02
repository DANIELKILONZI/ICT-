"""
Strategy Engine – Liquidity Engine

Detects:
  1. Equal Highs / Equal Lows  (potential liquidity pools)
  2. Stop Hunts  (wick breaks level, close returns inside range)
  3. Liquidity Sweeps:
       Bullish sweep: price takes buy-side liquidity then closes back below level
       Bearish sweep: price takes sell-side liquidity then closes back above level
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from python.config import STRATEGY, pip_size_for
from python.strategy_engine.market_structure import SwingPoint

_EQUAL_PIPS = STRATEGY.get("equal_level_pips", 3)


def _pips_to_price(pips: float, symbol: str = "EURUSD") -> float:
    """Convert pip count to price distance using the symbol-aware pip size."""
    return pips * pip_size_for(symbol)


@dataclass
class LiquidityLevel:
    price: float
    kind: str           # "EQUAL_HIGH" | "EQUAL_LOW"
    touches: int
    first_time: pd.Timestamp
    last_time: pd.Timestamp


@dataclass
class LiquiditySweep:
    index: int
    time: pd.Timestamp
    direction: str       # "BULLISH_SWEEP" | "BEARISH_SWEEP"
    swept_level: float
    close_price: float


def detect_equal_levels(
    swing_points: list[SwingPoint],
    pip_threshold: float = None,
    symbol: str = "EURUSD",
) -> list[LiquidityLevel]:
    """
    Group swing points whose prices are within *pip_threshold* pips of each other.
    Groups with 2+ members are equal highs or equal lows.
    """
    threshold = _pips_to_price(pip_threshold or _EQUAL_PIPS, symbol)
    levels: list[LiquidityLevel] = []
    used = set()

    points_sorted = sorted(swing_points, key=lambda s: s.price)

    for i, sp in enumerate(points_sorted):
        if i in used:
            continue
        group = [sp]
        for j in range(i + 1, len(points_sorted)):
            if abs(points_sorted[j].price - sp.price) <= threshold:
                group.append(points_sorted[j])
                used.add(j)
            else:
                break
        if len(group) >= 2:
            kind = "EQUAL_HIGH" if sp.kind == "HIGH" else "EQUAL_LOW"
            avg_price = sum(g.price for g in group) / len(group)
            levels.append(
                LiquidityLevel(
                    price=avg_price,
                    kind=kind,
                    touches=len(group),
                    first_time=min(g.time for g in group),
                    last_time=max(g.time for g in group),
                )
            )
    return levels


def detect_stop_hunts(
    df: pd.DataFrame,
    levels: list[LiquidityLevel],
    symbol: str = "EURUSD",
    pip_threshold: float = None,
) -> list[dict]:
    """
    Stop hunt: a candle wick pierces a liquidity level but the candle CLOSES
    back inside the range (above level for equal lows, below for equal highs).

    Returns list of dicts with keys: index, time, direction, level_price, close.
    """
    threshold = _pips_to_price(pip_threshold or _EQUAL_PIPS, symbol)
    hunts = []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    times = df["time"].tolist()

    for i in range(len(df)):
        for level in levels:
            if level.kind == "EQUAL_HIGH":
                # Wick above level, close below
                if highs[i] > level.price + threshold and closes[i] < level.price:
                    hunts.append(
                        {
                            "index": i,
                            "time": times[i],
                            "direction": "BEARISH_HUNT",
                            "level_price": level.price,
                            "close": closes[i],
                        }
                    )
            elif level.kind == "EQUAL_LOW":
                # Wick below level, close above
                if lows[i] < level.price - threshold and closes[i] > level.price:
                    hunts.append(
                        {
                            "index": i,
                            "time": times[i],
                            "direction": "BULLISH_HUNT",
                            "level_price": level.price,
                            "close": closes[i],
                        }
                    )
    return hunts


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
    symbol: str = "EURUSD",
    pip_threshold: float = None,
) -> list[LiquiditySweep]:
    """
    Bullish sweep:
        Price wicks *above* a previous swing high then candle closes *below* it.
    Bearish sweep:
        Price wicks *below* a previous swing low then candle closes *above* it.

    Complexity: O(n log k) where n is candle count and k is the number of
    swing points.  The prior-swing lists are built incrementally with a
    two-pointer so no per-candle list comprehension is needed.
    """
    threshold = _pips_to_price(pip_threshold or _EQUAL_PIPS, symbol)
    sweeps: list[LiquiditySweep] = []

    # Sort swing points by candle index so we can advance a pointer as we scan
    sorted_highs = sorted(swing_highs, key=lambda s: s.index)
    sorted_lows  = sorted(swing_lows,  key=lambda s: s.index)

    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    times  = df["time"].tolist()

    # Rolling collections of *active* (already-formed) swing prices
    active_high_prices: list[float] = []
    active_low_prices:  list[float] = []
    h_ptr = 0
    l_ptr = 0

    for i in range(len(df)):
        # Advance pointers: admit swings whose index < i (formed before this candle)
        while h_ptr < len(sorted_highs) and sorted_highs[h_ptr].index < i:
            active_high_prices.append(sorted_highs[h_ptr].price)
            h_ptr += 1

        while l_ptr < len(sorted_lows) and sorted_lows[l_ptr].index < i:
            active_low_prices.append(sorted_lows[l_ptr].price)
            l_ptr += 1

        # Check against all prior swing highs (sell-side liquidity above)
        for ph in active_high_prices:
            if highs[i] > ph + threshold and closes[i] < ph:
                sweeps.append(
                    LiquiditySweep(
                        index=i,
                        time=times[i],
                        direction="BULLISH_SWEEP",
                        swept_level=ph,
                        close_price=closes[i],
                    )
                )

        # Check against all prior swing lows (buy-side liquidity below)
        for pl in active_low_prices:
            if lows[i] < pl - threshold and closes[i] > pl:
                sweeps.append(
                    LiquiditySweep(
                        index=i,
                        time=times[i],
                        direction="BEARISH_SWEEP",
                        swept_level=pl,
                        close_price=closes[i],
                    )
                )

    return sweeps


def latest_sweep(
    sweeps: list[LiquiditySweep], direction: Optional[str] = None
) -> Optional[LiquiditySweep]:
    filtered = [s for s in sweeps if direction is None or s.direction == direction]
    return max(filtered, key=lambda s: s.index) if filtered else None
