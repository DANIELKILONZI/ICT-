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

from python.config import STRATEGY
from python.strategy_engine.market_structure import SwingPoint

_EQUAL_PIPS = STRATEGY.get("equal_level_pips", 3)


def _pips_to_price(pips: float, symbol: str = "EURUSD") -> float:
    """Convert pip count to price distance (default 5-digit broker)."""
    return pips * 0.00010


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

    for i, row in df.iterrows():
        for level in levels:
            if level.kind == "EQUAL_HIGH":
                # Wick above level, close below
                if row["high"] > level.price + threshold and row["close"] < level.price:
                    hunts.append(
                        {
                            "index": int(i),
                            "time": row["time"],
                            "direction": "BEARISH_HUNT",
                            "level_price": level.price,
                            "close": row["close"],
                        }
                    )
            elif level.kind == "EQUAL_LOW":
                # Wick below level, close above
                if row["low"] < level.price - threshold and row["close"] > level.price:
                    hunts.append(
                        {
                            "index": int(i),
                            "time": row["time"],
                            "direction": "BULLISH_HUNT",
                            "level_price": level.price,
                            "close": row["close"],
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
    """
    threshold = _pips_to_price(pip_threshold or _EQUAL_PIPS, symbol)
    sweeps: list[LiquiditySweep] = []

    high_prices = {sp.index: sp.price for sp in swing_highs}
    low_prices = {sp.index: sp.price for sp in swing_lows}

    for i, row in df.iterrows():
        row_idx = int(i)
        # Check against all prior swing highs (sell-side liquidity above)
        prior_highs = [p for idx, p in high_prices.items() if idx < row_idx]
        for ph in prior_highs:
            if row["high"] > ph + threshold and row["close"] < ph:
                sweeps.append(
                    LiquiditySweep(
                        index=row_idx,
                        time=row["time"],
                        direction="BULLISH_SWEEP",
                        swept_level=ph,
                        close_price=row["close"],
                    )
                )

        # Check against all prior swing lows (buy-side liquidity below)
        prior_lows = [p for idx, p in low_prices.items() if idx < row_idx]
        for pl in prior_lows:
            if row["low"] < pl - threshold and row["close"] > pl:
                sweeps.append(
                    LiquiditySweep(
                        index=row_idx,
                        time=row["time"],
                        direction="BEARISH_SWEEP",
                        swept_level=pl,
                        close_price=row["close"],
                    )
                )

    return sweeps


def latest_sweep(
    sweeps: list[LiquiditySweep], direction: Optional[str] = None
) -> Optional[LiquiditySweep]:
    filtered = [s for s in sweeps if direction is None or s.direction == direction]
    return max(filtered, key=lambda s: s.index) if filtered else None
