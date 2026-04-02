"""
Strategy Engine – Premium / Discount Model

Uses Fibonacci levels derived from the last confirmed swing high and swing low.

  0.0  = swing low (discount extreme)
  0.5  = equilibrium
  1.0  = swing high (premium extreme)

Rules:
  BUY  only in discount zone  (price < 50% level)
  SELL only in premium zone   (price > 50% level)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from python.strategy_engine.market_structure import SwingPoint


FIB_LEVELS = {
    "0.0": 0.0,
    "0.236": 0.236,
    "0.382": 0.382,
    "0.5": 0.5,
    "0.618": 0.618,
    "0.786": 0.786,
    "1.0": 1.0,
}


@dataclass
class FibRange:
    swing_low: SwingPoint
    swing_high: SwingPoint
    levels: dict[str, float]   # label -> price
    equilibrium: float


def compute_fib_range(
    swing_high: SwingPoint, swing_low: SwingPoint
) -> Optional[FibRange]:
    """
    Compute Fibonacci price levels between *swing_low* and *swing_high*.
    Returns None if swing_high.price <= swing_low.price.
    """
    if swing_high.price <= swing_low.price:
        return None

    rng = swing_high.price - swing_low.price
    levels = {
        label: swing_low.price + ratio * rng
        for label, ratio in FIB_LEVELS.items()
    }
    return FibRange(
        swing_low=swing_low,
        swing_high=swing_high,
        levels=levels,
        equilibrium=levels["0.5"],
    )


def price_zone(fib_range: FibRange, price: float) -> str:
    """
    Return 'DISCOUNT', 'PREMIUM', or 'EQUILIBRIUM' for a given price.
    """
    eq = fib_range.equilibrium
    if price < eq:
        return "DISCOUNT"
    if price > eq:
        return "PREMIUM"
    return "EQUILIBRIUM"


def is_discount(fib_range: FibRange, price: float) -> bool:
    return price < fib_range.equilibrium


def is_premium(fib_range: FibRange, price: float) -> bool:
    return price > fib_range.equilibrium


def fib_level_at_price(fib_range: FibRange, price: float) -> float:
    """Return the normalised Fibonacci ratio (0–1) for a given price."""
    rng = fib_range.swing_high.price - fib_range.swing_low.price
    if rng == 0:
        return 0.0
    return (price - fib_range.swing_low.price) / rng
