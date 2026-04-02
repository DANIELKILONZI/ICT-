"""
Strategy Engine – Order Block Detector

Bullish OB:
    Last BEARISH candle immediately before a bullish displacement that:
      • Results in a Bullish BOS
      • Displacement candle exceeds ATR threshold

Bearish OB:
    Last BULLISH candle immediately before a bearish displacement that:
      • Results in a Bearish BOS
      • Displacement candle exceeds ATR threshold
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from python.config import STRATEGY
from python.strategy_engine.bos_detector import BOSEvent


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


@dataclass
class OrderBlock:
    index: int               # DataFrame index of the OB candle
    time: pd.Timestamp
    direction: str           # "BULLISH" | "BEARISH"
    ob_high: float
    ob_low: float
    ob_open: float
    ob_close: float
    bos_index: int           # index of the confirming BOS candle
    mitigated: bool = False  # True once price enters the OB zone


def detect_order_blocks(
    df: pd.DataFrame,
    bos_events: list[BOSEvent],
    atr_period: int = None,
    atr_multiplier: float = None,
) -> list[OrderBlock]:
    """
    For each BOS event, look back to find the last opposing candle before the
    displacement move that caused the BOS.
    """
    period = atr_period or STRATEGY.get("atr_period", 14)
    mult = atr_multiplier or STRATEGY.get("atr_multiplier", 1.5)
    ob_window   = STRATEGY.get("ob_search_window",   10)  # candles back from displacement to find OB
    disp_window = STRATEGY.get("disp_search_window",  6)  # candles back from BOS to find displacement

    atr_series = _atr(df, period).to_numpy()
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    times = df["time"].tolist()

    order_blocks: list[OrderBlock] = []

    for event in bos_events:
        bos_idx = event.index
        if bos_idx < 2:
            continue

        # ── Step 1: find the displacement candle ────────────────────────────
        # Per ICT theory the OB is defined relative to the displacement move
        # that *caused* the BOS, not relative to the BOS candle itself.
        # We search backward from the BOS candle to locate the displacement:
        # the most recent candle whose body exceeds ATR × multiplier and whose
        # close is in the direction of the BOS.
        for disp_idx in range(bos_idx, max(bos_idx - disp_window, 0), -1):
            candle_range = abs(closes[disp_idx] - opens[disp_idx])
            atr_val = atr_series[disp_idx]
            if np.isnan(atr_val) or atr_val == 0:
                continue
            if candle_range < mult * atr_val:
                continue

            # Found a valid displacement candle
            if event.direction == "BULLISH":
                # Displacement must be a bullish candle
                if closes[disp_idx] <= opens[disp_idx]:
                    continue
                # ── Step 2: find the OB ─────────────────────────────────────
                # The OB is the LAST (most recent) bearish candle immediately
                # before the displacement move – i.e. we walk backward from
                # disp_idx-1 and stop at the first opposing candle we find.
                for ob_idx in range(disp_idx - 1, max(disp_idx - ob_window, -1), -1):
                    if ob_idx < 0:
                        break
                    if closes[ob_idx] < opens[ob_idx]:  # bearish candle
                        order_blocks.append(
                            OrderBlock(
                                index=ob_idx,
                                time=times[ob_idx],
                                direction="BULLISH",
                                ob_high=highs[ob_idx],
                                ob_low=lows[ob_idx],
                                ob_open=opens[ob_idx],
                                ob_close=closes[ob_idx],
                                bos_index=bos_idx,
                            )
                        )
                        break
                break

            elif event.direction == "BEARISH":
                # Displacement must be a bearish candle
                if closes[disp_idx] >= opens[disp_idx]:
                    continue
                # ── Step 2: find the OB ─────────────────────────────────────
                # The OB is the LAST (most recent) bullish candle immediately
                # before the displacement move.
                for ob_idx in range(disp_idx - 1, max(disp_idx - ob_window, -1), -1):
                    if ob_idx < 0:
                        break
                    if closes[ob_idx] > opens[ob_idx]:  # bullish candle
                        order_blocks.append(
                            OrderBlock(
                                index=ob_idx,
                                time=times[ob_idx],
                                direction="BEARISH",
                                ob_high=highs[ob_idx],
                                ob_low=lows[ob_idx],
                                ob_open=opens[ob_idx],
                                ob_close=closes[ob_idx],
                                bos_index=bos_idx,
                            )
                        )
                        break
                break

    # Deduplicate by index
    seen: set[int] = set()
    unique_obs: list[OrderBlock] = []
    for ob in order_blocks:
        if ob.index not in seen:
            seen.add(ob.index)
            unique_obs.append(ob)

    return unique_obs


def mark_mitigated_obs(obs: list[OrderBlock], df: pd.DataFrame) -> None:
    """Update *mitigated* flag in-place when price enters the OB zone."""
    for ob in obs:
        future = df[df.index > ob.index]
        if ob.direction == "BULLISH":
            if (future["low"] <= ob.ob_high).any():
                ob.mitigated = True
        else:
            if (future["high"] >= ob.ob_low).any():
                ob.mitigated = True


def active_obs(
    obs: list[OrderBlock], direction: Optional[str] = None
) -> list[OrderBlock]:
    return [
        ob
        for ob in obs
        if not ob.mitigated and (direction is None or ob.direction == direction)
    ]


def nearest_ob(
    obs: list[OrderBlock], current_price: float, direction: Optional[str] = None
) -> Optional[OrderBlock]:
    candidates = active_obs(obs, direction)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda ob: abs(((ob.ob_high + ob.ob_low) / 2) - current_price),
    )
