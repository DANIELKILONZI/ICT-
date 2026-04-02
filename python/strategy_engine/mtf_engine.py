"""
Strategy Engine – Multi-Timeframe (MTF) Logic Engine

Hierarchy:
    D1  → macro bias
    H1  → structure confirmation
    M5  → entry execution

Rules for a valid trade opportunity:
  1. D1 trend aligns with H1 BOS direction
  2. H1 has an active OB or FVG in the direction of D1 bias
  3. M5 shows a liquidity sweep + FVG alignment
  4. Entry price retraces into a valid OB/FVG zone
  5. Price is in discount (BUY) or premium (SELL) per Fibonacci model
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from python.config import SCORING, STRATEGY
from python.strategy_engine import (
    bos_detector,
    fvg_detector,
    liquidity_engine,
    market_structure,
    order_block_detector,
    premium_discount,
)
from python.strategy_engine.market_structure import TrendDirection
from python.strategy_engine.util import atr_value as _atr_value

logger = logging.getLogger(__name__)

_FIB_LOOKBACK = STRATEGY.get("fib_swing_lookback", 50)

_SC_BIAS        = SCORING.get("bias_alignment",    0.20)
_SC_ZONE_OK     = SCORING.get("correct_zone",      0.15)
_SC_ZONE_WRONG  = SCORING.get("wrong_zone_penalty", 0.30)
_SC_H1_OB       = SCORING.get("h1_ob",             0.20)
_SC_H1_FVG      = SCORING.get("h1_fvg",            0.15)
_SC_M5_SWEEP    = SCORING.get("m5_sweep",          0.20)
_SC_M5_FVG      = SCORING.get("m5_fvg",            0.10)
_SC_MIN_VALID   = SCORING.get("min_valid_score",   0.50)


@dataclass
class MTFAnalysis:
    symbol: str
    d1_trend: TrendDirection
    h1_trend: TrendDirection
    h1_bos_direction: Optional[str]           # "BULLISH" | "BEARISH" | None
    m5_sweep: Optional[liquidity_engine.LiquiditySweep]
    m5_fvg: Optional[fvg_detector.FVGZone]
    h1_ob: Optional[order_block_detector.OrderBlock]
    h1_fvg: Optional[fvg_detector.FVGZone]
    fib_range: Optional[premium_discount.FibRange]
    price_zone: str                            # "DISCOUNT" | "PREMIUM" | "EQUILIBRIUM"
    signal_direction: Optional[str]           # "BUY" | "SELL" | None
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confluence_score: float = 0.0
    valid: bool = False
    reasons: list[str] = field(default_factory=list)


def _analyse_tf(df: pd.DataFrame) -> dict:
    ms = market_structure.analyse(df)
    bos_events = bos_detector.detect_bos(df, ms.swing_highs, ms.swing_lows)
    fvg_zones = fvg_detector.detect_fvg(df)
    fvg_detector.mark_filled_fvg(fvg_zones, df)
    obs = order_block_detector.detect_order_blocks(df, bos_events)
    order_block_detector.mark_mitigated_obs(obs, df)
    sweeps = liquidity_engine.detect_liquidity_sweeps(df, ms.swing_highs, ms.swing_lows)
    return {
        "ms": ms,
        "bos": bos_events,
        "fvg": fvg_zones,
        "obs": obs,
        "sweeps": sweeps,
    }


def analyse(
    symbol: str,
    df_d1: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
) -> MTFAnalysis:
    """
    Run full multi-timeframe ICT analysis and return an MTFAnalysis result.
    """
    current_price = float(df_m5["close"].iloc[-1])
    reasons: list[str] = []
    score = 0.0

    # ── Timeframe analysis ────────────────────────────────────────────────
    d1 = _analyse_tf(df_d1)
    h1 = _analyse_tf(df_h1)
    m5 = _analyse_tf(df_m5)

    d1_trend: TrendDirection = d1["ms"].trend
    h1_trend: TrendDirection = h1["ms"].trend

    # ── Latest BOS on H1 ──────────────────────────────────────────────────
    h1_latest_bos = bos_detector.latest_bos(h1["bos"])
    h1_bos_dir: Optional[str] = h1_latest_bos.direction if h1_latest_bos else None

    # ── Determine potential trade direction ───────────────────────────────
    signal_dir: Optional[str] = None

    if d1_trend == TrendDirection.BULLISH and h1_bos_dir == "BULLISH":
        signal_dir = "BUY"
        score += _SC_BIAS
        reasons.append("D1 bullish + H1 bullish BOS")
    elif d1_trend == TrendDirection.BEARISH and h1_bos_dir == "BEARISH":
        signal_dir = "SELL"
        score += _SC_BIAS
        reasons.append("D1 bearish + H1 bearish BOS")
    else:
        return MTFAnalysis(
            symbol=symbol,
            d1_trend=d1_trend,
            h1_trend=h1_trend,
            h1_bos_direction=h1_bos_dir,
            m5_sweep=None,
            m5_fvg=None,
            h1_ob=None,
            h1_fvg=None,
            fib_range=None,
            price_zone="UNKNOWN",
            signal_direction=None,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            confluence_score=0.0,
            valid=False,
            reasons=["D1/H1 bias misalignment"],
        )

    # ── Fibonacci premium/discount ─────────────────────────────────────────
    ms_d1 = d1["ms"]
    fib_range: Optional[premium_discount.FibRange] = None
    pzone = "UNKNOWN"

    if ms_d1.last_swing_high and ms_d1.last_swing_low:
        fib_range = premium_discount.compute_fib_range(
            ms_d1.last_swing_high, ms_d1.last_swing_low
        )
        if fib_range:
            pzone = premium_discount.price_zone(fib_range, current_price)
            if (signal_dir == "BUY" and pzone == "DISCOUNT") or (
                signal_dir == "SELL" and pzone == "PREMIUM"
            ):
                score += _SC_ZONE_OK
                reasons.append(f"Price in {pzone} zone")
            else:
                # Wrong zone – reduce score
                score -= _SC_ZONE_WRONG
                reasons.append(f"Price in {pzone} zone (wrong for {signal_dir})")

    # ── H1 Order Block ────────────────────────────────────────────────────
    h1_ob = order_block_detector.nearest_ob(h1["obs"], current_price, signal_dir)
    if h1_ob:
        score += _SC_H1_OB
        reasons.append(f"H1 OB at {h1_ob.ob_low:.5f}-{h1_ob.ob_high:.5f}")

    # ── H1 FVG ────────────────────────────────────────────────────────────
    h1_fvg = fvg_detector.nearest_fvg(h1["fvg"], current_price, signal_dir)
    if h1_fvg:
        score += _SC_H1_FVG
        reasons.append(f"H1 FVG at {h1_fvg.gap_low:.5f}-{h1_fvg.gap_high:.5f}")

    # ── M5 liquidity sweep ────────────────────────────────────────────────
    sweep_dir = "BULLISH_SWEEP" if signal_dir == "BUY" else "BEARISH_SWEEP"
    m5_sweep = liquidity_engine.latest_sweep(m5["sweeps"], sweep_dir)
    if m5_sweep:
        # Only count recent sweeps (last 10 candles on M5)
        if (len(df_m5) - 1 - m5_sweep.index) <= 10:
            score += _SC_M5_SWEEP
            reasons.append("M5 liquidity sweep confirmed")

    # ── M5 FVG ────────────────────────────────────────────────────────────
    m5_fvg = fvg_detector.nearest_fvg(m5["fvg"], current_price, signal_dir)
    if m5_fvg:
        score += _SC_M5_FVG
        reasons.append(f"M5 FVG at {m5_fvg.gap_low:.5f}-{m5_fvg.gap_high:.5f}")

    # ── Entry / SL / TP calculation ───────────────────────────────────────
    atr_m5 = _atr_value(df_m5)
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    if signal_dir == "BUY":
        # Entry at OB high or FVG low, whichever is closer to current price
        candidates = []
        if h1_ob:
            candidates.append(h1_ob.ob_high)
        if h1_fvg:
            candidates.append(h1_fvg.gap_low)
        if m5_fvg:
            candidates.append(m5_fvg.gap_low)
        entry_price = min(candidates, key=lambda p: abs(p - current_price)) if candidates else current_price
        stop_loss = entry_price - 2 * atr_m5
        take_profit = entry_price + (entry_price - stop_loss) * 2.0  # 2:1 RR

    elif signal_dir == "SELL":
        candidates = []
        if h1_ob:
            candidates.append(h1_ob.ob_low)
        if h1_fvg:
            candidates.append(h1_fvg.gap_high)
        if m5_fvg:
            candidates.append(m5_fvg.gap_high)
        entry_price = min(candidates, key=lambda p: abs(p - current_price)) if candidates else current_price
        stop_loss = entry_price + 2 * atr_m5
        take_profit = entry_price - (stop_loss - entry_price) * 2.0

    # ── Validity gate ─────────────────────────────────────────────────────
    valid = (
        score >= _SC_MIN_VALID
        and entry_price is not None
        and stop_loss is not None
        and take_profit is not None
        and m5_sweep is not None
    )

    score = max(0.0, min(1.0, score))

    return MTFAnalysis(
        symbol=symbol,
        d1_trend=d1_trend,
        h1_trend=h1_trend,
        h1_bos_direction=h1_bos_dir,
        m5_sweep=m5_sweep,
        m5_fvg=m5_fvg,
        h1_ob=h1_ob,
        h1_fvg=h1_fvg,
        fib_range=fib_range,
        price_zone=pzone,
        signal_direction=signal_dir,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confluence_score=score,
        valid=valid,
        reasons=reasons,
    )
