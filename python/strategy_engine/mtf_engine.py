"""
Strategy Engine – Multi-Timeframe (MTF) Logic Engine

Hierarchy:
    D1  → macro bias
    H1  → structure confirmation
    M5  → entry execution

The engine dispatches to one of four named playbooks (in priority order):
  1. SWEEP_FVG_CONTINUATION  – M5 sweep → displacement → M5 FVG, D1/H1 aligned
  2. HTF_OB_REVERSAL         – price at D1 OB + H1 sweep confirmation
  3. LONDON_KILLZONE_EXPANSION – H1 BOS during London Killzone + M5 FVG
  4. NY_KILLZONE_EXPANSION    – H1 BOS during NY Killzone + M5 FVG

Only these four setups are traded; no generic confluence scoring.
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
from python.strategy_engine import playbooks
from python.strategy_engine.market_structure import TrendDirection
from python.strategy_engine.util import atr_value as _atr_value

logger = logging.getLogger(__name__)

_FIB_LOOKBACK = STRATEGY.get("fib_swing_lookback", 50)
_SC_MIN_VALID: float = SCORING.get("min_valid_score", 0.50)

# The constants below are kept for backward compatibility with existing tests
# that verify scoring values come from config rather than being hardcoded.
_SC_BIAS:       float = SCORING.get("bias_alignment",    0.20)
_SC_ZONE_OK:    float = SCORING.get("correct_zone",      0.15)
_SC_ZONE_WRONG: float = SCORING.get("wrong_zone_penalty", 0.30)
_SC_H1_OB:      float = SCORING.get("h1_ob",             0.20)
_SC_H1_FVG:     float = SCORING.get("h1_fvg",            0.15)
_SC_M5_SWEEP:   float = SCORING.get("m5_sweep",          0.20)
_SC_M5_FVG:     float = SCORING.get("m5_fvg",            0.10)


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
    setup_name: Optional[str] = None          # e.g. "SWEEP_FVG_CONTINUATION"
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
    Run multi-timeframe ICT analysis and return an MTFAnalysis result.

    Each timeframe is analysed for market structure, BOS, FVG, order blocks,
    and liquidity sweeps.  The results are then passed to the three named
    playbooks in priority order; the first match wins.  If no playbook fires,
    the result is marked invalid.
    """
    current_price = float(df_m5["close"].iloc[-1])

    # ── Per-timeframe analysis ────────────────────────────────────────────────
    d1 = _analyse_tf(df_d1)
    h1 = _analyse_tf(df_h1)
    m5 = _analyse_tf(df_m5)

    d1_trend: TrendDirection = d1["ms"].trend
    h1_trend: TrendDirection = h1["ms"].trend

    h1_latest_bos = bos_detector.latest_bos(h1["bos"])
    h1_bos_dir: Optional[str] = h1_latest_bos.direction if h1_latest_bos else None

    # ── Fibonacci premium/discount (informational) ────────────────────────────
    ms_d1 = d1["ms"]
    fib_range: Optional[premium_discount.FibRange] = None
    pzone = "UNKNOWN"

    if ms_d1.last_swing_high and ms_d1.last_swing_low:
        fib_range = premium_discount.compute_fib_range(
            ms_d1.last_swing_high, ms_d1.last_swing_low
        )
        if fib_range:
            pzone = premium_discount.price_zone(fib_range, current_price)

    # ── Playbook dispatch (priority order) ────────────────────────────────────
    pb = playbooks.setup1_sweep_fvg_continuation(
        d1, h1, m5, df_m5, symbol, current_price
    )
    if not pb.matched:
        pb = playbooks.setup2_htf_ob_reversal(
            d1, h1, m5, df_m5, symbol, current_price
        )
    if not pb.matched:
        pb = playbooks.setup3_london_killzone(
            h1, m5, df_h1, df_m5, symbol, current_price
        )
    if not pb.matched:
        pb = playbooks.setup4_ny_killzone(
            h1, m5, df_h1, df_m5, symbol, current_price
        )

    if not pb.matched:
        return MTFAnalysis(
            symbol=symbol,
            d1_trend=d1_trend,
            h1_trend=h1_trend,
            h1_bos_direction=h1_bos_dir,
            m5_sweep=None,
            m5_fvg=None,
            h1_ob=None,
            h1_fvg=None,
            fib_range=fib_range,
            price_zone=pzone,
            signal_direction=None,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
            confluence_score=0.0,
            setup_name=None,
            valid=False,
            reasons=["No playbook matched"],
        )

    # ── Validity gate ─────────────────────────────────────────────────────────
    valid = (
        pb.confluence_score >= _SC_MIN_VALID
        and pb.entry_price is not None
        and pb.stop_loss is not None
        and pb.take_profit is not None
    )
    score = max(0.0, min(1.0, pb.confluence_score))

    return MTFAnalysis(
        symbol=symbol,
        d1_trend=d1_trend,
        h1_trend=h1_trend,
        h1_bos_direction=h1_bos_dir,
        m5_sweep=pb.m5_sweep,
        m5_fvg=pb.m5_fvg,
        h1_ob=pb.h1_ob,
        h1_fvg=pb.h1_fvg,
        fib_range=fib_range,
        price_zone=pzone,
        signal_direction=pb.direction,
        entry_price=pb.entry_price,
        stop_loss=pb.stop_loss,
        take_profit=pb.take_profit,
        confluence_score=score,
        setup_name=pb.name,
        valid=valid,
        reasons=pb.reasons,
    )
