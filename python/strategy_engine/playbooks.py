"""
Strategy Engine – ICT Trading Playbooks

Three focused setups the system is authorised to trade:

  Setup 1 – SWEEP_FVG_CONTINUATION
      M5 liquidity sweep → displacement candle → M5 FVG in the same direction.
      D1 / H1 trend alignment required.

  Setup 2 – HTF_OB_REVERSAL
      Price returns into an active D1 Order Block.
      H1 liquidity sweep confirms the reversal direction.

  Setup 3 – LONDON_KILLZONE_EXPANSION
      Active only during London Killzone (default 07-10 UTC).
      Recent H1 BOS establishes direction; M5 FVG gives entry.

  Setup 4 – NY_KILLZONE_EXPANSION
      Active only during New York Killzone (default 13-22 UTC).
      Recent H1 BOS establishes direction; M5 FVG gives entry.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from python.config import SCORING, STRATEGY
from python.strategy_engine import (
    bos_detector,
    fvg_detector,
    liquidity_engine,
    order_block_detector,
)
from python.strategy_engine.market_structure import TrendDirection
from python.strategy_engine.util import atr_value as _atr_value

logger = logging.getLogger(__name__)

# ── Scoring weights (shared with config where keys match) ─────────────────────
_SC_H1_OB: float  = SCORING.get("h1_ob",  0.20)
_SC_H1_FVG: float = SCORING.get("h1_fvg", 0.15)

# ── Parameters ────────────────────────────────────────────────────────────────
_SWEEP_RECENCY: int      = STRATEGY.get("sweep_recency", 10)
_LKZ_OPEN: int           = STRATEGY.get("london_killzone_open",  7)
_LKZ_CLOSE: int          = STRATEGY.get("london_killzone_close", 10)
_KILLZONE_BOS_LOOKBACK: int = STRATEGY.get("killzone_bos_lookback", 3)
_NYZ_OPEN: int           = STRATEGY.get("ny_killzone_open",  13)
_NYZ_CLOSE: int          = STRATEGY.get("ny_killzone_close", 22)


@dataclass
class PlaybookResult:
    """Return value of every playbook evaluator function."""

    matched: bool
    name: str = ""
    direction: Optional[str] = None     # "BUY" | "SELL"
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    confluence_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    # Component references forwarded to MTFAnalysis for downstream use
    m5_sweep: Optional[Any] = None
    m5_fvg: Optional[Any] = None
    h1_ob: Optional[Any] = None
    h1_fvg: Optional[Any] = None


def _no_match() -> PlaybookResult:
    return PlaybookResult(matched=False)


# ── Setup 1 ───────────────────────────────────────────────────────────────────

def setup1_sweep_fvg_continuation(
    d1_tf: dict,
    h1_tf: dict,
    m5_tf: dict,
    df_m5: pd.DataFrame,
    symbol: str,
    current_price: float,
) -> PlaybookResult:
    """
    Setup 1 – SWEEP_FVG_CONTINUATION

    Sequence: D1/H1 trends align → recent M5 liquidity sweep → M5 FVG formed
    *after* the sweep candle (displacement→imbalance) → enter at the FVG.

    Required
    --------
    • D1 and H1 trends are both bullish or both bearish.
    • A matching M5 liquidity sweep within the last _SWEEP_RECENCY candles.
    • An unfilled M5 FVG (matching direction) with its index > sweep.index.

    Bonuses
    -------
    • Active H1 OB near current price: +h1_ob weight
    • Active H1 FVG near current price: +h1_fvg weight
    """
    d1_trend: TrendDirection = d1_tf["ms"].trend
    h1_trend: TrendDirection = h1_tf["ms"].trend

    if d1_trend == TrendDirection.BULLISH and h1_trend == TrendDirection.BULLISH:
        direction, sweep_dir, fvg_dir = "BUY", "BULLISH_SWEEP", "BULLISH"
    elif d1_trend == TrendDirection.BEARISH and h1_trend == TrendDirection.BEARISH:
        direction, sweep_dir, fvg_dir = "SELL", "BEARISH_SWEEP", "BEARISH"
    else:
        return _no_match()

    reasons: list[str] = [f"D1 {d1_trend.value} + H1 {h1_trend.value} trend aligned"]
    score = 0.40  # base for required trend alignment

    # ── Recent M5 sweep ───────────────────────────────────────────────────────
    n_m5 = len(df_m5)
    m5_sweep = liquidity_engine.latest_sweep(m5_tf["sweeps"], sweep_dir)
    if m5_sweep is None or (n_m5 - 1 - m5_sweep.index) > _SWEEP_RECENCY:
        return _no_match()

    reasons.append(f"M5 {sweep_dir} at candle {m5_sweep.index}")
    score += 0.20

    # ── M5 FVG formed AFTER the sweep ─────────────────────────────────────────
    post_sweep_fvgs = [
        z for z in fvg_detector.active_fvg(m5_tf["fvg"], fvg_dir)
        if z.index > m5_sweep.index
    ]
    if not post_sweep_fvgs:
        return _no_match()

    m5_fvg = min(post_sweep_fvgs, key=lambda z: abs(z.midpoint - current_price))
    reasons.append(f"M5 FVG {m5_fvg.gap_low:.5f}-{m5_fvg.gap_high:.5f} post-sweep")
    score += 0.20

    # ── Optional bonuses ──────────────────────────────────────────────────────
    h1_ob = order_block_detector.nearest_ob(h1_tf["obs"], current_price, direction)
    if h1_ob:
        score += _SC_H1_OB
        reasons.append(f"H1 OB {h1_ob.ob_low:.5f}-{h1_ob.ob_high:.5f}")

    h1_fvg = fvg_detector.nearest_fvg(h1_tf["fvg"], current_price, fvg_dir)
    if h1_fvg:
        score += _SC_H1_FVG
        reasons.append(f"H1 FVG {h1_fvg.gap_low:.5f}-{h1_fvg.gap_high:.5f}")

    # ── Entry / SL / TP ───────────────────────────────────────────────────────
    atr = _atr_value(df_m5)
    if atr != atr:  # NaN guard
        return _no_match()

    entry = m5_fvg.midpoint
    if direction == "BUY":
        sl = m5_fvg.gap_low - atr
        tp = entry + (entry - sl) * 2.0
    else:
        sl = m5_fvg.gap_high + atr
        tp = entry - (sl - entry) * 2.0

    return PlaybookResult(
        matched=True,
        name="SWEEP_FVG_CONTINUATION",
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        confluence_score=min(1.0, score),
        reasons=reasons,
        m5_sweep=m5_sweep,
        m5_fvg=m5_fvg,
        h1_ob=h1_ob,
        h1_fvg=h1_fvg,
    )


# ── Setup 2 ───────────────────────────────────────────────────────────────────

def setup2_htf_ob_reversal(
    d1_tf: dict,
    h1_tf: dict,
    m5_tf: dict,
    df_m5: pd.DataFrame,
    symbol: str,
    current_price: float,
) -> PlaybookResult:
    """
    Setup 2 – HTF_OB_REVERSAL

    Sequence: Price returns into an active D1 Order Block (the high-timeframe
    anchor) → H1 liquidity sweep confirms the reversal → M5 FVG for precision
    entry (optional).

    Required
    --------
    • An active D1 OB whose zone contains or is within 2×ATR(M5) of current price.
    • A matching H1 liquidity sweep (confirming the reversal direction).

    Bonuses
    -------
    • Recent M5 sweep (last _SWEEP_RECENCY candles): +0.15
    • Active M5 FVG for precision entry: +0.15
    • D1 trend aligns with the OB direction: +0.10
    """
    atr_m5 = _atr_value(df_m5)
    if atr_m5 != atr_m5:  # NaN guard
        return _no_match()

    # ── Find D1 OB with price at/near its zone ────────────────────────────────
    d1_ob: Optional[Any] = None
    direction: Optional[str] = None

    for ob in order_block_detector.active_obs(d1_tf["obs"], "BULLISH"):
        if ob.ob_low <= current_price <= ob.ob_high + 2 * atr_m5:
            d1_ob, direction = ob, "BUY"
            break

    if d1_ob is None:
        for ob in order_block_detector.active_obs(d1_tf["obs"], "BEARISH"):
            if ob.ob_low - 2 * atr_m5 <= current_price <= ob.ob_high:
                d1_ob, direction = ob, "SELL"
                break

    if d1_ob is None:
        return _no_match()

    reasons: list[str] = [f"D1 {direction} OB {d1_ob.ob_low:.5f}-{d1_ob.ob_high:.5f}"]
    score = 0.40  # base for required D1 OB presence

    # ── H1 sweep confirming the reversal ──────────────────────────────────────
    h1_sweep_dir = "BULLISH_SWEEP" if direction == "BUY" else "BEARISH_SWEEP"
    h1_sweep = liquidity_engine.latest_sweep(h1_tf["sweeps"], h1_sweep_dir)
    if h1_sweep is None:
        return _no_match()

    reasons.append(f"H1 {h1_sweep_dir} confirms OB reversal")
    score += 0.20

    # ── Optional bonuses ──────────────────────────────────────────────────────
    n_m5 = len(df_m5)
    m5_sweep_dir = h1_sweep_dir
    m5_sweep = liquidity_engine.latest_sweep(m5_tf["sweeps"], m5_sweep_dir)
    if m5_sweep and (n_m5 - 1 - m5_sweep.index) <= _SWEEP_RECENCY:
        score += 0.15
        reasons.append("M5 sweep corroborates H1 reversal")
    else:
        m5_sweep = None

    fvg_dir = "BULLISH" if direction == "BUY" else "BEARISH"
    m5_fvg = fvg_detector.nearest_fvg(m5_tf["fvg"], current_price, fvg_dir)
    if m5_fvg:
        score += 0.15
        reasons.append(f"M5 FVG entry {m5_fvg.gap_low:.5f}-{m5_fvg.gap_high:.5f}")

    d1_trend: TrendDirection = d1_tf["ms"].trend
    if (direction == "BUY" and d1_trend == TrendDirection.BULLISH) or (
        direction == "SELL" and d1_trend == TrendDirection.BEARISH
    ):
        score += 0.10
        reasons.append(f"D1 {d1_trend.value} trend supports OB direction")

    # ── Entry / SL / TP ───────────────────────────────────────────────────────
    if m5_fvg:
        entry = m5_fvg.midpoint
    elif direction == "BUY":
        entry = d1_ob.ob_high
    else:
        entry = d1_ob.ob_low

    if direction == "BUY":
        sl = d1_ob.ob_low - 0.5 * atr_m5
        tp = entry + (entry - sl) * 2.0
    else:
        sl = d1_ob.ob_high + 0.5 * atr_m5
        tp = entry - (sl - entry) * 2.0

    return PlaybookResult(
        matched=True,
        name="HTF_OB_REVERSAL",
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        confluence_score=min(1.0, score),
        reasons=reasons,
        m5_sweep=m5_sweep,
        m5_fvg=m5_fvg,
        h1_ob=None,   # D1 OB is the anchor; H1 OB not a component here
        h1_fvg=None,
    )


# ── Setup 3 ───────────────────────────────────────────────────────────────────

def setup3_london_killzone(
    h1_tf: dict,
    m5_tf: dict,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: str,
    current_price: float,
    _now_hour: Optional[int] = None,
) -> PlaybookResult:
    """
    Setup 3 – LONDON_KILLZONE_EXPANSION

    Sequence: London Killzone opens → H1 BOS establishes expansion direction →
    M5 FVG (imbalance / displacement) provides the entry trigger.

    Required
    --------
    • Current UTC hour is within [london_killzone_open, london_killzone_close).
    • A recent H1 BOS (within _KILLZONE_BOS_LOOKBACK candles of the end of df_h1).
    • An active M5 FVG in the BOS direction.

    Bonuses
    -------
    • Recent M5 sweep in the BOS direction: +0.15
    • Active H1 OB near current price: +0.10

    Parameters
    ----------
    _now_hour
        Override for the current UTC hour (for testing only). When ``None``
        the real clock is used.
    """
    now_h = _now_hour if _now_hour is not None else datetime.now(timezone.utc).hour
    if not (_LKZ_OPEN <= now_h < _LKZ_CLOSE):
        return _no_match()

    reasons: list[str] = [f"London Killzone active ({now_h:02d}:00 UTC)"]
    score = 0.30  # base for killzone time filter

    # ── Recent H1 BOS ─────────────────────────────────────────────────────────
    latest_h1_bos = bos_detector.latest_bos(h1_tf["bos"])
    if latest_h1_bos is None:
        return _no_match()

    n_h1 = len(df_h1)
    if (n_h1 - 1 - latest_h1_bos.index) > _KILLZONE_BOS_LOOKBACK:
        return _no_match()

    direction = "BUY" if latest_h1_bos.direction == "BULLISH" else "SELL"
    fvg_dir = latest_h1_bos.direction  # "BULLISH" | "BEARISH"
    reasons.append(f"H1 {latest_h1_bos.direction} BOS → {direction}")
    score += 0.30

    # ── M5 FVG for entry ──────────────────────────────────────────────────────
    m5_fvg = fvg_detector.nearest_fvg(m5_tf["fvg"], current_price, fvg_dir)
    if m5_fvg is None:
        return _no_match()

    reasons.append(f"M5 FVG imbalance {m5_fvg.gap_low:.5f}-{m5_fvg.gap_high:.5f}")
    score += 0.20

    # ── Optional bonuses ──────────────────────────────────────────────────────
    sweep_dir = "BULLISH_SWEEP" if direction == "BUY" else "BEARISH_SWEEP"
    n_m5 = len(df_m5)
    m5_sweep = liquidity_engine.latest_sweep(m5_tf["sweeps"], sweep_dir)
    if m5_sweep and (n_m5 - 1 - m5_sweep.index) <= _SWEEP_RECENCY:
        score += 0.15
        reasons.append("M5 sweep aligns with killzone expansion")
    else:
        m5_sweep = None

    h1_ob = order_block_detector.nearest_ob(h1_tf["obs"], current_price, direction)
    if h1_ob:
        score += 0.10
        reasons.append(f"H1 OB {h1_ob.ob_low:.5f}-{h1_ob.ob_high:.5f} supports entry")

    # ── Entry / SL / TP ───────────────────────────────────────────────────────
    atr = _atr_value(df_m5)
    if atr != atr:  # NaN guard
        return _no_match()

    entry = m5_fvg.midpoint
    if direction == "BUY":
        sl = m5_fvg.gap_low - atr
        tp = entry + (entry - sl) * 2.0
    else:
        sl = m5_fvg.gap_high + atr
        tp = entry - (sl - entry) * 2.0

    return PlaybookResult(
        matched=True,
        name="LONDON_KILLZONE_EXPANSION",
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        confluence_score=min(1.0, score),
        reasons=reasons,
        m5_sweep=m5_sweep,
        m5_fvg=m5_fvg,
        h1_ob=h1_ob,
        h1_fvg=None,
    )


# ── Setup 4 ───────────────────────────────────────────────────────────────────

def setup4_ny_killzone(
    h1_tf: dict,
    m5_tf: dict,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: str,
    current_price: float,
    _now_hour: Optional[int] = None,
) -> PlaybookResult:
    """
    Setup 4 – NY_KILLZONE_EXPANSION

    Sequence: New York Killzone opens → H1 BOS establishes expansion direction →
    M5 FVG (imbalance / displacement) provides the entry trigger.

    Required
    --------
    • Current UTC hour is within [ny_killzone_open, ny_killzone_close).
    • A recent H1 BOS (within _KILLZONE_BOS_LOOKBACK candles of the end of df_h1).
    • An active M5 FVG in the BOS direction.

    Bonuses
    -------
    • Recent M5 sweep in the BOS direction: +0.15
    • Active H1 OB near current price: +0.10

    Parameters
    ----------
    _now_hour
        Override for the current UTC hour (for testing only). When ``None``
        the real clock is used.
    """
    now_h = _now_hour if _now_hour is not None else datetime.now(timezone.utc).hour
    if not (_NYZ_OPEN <= now_h < _NYZ_CLOSE):
        return _no_match()

    reasons: list[str] = [f"NY Killzone active ({now_h:02d}:00 UTC)"]
    score = 0.30  # base for killzone time filter

    # ── Recent H1 BOS ─────────────────────────────────────────────────────────
    latest_h1_bos = bos_detector.latest_bos(h1_tf["bos"])
    if latest_h1_bos is None:
        return _no_match()

    n_h1 = len(df_h1)
    if (n_h1 - 1 - latest_h1_bos.index) > _KILLZONE_BOS_LOOKBACK:
        return _no_match()

    direction = "BUY" if latest_h1_bos.direction == "BULLISH" else "SELL"
    fvg_dir = latest_h1_bos.direction  # "BULLISH" | "BEARISH"
    reasons.append(f"H1 {latest_h1_bos.direction} BOS → {direction}")
    score += 0.30

    # ── M5 FVG for entry ──────────────────────────────────────────────────────
    m5_fvg = fvg_detector.nearest_fvg(m5_tf["fvg"], current_price, fvg_dir)
    if m5_fvg is None:
        return _no_match()

    reasons.append(f"M5 FVG imbalance {m5_fvg.gap_low:.5f}-{m5_fvg.gap_high:.5f}")
    score += 0.20

    # ── Optional bonuses ──────────────────────────────────────────────────────
    sweep_dir = "BULLISH_SWEEP" if direction == "BUY" else "BEARISH_SWEEP"
    n_m5 = len(df_m5)
    m5_sweep = liquidity_engine.latest_sweep(m5_tf["sweeps"], sweep_dir)
    if m5_sweep and (n_m5 - 1 - m5_sweep.index) <= _SWEEP_RECENCY:
        score += 0.15
        reasons.append("M5 sweep aligns with NY killzone expansion")
    else:
        m5_sweep = None

    h1_ob = order_block_detector.nearest_ob(h1_tf["obs"], current_price, direction)
    if h1_ob:
        score += 0.10
        reasons.append(f"H1 OB {h1_ob.ob_low:.5f}-{h1_ob.ob_high:.5f} supports entry")

    # ── Entry / SL / TP ───────────────────────────────────────────────────────
    atr = _atr_value(df_m5)
    if atr != atr:  # NaN guard
        return _no_match()

    entry = m5_fvg.midpoint
    if direction == "BUY":
        sl = m5_fvg.gap_low - atr
        tp = entry + (entry - sl) * 2.0
    else:
        sl = m5_fvg.gap_high + atr
        tp = entry - (sl - entry) * 2.0

    return PlaybookResult(
        matched=True,
        name="NY_KILLZONE_EXPANSION",
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        confluence_score=min(1.0, score),
        reasons=reasons,
        m5_sweep=m5_sweep,
        m5_fvg=m5_fvg,
        h1_ob=h1_ob,
        h1_fvg=None,
    )
