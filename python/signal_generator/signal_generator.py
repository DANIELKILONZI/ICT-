"""
Signal Generation Engine

Produces structured trade signal dicts when ALL ICT confluence conditions align.

Output format:
{
  "symbol": "EURUSD",
  "direction": "BUY",
  "entry_type": "LIMIT",
  "entry_price": 1.08500,
  "stop_loss": 1.08300,
  "take_profit": 1.09000,
  "risk_percent": 1.0,
  "timeframe_alignment": "D1-H1-M5",
  "setup_type": "ICT_FVG_OB",
  "confidence_score": 0.82,
  "timestamp": "2024-01-15T10:30:00+00:00"
}
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from python.config import SIGNAL_CFG, SIGNAL_OUTPUT_PATH, RISK
from python.config import pip_size_for
from python.exceptions import SignalValidationError
from python.strategy_engine.mtf_engine import MTFAnalysis

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE = SIGNAL_CFG.get("min_confidence", 0.65)
_MIN_RR = SIGNAL_CFG.get("min_risk_reward", 2.0)
_SPREAD_PIPS = SIGNAL_CFG.get("spread_pips", 1.0)
_RISK_PERCENT = RISK.get("risk_percent", 1.0)


def _setup_type(analysis: MTFAnalysis) -> str:
    """Determine the ICT setup label from active components."""
    parts = []
    if analysis.h1_fvg:
        parts.append("FVG")
    if analysis.h1_ob:
        parts.append("OB")
    if analysis.m5_sweep:
        parts.append("SWEEP")
    if analysis.m5_fvg:
        parts.append("M5FVG")
    return "ICT_" + "_".join(parts) if parts else "ICT_SETUP"


def _risk_reward(entry: float, sl: float, tp: float) -> float:
    risk = abs(entry - sl)
    reward = abs(tp - entry)
    return reward / risk if risk > 0 else 0.0


def _spread_adjusted_rr(
    entry: float,
    sl: float,
    tp: float,
    direction: str,
    symbol: str,
    spread_pips: float,
) -> tuple[float, float, float, float]:
    """
    Return (adj_entry, adj_sl, adj_tp, adj_rr) after accounting for spread cost.

    For a BUY trade the broker fills at *Ask = mid + half-spread*, so the
    effective entry is higher by one spread.  The SL is also widened by one
    spread because a stop is triggered at the *Bid*.  TP is unaffected in
    price terms but the effective reward is reduced by the spread paid at
    entry.

    For a SELL trade the effective entry is lower by one spread (filled at
    Bid), the SL is widened by one spread (triggered at Ask), and again the
    reward is reduced by the spread.

    We use the full spread (not half) as a conservative worst-case.
    """
    pip = pip_size_for(symbol)
    spread_price = spread_pips * pip

    if direction == "BUY":
        adj_entry = entry + spread_price   # filled at ask
        adj_sl    = sl    - spread_price   # SL triggered at bid → distance widens
        adj_tp    = tp                     # TP filled at bid (no change needed for distance calc)
    else:  # SELL
        adj_entry = entry - spread_price   # filled at bid
        adj_sl    = sl    + spread_price   # SL triggered at ask → distance widens
        adj_tp    = tp

    adj_rr = _risk_reward(adj_entry, adj_sl, adj_tp)
    return adj_entry, adj_sl, adj_tp, adj_rr


def generate_signal(analysis: MTFAnalysis) -> Optional[dict]:
    """
    Convert an MTFAnalysis into a trade signal dict.
    Returns None if conditions are not met.
    """
    if not analysis.valid:
        logger.debug(
            "Signal rejected for %s – not valid. Reasons: %s",
            analysis.symbol,
            analysis.reasons,
        )
        return None

    if analysis.confluence_score < _MIN_CONFIDENCE:
        logger.debug(
            "Signal rejected for %s – confidence %.2f < %.2f",
            analysis.symbol,
            analysis.confluence_score,
            _MIN_CONFIDENCE,
        )
        return None

    entry = analysis.entry_price
    sl = analysis.stop_loss
    tp = analysis.take_profit

    if entry is None or sl is None or tp is None:
        logger.debug("Signal rejected for %s – missing price levels", analysis.symbol)
        return None

    # Compute spread-adjusted R:R.  The unadjusted R:R is stored for reference
    # while the adjusted figure is used for the minimum threshold check.
    raw_rr = _risk_reward(entry, sl, tp)
    adj_entry, adj_sl, adj_tp, adj_rr = _spread_adjusted_rr(
        entry, sl, tp,
        direction=analysis.signal_direction or "",
        symbol=analysis.symbol,
        spread_pips=_SPREAD_PIPS,
    )

    if adj_rr < _MIN_RR:
        logger.debug(
            "Signal rejected for %s – spread-adjusted R:R %.2f < %.2f",
            analysis.symbol,
            adj_rr,
            _MIN_RR,
        )
        return None

    signal = {
        "symbol": analysis.symbol,
        "direction": analysis.signal_direction,
        "entry_type": "LIMIT",
        "entry_price": round(adj_entry, 5),
        "stop_loss": round(adj_sl, 5),
        "take_profit": round(adj_tp, 5),
        "risk_reward": round(adj_rr, 2),
        "risk_reward_raw": round(raw_rr, 2),
        "spread_pips": _SPREAD_PIPS,
        "risk_percent": _RISK_PERCENT,
        "timeframe_alignment": "D1-H1-M5",
        "setup_type": _setup_type(analysis),
        "confidence_score": round(analysis.confluence_score, 4),
        "price_zone": analysis.price_zone,
        "d1_trend": str(analysis.d1_trend.value),
        "h1_bos": analysis.h1_bos_direction,
        "reasons": analysis.reasons,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info(
        "Signal generated: %s %s @ %.5f  SL=%.5f  TP=%.5f  adjRR=%.2f  conf=%.2f",
        signal["symbol"],
        signal["direction"],
        signal["entry_price"],
        signal["stop_loss"],
        signal["take_profit"],
        signal["risk_reward"],
        signal["confidence_score"],
    )
    return signal


def _sanitize_signal(signal: dict) -> dict:
    """
    Sanitize all string values in *signal* so they are safe for the MQL5
    simple regex-based JSON parser.

    The MQL5 parser (``ExtractJsonString``) finds the first ``"`` after a key
    name and reads until the next ``"``.  It cannot handle:
      - escaped double-quotes  (``\"``)  inside a value
      - backslashes            (``\\``)  inside a value
      - bare control characters (CR, LF, TAB)

    We strip those characters from string fields.  Numeric and boolean fields
    are not affected.  The ``reasons`` list (not parsed by MQL5) is left as-is.
    """
    _unsafe = re.compile(r'["\\\r\n\t]')
    sanitized = {}
    for k, v in signal.items():
        if isinstance(v, str):
            sanitized[k] = _unsafe.sub("", v)
        else:
            sanitized[k] = v
    return sanitized


def save_signal(signal: dict) -> None:
    """
    Persist the latest signal to the configured JSON file.

    Raises
    ------
    SignalValidationError
        When *signal* is missing a required field.
    """
    required = {"symbol", "direction", "entry_price", "stop_loss", "take_profit"}
    for field in required:
        if field not in signal:
            raise SignalValidationError(
                f"Signal is missing required field {field!r} before save.",
                field=field,
            )
    SIGNAL_OUTPUT_PATH.write_text(json.dumps(_sanitize_signal(signal), indent=2))
    logger.debug("Signal saved to %s", SIGNAL_OUTPUT_PATH)


def load_latest_signal() -> Optional[dict]:
    """Read the latest signal from disk. Returns None if missing or expired."""
    import time

    if not SIGNAL_OUTPUT_PATH.exists():
        return None
    try:
        data = json.loads(SIGNAL_OUTPUT_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # Check TTL
    from python.config import INTEGRATION

    ttl = INTEGRATION.get("signal_ttl_seconds", 300)
    ts = data.get("timestamp")
    if ts:
        signal_time = datetime.fromisoformat(ts)
        age = (datetime.now(timezone.utc) - signal_time).total_seconds()
        if age > ttl:
            logger.debug("Signal expired (age %.0fs > TTL %ds)", age, ttl)
            return None

    return data
