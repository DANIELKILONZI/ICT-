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
from datetime import datetime, timezone
from typing import Optional

from python.config import SIGNAL_CFG, SIGNAL_OUTPUT_PATH, RISK
from python.strategy_engine.mtf_engine import MTFAnalysis

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE = SIGNAL_CFG.get("min_confidence", 0.65)
_MIN_RR = SIGNAL_CFG.get("min_risk_reward", 2.0)
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

    rr = _risk_reward(entry, sl, tp)
    if rr < _MIN_RR:
        logger.debug(
            "Signal rejected for %s – R:R %.2f < %.2f", analysis.symbol, rr, _MIN_RR
        )
        return None

    signal = {
        "symbol": analysis.symbol,
        "direction": analysis.signal_direction,
        "entry_type": "LIMIT",
        "entry_price": round(entry, 5),
        "stop_loss": round(sl, 5),
        "take_profit": round(tp, 5),
        "risk_reward": round(rr, 2),
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
        "Signal generated: %s %s @ %.5f  SL=%.5f  TP=%.5f  conf=%.2f",
        signal["symbol"],
        signal["direction"],
        signal["entry_price"],
        signal["stop_loss"],
        signal["take_profit"],
        signal["confidence_score"],
    )
    return signal


def save_signal(signal: dict) -> None:
    """Persist the latest signal to the configured JSON file."""
    SIGNAL_OUTPUT_PATH.write_text(json.dumps(signal, indent=2))
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
