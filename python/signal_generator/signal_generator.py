"""
Signal Generation Engine

Produces structured trade signal dicts when ALL ICT confluence conditions align.

Output format:
{
  "signal_id": "EURUSD-20260525-083000-ICT_FVG_OB_SWEEP",
  "created_at": "2026-05-25T08:30:00Z",
  "expires_at": "2026-05-25T08:35:00Z",
  "valid_from": "2026-05-25T08:30:00Z",
  "engine_cycle_id": "cycle-000381",
  "status": "ACTIVE",
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
  "payload_hash": "sha256:...",
  "timestamp": "2024-01-15T10:30:00+00:00"
}
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Optional

from python.config import INTEGRATION, RISK, SIGNAL_CFG, signal_path_for
from python.config import pip_size_for
from python.exceptions import SignalValidationError
from python.strategy_engine.mtf_engine import MTFAnalysis

logger = logging.getLogger(__name__)

_MIN_CONFIDENCE = SIGNAL_CFG.get("min_confidence", 0.65)
_MIN_RR = SIGNAL_CFG.get("min_risk_reward", 2.0)
_SPREAD_PIPS = SIGNAL_CFG.get("spread_pips", 1.0)
_RISK_PERCENT = RISK.get("risk_percent", 1.0)

_SAFE_SYMBOL_RE = re.compile(r'[^A-Z0-9]')


def _validated_symbol(symbol: str) -> str:
    """
    Return a filesystem-safe version of *symbol* by stripping every character
    that is not an uppercase ASCII letter or digit.

    This prevents path traversal when the symbol value is used to construct a
    file path (e.g. ``signals/active/EURUSD.json``).  Characters like ``/``,
    ``..``, whitespace, and null bytes are removed before the value is used.

    Raises ``SignalValidationError`` if the cleaned result is empty or too long
    (> 20 characters), which would indicate an invalid or spoofed symbol.
    """
    clean = _SAFE_SYMBOL_RE.sub("", symbol.strip().upper())
    if not clean or len(clean) > 20:
        raise SignalValidationError(
            f"Symbol {symbol!r} is not a valid trading symbol (must be 1-20 "
            "alphanumeric characters after stripping unsafe chars).",
            field="symbol",
        )
    return clean


def _safe_signal_path(symbol: str) -> Path:
    """
    Return the filesystem path for *symbol*'s active signal file.

    To avoid any path-injection risk, the return value is built exclusively
    from pre-configured values in ``SYMBOLS`` (loaded from config.yaml), not
    from the caller-supplied string.  The caller's *symbol* is only used as a
    dictionary look-up key; the dict values come from the application config.

    Raises ``SignalValidationError`` if *symbol* (after stripping unsafe
    characters) is not present in the configured symbols list.
    """
    from python.config import SIGNAL_ACTIVE_DIR, SYMBOLS

    clean = _validated_symbol(symbol)
    # Build the map from config (not from user input) and look up.
    # The returned path comes from config values, breaking the taint chain.
    symbol_map: dict[str, Path] = {
        s.upper(): SIGNAL_ACTIVE_DIR / f"{s.upper()}.json"
        for s in SYMBOLS
    }
    path = symbol_map.get(clean)
    if path is None:
        raise SignalValidationError(
            f"Symbol {symbol!r} is not in the configured symbols list.",
            field="symbol",
        )
    return path


def _signal_ttl() -> int:
    """Return signal TTL in seconds from integration config."""
    return int(INTEGRATION.get("signal_ttl_seconds", 300))


def _setup_type(analysis: MTFAnalysis) -> str:
    """Return the ICT setup label, preferring the named playbook when available."""
    if analysis.setup_name:
        return f"ICT_{analysis.setup_name}"
    # Fallback: derive from active components (backward compatibility)
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


def generate_signal(analysis: MTFAnalysis, cycle_id: str = "") -> Optional[dict]:
    """
    Convert an MTFAnalysis into a trade signal dict.
    Returns None if conditions are not met.

    Parameters
    ----------
    analysis:
        Result of multi-timeframe ICT analysis.
    cycle_id:
        Opaque identifier for the engine cycle that produced this signal.
        Used for traceability (e.g. "cycle-000381").
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

    now_utc = datetime.now(timezone.utc)
    setup_type = _setup_type(analysis)
    # Build a deterministic, human-readable signal ID
    signal_id = (
        f"{analysis.symbol}"
        f"-{now_utc.strftime('%Y%m%d-%H%M%S')}"
        f"-{setup_type}"
    )
    ttl = _signal_ttl()
    expires_at = now_utc + timedelta(seconds=ttl)

    signal = {
        "signal_id": signal_id,
        "created_at": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expires_at": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "valid_from": now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "engine_cycle_id": cycle_id,
        "status": "ACTIVE",
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
        "setup_type": setup_type,
        "confidence_score": round(analysis.confluence_score, 4),
        "price_zone": analysis.price_zone,
        "d1_trend": str(analysis.d1_trend.value),
        "h1_bos": analysis.h1_bos_direction,
        "reasons": analysis.reasons,
        "timestamp": now_utc.isoformat(),
    }
    signal["payload_hash"] = _compute_payload_hash(signal)

    logger.info(
        "Signal generated: %s %s @ %.5f  SL=%.5f  TP=%.5f  adjRR=%.2f  conf=%.2f  id=%s",
        signal["symbol"],
        signal["direction"],
        signal["entry_price"],
        signal["stop_loss"],
        signal["take_profit"],
        signal["risk_reward"],
        signal["confidence_score"],
        signal["signal_id"],
    )
    return signal


def _compute_payload_hash(signal: dict) -> str:
    """
    Compute a SHA-256 hash over all signal fields except ``payload_hash``
    itself.  Keys are sorted for determinism.

    Returns a string like ``"sha256:abcdef1234..."``.
    """
    payload = {k: v for k, v in signal.items() if k != "payload_hash"}
    serialised = json.dumps(payload, sort_keys=True, default=str).encode()
    digest = hashlib.sha256(serialised).hexdigest()
    return f"sha256:{digest}"


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
    Persist *signal* to ``signals/active/{symbol}.json`` atomically.

    The write uses a temporary file in the same directory followed by an
    ``os.replace()`` rename so the EA never reads a partially-written file.

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

    final_path = _safe_signal_path(signal["symbol"])
    sanitized = _sanitize_signal(signal)
    payload = json.dumps(sanitized, indent=2)

    # Atomic write: temp file → fsync → rename
    dir_path = final_path.parent
    dir_path.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".tmp.json")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, final_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    logger.debug("Signal saved atomically to %s", final_path)


def load_latest_signal(symbol: str) -> Optional[dict]:
    """
    Read the active signal for *symbol* from disk.

    Returns ``None`` when:
    - the file does not exist,
    - the file cannot be parsed,
    - the signal's ``expires_at`` timestamp has passed (TTL expired).

    Parameters
    ----------
    symbol:
        The trading symbol whose signal file should be read
        (e.g. ``"EURUSD"``).
    """
    path = _safe_signal_path(symbol)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # Prefer the explicit expires_at field; fall back to timestamp + TTL
    expires_at_str = data.get("expires_at")
    if expires_at_str:
        try:
            expires_at = datetime.fromisoformat(
                expires_at_str.replace("Z", "+00:00")
            )
            if datetime.now(timezone.utc) > expires_at:
                logger.debug(
                    "Signal for %s expired at %s", symbol, expires_at_str
                )
                return None
        except ValueError:
            pass
    else:
        # Legacy fallback: check age against TTL
        ttl = INTEGRATION.get("signal_ttl_seconds", 300)
        ts = data.get("timestamp")
        if ts:
            try:
                signal_time = datetime.fromisoformat(ts)
                age = (datetime.now(timezone.utc) - signal_time).total_seconds()
                if age > ttl:
                    logger.debug(
                        "Signal for %s expired (age %.0fs > TTL %ds)", symbol, age, ttl
                    )
                    return None
            except ValueError:
                pass

    return data
