"""
Performance Tracking Module – Three-Layer Architecture

Layer 1  logs/candidates.csv    – every setup that passed ICT hard gates
Layer 2  logs/signals.csv       – every signal published to the EA
Layer 3  logs/executions.csv    – trades the EA actually placed
         logs/rejections.csv    – signals the EA rejected and why

``logs/performance.csv`` is kept for backward compatibility and mirrors
the signals log.

The EA writes execution feedback to ``signals/feedback/{signal_id}.json``.
Call ``ingest_execution_feedback()`` (or run
``python.integration.execution_feedback.poll()``) to reconcile layers 2 and 3.
"""
from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from python.config import (
    CANDIDATES_LOG,
    EXECUTIONS_LOG,
    PERF_LOG,
    REJECTIONS_LOG,
    SIGNALS_LOG,
    TELEGRAM,
    pip_size_for,
)

logger = logging.getLogger(__name__)

_csv_lock = threading.Lock()

# ── Column schemas ────────────────────────────────────────────────────────────

_SIGNAL_HEADERS = [
    "timestamp", "signal_id", "symbol", "direction",
    "entry_price", "stop_loss", "take_profit",
    "exit_price", "result",           # WIN | LOSS | OPEN | REJECTED
    "pnl_pips", "risk_percent", "confidence_score", "setup_type",
]

# Legacy: keep identical to _SIGNAL_HEADERS so existing consumers still work.
_HEADERS = _SIGNAL_HEADERS

_CANDIDATE_HEADERS = [
    "timestamp", "symbol", "direction", "confidence_score",
    "setup_type", "reasons",
]

_EXECUTION_HEADERS = [
    "timestamp", "signal_id", "symbol", "direction",
    "lots", "entry_price", "stop_loss", "take_profit",
    "result",           # EXECUTED | REJECTED
    "pnl_pips", "reason",
]

_REJECTION_HEADERS = [
    "timestamp", "signal_id", "symbol", "direction",
    "reason", "spread_pips", "detail",
]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _ensure(path: Path, headers: list[str]) -> None:
    """Create CSV file with header row if it does not exist yet."""
    if not path.exists():
        with open(path, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=headers).writeheader()


def _append(path: Path, headers: list[str], row: dict) -> None:
    """Thread-safe append of *row* to *path*."""
    with _csv_lock:
        _ensure(path, headers)
        with open(path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=headers).writerow(row)


# ── Layer 1 – Candidate setups ───────────────────────────────────────────────

def log_candidate(
    symbol: str,
    direction: str,
    confidence_score: float,
    setup_type: str,
    reasons: list[str] | None = None,
) -> None:
    """
    Record a setup that passed all ICT hard gates (before signal generation).

    Call this as soon as ``analyse()`` returns a valid MTFAnalysis, before
    ``generate_signal()`` applies spread / R:R / confidence filters.
    """
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "direction": direction,
        "confidence_score": round(confidence_score, 4),
        "setup_type": setup_type,
        "reasons": "|".join(reasons or []),
    }
    _append(CANDIDATES_LOG, _CANDIDATE_HEADERS, row)
    logger.debug("Candidate logged: %s %s conf=%.2f", symbol, direction, confidence_score)


# ── Layer 2 – Published signals ──────────────────────────────────────────────

def _ensure_file() -> None:
    """Backward-compat shim used by callers that pre-date the 3-layer design."""
    _ensure(PERF_LOG, _SIGNAL_HEADERS)
    _ensure(SIGNALS_LOG, _SIGNAL_HEADERS)


def log_signal(signal: dict) -> None:
    """
    Record a published signal.

    Writes to both ``signals.csv`` (canonical) and ``performance.csv``
    (backward-compat alias).
    """
    row = {h: signal.get(h, "") for h in _SIGNAL_HEADERS}
    row["timestamp"] = signal.get("timestamp", datetime.now(timezone.utc).isoformat())
    row["signal_id"] = signal.get("signal_id", "")
    row["result"] = "OPEN"

    with _csv_lock:
        for path in (SIGNALS_LOG, PERF_LOG):
            _ensure(path, _SIGNAL_HEADERS)
            with open(path, "a", newline="") as fh:
                csv.DictWriter(fh, fieldnames=_SIGNAL_HEADERS).writerow(row)

    logger.debug("Signal logged: %s", signal.get("signal_id", signal.get("symbol")))


def update_trade_result(
    symbol: str,
    entry_price: float,
    exit_price: float,
    direction: str,
) -> None:
    """
    Update the most recent OPEN trade for *symbol* with its actual exit price.
    Updates both performance.csv and signals.csv.
    """
    for path in (SIGNALS_LOG, PERF_LOG):
        _update_result_in_file(path, symbol, entry_price, exit_price, direction)


def _update_result_in_file(
    path: Path,
    symbol: str,
    entry_price: float,
    exit_price: float,
    direction: str,
) -> None:
    """Scan *path* bottom-up for the most recent OPEN trade and update it."""
    if not path.exists():
        return

    rows: list[dict] = []
    with _csv_lock:
        with open(path, "r", newline="") as fh:
            rows = list(csv.DictReader(fh))

        updated = False
        for row in reversed(rows):
            if row.get("symbol") == symbol and row.get("result") == "OPEN":
                try:
                    ep = float(row["entry_price"])
                    pip_size = pip_size_for(symbol)
                    pnl_pips = (
                        (exit_price - ep) / pip_size
                        if direction == "BUY"
                        else (ep - exit_price) / pip_size
                    )
                    row["exit_price"] = str(round(exit_price, 5))
                    row["pnl_pips"] = str(round(pnl_pips, 1))
                    row["result"] = "WIN" if pnl_pips > 0 else "LOSS"
                except (ValueError, ZeroDivisionError):
                    pass
                updated = True
                break

        if updated:
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=_SIGNAL_HEADERS)
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Trade result updated in %s for %s", path.name, symbol)


# ── Layer 3 – Execution feedback from EA ─────────────────────────────────────

def log_execution_feedback(feedback: dict) -> None:
    """
    Process a feedback dict written by the EA after executing or rejecting a signal.

    Expected fields:
        signal_id, status (EXECUTED|REJECTED), reason, spread_pips,
        timestamp, lots (optional), entry_price (optional), detail (optional)

    Writes to executions.csv or rejections.csv depending on ``status``.
    Also marks the corresponding signals.csv row as EXECUTED or REJECTED.
    """
    status = feedback.get("status", "").upper()
    signal_id = feedback.get("signal_id", "")
    symbol = feedback.get("symbol", "")
    direction = feedback.get("direction", "")
    ts = feedback.get("timestamp", datetime.now(timezone.utc).isoformat())

    if status == "EXECUTED":
        row: dict = {
            "timestamp": ts,
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "lots": feedback.get("lots", ""),
            "entry_price": feedback.get("entry_price", ""),
            "stop_loss": feedback.get("stop_loss", ""),
            "take_profit": feedback.get("take_profit", ""),
            "result": "EXECUTED",
            "pnl_pips": "",
            "reason": "",
        }
        _append(EXECUTIONS_LOG, _EXECUTION_HEADERS, row)
        logger.info("Execution confirmed: %s %s %s", signal_id, symbol, direction)

    elif status == "REJECTED":
        row = {
            "timestamp": ts,
            "signal_id": signal_id,
            "symbol": symbol,
            "direction": direction,
            "reason": feedback.get("reason", ""),
            "spread_pips": feedback.get("spread_pips", ""),
            "detail": feedback.get("detail", ""),
        }
        _append(REJECTIONS_LOG, _REJECTION_HEADERS, row)
        logger.info(
            "Signal rejected by EA: %s %s reason=%s",
            signal_id, symbol, feedback.get("reason"),
        )
        # Mark signal as REJECTED in the signals log
        _mark_signal_rejected(signal_id)

    else:
        logger.warning("Unknown feedback status %r for signal %s", status, signal_id)


def _mark_signal_rejected(signal_id: str) -> None:
    """Update signals.csv to mark the given signal_id as REJECTED."""
    for path in (SIGNALS_LOG, PERF_LOG):
        if not path.exists():
            continue
        rows: list[dict] = []
        changed = False
        with _csv_lock:
            with open(path, "r", newline="") as fh:
                rows = list(csv.DictReader(fh))
            for row in rows:
                if row.get("signal_id") == signal_id and row.get("result") == "OPEN":
                    row["result"] = "REJECTED"
                    changed = True
                    break
            if changed:
                with open(path, "w", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=_SIGNAL_HEADERS)
                    writer.writeheader()
                    writer.writerows(rows)


# ── Statistics ────────────────────────────────────────────────────────────────

def statistics() -> dict:
    """
    Compute win rate, expectancy, and max drawdown from the signals log.

    Only closed trades (WIN/LOSS) are included.  OPEN and REJECTED rows are
    excluded so that the stats represent actual closed positions.
    """
    path = SIGNALS_LOG if SIGNALS_LOG.exists() else PERF_LOG
    with _csv_lock:
        _ensure(path, _SIGNAL_HEADERS)
        with open(path, "r", newline="") as fh:
            rows = list(csv.DictReader(fh))

    rejected = sum(1 for r in rows if r.get("result") == "REJECTED")
    closed = [r for r in rows if r.get("result") in ("WIN", "LOSS")]

    if not closed:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "rejected_by_ea": rejected,
        }

    wins = sum(1 for r in closed if r["result"] == "WIN")
    losses = len(closed) - wins
    win_rate = wins / len(closed)

    pnls: list[float] = []
    for r in closed:
        try:
            pnls.append(float(r["pnl_pips"]))
        except (ValueError, KeyError):
            pass

    avg_win = sum(p for p in pnls if p > 0) / max(wins, 1)
    avg_loss = abs(sum(p for p in pnls if p < 0)) / max(losses, 1)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    equity_curve = [0.0]
    for p in pnls:
        equity_curve.append(equity_curve[-1] + p)
    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    return {
        "total": len(closed),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "expectancy_pips": round(expectancy, 2),
        "avg_win_pips": round(avg_win, 2),
        "avg_loss_pips": round(avg_loss, 2),
        "max_drawdown_pips": round(max_dd, 2),
        "rejected_by_ea": rejected,
    }


# ── Optional Telegram ─────────────────────────────────────────────────────────

def _send_telegram(message: str) -> None:
    if not TELEGRAM.get("enabled"):
        return
    try:
        import requests  # type: ignore

        token = TELEGRAM["bot_token"]
        chat_id = TELEGRAM["chat_id"]
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Telegram notification failed: %s", exc)


def notify_signal(signal: dict) -> None:
    msg = (
        f"📊 ICT Signal\n"
        f"Symbol: {signal.get('symbol')}\n"
        f"Direction: {signal.get('direction')}\n"
        f"Entry: {signal.get('entry_price')}\n"
        f"SL: {signal.get('stop_loss')}\n"
        f"TP: {signal.get('take_profit')}\n"
        f"Confidence: {signal.get('confidence_score')}\n"
        f"Setup: {signal.get('setup_type')}"
    )
    _send_telegram(msg)
