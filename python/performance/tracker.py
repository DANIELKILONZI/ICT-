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


# ── Risk guard helpers ────────────────────────────────────────────────────────

def has_open_trade(symbol: str) -> bool:
    """
    Return True if *symbol* has at least one signal marked OPEN in signals.csv.

    This allows the Python engine to avoid publishing duplicate signals for a
    symbol that already has an active trade on the EA side (the EA also checks
    this, but filtering here reduces noise and unnecessary file writes).
    """
    path = SIGNALS_LOG if SIGNALS_LOG.exists() else PERF_LOG
    if not path.exists():
        return False
    with _csv_lock:
        with open(path, "r", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("symbol") == symbol and row.get("result") == "OPEN":
                    return True
    return False


def daily_loss_reached() -> bool:
    """
    Return True if today's closed-trade losses have exceeded the configured
    ``risk.max_daily_loss_percent`` threshold.

    The calculation sums all pnl_pips for today's closed trades (WIN+LOSS)
    and compares the net result against the threshold.  Because Python does
    not have access to the actual equity curve, we use a simplified approach:
    if the net loss in pips today exceeds a proxy limit derived from config.
    """
    from python.config import RISK

    max_loss_pct = RISK.get("max_daily_loss_percent", 3.0)
    # Proxy: each 1% of risk ~ roughly 100 pips net movement
    # This is a conservative heuristic; the EA has the real equity check.
    max_loss_pips = max_loss_pct * 100

    path = SIGNALS_LOG if SIGNALS_LOG.exists() else PERF_LOG
    if not path.exists():
        return False

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    net_pips = 0.0

    with _csv_lock:
        with open(path, "r", newline="") as fh:
            for row in csv.DictReader(fh):
                ts = row.get("timestamp", "")
                if not ts.startswith(today_str):
                    continue
                if row.get("result") not in ("WIN", "LOSS"):
                    continue
                try:
                    net_pips += float(row["pnl_pips"])
                except (ValueError, KeyError):
                    pass

    return net_pips < -max_loss_pips


def trades_today_count() -> int:
    """
    Return the number of signals published (OPEN, WIN, LOSS, EXECUTED) today.

    Used to enforce ``risk.max_trades_per_day`` on the Python side before
    publishing additional signals.
    """
    path = SIGNALS_LOG if SIGNALS_LOG.exists() else PERF_LOG
    if not path.exists():
        return 0

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    count = 0

    with _csv_lock:
        with open(path, "r", newline="") as fh:
            for row in csv.DictReader(fh):
                ts = row.get("timestamp", "")
                if ts.startswith(today_str):
                    count += 1
    return count


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
        _mark_signal_executed(signal_id)
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
    _mark_signal_result(signal_id, "REJECTED")


def _mark_signal_executed(signal_id: str) -> None:
    """Update signals.csv to mark the given signal_id as EXECUTED."""
    _mark_signal_result(signal_id, "EXECUTED")


def _mark_signal_result(signal_id: str, result: str) -> None:
    """Update OPEN signal rows in signals/performance logs to the given result."""
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
                    row["result"] = result
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
