"""
Performance Tracking Module

Logs all signals generated and trades executed.
Tracks: win/loss, drawdown, expectancy, and exports CSV reports.
Optionally sends Telegram notifications.

Guards:
  has_open_trade(symbol) → True if an OPEN row exists for the symbol.
  daily_loss_reached()   → True when today's realised losses exceed the
                           configured max_daily_loss_percent limit.
"""
from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from python.config import PERF_LOG, RISK, TELEGRAM, pip_size_for

logger = logging.getLogger(__name__)

_csv_lock = threading.Lock()

_HEADERS = [
    "timestamp",
    "symbol",
    "direction",
    "entry_price",
    "stop_loss",
    "take_profit",
    "exit_price",
    "result",           # WIN | LOSS | OPEN
    "pnl_pips",
    "risk_percent",
    "confidence_score",
    "setup_type",
]


def _ensure_file() -> None:
    if not PERF_LOG.exists():
        with open(PERF_LOG, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_HEADERS)
            writer.writeheader()


def has_open_trade(symbol: str) -> bool:
    """Return True if there is already an OPEN trade logged for *symbol*."""
    with _csv_lock:
        _ensure_file()
        with open(PERF_LOG, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    return any(r.get("symbol") == symbol and r.get("result") == "OPEN" for r in rows)


def trades_today_count() -> int:
    """
    Return the number of signals logged today (UTC).

    Counts every row whose timestamp starts with today's date, regardless of
    result status.  This includes OPEN trades, so the guard fires as soon as the
    *n*-th signal is generated, not only after it closes.
    """
    today: str = datetime.now(timezone.utc).date().isoformat()
    with _csv_lock:
        _ensure_file()
        with open(PERF_LOG, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    return sum(1 for r in rows if r.get("timestamp", "").startswith(today))


def daily_loss_reached() -> bool:
    """
    Return True when today's estimated losses meet or exceed
    ``risk.max_daily_loss_percent`` from config.yaml.

    Approximation: each closed LOSS trade is assumed to have cost
    ``risk_percent`` of account equity (the amount that was risked on the SL).
    Each WIN trade recovers nothing in this conservative estimate.
    The guard resets automatically because only *today's* rows are examined.
    """
    limit: float = float(RISK.get("max_daily_loss_percent", 3.0))
    today: str = datetime.now(timezone.utc).date().isoformat()

    with _csv_lock:
        _ensure_file()
        with open(PERF_LOG, "r", newline="") as f:
            rows = list(csv.DictReader(f))

    total_loss_pct: float = 0.0
    for r in rows:
        if not r.get("timestamp", "").startswith(today):
            continue
        if r.get("result") != "LOSS":
            continue
        try:
            total_loss_pct += float(r.get("risk_percent", 1.0))
        except (ValueError, TypeError):
            pass

    if total_loss_pct >= limit:
        logger.warning(
            "Daily loss guard triggered: estimated %.2f%% loss today (limit %.2f%%)",
            total_loss_pct,
            limit,
        )
        return True
    return False


def log_signal(signal: dict) -> None:
    """Append a generated signal to the performance CSV."""
    with _csv_lock:
        _ensure_file()
        row = {h: signal.get(h, "") for h in _HEADERS}
        row["timestamp"] = signal.get("timestamp", datetime.now(timezone.utc).isoformat())
        row["result"] = "OPEN"
        with open(PERF_LOG, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_HEADERS).writerow(row)
    logger.debug("Signal logged to performance CSV.")


def update_trade_result(
    symbol: str,
    entry_price: float,
    exit_price: float,
    direction: str,
) -> None:
    """
    Update the OPEN trade with the actual exit price and WIN/LOSS result.
    Scans the CSV from the bottom to find the most recent OPEN trade for the symbol.
    """
    _ensure_file()
    rows: list[dict] = []
    with _csv_lock:
        with open(PERF_LOG, "r", newline="") as f:
            rows = list(csv.DictReader(f))

        updated = False
        for row in reversed(rows):
            if row["symbol"] == symbol and row["result"] == "OPEN":
                try:
                    ep = float(row["entry_price"])
                    sl = float(row["stop_loss"])
                    pip_size = pip_size_for(symbol)

                    if direction == "BUY":
                        pnl_pips = (exit_price - ep) / pip_size
                    else:
                        pnl_pips = (ep - exit_price) / pip_size

                    row["exit_price"] = str(round(exit_price, 5))
                    row["pnl_pips"] = str(round(pnl_pips, 1))
                    row["result"] = "WIN" if pnl_pips > 0 else "LOSS"
                except (ValueError, ZeroDivisionError):
                    pass
                updated = True
                break

        if updated:
            with open(PERF_LOG, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_HEADERS)
                writer.writeheader()
                writer.writerows(rows)
            logger.info("Trade result updated for %s", symbol)


def statistics() -> dict:
    """Compute win rate, expectancy, and max drawdown from the log."""
    with _csv_lock:
        _ensure_file()
        with open(PERF_LOG, "r", newline="") as f:
            rows = list(csv.DictReader(f))

    closed = [r for r in rows if r["result"] in ("WIN", "LOSS")]
    if not closed:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "expectancy": 0.0}

    wins = sum(1 for r in closed if r["result"] == "WIN")
    losses = len(closed) - wins
    win_rate = wins / len(closed)

    pnls = []
    for r in closed:
        try:
            pnls.append(float(r["pnl_pips"]))
        except (ValueError, KeyError):
            pass

    avg_win = sum(p for p in pnls if p > 0) / max(wins, 1)
    # When there are no losses avg_loss is 0.0 (the max(losses, 1) denominator is
    # correct; the numerator is simply 0).  In that all-win scenario expectancy
    # equals win_rate * avg_win, which is the correct positive result.
    avg_loss = abs(sum(p for p in pnls if p < 0)) / max(losses, 1)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Max drawdown
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
    }


# ── Optional Telegram ────────────────────────────────────────────────────────

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


def notify_alert(message: str) -> None:
    """Send a free-form alert string via Telegram."""
    _send_telegram(message)
