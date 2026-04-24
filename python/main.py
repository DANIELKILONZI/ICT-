"""
ICT Trading System – Main Entry Point

Orchestrates:
  1. Data fetching for all configured symbols and timeframes
  2. ICT strategy analysis (multi-timeframe)
  3. Signal generation
  4. Performance logging
  5. Notification dispatch
  6. Integration (file / HTTP / socket)
"""
from __future__ import annotations

import json as _json
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone

import yaml

from python.config import CONFIG, SYMBOLS, TIMEFRAMES
from python.data_engine.data_store import get_ohlcv, refresh
from python.ml.signal_filter import passes_ml_filter
from python.performance.tracker import log_signal, notify_alert, notify_signal, statistics
from python.performance.tracker import daily_loss_reached, has_open_trade, trades_today_count
from python.signal_generator.signal_generator import generate_signal, save_signal
from python.strategy_engine.mtf_engine import analyse
from python.strategy_engine.util import atr_value as _atr_value

# ── Logging setup ────────────────────────────────────────────────────────────


class _JsonLogFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return _json.dumps(payload)


_log_format = CONFIG["system"].get("log_format", "text")
_formatter: logging.Formatter = (
    _JsonLogFormatter()
    if _log_format == "json"
    else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)
_log_handlers: list[logging.Handler] = [
    logging.StreamHandler(sys.stdout),
    logging.FileHandler(CONFIG["system"]["log_file"]),
]
for _h in _log_handlers:
    _h.setFormatter(_formatter)
logging.basicConfig(
    level=getattr(logging, CONFIG["system"].get("log_level", "INFO")),
    handlers=_log_handlers,
)
logger = logging.getLogger(__name__)

_RUNNING = True

# ── Watchdog state ────────────────────────────────────────────────────────────
_last_cycle_time: float = 0.0
_last_cycle_lock = threading.Lock()


def _record_cycle() -> None:
    """Update the heartbeat timestamp used by the watchdog."""
    global _last_cycle_time
    with _last_cycle_lock:
        _last_cycle_time = time.monotonic()


def _watchdog(scan_interval: int) -> None:
    """
    Background thread: log an error (and optionally Telegram-alert) if no
    analysis cycle has completed within 2 × scan_interval_seconds.

    The initial sleep of 2 × scan_interval gives the main loop time to
    complete its first cycle before the watchdog starts checking.
    """
    threshold = scan_interval * 2
    time.sleep(threshold)
    while _RUNNING:
        with _last_cycle_lock:
            last = _last_cycle_time
        if last > 0:
            age = time.monotonic() - last
            if age > threshold:
                msg = (
                    f"No analysis cycle completed in the last {age:.0f}s "
                    f"(threshold {threshold}s). System may be stalled."
                )
                logger.error("WATCHDOG: %s", msg)
                try:
                    notify_alert(f"⚠️ WATCHDOG: {msg}")
                except Exception:  # noqa: BLE001
                    pass
        time.sleep(scan_interval)


def _shutdown(signum, frame):
    global _RUNNING
    logger.info("Shutdown signal received.")
    _RUNNING = False


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── Session filter ────────────────────────────────────────────────────────────

def _in_trading_session() -> bool:
    """Return True during London or New York trading sessions (UTC)."""
    hours = CONFIG["risk"].get("trading_hours", {})
    now_h = datetime.now(timezone.utc).hour
    london = hours.get("london_open", 8) <= now_h < hours.get("london_close", 17)
    ny = hours.get("ny_open", 13) <= now_h < hours.get("ny_close", 22)
    return london or ny


# ── Integration startup ───────────────────────────────────────────────────────

def _start_integration() -> None:
    mode = CONFIG["integration"].get("mode", "file")
    if mode == "http":
        from python.integration.api_server import run_server

        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        logger.info("HTTP integration server started.")
    elif mode == "socket":
        from python.integration.socket_bridge import _get_socket

        _get_socket()  # eagerly bind the socket so errors surface at startup
        logger.info("ZeroMQ socket integration started.")
    elif mode == "file":
        logger.info("File integration mode – signals written to %s", CONFIG["system"]["signal_output_path"])


# ── Main analysis loop ────────────────────────────────────────────────────────

def run_analysis_cycle() -> None:
    """Run one full cycle of MTF ICT analysis for all symbols."""
    _record_cycle()  # update watchdog heartbeat unconditionally
    if not _in_trading_session():
        logger.debug("Outside trading session – skipping analysis.")
        return

    # ── Daily loss guard ────────────────────────────────────────────────────
    if daily_loss_reached():
        logger.warning("Daily loss limit reached – no new signals will be emitted today.")
        return

    tf_macro = TIMEFRAMES.get("macro", "D1")
    tf_struct = TIMEFRAMES.get("structure", "H1")
    tf_entry = TIMEFRAMES.get("entry", "M5")
    count = CONFIG["data"].get("candle_history", 500)

    allow_multiple = CONFIG["risk"].get("allow_multiple_positions_per_symbol", False)

    for symbol in SYMBOLS:
        try:
            # ── Duplicate signal prevention ─────────────────────────────────
            if not allow_multiple and has_open_trade(symbol):
                logger.debug("%s: open trade exists – skipping new signal.", symbol)
                continue

            df_d1 = get_ohlcv(symbol, tf_macro, count)
            df_h1 = get_ohlcv(symbol, tf_struct, count)
            df_m5 = get_ohlcv(symbol, tf_entry, count)

            if df_d1.empty or df_h1.empty or df_m5.empty:
                logger.warning("Insufficient data for %s – skipping.", symbol)
                continue

            analysis = analyse(symbol, df_d1, df_h1, df_m5)

            if not analysis.valid:
                logger.debug("%s: No valid setup. Reasons: %s", symbol, analysis.reasons)
                continue

            # Optional ML filter
            hour_utc = datetime.now(timezone.utc).hour
            atr_m5 = _atr_value(df_m5) if CONFIG["ml"].get("enabled") else 0.0
            if not passes_ml_filter(analysis, atr_m5=atr_m5, hour_utc=hour_utc):
                logger.info("%s: Signal filtered out by ML model.", symbol)
                continue

            signal = generate_signal(analysis)
            if signal:
                save_signal(signal)
                log_signal(signal)
                notify_signal(signal)
                if CONFIG["integration"].get("mode") == "socket":
                    from python.integration.socket_bridge import publish_signal
                    publish_signal(signal)
                logger.info("✅  Signal saved for %s %s", symbol, signal["direction"])

        except Exception as exc:  # noqa: BLE001
            logger.error("Error analysing %s: %s", symbol, exc, exc_info=True)


def main() -> None:
    logger.info("=== ICT Multi-Timeframe Trading System Starting ===")
    _start_integration()

    scan_interval = CONFIG.get("system", {}).get("scan_interval_seconds", 60)

    # Start watchdog before the main loop so it begins timing from the first cycle
    watchdog_thread = threading.Thread(
        target=_watchdog, args=(scan_interval,), daemon=True, name="watchdog"
    )
    watchdog_thread.start()
    logger.debug("Watchdog thread started (threshold=%ds).", scan_interval * 2)

    while _RUNNING:
        run_analysis_cycle()
        logger.debug("Cycle complete. Sleeping %ds.", scan_interval)
        time.sleep(scan_interval)

    # Print final stats before exit
    stats = statistics()
    logger.info("Performance stats: %s", stats)
    if CONFIG["integration"].get("mode") == "socket":
        from python.integration.socket_bridge import close as _close_socket
        _close_socket()
    logger.info("=== System stopped ===")


if __name__ == "__main__":
    main()
