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
from python.performance.tracker import log_signal, notify_signal, statistics
from python.signal_generator.signal_generator import generate_signal, save_signal
from python.strategy_engine.mtf_engine import analyse
from python.strategy_engine.util import atr_value as _atr_value

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, CONFIG["system"].get("log_level", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(CONFIG["system"]["log_file"]),
    ],
)
logger = logging.getLogger(__name__)

_RUNNING = True


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
    if not _in_trading_session():
        logger.debug("Outside trading session – skipping analysis.")
        return

    tf_macro = TIMEFRAMES.get("macro", "D1")
    tf_struct = TIMEFRAMES.get("structure", "H1")
    tf_entry = TIMEFRAMES.get("entry", "M5")
    count = CONFIG["data"].get("candle_history", 500)

    for symbol in SYMBOLS:
        try:
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
            atr_m5 = _atr_value(df_m5)
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
