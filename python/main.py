"""
ICT Trading System – Main Entry Point

Orchestrates:
  1. Data fetching for all configured symbols and timeframes
  2. ICT strategy analysis (multi-timeframe)
  3. Signal generation
  4. Performance logging (three-layer: candidates → signals → executions)
  5. Notification dispatch
  6. Integration (file / HTTP / socket)
  7. EA feedback ingestion (execution_feedback poller)

Backtest mode (Issue 10):
  python -m python.main --backtest --from 2024-01-01 --to 2024-12-31 --symbol EURUSD
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from datetime import datetime, timezone

from python.config import CONFIG, SYMBOLS, TIMEFRAMES, session_is_open
from python.data_engine.data_store import get_ohlcv
from python.integration.execution_feedback import ingest_pending as ingest_ea_feedback
from python.ml.signal_filter import ml_evaluate
from python.ml.signal_policy import build_signal_ml_metadata, should_publish
from python.performance.tracker import (
    daily_loss_reached,
    has_open_trade,
    log_candidate,
    log_signal,
    notify_signal,
    statistics,
    trades_today_count,
)
from python.signal_generator.signal_generator import generate_signal, save_signal
from python.strategy_engine.mtf_engine import analyse

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
_cycle_counter: int = 0


def _shutdown(signum, frame):
    global _RUNNING
    logger.info("Shutdown signal received.")
    _RUNNING = False


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ── Session filter ────────────────────────────────────────────────────────────

def _in_trading_session(symbol: str) -> bool:
    """Return True when *symbol* is inside its configured UTC session."""
    return session_is_open(symbol, now_utc=datetime.now(timezone.utc))


# ── Stale signal cleanup ──────────────────────────────────────────────────────

def _purge_expired_signals() -> None:
    """
    Remove signal files from signals/active/ whose expires_at has passed.

    This prevents the EA from reading stale signals and ensures disk state
    accurately reflects current system intent.
    """
    from python.config import SIGNAL_ACTIVE_DIR
    from python.signal_generator.signal_generator import load_latest_signal

    for path in SIGNAL_ACTIVE_DIR.glob("*.json"):
        symbol = path.stem  # e.g. "EURUSD"
        # load_latest_signal returns None if the signal has expired
        if load_latest_signal(symbol) is None and path.exists():
            try:
                path.unlink()
                logger.debug("Purged expired signal file: %s", path.name)
            except OSError as exc:
                logger.warning("Could not purge %s: %s", path.name, exc)


# ── Integration startup ───────────────────────────────────────────────────────

def _start_integration() -> None:
    mode = CONFIG["integration"].get("mode", "file")
    if mode == "http":
        from python.integration.api_server import run_server

        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        logger.info("HTTP integration server started.")
    elif mode == "file":
        logger.info("File integration mode – signals written to %s", CONFIG["system"]["signal_output_path"])
    # Socket mode can be added here


# ── Main analysis loop ────────────────────────────────────────────────────────

def run_analysis_cycle() -> None:
    """Run one full cycle of MTF ICT analysis for all symbols."""
    global _cycle_counter
    _cycle_counter += 1
    cycle_id = f"cycle-{_cycle_counter:06d}"

    # ── Portfolio-level risk guards ────────────────────────────────────────
    if daily_loss_reached():
        logger.info("Daily loss limit reached – no new signals this cycle.")
        return

    max_trades = CONFIG["risk"].get("max_trades_per_day", 5)
    if trades_today_count() >= max_trades:
        logger.info("Max trades per day (%d) reached – no new signals.", max_trades)
        return

    tf_macro = TIMEFRAMES.get("macro", "D1")
    tf_struct = TIMEFRAMES.get("structure", "H1")
    tf_entry = TIMEFRAMES.get("entry", "M5")
    count = CONFIG["data"].get("candle_history", 500)
    allow_multiple = CONFIG["risk"].get("allow_multiple_positions_per_symbol", False)

    for symbol in SYMBOLS:
        try:
            if not _in_trading_session(symbol):
                logger.debug("%s: Outside configured trading session – skipping.", symbol)
                continue

            # ── Per-symbol risk gate ───────────────────────────────────────
            if not allow_multiple and has_open_trade(symbol):
                logger.debug("%s: Open trade exists – skipping.", symbol)
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

            # Layer 1 – record the candidate setup (all hard gates passed)
            log_candidate(
                symbol=symbol,
                direction=analysis.signal_direction or "",
                confidence_score=analysis.confluence_score,
                setup_type=str(getattr(analysis, "setup_name", "") or ""),
                reasons=analysis.reasons,
            )

            # ML evaluation (advisory – never mutates entry/SL/TP)
            hour_utc = datetime.now(timezone.utc).hour
            ml_result = ml_evaluate(analysis, hour_utc=hour_utc)

            # Risk gate: verified above (daily loss OK, trade count OK, no duplicate)
            risk_gate_passed = True

            # Signal policy decides whether to publish
            publish, policy_reason = should_publish(
                ict_valid=True,  # already confirmed by analysis.valid
                risk_gate_passed=risk_gate_passed,
                ml_result=ml_result,
            )
            if not publish:
                logger.info("%s: Signal not published – %s", symbol, policy_reason)
                continue

            # Build ML metadata to embed in the signal
            ml_metadata = build_signal_ml_metadata(ml_result, ict_valid=True)

            signal = generate_signal(analysis, cycle_id=cycle_id, ml_metadata=ml_metadata)
            if signal:
                save_signal(signal)
                # Layer 2 – record the published signal
                log_signal(signal)
                notify_signal(signal)
                logger.info("✅  Signal saved for %s %s", symbol, signal["direction"])

        except Exception as exc:  # noqa: BLE001
            logger.error("Error analysing %s: %s", symbol, exc, exc_info=True)

    # Layer 3 – ingest any EA feedback written since the last cycle
    try:
        ingest_ea_feedback()
    except Exception as exc:  # noqa: BLE001
        logger.warning("EA feedback ingestion failed: %s", exc)

    # Purge expired signal files from disk
    _purge_expired_signals()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ICT Multi-Timeframe Trading System"
    )
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="Run in backtest export mode instead of live scanning.",
    )
    parser.add_argument(
        "--from",
        dest="date_from",
        metavar="YYYY-MM-DD",
        help="Backtest start date (required with --backtest).",
    )
    parser.add_argument(
        "--to",
        dest="date_to",
        metavar="YYYY-MM-DD",
        help="Backtest end date (required with --backtest).",
    )
    parser.add_argument(
        "--symbol",
        metavar="SYMBOL",
        help="Symbol to backtest (required with --backtest).",
    )
    args = parser.parse_args()

    if args.backtest:
        _run_backtest_mode(args)
    else:
        _run_live_mode()


def _run_backtest_mode(args: argparse.Namespace) -> None:
    """Export historical signals to JSONL for Strategy Tester replay."""
    missing = [
        f for f, v in [("--from", args.date_from), ("--to", args.date_to), ("--symbol", args.symbol)]
        if not v
    ]
    if missing:
        logger.error("Backtest mode requires: %s", ", ".join(missing))
        sys.exit(1)

    from python.backtest.runner import run_backtest

    logger.info("=== ICT Backtest Export Mode ===")
    try:
        out = run_backtest(
            symbol=args.symbol.upper(),
            start_date=args.date_from,
            end_date=args.date_to,
        )
        logger.info("Backtest signals written to: %s", out)
    except Exception as exc:
        logger.error("Backtest failed: %s", exc, exc_info=True)
        sys.exit(1)


def _run_live_mode() -> None:
    """Run continuous live analysis loop."""
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
    logger.info("=== System stopped ===")


if __name__ == "__main__":
    main()
