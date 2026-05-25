"""
Backtest Export Runner

Replays ICT analysis over historical CSV data and exports every generated
signal to a JSONL file.  The output can be replayed in MT5 Strategy Tester
(or a custom harness) to give a more honest end-to-end test of the full
Python → signal → EA pipeline.

Usage::

    python -m python.main --backtest \
        --from 2024-01-01 --to 2024-12-31 --symbol EURUSD

Output file: ``backtests/signals/EURUSD_2024-01-01_2024-12-31.jsonl``

Each line in the JSONL file is a self-contained signal dict (same schema as
``signals/active/{SYMBOL}.json``) that includes ``created_at`` and
``expires_at`` set to the *simulated* bar time, not wall-clock time.

The EA reads the JSONL in a ``CHistoricSignalReader`` (see
``mql5/ICT_EA.mqh``) by matching the signal's ``valid_from`` against the
current bar's open time during Strategy Tester playback.
"""
from __future__ import annotations

import json
import logging
from datetime import timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from python.config import BACKTEST_OUTPUT_DIR, CONFIG
from python.data_engine.csv_loader import load_csv
from python.signal_generator.signal_generator import generate_signal
from python.strategy_engine.mtf_engine import analyse

logger = logging.getLogger(__name__)

# Minimum candles required before the first bar is analysed
_MIN_HISTORY = 100


def run_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Replay ICT analysis over historical data for *symbol* and export signals.

    Parameters
    ----------
    symbol:
        Uppercase trading symbol, e.g. ``"EURUSD"``.
    start_date:
        ISO-8601 date string (``"YYYY-MM-DD"``).  Analysis starts at the
        first M5 bar on or after this date.
    end_date:
        ISO-8601 date string (``"YYYY-MM-DD"``).  Analysis stops at the last
        M5 bar on or before this date.
    output_dir:
        Directory to write the JSONL file.  Defaults to ``BACKTEST_OUTPUT_DIR``
        from config (``backtests/signals/``).

    Returns
    -------
    Path
        Path of the JSONL output file.
    """
    out_dir = output_dir or BACKTEST_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    tf_macro = CONFIG.get("timeframes", {}).get("macro", "D1")
    tf_struct = CONFIG.get("timeframes", {}).get("structure", "H1")
    tf_entry = CONFIG.get("timeframes", {}).get("entry", "M5")
    history = CONFIG.get("data", {}).get("candle_history", 500)
    ttl = CONFIG.get("integration", {}).get("signal_ttl_seconds", 300)

    logger.info(
        "Backtest: loading data for %s (%s/%s/%s)", symbol, tf_macro, tf_struct, tf_entry
    )

    df_d1 = load_csv(symbol, tf_macro, count=None)      # all available history
    df_h1 = load_csv(symbol, tf_struct, count=None)
    df_m5 = load_csv(symbol, tf_entry, count=None)

    if df_d1.empty or df_h1.empty or df_m5.empty:
        raise RuntimeError(
            f"No CSV data found for {symbol}. "
            "Ensure CSV files exist in the configured csv_path directory."
        )

    # Filter M5 to the requested date range
    start_dt = pd.Timestamp(start_date, tz="UTC")
    end_dt = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    df_m5_range = df_m5[(df_m5["time"] >= start_dt) & (df_m5["time"] <= end_dt)].reset_index(
        drop=True
    )

    if df_m5_range.empty:
        raise RuntimeError(
            f"No M5 bars for {symbol} between {start_date} and {end_date}."
        )

    logger.info(
        "Backtest: replaying %d M5 bars (%s → %s)",
        len(df_m5_range),
        df_m5_range["time"].iloc[0],
        df_m5_range["time"].iloc[-1],
    )

    out_file = out_dir / f"{symbol}_{start_date}_{end_date}.jsonl"
    signals_emitted = 0
    cycle_counter = 0

    with open(out_file, "w") as fh:
        for bar_idx in range(_MIN_HISTORY, len(df_m5_range)):
            bar_time: pd.Timestamp = df_m5_range["time"].iloc[bar_idx]

            # Slice historical windows for each timeframe up to the current bar
            d1_slice = df_d1[df_d1["time"] <= bar_time].tail(history)
            h1_slice = df_h1[df_h1["time"] <= bar_time].tail(history)
            m5_slice = df_m5_range.iloc[max(0, bar_idx - history) : bar_idx]

            if len(d1_slice) < 10 or len(h1_slice) < 20 or len(m5_slice) < _MIN_HISTORY:
                continue

            try:
                analysis = analyse(symbol, d1_slice, h1_slice, m5_slice)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Analysis error at bar %s: %s", bar_time, exc)
                continue

            if not analysis.valid:
                continue

            cycle_counter += 1
            cycle_id = f"bt-{cycle_counter:06d}"

            signal = generate_signal(analysis, cycle_id=cycle_id)
            if signal is None:
                continue

            # Override timestamps with simulated bar time (not wall clock)
            bar_utc = bar_time.to_pydatetime()
            if bar_utc.tzinfo is None:
                bar_utc = bar_utc.replace(tzinfo=timezone.utc)
            expires_utc = bar_utc + timedelta(seconds=ttl)

            signal["created_at"] = bar_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            signal["valid_from"] = bar_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            signal["expires_at"] = expires_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            signal["timestamp"] = bar_utc.isoformat()

            fh.write(json.dumps(signal) + "\n")
            signals_emitted += 1

    logger.info(
        "Backtest complete: %d signals written to %s", signals_emitted, out_file
    )
    return out_file
