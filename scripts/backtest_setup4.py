#!/usr/bin/env python
"""
Backtest Setup 4 – NY Killzone Expansion

Scans historical CSV data bar-by-bar and records every candle where
Setup 4 fires.  For each matched signal, the trade outcome is forward-
simulated by checking whether the take-profit or stop-loss is reached
first within `--max-bars` M5 candles.

The output CSV contains the feature columns expected by the ML filter
(build_feature_vector()) plus a binary `label` column (1=WIN, 0=LOSS),
making it directly usable as training data for scripts/train_ml.py.

Usage:
    # Single run with parameters from config.yaml
    python scripts/backtest_setup4.py

    # Override symbol and output path
    python scripts/backtest_setup4.py --symbol GBPUSD --output data/gbpusd_labels.csv

    # Grid-search ny_open / ny_close / bos_lookback
    python scripts/backtest_setup4.py --sweep

CSV data requirements:
    Place files matching the naming convention <SYMBOL>_<TF>.csv in the
    directory configured as `data.csv_path` in config.yaml (default: data/csv/).
    Required timeframes: D1, H1, M5.

    Expected columns (case-insensitive): time, open, high, low, close[, volume]
    The time column must be parseable by pandas.to_datetime().
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from python.config import STRATEGY
from python.data_engine.csv_loader import load_csv
from python.ml.signal_filter import _trend_int, _zone_int  # private helpers – stable
from python.strategy_engine import bos_detector as _bos_detector
from python.strategy_engine import playbooks as _pb_module
from python.strategy_engine.mtf_engine import MTFAnalysis, _analyse_tf
from python.strategy_engine.playbooks import setup4_ny_killzone
from python.strategy_engine.premium_discount import compute_fib_range
from python.strategy_engine.premium_discount import price_zone as _price_zone
from python.strategy_engine.util import atr_value

# ── Scan constants ────────────────────────────────────────────────────────────
ANALYSIS_LOOKBACK_M5 = 200   # M5 bars used for per-step analysis window
ANALYSIS_LOOKBACK_H1 = 200   # H1 bars used for per-step analysis window
ANALYSIS_LOOKBACK_D1 = 100   # D1 bars used for per-step analysis window
MIN_HISTORY_M5       = 100   # bars of M5 history required before scanning
MAX_BARS_FWD         = 200   # default max M5 bars to look ahead for TP/SL

# ── Parameter grids (used by --sweep mode) ────────────────────────────────────
GRID_NY_OPEN      = [12, 13, 14]
GRID_NY_CLOSE     = [20, 21, 22]
GRID_BOS_LOOKBACK = [2, 3, 5]


def _simulate_outcome(
    df_m5: pd.DataFrame,
    from_idx: int,
    direction: str,
    sl: float,
    tp: float,
    max_bars: int = MAX_BARS_FWD,
) -> tuple[str, int]:
    """
    Forward-scan M5 candles from *from_idx* to find which level is hit first.

    Stops at ``from_idx + max_bars`` or end of data, whichever comes first.
    The stop-loss is checked before take-profit (conservative assumption: on a
    bar that touches both levels the adverse fill is assumed to occur first).

    Returns (outcome, bars_held) where outcome is ``'WIN'``, ``'LOSS'``, or
    ``'OPEN'`` (neither level reached within *max_bars*).
    """
    end = min(from_idx + max_bars, len(df_m5))
    for j in range(from_idx, end):
        high = df_m5["high"].iloc[j]
        low  = df_m5["low"].iloc[j]
        if direction == "BUY":
            if low  <= sl:  return "LOSS", j - from_idx + 1
            if high >= tp:  return "WIN",  j - from_idx + 1
        else:               # SELL
            if high >= sl:  return "LOSS", j - from_idx + 1
            if low  <= tp:  return "WIN",  j - from_idx + 1
    return "OPEN", max_bars


def _extract_features(analysis: MTFAnalysis, atr_m5: float, hour_utc: int) -> dict:
    """Return the 10 ML feature values as a flat dict (matches build_feature_vector())."""
    return {
        "d1_trend":         _trend_int(analysis.d1_trend),
        "h1_trend":         _trend_int(analysis.h1_trend),
        "atr_m5":           round(atr_m5, 6),
        "hour_utc":         hour_utc,
        "m5_sweep":         1 if analysis.m5_sweep is not None else 0,
        "price_zone":       _zone_int(analysis.price_zone),
        "h1_ob":            1 if analysis.h1_ob  is not None else 0,
        "h1_fvg":           1 if analysis.h1_fvg is not None else 0,
        "m5_fvg":           1 if analysis.m5_fvg is not None else 0,
        "confluence_score": round(analysis.confluence_score, 4),
    }


def run_backtest(
    df_d1: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: str,
    ny_open: int,
    ny_close: int,
    bos_lookback: int,
    max_bars: int = MAX_BARS_FWD,
) -> list[dict]:
    """
    Scan *df_m5* bar-by-bar and return a list of trade records for each bar
    where Setup 4 fires with the given parameters.

    Parameters
    ----------
    df_d1, df_h1, df_m5
        Full historical DataFrames.  Slices respecting the current timestamp
        are used at each step to prevent lookahead bias.
    ny_open, ny_close
        UTC hour boundaries for the NY Killzone time gate.
    bos_lookback
        Maximum H1 candle lag between the latest BOS and the current bar.
    max_bars
        Forward-scan horizon for TP/SL simulation.
    """
    records: list[dict] = []
    n = len(df_m5)

    for i in range(MIN_HISTORY_M5, n):
        current_ts = df_m5["time"].iloc[i]
        hour_utc   = int(current_ts.hour)

        # Pre-filter: only run during the NY Killzone window
        if not (ny_open <= hour_utc < ny_close):
            continue

        # ── Windowed slices (no lookahead) ────────────────────────────────────
        m5_slice = df_m5.iloc[max(0, i - ANALYSIS_LOOKBACK_M5 + 1): i + 1].copy()
        h1_slice = df_h1[df_h1["time"] <= current_ts].tail(ANALYSIS_LOOKBACK_H1)
        d1_slice = df_d1[df_d1["time"] <= current_ts].tail(ANALYSIS_LOOKBACK_D1)

        if len(h1_slice) < 10 or d1_slice.empty:
            continue

        # ── Per-timeframe analysis ────────────────────────────────────────────
        m5_tf = _analyse_tf(m5_slice)
        h1_tf = _analyse_tf(h1_slice)
        d1_tf = _analyse_tf(d1_slice)
        current_price = float(m5_slice["close"].iloc[-1])

        # ── Setup 4 evaluation with overridden parameters ─────────────────────
        with patch.object(_pb_module, "_NYZ_OPEN",              ny_open), \
             patch.object(_pb_module, "_NYZ_CLOSE",             ny_close), \
             patch.object(_pb_module, "_KILLZONE_BOS_LOOKBACK", bos_lookback):
            pb = setup4_ny_killzone(
                h1_tf=h1_tf,
                m5_tf=m5_tf,
                df_h1=h1_slice,
                df_m5=m5_slice,
                symbol=symbol,
                current_price=current_price,
                _now_hour=hour_utc,
            )

        if not pb.matched:
            continue

        # ── Build MTFAnalysis for ML feature extraction ────────────────────────
        ms_d1     = d1_tf["ms"]
        fib_range = None
        pzone     = "EQUILIBRIUM"
        if ms_d1.last_swing_high and ms_d1.last_swing_low:
            fib_range = compute_fib_range(ms_d1.last_swing_high, ms_d1.last_swing_low)
            if fib_range:
                pzone = _price_zone(fib_range, current_price)

        h1_latest_bos = _bos_detector.latest_bos(h1_tf["bos"])
        analysis = MTFAnalysis(
            symbol=symbol,
            d1_trend=d1_tf["ms"].trend,
            h1_trend=h1_tf["ms"].trend,
            h1_bos_direction=h1_latest_bos.direction if h1_latest_bos else None,
            m5_sweep=pb.m5_sweep,
            m5_fvg=pb.m5_fvg,
            h1_ob=pb.h1_ob,
            h1_fvg=pb.h1_fvg,
            fib_range=fib_range,
            price_zone=pzone,
            signal_direction=pb.direction,
            entry_price=pb.entry_price,
            stop_loss=pb.stop_loss,
            take_profit=pb.take_profit,
            confluence_score=pb.confluence_score,
            setup_name="NY_KILLZONE_EXPANSION",
            valid=True,
        )

        # ── Forward simulation ─────────────────────────────────────────────────
        atr_m5   = atr_value(m5_slice)
        features = _extract_features(analysis, atr_m5, hour_utc)
        outcome, bars_held = _simulate_outcome(
            df_m5, i + 1,
            pb.direction, pb.stop_loss, pb.take_profit,
            max_bars,
        )

        records.append({
            "timestamp":        str(current_ts),
            "symbol":           symbol,
            "direction":        pb.direction,
            "entry_price":      round(pb.entry_price, 5),
            "stop_loss":        round(pb.stop_loss,   5),
            "take_profit":      round(pb.take_profit, 5),
            "confluence_score": round(pb.confluence_score, 4),
            "outcome":          outcome,
            "bars_held":        bars_held,
            "label":            1 if outcome == "WIN" else 0,
            **features,
        })

    return records


def _print_summary(records: list[dict], label: str = "") -> None:
    """Print a one-line trade summary."""
    tag    = f"[{label}] " if label else ""
    total  = len(records)
    wins   = sum(1 for r in records if r["outcome"] == "WIN")
    losses = sum(1 for r in records if r["outcome"] == "LOSS")
    open_  = sum(1 for r in records if r["outcome"] == "OPEN")
    closed = wins + losses
    wr     = wins / closed if closed else 0.0
    # Setup 4 always constructs a 2:1 R:R (TP = entry ± 2× risk distance)
    exp    = (wr * 2.0 - (1 - wr) * 1.0) if closed else 0.0

    if not total:
        print(f"  {tag}No signals fired.")
        return
    print(
        f"  {tag}Signals={total:3d}  Closed={closed:3d}  "
        f"W={wins}  L={losses}  Open={open_:2d}  "
        f"WinRate={wr:5.1%}  Expectancy={exp:+.2f}R"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest Setup 4 – NY Killzone Expansion",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol",   default="EURUSD",
        help="Trading symbol (must have matching CSVs in data/csv/)",
    )
    parser.add_argument(
        "--sweep",    action="store_true",
        help="Grid-search ny_open / ny_close / bos_lookback and print comparison table",
    )
    parser.add_argument(
        "--output",   default="data/backtest_labels.csv",
        help="Path to save the labeled trade CSV for ML training",
    )
    parser.add_argument(
        "--max-bars", type=int, default=MAX_BARS_FWD,
        help=f"Max M5 bars to look ahead when simulating TP/SL outcome",
    )
    args = parser.parse_args()

    symbol = args.symbol.upper()
    print(f"Loading CSV data for {symbol} …")
    df_d1 = load_csv(symbol, "D1")
    df_h1 = load_csv(symbol, "H1")
    df_m5 = load_csv(symbol, "M5")

    for tf, df in [("D1", df_d1), ("H1", df_h1), ("M5", df_m5)]:
        if df.empty:
            print(
                f"ERROR: No {tf} data for {symbol}.\n"
                f"  Place {symbol}_{tf}.csv in data/csv/ and retry.\n"
                f"  Required columns: time, open, high, low, close"
            )
            sys.exit(1)

    print(
        f"Loaded: D1={len(df_d1)} bars  H1={len(df_h1)} bars  M5={len(df_m5)} bars\n"
    )

    if args.sweep:
        print("Parameter sweep (Setup 4 – NY Killzone):")
        print(f"  ny_open      : {GRID_NY_OPEN}")
        print(f"  ny_close     : {GRID_NY_CLOSE}")
        print(f"  bos_lookback : {GRID_BOS_LOOKBACK}")
        print()

        best_label    = ""
        best_exp      = float("-inf")
        best_records: list[dict] = []

        for ny_open in GRID_NY_OPEN:
            for ny_close in GRID_NY_CLOSE:
                if ny_open >= ny_close:
                    continue
                for bos_lb in GRID_BOS_LOOKBACK:
                    recs = run_backtest(
                        df_d1, df_h1, df_m5, symbol,
                        ny_open, ny_close, bos_lb, args.max_bars,
                    )
                    lbl = f"open={ny_open} close={ny_close} bos_lb={bos_lb}"
                    _print_summary(recs, lbl)
                    closed = [r for r in recs if r["outcome"] != "OPEN"]
                    if closed:
                        wins_n = sum(1 for r in closed if r["outcome"] == "WIN")
                        wr     = wins_n / len(closed)
                        exp    = wr * 2.0 - (1 - wr) * 1.0
                        if exp > best_exp:
                            best_exp, best_label, best_records = exp, lbl, recs

        if best_label:
            print(f"\nBest parameters : {best_label}  (expectancy {best_exp:+.2f}R)")
            if best_records:
                out = REPO_ROOT / args.output
                out.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(best_records).to_csv(out, index=False)
                print(f"Labeled trades  → {out}")
        else:
            print("\nNo signals fired for any parameter combination.")

    else:
        ny_open = STRATEGY.get("ny_killzone_open",    13)
        ny_close = STRATEGY.get("ny_killzone_close",   22)
        bos_lb   = STRATEGY.get("killzone_bos_lookback", 3)

        print(
            f"Running backtest: ny_open={ny_open}  ny_close={ny_close}  "
            f"bos_lookback={bos_lb}  max_bars_fwd={args.max_bars}"
        )
        records = run_backtest(
            df_d1, df_h1, df_m5, symbol, ny_open, ny_close, bos_lb, args.max_bars,
        )
        _print_summary(records)

        if records:
            out = REPO_ROOT / args.output
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(records).to_csv(out, index=False)
            print(f"Labeled trades → {out}")
        else:
            print(
                "No signals fired.  Check that your CSV data covers NY session hours "
                f"({ny_open}:00–{ny_close}:00 UTC) and has sufficient bars (> {MIN_HISTORY_M5})."
            )


if __name__ == "__main__":
    main()
