#!/usr/bin/env python
"""
Backtest All ICT Setups (1–4)

Scans historical CSV data bar-by-bar and records every signal fired by each
of the four ICT playbooks.  For each matched signal the trade outcome is
forward-simulated by checking whether take-profit or stop-loss is reached
first within `--max-bars` M5 candles.

The output CSV can be fed into scripts/train_ml.py to train the XGBoost
signal filter with labels from *all* session types.

Usage:
    # Run all four setups (EURUSD, parameters from config.yaml)
    python scripts/backtest_all_setups.py

    # Single setup only
    python scripts/backtest_all_setups.py --setup 1

    # Override symbol and output path
    python scripts/backtest_all_setups.py --symbol GBPUSD --output data/all_labels.csv

    # Grid-search killzone hours and BOS lookback for setups 3 & 4
    python scripts/backtest_all_setups.py --sweep

CSV data requirements:
    Place files matching the naming convention <SYMBOL>_<TF>.csv in the
    directory configured as `data.csv_path` in config.yaml (default: data/csv/).
    Required timeframes: D1, H1, M5.
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
from python.ml.signal_filter import _trend_int, _zone_int
from python.strategy_engine import bos_detector as _bos_detector
from python.strategy_engine import playbooks as _pb_module
from python.strategy_engine.mtf_engine import MTFAnalysis, _analyse_tf
from python.strategy_engine.playbooks import (
    setup1_sweep_fvg_continuation,
    setup2_htf_ob_reversal,
    setup3_london_killzone,
    setup4_ny_killzone,
)
from python.strategy_engine.premium_discount import compute_fib_range
from python.strategy_engine.premium_discount import price_zone as _price_zone
from python.strategy_engine.util import atr_value
from scripts.backtest_setup4 import (
    ANALYSIS_LOOKBACK_D1,
    ANALYSIS_LOOKBACK_H1,
    ANALYSIS_LOOKBACK_M5,
    MAX_BARS_FWD,
    MIN_HISTORY_M5,
    _extract_features,
    _simulate_outcome,
)

# ── Parameter grids (used by --sweep mode for setups 3 & 4) ──────────────────
GRID_KZ_OPEN      = [6, 7, 8]
GRID_KZ_CLOSE     = [9, 10, 11]
GRID_NY_OPEN      = [12, 13, 14]
GRID_NY_CLOSE     = [20, 21, 22]
GRID_BOS_LOOKBACK = [2, 3, 5]


def _build_analysis_and_features(
    d1_tf, h1_tf, m5_tf, m5_slice, h1_slice, d1_slice,
    pb, symbol, current_price, hour_utc, setup_name,
) -> tuple[MTFAnalysis, dict, float]:
    """Construct MTFAnalysis and ML feature dict from per-step analysis slices."""
    ms_d1 = d1_tf["ms"]
    fib_range = None
    pzone = "EQUILIBRIUM"
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
        setup_name=setup_name,
        valid=True,
    )
    atr_m5 = atr_value(m5_slice)
    features = _extract_features(analysis, atr_m5, hour_utc)
    return analysis, features, atr_m5


def _record(
    pb, symbol, current_ts, hour_utc, features, outcome, bars_held,
) -> dict:
    return {
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
    }


def run_backtest_setup1(
    df_d1: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: str,
    sweep_recency: int | None = None,
    max_bars: int = MAX_BARS_FWD,
) -> list[dict]:
    """Bar-by-bar backtest for Setup 1 – SWEEP_FVG_CONTINUATION."""
    recency = sweep_recency if sweep_recency is not None else STRATEGY.get("sweep_recency", 10)
    records: list[dict] = []
    n = len(df_m5)
    for i in range(MIN_HISTORY_M5, n):
        current_ts = df_m5["time"].iloc[i]
        hour_utc   = int(current_ts.hour)

        m5_slice = df_m5.iloc[max(0, i - ANALYSIS_LOOKBACK_M5 + 1): i + 1].copy()
        h1_slice = df_h1[df_h1["time"] <= current_ts].tail(ANALYSIS_LOOKBACK_H1)
        d1_slice = df_d1[df_d1["time"] <= current_ts].tail(ANALYSIS_LOOKBACK_D1)

        if len(h1_slice) < 10 or d1_slice.empty:
            continue

        m5_tf = _analyse_tf(m5_slice)
        h1_tf = _analyse_tf(h1_slice)
        d1_tf = _analyse_tf(d1_slice)
        current_price = float(m5_slice["close"].iloc[-1])

        with patch.object(_pb_module, "_SWEEP_RECENCY", recency):
            pb = setup1_sweep_fvg_continuation(
                d1_tf=d1_tf,
                h1_tf=h1_tf,
                m5_tf=m5_tf,
                df_m5=m5_slice,
                symbol=symbol,
                current_price=current_price,
            )

        if not pb.matched:
            continue

        _, features, atr_m5 = _build_analysis_and_features(
            d1_tf, h1_tf, m5_tf, m5_slice, h1_slice, d1_slice,
            pb, symbol, current_price, hour_utc, "SWEEP_FVG_CONTINUATION",
        )
        outcome, bars_held = _simulate_outcome(
            df_m5, i + 1, pb.direction, pb.stop_loss, pb.take_profit, max_bars,
        )
        records.append(_record(pb, symbol, current_ts, hour_utc, features, outcome, bars_held))
    return records


def run_backtest_setup2(
    df_d1: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: str,
    max_bars: int = MAX_BARS_FWD,
) -> list[dict]:
    """Bar-by-bar backtest for Setup 2 – HTF_OB_REVERSAL."""
    records: list[dict] = []
    n = len(df_m5)
    for i in range(MIN_HISTORY_M5, n):
        current_ts = df_m5["time"].iloc[i]
        hour_utc   = int(current_ts.hour)

        m5_slice = df_m5.iloc[max(0, i - ANALYSIS_LOOKBACK_M5 + 1): i + 1].copy()
        h1_slice = df_h1[df_h1["time"] <= current_ts].tail(ANALYSIS_LOOKBACK_H1)
        d1_slice = df_d1[df_d1["time"] <= current_ts].tail(ANALYSIS_LOOKBACK_D1)

        if len(h1_slice) < 10 or d1_slice.empty:
            continue

        m5_tf = _analyse_tf(m5_slice)
        h1_tf = _analyse_tf(h1_slice)
        d1_tf = _analyse_tf(d1_slice)
        current_price = float(m5_slice["close"].iloc[-1])

        pb = setup2_htf_ob_reversal(
            d1_tf=d1_tf,
            h1_tf=h1_tf,
            m5_tf=m5_tf,
            df_m5=m5_slice,
            symbol=symbol,
            current_price=current_price,
        )

        if not pb.matched:
            continue

        _, features, _ = _build_analysis_and_features(
            d1_tf, h1_tf, m5_tf, m5_slice, h1_slice, d1_slice,
            pb, symbol, current_price, hour_utc, "HTF_OB_REVERSAL",
        )
        outcome, bars_held = _simulate_outcome(
            df_m5, i + 1, pb.direction, pb.stop_loss, pb.take_profit, max_bars,
        )
        records.append(_record(pb, symbol, current_ts, hour_utc, features, outcome, bars_held))
    return records


def run_backtest_setup3(
    df_d1: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: str,
    lkz_open: int = 7,
    lkz_close: int = 10,
    bos_lookback: int = 3,
    max_bars: int = MAX_BARS_FWD,
) -> list[dict]:
    """Bar-by-bar backtest for Setup 3 – LONDON_KILLZONE_EXPANSION."""
    records: list[dict] = []
    n = len(df_m5)
    for i in range(MIN_HISTORY_M5, n):
        current_ts = df_m5["time"].iloc[i]
        hour_utc   = int(current_ts.hour)

        if not (lkz_open <= hour_utc < lkz_close):
            continue

        m5_slice = df_m5.iloc[max(0, i - ANALYSIS_LOOKBACK_M5 + 1): i + 1].copy()
        h1_slice = df_h1[df_h1["time"] <= current_ts].tail(ANALYSIS_LOOKBACK_H1)
        d1_slice = df_d1[df_d1["time"] <= current_ts].tail(ANALYSIS_LOOKBACK_D1)

        if len(h1_slice) < 10 or d1_slice.empty:
            continue

        m5_tf = _analyse_tf(m5_slice)
        h1_tf = _analyse_tf(h1_slice)
        d1_tf = _analyse_tf(d1_slice)
        current_price = float(m5_slice["close"].iloc[-1])

        with patch.object(_pb_module, "_LKZ_OPEN",              lkz_open), \
             patch.object(_pb_module, "_LKZ_CLOSE",             lkz_close), \
             patch.object(_pb_module, "_KILLZONE_BOS_LOOKBACK", bos_lookback):
            pb = setup3_london_killzone(
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

        _, features, _ = _build_analysis_and_features(
            d1_tf, h1_tf, m5_tf, m5_slice, h1_slice, d1_slice,
            pb, symbol, current_price, hour_utc, "LONDON_KILLZONE_EXPANSION",
        )
        outcome, bars_held = _simulate_outcome(
            df_m5, i + 1, pb.direction, pb.stop_loss, pb.take_profit, max_bars,
        )
        records.append(_record(pb, symbol, current_ts, hour_utc, features, outcome, bars_held))
    return records


def run_backtest_setup4(
    df_d1: pd.DataFrame,
    df_h1: pd.DataFrame,
    df_m5: pd.DataFrame,
    symbol: str,
    ny_open: int = 13,
    ny_close: int = 22,
    bos_lookback: int = 3,
    max_bars: int = MAX_BARS_FWD,
) -> list[dict]:
    """Bar-by-bar backtest for Setup 4 – NY_KILLZONE_EXPANSION (delegates to backtest_setup4)."""
    from scripts.backtest_setup4 import run_backtest
    return run_backtest(df_d1, df_h1, df_m5, symbol, ny_open, ny_close, bos_lookback, max_bars)


# ── Dispatch table ────────────────────────────────────────────────────────────

_SETUP_RUNNERS = {
    "1": ("SWEEP_FVG_CONTINUATION", run_backtest_setup1),
    "2": ("HTF_OB_REVERSAL",        run_backtest_setup2),
    "3": ("LONDON_KILLZONE",        run_backtest_setup3),
    "4": ("NY_KILLZONE",            run_backtest_setup4),
}


def _print_summary(records: list[dict], label: str = "") -> None:
    tag    = f"[{label}] " if label else ""
    total  = len(records)
    wins   = sum(1 for r in records if r["outcome"] == "WIN")
    losses = sum(1 for r in records if r["outcome"] == "LOSS")
    open_  = sum(1 for r in records if r["outcome"] == "OPEN")
    closed = wins + losses
    wr     = wins / closed if closed else 0.0
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
        description="Backtest all four ICT setups",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbol", default="EURUSD",
        help="Trading symbol (must have matching CSVs in data/csv/)",
    )
    parser.add_argument(
        "--setup", default="all",
        choices=["all", "1", "2", "3", "4"],
        help="Which setup to backtest",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Grid-search killzone hours and BOS lookback for setups 3 & 4",
    )
    parser.add_argument(
        "--output", default="data/backtest_all_labels.csv",
        help="Path to save the combined labeled CSV",
    )
    parser.add_argument(
        "--max-bars", type=int, default=MAX_BARS_FWD,
        help="Max M5 bars to look ahead when simulating TP/SL outcome",
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
                f"  Place {symbol}_{tf}.csv in data/csv/ and retry."
            )
            sys.exit(1)

    print(f"Loaded: D1={len(df_d1)} bars  H1={len(df_h1)} bars  M5={len(df_m5)} bars\n")

    setups_to_run = list(_SETUP_RUNNERS.keys()) if args.setup == "all" else [args.setup]
    all_records: list[dict] = []

    for s_key in setups_to_run:
        name, runner = _SETUP_RUNNERS[s_key]
        print(f"Setup {s_key} – {name}:")

        if args.sweep and s_key in ("3", "4"):
            grid_open  = GRID_KZ_OPEN if s_key == "3" else GRID_NY_OPEN
            grid_close = GRID_KZ_CLOSE if s_key == "3" else GRID_NY_CLOSE
            for kz_open in grid_open:
                for kz_close in grid_close:
                    if kz_open >= kz_close:
                        continue
                    for bos_lb in GRID_BOS_LOOKBACK:
                        kw = dict(
                            lkz_open=kz_open, lkz_close=kz_close,
                            bos_lookback=bos_lb, max_bars=args.max_bars,
                        ) if s_key == "3" else dict(
                            ny_open=kz_open, ny_close=kz_close,
                            bos_lookback=bos_lb, max_bars=args.max_bars,
                        )
                        recs = runner(df_d1, df_h1, df_m5, symbol, **kw)
                        lbl = f"open={kz_open} close={kz_close} bos_lb={bos_lb}"
                        _print_summary(recs, lbl)
        else:
            kw: dict = {}
            if s_key in ("3", "4"):
                kw["bos_lookback"] = STRATEGY.get("killzone_bos_lookback", 3)
                if s_key == "3":
                    kw["lkz_open"]  = STRATEGY.get("london_killzone_open",  7)
                    kw["lkz_close"] = STRATEGY.get("london_killzone_close", 10)
                else:
                    kw["ny_open"]  = STRATEGY.get("ny_killzone_open",  13)
                    kw["ny_close"] = STRATEGY.get("ny_killzone_close", 22)
            kw["max_bars"] = args.max_bars
            recs = runner(df_d1, df_h1, df_m5, symbol, **kw)
            _print_summary(recs)
            for r in recs:
                r["setup_num"] = s_key
            all_records.extend(recs)

    if all_records and not args.sweep:
        out = REPO_ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_records).to_csv(out, index=False)
        print(f"\nLabeled trades → {out}  ({len(all_records)} total)")
    elif not all_records and not args.sweep:
        print("\nNo signals fired for any setup.")


if __name__ == "__main__":
    main()
