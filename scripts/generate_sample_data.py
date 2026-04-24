#!/usr/bin/env python
"""
Generate synthetic OHLCV CSV data for offline development and testing.

Produces realistic-looking price series using geometric Brownian motion (GBM)
seeded deterministically (seed=42 by default) so results are reproducible.

Output files are written to the directory configured as ``data.csv_path`` in
config.yaml (default: ``data/csv/``).  Existing files are overwritten only
when ``--force`` is passed.

Generated symbols / timeframes
-------------------------------
  EURUSD  D1  (5 years  ≈ 1 300 bars)
  EURUSD  H1  (2 years  ≈ 17 000 bars)
  EURUSD  M5  (90 days  ≈ 26 000 bars)
  GBPUSD  D1 / H1 / M5  (same ranges)

Usage
-----
  python scripts/generate_sample_data.py
  python scripts/generate_sample_data.py --output data/csv/ --seed 123 --force
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Repository root on sys.path so we can import python.config
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from python.config import CONFIG  # noqa: E402  (import after sys.path fix)


# ---------------------------------------------------------------------------
# GBM simulation helpers
# ---------------------------------------------------------------------------

def _simulate_closes(
    n_bars: int,
    s0: float,
    mu: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an array of *n_bars* close prices using GBM."""
    z = rng.standard_normal(n_bars)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z
    closes = s0 * np.exp(np.cumsum(log_returns))
    return closes


def _build_ohlcv(
    closes: np.ndarray,
    pip_size: float,
    rng: np.random.Generator,
    spread_pips: float = 2.0,
) -> pd.DataFrame:
    """Convert a close series into an OHLCV DataFrame with realistic wicks."""
    n = len(closes)
    # Intrabar range ≈ 5–25 pips; distributed between high and low sides
    range_pips = rng.uniform(5, 25, size=n) * pip_size
    wick_up = rng.uniform(0.1, 0.9, size=n) * range_pips
    wick_down = range_pips - wick_up

    # Determine open from previous close (small gap)
    opens = np.empty(n)
    opens[0] = closes[0] * (1 + rng.uniform(-0.0001, 0.0001))
    opens[1:] = closes[:-1] * (1 + rng.uniform(-0.00005, 0.00005, size=n - 1))

    highs = np.maximum(opens, closes) + wick_up
    lows = np.minimum(opens, closes) - wick_down
    volumes = rng.integers(100, 5000, size=n).astype(float)

    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volumes})


def _make_timestamps(n: int, end: datetime, freq_minutes: int) -> list[datetime]:
    """Generate *n* timestamps ending at *end* with *freq_minutes* spacing."""
    start = end - timedelta(minutes=freq_minutes * (n - 1))
    return [start + timedelta(minutes=freq_minutes * i) for i in range(n)]


# ---------------------------------------------------------------------------
# Per-symbol parameters
# ---------------------------------------------------------------------------

_SYMBOL_PARAMS: dict[str, dict] = {
    "EURUSD": {"s0": 1.0850, "mu": 0.00,  "sigma": 0.08, "pip_size": 0.0001},
    "GBPUSD": {"s0": 1.2700, "mu": 0.00,  "sigma": 0.10, "pip_size": 0.0001},
    "USDJPY": {"s0": 148.50, "mu": 0.00,  "sigma": 0.08, "pip_size": 0.01},
    "XAUUSD": {"s0": 1980.0, "mu": 0.02,  "sigma": 0.15, "pip_size": 0.10},
}

_TF_CONFIG: dict[str, dict] = {
    "D1":  {"freq_minutes": 1440,  "n_bars": 5 * 365},       # 5 years
    "H1":  {"freq_minutes": 60,    "n_bars": 2 * 365 * 24},  # 2 years
    "M5":  {"freq_minutes": 5,     "n_bars": 90 * 24 * 12},  # 90 days
    "M15": {"freq_minutes": 15,    "n_bars": 180 * 24 * 4},  # 180 days
    "H4":  {"freq_minutes": 240,   "n_bars": 4 * 365 * 6},   # 4 years
    "M1":  {"freq_minutes": 1,     "n_bars": 30 * 24 * 60},  # 30 days
}


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------

def generate(
    symbol: str,
    timeframe: str,
    output_dir: Path,
    rng: np.random.Generator,
    force: bool = False,
) -> Path:
    """Generate one CSV file for *symbol* / *timeframe*."""
    params = _SYMBOL_PARAMS.get(symbol)
    if params is None:
        raise ValueError(f"No parameters defined for symbol '{symbol}'")
    tf_cfg = _TF_CONFIG.get(timeframe)
    if tf_cfg is None:
        raise ValueError(f"Unsupported timeframe '{timeframe}'")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{symbol}_{timeframe}.csv"

    if out_path.exists() and not force:
        print(f"  skip  {out_path.name}  (already exists; use --force to overwrite)")
        return out_path

    n = tf_cfg["n_bars"]
    freq = tf_cfg["freq_minutes"]
    dt = freq / (365 * 24 * 60)  # fraction of a year per bar

    closes = _simulate_closes(
        n, params["s0"], params["mu"], params["sigma"], dt, rng
    )
    df = _build_ohlcv(closes, params["pip_size"], rng)
    end_ts = datetime(2024, 12, 31, 23, 59, tzinfo=timezone.utc)
    df["time"] = _make_timestamps(n, end_ts, freq)
    df = df[["time", "open", "high", "low", "close", "volume"]]
    df["time"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    df.to_csv(out_path, index=False)
    print(f"  wrote {out_path.name}  ({n} bars)")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    csv_default = str(_REPO / CONFIG["data"].get("csv_path", "data/csv/"))
    parser = argparse.ArgumentParser(
        description="Generate synthetic OHLCV CSV files for offline development."
    )
    parser.add_argument(
        "--output", default=csv_default,
        help=f"Output directory (default: {csv_default})"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["EURUSD", "GBPUSD"],
        help="Symbols to generate (default: EURUSD GBPUSD)"
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=["D1", "H1", "M5"],
        help="Timeframes to generate (default: D1 H1 M5)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing CSV files"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output)
    rng = np.random.default_rng(args.seed)

    print(f"Generating synthetic OHLCV data → {output_dir}")
    print(f"  symbols:    {args.symbols}")
    print(f"  timeframes: {args.timeframes}")
    print(f"  seed:       {args.seed}")
    print()

    for symbol in args.symbols:
        for tf in args.timeframes:
            try:
                generate(symbol, tf, output_dir, rng, force=args.force)
            except ValueError as exc:
                print(f"  ERROR: {exc}")

    print()
    print("Done.  Set data.source: csv in config.yaml to use these files.")


if __name__ == "__main__":
    main()
