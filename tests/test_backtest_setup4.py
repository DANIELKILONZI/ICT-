"""
End-to-end test for scripts/backtest_setup4.py

Generates synthetic EURUSD CSV data (D1 / H1 / M5) using
scripts/generate_sample_data.py, then runs the backtest against a small
slice of that data and asserts:
  - the run_backtest() function returns a list (may be empty on small data)
  - when records are returned they contain the required columns
  - forward-simulation outcomes are one of WIN / LOSS / OPEN
  - label column is consistent with outcome

The fixture uses a short slice (300 M5 bars ≈ 1 day) to keep the test fast.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make sure the repo root is importable
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.generate_sample_data import generate  # noqa: E402
from scripts.backtest_setup4 import run_backtest    # noqa: E402

# ── Expected columns in every output record ───────────────────────────────────

_EXPECTED_COLUMNS = {
    "timestamp",
    "symbol",
    "direction",
    "entry_price",
    "stop_loss",
    "take_profit",
    "confluence_score",
    "outcome",
    "bars_held",
    "label",
    # ML feature columns
    "d1_trend",
    "h1_trend",
    "atr_m5",
    "hour_utc",
    "m5_sweep",
    "price_zone",
    "h1_ob",
    "h1_fvg",
    "m5_fvg",
}

_VALID_OUTCOMES = {"WIN", "LOSS", "OPEN"}


# ── Fixture: generate CSV data once per test session ─────────────────────────

@pytest.fixture(scope="module")
def sample_csv_dir(tmp_path_factory) -> Path:
    """Write EURUSD D1/H1/M5 CSVs to a temporary directory."""
    csv_dir = tmp_path_factory.mktemp("sample_csv")
    rng = np.random.default_rng(seed=42)
    for tf in ("D1", "H1", "M5"):
        generate("EURUSD", tf, csv_dir, rng, force=True)
    return csv_dir


@pytest.fixture(scope="module")
def sample_dataframes(sample_csv_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the generated CSVs. Use a small M5 tail to keep the test fast."""
    from python.data_engine.csv_loader import load_csv

    df_d1 = load_csv("EURUSD", "D1", csv_path=str(sample_csv_dir / "EURUSD_D1.csv"), count=0)
    df_h1 = load_csv("EURUSD", "H1", csv_path=str(sample_csv_dir / "EURUSD_H1.csv"), count=0)
    # Limit M5 to 500 bars so the bar-by-bar scan finishes quickly
    df_m5 = load_csv("EURUSD", "M5", csv_path=str(sample_csv_dir / "EURUSD_M5.csv"), count=500)
    return df_d1, df_h1, df_m5


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestBacktestSetup4:
    def test_sample_data_loads_correctly(self, sample_dataframes):
        """The generated CSVs must load as non-empty DataFrames with OHLCV columns."""
        df_d1, df_h1, df_m5 = sample_dataframes
        for label, df in [("D1", df_d1), ("H1", df_h1), ("M5", df_m5)]:
            assert not df.empty, f"{label} DataFrame is empty"
            for col in ("time", "open", "high", "low", "close", "volume"):
                assert col in df.columns, f"Missing column '{col}' in {label}"

    def test_backtest_returns_list(self, sample_dataframes):
        """run_backtest() must return a list (may be empty for small datasets)."""
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest(
            df_d1, df_h1, df_m5,
            symbol="EURUSD",
            ny_open=0,
            ny_close=24,
            bos_lookback=3,
            max_bars=20,
        )
        assert isinstance(records, list)

    def test_records_have_required_columns(self, sample_dataframes):
        """Every record dict must contain all expected column names."""
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest(
            df_d1, df_h1, df_m5,
            symbol="EURUSD",
            ny_open=0,
            ny_close=24,
            bos_lookback=3,
            max_bars=20,
        )
        for i, rec in enumerate(records):
            missing = _EXPECTED_COLUMNS - rec.keys()
            assert not missing, (
                f"Record #{i} is missing columns: {sorted(missing)}"
            )

    def test_outcomes_are_valid(self, sample_dataframes):
        """outcome field must be one of WIN / LOSS / OPEN."""
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest(
            df_d1, df_h1, df_m5,
            symbol="EURUSD",
            ny_open=0,
            ny_close=24,
            bos_lookback=3,
            max_bars=20,
        )
        for rec in records:
            assert rec["outcome"] in _VALID_OUTCOMES, (
                f"Unexpected outcome value: {rec['outcome']!r}"
            )

    def test_label_matches_outcome(self, sample_dataframes):
        """label column must be 1 for WIN and 0 for LOSS/OPEN."""
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest(
            df_d1, df_h1, df_m5,
            symbol="EURUSD",
            ny_open=0,
            ny_close=24,
            bos_lookback=3,
            max_bars=20,
        )
        for rec in records:
            expected_label = 1 if rec["outcome"] == "WIN" else 0
            assert rec["label"] == expected_label, (
                f"label={rec['label']} but outcome={rec['outcome']!r}"
            )

    def test_dataframe_from_records_is_valid(self, sample_dataframes):
        """Converting records to a DataFrame should succeed and have expected columns."""
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest(
            df_d1, df_h1, df_m5,
            symbol="EURUSD",
            ny_open=0,
            ny_close=24,
            bos_lookback=3,
            max_bars=20,
        )
        if not records:
            pytest.skip("No signals fired on this small slice – column check skipped.")
        out_df = pd.DataFrame(records)
        for col in _EXPECTED_COLUMNS:
            assert col in out_df.columns, f"Output DataFrame missing column '{col}'"

