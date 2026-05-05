"""
End-to-end tests for scripts/backtest_all_setups.py

Re-uses the session-scoped sample_csv_dir fixture from conftest.py so that
CSV generation happens only once per test session.

Each test runs the backtest over a small M5 slice (500 bars) to stay fast,
then asserts structural correctness regardless of whether any signals fired.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.backtest_all_setups import (  # noqa: E402
    run_backtest_setup1,
    run_backtest_setup2,
    run_backtest_setup3,
    run_backtest_setup4,
)

# ── Expected columns in every output record ───────────────────────────────────

_EXPECTED_COLUMNS = {
    "timestamp", "symbol", "direction",
    "entry_price", "stop_loss", "take_profit",
    "confluence_score", "outcome", "bars_held", "label",
    "d1_trend", "h1_trend", "atr_m5", "hour_utc",
    "m5_sweep", "price_zone", "h1_ob", "h1_fvg", "m5_fvg",
}

_VALID_OUTCOMES = {"WIN", "LOSS", "OPEN"}


# ── Shared data loading fixture ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_dataframes(sample_csv_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the generated CSVs.  Limit M5 to 500 bars for speed."""
    from python.data_engine.csv_loader import load_csv

    df_d1 = load_csv("EURUSD", "D1", csv_path=str(sample_csv_dir / "EURUSD_D1.csv"), count=0)
    df_h1 = load_csv("EURUSD", "H1", csv_path=str(sample_csv_dir / "EURUSD_H1.csv"), count=0)
    df_m5 = load_csv("EURUSD", "M5", csv_path=str(sample_csv_dir / "EURUSD_M5.csv"), count=500)
    return df_d1, df_h1, df_m5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_records_valid(records: list[dict]) -> None:
    for i, rec in enumerate(records):
        missing = _EXPECTED_COLUMNS - rec.keys()
        assert not missing, f"Record #{i} missing columns: {sorted(missing)}"
        assert rec["outcome"] in _VALID_OUTCOMES
        assert rec["label"] == (1 if rec["outcome"] == "WIN" else 0)


# ── Setup 1 ───────────────────────────────────────────────────────────────────

class TestBacktestSetup1:
    def test_returns_list(self, sample_dataframes):
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest_setup1(df_d1, df_h1, df_m5, "EURUSD",
                                      sweep_recency=50, max_bars=20)
        assert isinstance(records, list)

    def test_records_valid(self, sample_dataframes):
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest_setup1(df_d1, df_h1, df_m5, "EURUSD",
                                      sweep_recency=50, max_bars=20)
        _assert_records_valid(records)


# ── Setup 2 ───────────────────────────────────────────────────────────────────

class TestBacktestSetup2:
    def test_returns_list(self, sample_dataframes):
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest_setup2(df_d1, df_h1, df_m5, "EURUSD", max_bars=20)
        assert isinstance(records, list)

    def test_records_valid(self, sample_dataframes):
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest_setup2(df_d1, df_h1, df_m5, "EURUSD", max_bars=20)
        _assert_records_valid(records)


# ── Setup 3 ───────────────────────────────────────────────────────────────────

class TestBacktestSetup3:
    def test_returns_list(self, sample_dataframes):
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest_setup3(df_d1, df_h1, df_m5, "EURUSD",
                                      lkz_open=0, lkz_close=24,
                                      bos_lookback=3, max_bars=20)
        assert isinstance(records, list)

    def test_records_valid(self, sample_dataframes):
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest_setup3(df_d1, df_h1, df_m5, "EURUSD",
                                      lkz_open=0, lkz_close=24,
                                      bos_lookback=3, max_bars=20)
        _assert_records_valid(records)


# ── Setup 4 ───────────────────────────────────────────────────────────────────

class TestBacktestSetup4ViaAllSetups:
    def test_returns_list(self, sample_dataframes):
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest_setup4(df_d1, df_h1, df_m5, "EURUSD",
                                      ny_open=0, ny_close=24,
                                      bos_lookback=3, max_bars=20)
        assert isinstance(records, list)

    def test_records_valid(self, sample_dataframes):
        df_d1, df_h1, df_m5 = sample_dataframes
        records = run_backtest_setup4(df_d1, df_h1, df_m5, "EURUSD",
                                      ny_open=0, ny_close=24,
                                      bos_lookback=3, max_bars=20)
        _assert_records_valid(records)
