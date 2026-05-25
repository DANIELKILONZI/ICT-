"""Tests for the portfolio correlation exposure guard."""
from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def _risk_config():
    """Patch RISK config with correlation groups."""
    risk = {
        "risk_percent": 1.0,
        "max_daily_loss_percent": 3.0,
        "max_trades_per_day": 5,
        "max_spread_pips": 3.0,
        "max_slippage_pips": 2.0,
        "correlation_groups": {
            "USD_MAJORS": ["EURUSD", "GBPUSD", "USDJPY"],
            "METALS": ["XAUUSD"],
        },
        "max_correlated_exposure_percent": 2.0,
        "default_session": {},
        "sessions": {},
    }
    with patch("python.performance.portfolio_exposure.RISK", risk):
        yield risk


@pytest.fixture
def signals_csv(tmp_path):
    """Create a temporary signals CSV and patch SIGNALS_LOG."""
    csv_path = tmp_path / "signals.csv"
    with patch("python.performance.portfolio_exposure.SIGNALS_LOG", csv_path):
        yield csv_path


def _write_signals(path: Path, rows: list[dict]):
    """Write a signals CSV file with the given rows."""
    fieldnames = ["timestamp", "symbol", "direction", "result", "risk_percent", "signal_id"]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestCorrelationGroupLookup:
    def test_symbol_in_group(self, _risk_config):
        from python.performance.portfolio_exposure import _correlation_group_for

        assert _correlation_group_for("EURUSD") == "USD_MAJORS"
        assert _correlation_group_for("eurusd") == "USD_MAJORS"
        assert _correlation_group_for("XAUUSD") == "METALS"

    def test_symbol_not_in_any_group(self, _risk_config):
        from python.performance.portfolio_exposure import _correlation_group_for

        assert _correlation_group_for("NZDUSD") is None


class TestPortfolioExposureAllowed:
    def test_no_group_always_allowed(self, _risk_config, signals_csv):
        from python.performance.portfolio_exposure import is_portfolio_exposure_allowed

        # Symbol not in any group → always passes
        assert is_portfolio_exposure_allowed("NZDUSD", 1.0) is True

    def test_no_open_trades_allowed(self, _risk_config, signals_csv):
        from python.performance.portfolio_exposure import is_portfolio_exposure_allowed

        # No signals file → no open risk → allowed
        assert is_portfolio_exposure_allowed("EURUSD", 1.0) is True

    def test_within_limit(self, _risk_config, signals_csv):
        from python.performance.portfolio_exposure import is_portfolio_exposure_allowed

        _write_signals(signals_csv, [
            {"timestamp": "2026-05-25T10:00:00Z", "symbol": "GBPUSD",
             "direction": "BUY", "result": "OPEN", "risk_percent": "0.5",
             "signal_id": "sig-1"},
        ])
        # 0.5 + 1.0 = 1.5 <= 2.0 → allowed
        assert is_portfolio_exposure_allowed("EURUSD", 1.0) is True

    def test_exceeds_limit(self, _risk_config, signals_csv):
        from python.performance.portfolio_exposure import is_portfolio_exposure_allowed

        _write_signals(signals_csv, [
            {"timestamp": "2026-05-25T10:00:00Z", "symbol": "GBPUSD",
             "direction": "BUY", "result": "OPEN", "risk_percent": "1.0",
             "signal_id": "sig-1"},
            {"timestamp": "2026-05-25T10:30:00Z", "symbol": "USDJPY",
             "direction": "SELL", "result": "OPEN", "risk_percent": "0.5",
             "signal_id": "sig-2"},
        ])
        # 1.0 + 0.5 + 1.0 = 2.5 > 2.0 → blocked
        assert is_portfolio_exposure_allowed("EURUSD", 1.0) is False

    def test_closed_trades_not_counted(self, _risk_config, signals_csv):
        from python.performance.portfolio_exposure import is_portfolio_exposure_allowed

        _write_signals(signals_csv, [
            {"timestamp": "2026-05-25T10:00:00Z", "symbol": "GBPUSD",
             "direction": "BUY", "result": "WIN", "risk_percent": "1.0",
             "signal_id": "sig-1"},
            {"timestamp": "2026-05-25T10:30:00Z", "symbol": "USDJPY",
             "direction": "SELL", "result": "LOSS", "risk_percent": "1.0",
             "signal_id": "sig-2"},
        ])
        # Both are closed → current risk = 0 → 0 + 1.0 <= 2.0 → allowed
        assert is_portfolio_exposure_allowed("EURUSD", 1.0) is True

    def test_different_group_not_counted(self, _risk_config, signals_csv):
        from python.performance.portfolio_exposure import is_portfolio_exposure_allowed

        _write_signals(signals_csv, [
            {"timestamp": "2026-05-25T10:00:00Z", "symbol": "XAUUSD",
             "direction": "BUY", "result": "OPEN", "risk_percent": "1.5",
             "signal_id": "sig-1"},
        ])
        # XAUUSD is in METALS, not USD_MAJORS → EURUSD's group has 0 risk
        assert is_portfolio_exposure_allowed("EURUSD", 1.0) is True
