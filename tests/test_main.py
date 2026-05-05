"""
Integration tests for python.main.run_analysis_cycle()

Exercises the full cycle with mocked data/analysis to confirm that:
  - the daily-loss guard suppresses all signal output
  - the max-trades-per-day guard suppresses all signal output
  - the has_open_trade guard suppresses signal output for that symbol
  - a valid cycle (all guards pass) saves and logs a signal
  - being outside the trading session skips analysis
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import python.main as main_mod
from python.strategy_engine.market_structure import TrendDirection
from python.strategy_engine.mtf_engine import MTFAnalysis


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _valid_analysis(symbol: str = "EURUSD") -> MTFAnalysis:
    return MTFAnalysis(
        symbol=symbol,
        d1_trend=TrendDirection.BULLISH,
        h1_trend=TrendDirection.BULLISH,
        h1_bos_direction="BULLISH",
        m5_sweep=None,
        m5_fvg=None,
        h1_ob=None,
        h1_fvg=None,
        fib_range=None,
        price_zone="DISCOUNT",
        signal_direction="BUY",
        entry_price=1.10000,
        stop_loss=1.09000,
        take_profit=1.12200,
        confluence_score=0.85,
        valid=True,
        reasons=["bias aligned"],
    )


def _nonempty_df() -> pd.DataFrame:
    """Minimal OHLCV DataFrame that passes the empty-check in the cycle."""
    return pd.DataFrame(
        {
            "time":   pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC"),
            "open":   [1.10] * 5,
            "high":   [1.11] * 5,
            "low":    [1.09] * 5,
            "close":  [1.10] * 5,
            "volume": [1000] * 5,
        }
    )


# ---------------------------------------------------------------------------
# Guard: daily loss limit
# ---------------------------------------------------------------------------

class TestDailyLossGuard:
    def test_no_signal_when_daily_loss_reached(self):
        """When daily_loss_reached() is True the cycle returns immediately."""
        with (
            patch.object(main_mod, "_in_trading_session", return_value=True),
            patch.object(main_mod, "daily_loss_reached",  return_value=True),
            patch.object(main_mod, "save_signal") as mock_save,
            patch.object(main_mod, "log_signal")  as mock_log,
        ):
            main_mod.run_analysis_cycle()

        mock_save.assert_not_called()
        mock_log.assert_not_called()


# ---------------------------------------------------------------------------
# Guard: max trades per day
# ---------------------------------------------------------------------------

class TestMaxTradesGuard:
    def test_no_signal_when_max_trades_reached(self):
        """Cycle returns early when trades_today_count() >= max_trades_per_day."""
        with (
            patch.object(main_mod, "_in_trading_session",   return_value=True),
            patch.object(main_mod, "daily_loss_reached",    return_value=False),
            patch.object(main_mod, "trades_today_count",    return_value=999),
            patch.object(main_mod, "save_signal") as mock_save,
            patch.object(main_mod, "log_signal")  as mock_log,
        ):
            main_mod.run_analysis_cycle()

        mock_save.assert_not_called()
        mock_log.assert_not_called()

    def test_signal_emitted_when_below_max_trades(self):
        """Signal IS emitted when today's count is below the limit."""
        fake_signal = {
            "symbol":      "EURUSD",
            "direction":   "BUY",
            "entry_price": 1.10000,
            "stop_loss":   1.09000,
            "take_profit": 1.12000,
        }
        with (
            patch.object(main_mod, "_in_trading_session",   return_value=True),
            patch.object(main_mod, "daily_loss_reached",    return_value=False),
            patch.object(main_mod, "trades_today_count",    return_value=0),
            patch.object(main_mod, "has_open_trade",        return_value=False),
            patch.object(main_mod, "get_ohlcv",             return_value=_nonempty_df()),
            patch.object(main_mod, "analyse",               return_value=_valid_analysis()),
            patch.object(main_mod, "passes_ml_filter",      return_value=True),
            patch.object(main_mod, "generate_signal",       return_value=fake_signal),
            patch.object(main_mod, "save_signal") as mock_save,
            patch.object(main_mod, "log_signal")  as mock_log,
            patch.object(main_mod, "notify_signal"),
            patch.object(main_mod, "SYMBOLS", ["EURUSD"]),
        ):
            main_mod.run_analysis_cycle()

        mock_save.assert_called_once_with(fake_signal)
        mock_log.assert_called_once_with(fake_signal)


# ---------------------------------------------------------------------------
# Guard: open trade for symbol
# ---------------------------------------------------------------------------

class TestOpenTradeGuard:
    def test_no_signal_when_open_trade_exists(self):
        """has_open_trade suppresses new signal for that symbol."""
        with (
            patch.object(main_mod, "_in_trading_session", return_value=True),
            patch.object(main_mod, "daily_loss_reached",  return_value=False),
            patch.object(main_mod, "trades_today_count",  return_value=0),
            patch.object(main_mod, "has_open_trade",      return_value=True),
            patch.dict(main_mod.CONFIG["risk"], {"allow_multiple_positions_per_symbol": False}),
            patch.object(main_mod, "save_signal") as mock_save,
            patch.object(main_mod, "log_signal")  as mock_log,
            patch.object(main_mod, "SYMBOLS", ["EURUSD"]),
        ):
            main_mod.run_analysis_cycle()

        mock_save.assert_not_called()
        mock_log.assert_not_called()


# ---------------------------------------------------------------------------
# Guard: outside trading session
# ---------------------------------------------------------------------------

class TestTradingSessionGuard:
    def test_no_signal_outside_session(self):
        with (
            patch.object(main_mod, "_in_trading_session", return_value=False),
            patch.object(main_mod, "save_signal") as mock_save,
            patch.object(main_mod, "log_signal")  as mock_log,
        ):
            main_mod.run_analysis_cycle()

        mock_save.assert_not_called()
        mock_log.assert_not_called()


# ---------------------------------------------------------------------------
# Happy path: full cycle emits signal
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_signal_saved_and_logged_on_valid_cycle(self):
        fake_signal = {
            "symbol":      "EURUSD",
            "direction":   "BUY",
            "entry_price": 1.10000,
            "stop_loss":   1.09000,
            "take_profit": 1.12000,
        }
        with (
            patch.object(main_mod, "_in_trading_session", return_value=True),
            patch.object(main_mod, "daily_loss_reached",  return_value=False),
            patch.object(main_mod, "trades_today_count",  return_value=0),
            patch.object(main_mod, "has_open_trade",      return_value=False),
            patch.object(main_mod, "get_ohlcv",           return_value=_nonempty_df()),
            patch.object(main_mod, "analyse",             return_value=_valid_analysis()),
            patch.object(main_mod, "passes_ml_filter",    return_value=True),
            patch.object(main_mod, "generate_signal",     return_value=fake_signal),
            patch.object(main_mod, "save_signal") as mock_save,
            patch.object(main_mod, "log_signal")  as mock_log,
            patch.object(main_mod, "notify_signal"),
            patch.object(main_mod, "SYMBOLS", ["EURUSD"]),
        ):
            main_mod.run_analysis_cycle()

        mock_save.assert_called_once_with(fake_signal)
        mock_log.assert_called_once_with(fake_signal)

    def test_no_signal_when_analysis_invalid(self):
        """Cycle does not emit when analyse() returns valid=False."""
        invalid = _valid_analysis()
        object.__setattr__(invalid, "valid", False)
        with (
            patch.object(main_mod, "_in_trading_session", return_value=True),
            patch.object(main_mod, "daily_loss_reached",  return_value=False),
            patch.object(main_mod, "trades_today_count",  return_value=0),
            patch.object(main_mod, "has_open_trade",      return_value=False),
            patch.object(main_mod, "get_ohlcv",           return_value=_nonempty_df()),
            patch.object(main_mod, "analyse",             return_value=invalid),
            patch.object(main_mod, "save_signal") as mock_save,
            patch.object(main_mod, "log_signal")  as mock_log,
            patch.object(main_mod, "SYMBOLS", ["EURUSD"]),
        ):
            main_mod.run_analysis_cycle()

        mock_save.assert_not_called()
        mock_log.assert_not_called()

    def test_no_signal_when_ml_filter_rejects(self):
        """Cycle does not emit when passes_ml_filter() returns False."""
        with (
            patch.object(main_mod, "_in_trading_session", return_value=True),
            patch.object(main_mod, "daily_loss_reached",  return_value=False),
            patch.object(main_mod, "trades_today_count",  return_value=0),
            patch.object(main_mod, "has_open_trade",      return_value=False),
            patch.object(main_mod, "get_ohlcv",           return_value=_nonempty_df()),
            patch.object(main_mod, "analyse",             return_value=_valid_analysis()),
            patch.object(main_mod, "passes_ml_filter",    return_value=False),
            patch.object(main_mod, "save_signal") as mock_save,
            patch.object(main_mod, "log_signal")  as mock_log,
            patch.object(main_mod, "SYMBOLS", ["EURUSD"]),
        ):
            main_mod.run_analysis_cycle()

        mock_save.assert_not_called()
        mock_log.assert_not_called()
