"""
Unit tests for python.performance.tracker

Covers:
  - log_signal() appends a row with result=OPEN
  - update_trade_result() updates the most recent OPEN trade for the symbol
  - update_trade_result() computes WIN / LOSS correctly for BUY and SELL
  - statistics() returns zeros when there are no closed trades
  - statistics() computes correct win_rate, avg_win_pips, avg_loss_pips, expectancy
  - statistics() handles all-wins edge case (avg_loss_pips=0)
  - statistics() handles all-losses edge case (avg_win_pips=0)
  - CSV round-trip: headers are always written, rows survive write→read→write
"""
from __future__ import annotations

import csv
import pytest

import python.performance.tracker as tracker_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _patch_perf_log(tmp_path, monkeypatch):
    """Redirect PERF_LOG to a temp file so tests don't touch the real log."""
    log_path = tmp_path / "perf_test.csv"
    monkeypatch.setattr(tracker_mod, "PERF_LOG", log_path)
    return log_path


def _signal(
    symbol: str = "EURUSD",
    direction: str = "BUY",
    entry: float = 1.10000,
    sl: float = 1.09000,
    tp: float = 1.12000,
    confidence: float = 0.80,
    setup: str = "ICT_SETUP",
) -> dict:
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "confidence_score": confidence,
        "setup_type": setup,
    }


# ---------------------------------------------------------------------------
# log_signal
# ---------------------------------------------------------------------------

class TestLogSignal:
    def test_creates_file_with_header(self, tmp_path, monkeypatch):
        log = _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal())
        assert log.exists()
        with open(log) as f:
            lines = list(csv.DictReader(f))
        assert len(lines) == 1

    def test_row_has_open_result(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal())
        rows = list(csv.DictReader(open(tracker_mod.PERF_LOG)))
        assert rows[0]["result"] == "OPEN"

    def test_row_symbol_matches(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(symbol="GBPUSD"))
        rows = list(csv.DictReader(open(tracker_mod.PERF_LOG)))
        assert rows[0]["symbol"] == "GBPUSD"

    def test_multiple_signals_appended(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(symbol="EURUSD"))
        tracker_mod.log_signal(_signal(symbol="GBPUSD"))
        rows = list(csv.DictReader(open(tracker_mod.PERF_LOG)))
        assert len(rows) == 2


# ---------------------------------------------------------------------------
# update_trade_result
# ---------------------------------------------------------------------------

class TestUpdateTradeResult:
    def test_buy_win_updates_result(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(entry=1.10000))
        tracker_mod.update_trade_result("EURUSD", 1.10000, 1.11000, "BUY")
        rows = list(csv.DictReader(open(tracker_mod.PERF_LOG)))
        assert rows[0]["result"] == "WIN"

    def test_buy_loss_updates_result(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(entry=1.10000))
        tracker_mod.update_trade_result("EURUSD", 1.10000, 1.09500, "BUY")
        rows = list(csv.DictReader(open(tracker_mod.PERF_LOG)))
        assert rows[0]["result"] == "LOSS"

    def test_sell_win_updates_result(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(direction="SELL", entry=1.10000))
        tracker_mod.update_trade_result("EURUSD", 1.10000, 1.09000, "SELL")
        rows = list(csv.DictReader(open(tracker_mod.PERF_LOG)))
        assert rows[0]["result"] == "WIN"

    def test_sell_loss_updates_result(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(direction="SELL", entry=1.10000))
        tracker_mod.update_trade_result("EURUSD", 1.10000, 1.11000, "SELL")
        rows = list(csv.DictReader(open(tracker_mod.PERF_LOG)))
        assert rows[0]["result"] == "LOSS"

    def test_pnl_pips_computed(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(entry=1.10000))
        tracker_mod.update_trade_result("EURUSD", 1.10000, 1.11000, "BUY")
        rows = list(csv.DictReader(open(tracker_mod.PERF_LOG)))
        # 1.11000 - 1.10000 = 0.01 price; EURUSD pip=0.0001 → 100 pips
        assert float(rows[0]["pnl_pips"]) == pytest.approx(100.0)

    def test_only_most_recent_open_is_updated(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(entry=1.10000))
        tracker_mod.log_signal(_signal(entry=1.10500))
        # Update the most recent open
        tracker_mod.update_trade_result("EURUSD", 1.10500, 1.11500, "BUY")
        rows = list(csv.DictReader(open(tracker_mod.PERF_LOG)))
        # Row 0 (first) should remain OPEN
        assert rows[0]["result"] == "OPEN"
        # Row 1 (second) should be updated to WIN
        assert rows[1]["result"] == "WIN"

    def test_no_update_for_wrong_symbol(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(symbol="EURUSD"))
        tracker_mod.update_trade_result("GBPUSD", 1.10000, 1.11000, "BUY")
        rows = list(csv.DictReader(open(tracker_mod.PERF_LOG)))
        assert rows[0]["result"] == "OPEN"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_no_closed_trades_returns_zeros(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        stats = tracker_mod.statistics()
        assert stats["total"] == 0
        assert stats["win_rate"] == 0.0
        assert stats["expectancy"] == 0.0
        assert stats["profit_factor"] is None

    def _populate(self, tmp_path, monkeypatch, outcomes: list[tuple[str, float]]) -> None:
        """Log signals and set their results. outcomes: [(result, pnl_pips)]."""
        _patch_perf_log(tmp_path, monkeypatch)
        for result, pnl in outcomes:
            tracker_mod.log_signal(_signal(symbol="EURUSD", direction="BUY", entry=1.10000))
        # Re-write the file directly with known pnl values
        rows = list(csv.DictReader(open(tracker_mod.PERF_LOG)))
        for row, (result, pnl) in zip(rows, outcomes):
            row["result"] = result
            row["pnl_pips"] = str(pnl)
            row["exit_price"] = "1.10000"
        with open(tracker_mod.PERF_LOG, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tracker_mod._HEADERS)
            writer.writeheader()
            writer.writerows(rows)

    def test_win_rate_one_win_one_loss(self, tmp_path, monkeypatch):
        self._populate(tmp_path, monkeypatch, [("WIN", 100.0), ("LOSS", -50.0)])
        stats = tracker_mod.statistics()
        assert stats["win_rate"] == pytest.approx(0.5)
        assert stats["wins"] == 1
        assert stats["losses"] == 1

    def test_avg_win_avg_loss(self, tmp_path, monkeypatch):
        self._populate(tmp_path, monkeypatch, [("WIN", 100.0), ("LOSS", -50.0)])
        stats = tracker_mod.statistics()
        assert stats["avg_win_pips"] == pytest.approx(100.0)
        assert stats["avg_loss_pips"] == pytest.approx(50.0)

    def test_expectancy(self, tmp_path, monkeypatch):
        # 2 wins of 100, 1 loss of 50
        self._populate(tmp_path, monkeypatch, [("WIN", 100.0), ("WIN", 100.0), ("LOSS", -50.0)])
        stats = tracker_mod.statistics()
        # win_rate = 2/3, avg_win = 100, avg_loss = 50
        # expectancy = (2/3 * 100) - (1/3 * 50) ≈ 50
        assert stats["expectancy_pips"] == pytest.approx(50.0, abs=0.1)

    def test_all_wins_avg_loss_is_zero(self, tmp_path, monkeypatch):
        self._populate(tmp_path, monkeypatch, [("WIN", 80.0), ("WIN", 120.0)])
        stats = tracker_mod.statistics()
        assert stats["avg_loss_pips"] == pytest.approx(0.0)
        assert stats["losses"] == 0
        # expectancy = 1.0 * avg_win - 0.0 = avg_win
        assert stats["expectancy_pips"] == pytest.approx(100.0)

    def test_all_losses_avg_win_is_zero(self, tmp_path, monkeypatch):
        self._populate(tmp_path, monkeypatch, [("LOSS", -80.0), ("LOSS", -40.0)])
        stats = tracker_mod.statistics()
        assert stats["avg_win_pips"] == pytest.approx(0.0)
        assert stats["wins"] == 0
        assert stats["expectancy_pips"] < 0

    def test_max_drawdown_non_negative(self, tmp_path, monkeypatch):
        self._populate(tmp_path, monkeypatch, [("WIN", 50.0), ("LOSS", -100.0), ("WIN", 30.0)])
        stats = tracker_mod.statistics()
        assert stats["max_drawdown_pips"] >= 0.0

    def test_profit_factor_none_when_no_losses(self, tmp_path, monkeypatch):
        self._populate(tmp_path, monkeypatch, [("WIN", 80.0), ("WIN", 120.0)])
        stats = tracker_mod.statistics()
        assert stats["profit_factor"] is None

    def test_profit_factor_correct(self, tmp_path, monkeypatch):
        # gross wins = 100 + 100 = 200, gross losses = 50 → PF = 4.0
        self._populate(tmp_path, monkeypatch, [("WIN", 100.0), ("WIN", 100.0), ("LOSS", -50.0)])
        stats = tracker_mod.statistics()
        assert stats["profit_factor"] == pytest.approx(4.0)

    def test_profit_factor_none_when_no_trades(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        stats = tracker_mod.statistics()
        assert stats["profit_factor"] is None


# ---------------------------------------------------------------------------
# has_open_trade
# ---------------------------------------------------------------------------

class TestHasOpenTrade:
    def test_returns_false_when_no_trades(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        assert tracker_mod.has_open_trade("EURUSD") is False

    def test_returns_true_for_open_symbol(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(symbol="EURUSD"))
        assert tracker_mod.has_open_trade("EURUSD") is True

    def test_returns_false_for_different_symbol(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(symbol="EURUSD"))
        assert tracker_mod.has_open_trade("GBPUSD") is False

    def test_returns_false_after_trade_closed(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(symbol="EURUSD", entry=1.10000))
        tracker_mod.update_trade_result("EURUSD", 1.10000, 1.11000, "BUY")
        assert tracker_mod.has_open_trade("EURUSD") is False

    def test_returns_true_when_one_of_two_still_open(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal(symbol="EURUSD", entry=1.10000))
        tracker_mod.log_signal(_signal(symbol="EURUSD", entry=1.10500))
        # Close only the most recent one
        tracker_mod.update_trade_result("EURUSD", 1.10500, 1.11500, "BUY")
        assert tracker_mod.has_open_trade("EURUSD") is True


# ---------------------------------------------------------------------------
# daily_loss_reached
# ---------------------------------------------------------------------------

class TestDailyLossReached:
    def _write_rows(self, tmp_path, monkeypatch, rows_override: list[dict]) -> None:
        log_path = _patch_perf_log(tmp_path, monkeypatch)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=tracker_mod._HEADERS)
            writer.writeheader()
            writer.writerows(rows_override)

    def _today_ts(self) -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()

    def test_no_trades_returns_false(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        assert tracker_mod.daily_loss_reached() is False

    def test_only_open_trades_returns_false(self, tmp_path, monkeypatch):
        _patch_perf_log(tmp_path, monkeypatch)
        tracker_mod.log_signal(_signal())
        assert tracker_mod.daily_loss_reached() is False

    def test_losses_below_limit_returns_false(self, tmp_path, monkeypatch):
        # 2 losses × 1% risk = 2% < 3% limit
        rows = [
            {h: "" for h in tracker_mod._HEADERS}
            | {"symbol": "EURUSD", "result": "LOSS",
               "risk_percent": "1.0", "pnl_pips": "-50",
               "timestamp": self._today_ts()}
            for _ in range(2)
        ]
        self._write_rows(tmp_path, monkeypatch, rows)
        assert tracker_mod.daily_loss_reached() is False

    def test_losses_at_limit_returns_true(self, tmp_path, monkeypatch):
        # 3 losses × 1% = 3% == limit (3%)
        rows = [
            {h: "" for h in tracker_mod._HEADERS}
            | {"symbol": "EURUSD", "result": "LOSS",
               "risk_percent": "1.0", "pnl_pips": "-50",
               "timestamp": self._today_ts()}
            for _ in range(3)
        ]
        self._write_rows(tmp_path, monkeypatch, rows)
        assert tracker_mod.daily_loss_reached() is True

    def test_old_losses_are_ignored(self, tmp_path, monkeypatch):
        # Losses from yesterday should not count
        rows = [
            {h: "" for h in tracker_mod._HEADERS}
            | {"symbol": "EURUSD", "result": "LOSS",
               "risk_percent": "1.0", "pnl_pips": "-50",
               "timestamp": "2000-01-01T00:00:00+00:00"}
            for _ in range(10)
        ]
        self._write_rows(tmp_path, monkeypatch, rows)
        assert tracker_mod.daily_loss_reached() is False

    def test_wins_do_not_reduce_loss_total(self, tmp_path, monkeypatch):
        # 3 wins should not prevent the guard from triggering when 3 losses hit the limit
        today = self._today_ts()
        rows = (
            [
                {h: "" for h in tracker_mod._HEADERS}
                | {"symbol": "EURUSD", "result": "WIN",
                   "risk_percent": "1.0", "pnl_pips": "100",
                   "timestamp": today}
                for _ in range(3)
            ]
            + [
                {h: "" for h in tracker_mod._HEADERS}
                | {"symbol": "EURUSD", "result": "LOSS",
                   "risk_percent": "1.0", "pnl_pips": "-50",
                   "timestamp": today}
                for _ in range(3)
            ]
        )
        self._write_rows(tmp_path, monkeypatch, rows)
        assert tracker_mod.daily_loss_reached() is True

