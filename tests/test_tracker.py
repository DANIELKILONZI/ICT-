"""
Unit tests for python.performance.tracker (three-layer architecture)

Covers:
  - log_candidate() writes to candidates.csv
  - log_signal() writes to signals.csv and performance.csv
  - log_execution_feedback() writes to executions.csv (EXECUTED) or rejections.csv (REJECTED)
  - log_execution_feedback(REJECTED) marks the signal as REJECTED in signals.csv
  - statistics() excludes REJECTED rows from win/loss counts
  - statistics() includes rejected_by_ea count
  - update_trade_result() updates WIN/LOSS in both signals.csv and performance.csv
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

import python.config as cfg_mod
from python.performance.tracker import (
    log_candidate,
    log_execution_feedback,
    log_signal,
    statistics,
    update_trade_result,
)


# ---------------------------------------------------------------------------
# Fixtures: redirect all CSV paths to tmp_path
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_log_paths(tmp_path, monkeypatch):
    """Redirect every CSV path constant to isolated temp files."""
    monkeypatch.setattr(cfg_mod, "CANDIDATES_LOG", tmp_path / "candidates.csv")
    monkeypatch.setattr(cfg_mod, "SIGNALS_LOG",    tmp_path / "signals.csv")
    monkeypatch.setattr(cfg_mod, "PERF_LOG",       tmp_path / "performance.csv")
    monkeypatch.setattr(cfg_mod, "EXECUTIONS_LOG", tmp_path / "executions.csv")
    monkeypatch.setattr(cfg_mod, "REJECTIONS_LOG", tmp_path / "rejections.csv")

    # Reload the constants inside tracker (they're module-level references)
    import importlib
    import python.performance.tracker as tracker_mod
    monkeypatch.setattr(tracker_mod, "CANDIDATES_LOG", tmp_path / "candidates.csv")
    monkeypatch.setattr(tracker_mod, "SIGNALS_LOG",    tmp_path / "signals.csv")
    monkeypatch.setattr(tracker_mod, "PERF_LOG",       tmp_path / "performance.csv")
    monkeypatch.setattr(tracker_mod, "EXECUTIONS_LOG", tmp_path / "executions.csv")
    monkeypatch.setattr(tracker_mod, "REJECTIONS_LOG", tmp_path / "rejections.csv")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _make_signal(**overrides) -> dict:
    base = {
        "signal_id": "EURUSD-20260101-080000-ICT_FVG",
        "symbol": "EURUSD",
        "direction": "BUY",
        "entry_price": 1.08500,
        "stop_loss": 1.08300,
        "take_profit": 1.09000,
        "risk_percent": 1.0,
        "confidence_score": 0.75,
        "setup_type": "ICT_FVG_OB",
        "timestamp": "2026-01-01T08:00:00+00:00",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Layer 1 – candidates
# ---------------------------------------------------------------------------

class TestLogCandidate:
    def test_creates_file_on_first_write(self, tmp_path):
        path = tmp_path / "candidates.csv"
        import python.performance.tracker as t
        t.CANDIDATES_LOG = path
        log_candidate("EURUSD", "BUY", 0.80, "ICT_FVG_OB", ["reason1"])
        assert path.exists()

    def test_row_content(self, tmp_path):
        import python.performance.tracker as t
        t.CANDIDATES_LOG = tmp_path / "candidates.csv"
        log_candidate("GBPUSD", "SELL", 0.72, "ICT_OB_SWEEP", ["r1", "r2"])
        rows = _read_csv(t.CANDIDATES_LOG)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "GBPUSD"
        assert rows[0]["direction"] == "SELL"
        assert "0.72" in rows[0]["confidence_score"]
        assert "r1|r2" in rows[0]["reasons"]

    def test_multiple_candidates(self, tmp_path):
        import python.performance.tracker as t
        t.CANDIDATES_LOG = tmp_path / "candidates.csv"
        for i in range(3):
            log_candidate("EURUSD", "BUY", 0.70 + i * 0.05, "ICT_SETUP")
        rows = _read_csv(t.CANDIDATES_LOG)
        assert len(rows) == 3


# ---------------------------------------------------------------------------
# Layer 2 – signals
# ---------------------------------------------------------------------------

class TestLogSignal:
    def test_writes_to_signals_and_perf_logs(self, tmp_path):
        import python.performance.tracker as t
        t.SIGNALS_LOG = tmp_path / "signals.csv"
        t.PERF_LOG    = tmp_path / "performance.csv"
        sig = _make_signal()
        log_signal(sig)
        for path in (t.SIGNALS_LOG, t.PERF_LOG):
            rows = _read_csv(path)
            assert len(rows) == 1
            assert rows[0]["result"] == "OPEN"
            assert rows[0]["signal_id"] == sig["signal_id"]

    def test_result_defaults_to_open(self, tmp_path):
        import python.performance.tracker as t
        t.SIGNALS_LOG = tmp_path / "signals.csv"
        t.PERF_LOG    = tmp_path / "performance.csv"
        log_signal(_make_signal())
        rows = _read_csv(t.SIGNALS_LOG)
        assert rows[0]["result"] == "OPEN"


class TestUpdateTradeResult:
    def test_updates_to_win(self, tmp_path):
        import python.performance.tracker as t
        t.SIGNALS_LOG = tmp_path / "signals.csv"
        t.PERF_LOG    = tmp_path / "performance.csv"
        sig = _make_signal(entry_price=1.08500)
        log_signal(sig)
        update_trade_result("EURUSD", 1.08500, 1.09000, "BUY")
        rows = _read_csv(t.SIGNALS_LOG)
        assert rows[0]["result"] == "WIN"

    def test_updates_to_loss(self, tmp_path):
        import python.performance.tracker as t
        t.SIGNALS_LOG = tmp_path / "signals.csv"
        t.PERF_LOG    = tmp_path / "performance.csv"
        sig = _make_signal(entry_price=1.08500)
        log_signal(sig)
        update_trade_result("EURUSD", 1.08500, 1.08200, "BUY")
        rows = _read_csv(t.SIGNALS_LOG)
        assert rows[0]["result"] == "LOSS"


# ---------------------------------------------------------------------------
# Layer 3 – execution feedback
# ---------------------------------------------------------------------------

class TestLogExecutionFeedback:
    def test_executed_writes_to_executions(self, tmp_path):
        import python.performance.tracker as t
        t.EXECUTIONS_LOG = tmp_path / "executions.csv"
        t.REJECTIONS_LOG = tmp_path / "rejections.csv"
        log_execution_feedback({
            "signal_id": "EURUSD-20260101-080000-ICT_FVG",
            "symbol": "EURUSD",
            "direction": "BUY",
            "status": "EXECUTED",
            "lots": 0.10,
        })
        rows = _read_csv(t.EXECUTIONS_LOG)
        assert len(rows) == 1
        assert rows[0]["result"] == "EXECUTED"
        assert not _read_csv(t.REJECTIONS_LOG)

    def test_rejected_writes_to_rejections(self, tmp_path):
        import python.performance.tracker as t
        t.EXECUTIONS_LOG = tmp_path / "executions.csv"
        t.REJECTIONS_LOG = tmp_path / "rejections.csv"
        log_execution_feedback({
            "signal_id": "EURUSD-20260101-080000-ICT_FVG",
            "symbol": "EURUSD",
            "direction": "BUY",
            "status": "REJECTED",
            "reason": "SPREAD_TOO_HIGH",
            "spread_pips": 4.2,
        })
        rows = _read_csv(t.REJECTIONS_LOG)
        assert len(rows) == 1
        assert rows[0]["reason"] == "SPREAD_TOO_HIGH"
        assert not _read_csv(t.EXECUTIONS_LOG)

    def test_rejected_marks_signal_in_signals_csv(self, tmp_path):
        import python.performance.tracker as t
        t.SIGNALS_LOG    = tmp_path / "signals.csv"
        t.PERF_LOG       = tmp_path / "performance.csv"
        t.REJECTIONS_LOG = tmp_path / "rejections.csv"
        t.EXECUTIONS_LOG = tmp_path / "executions.csv"

        sig = _make_signal()
        log_signal(sig)
        # Confirm OPEN before feedback
        assert _read_csv(t.SIGNALS_LOG)[0]["result"] == "OPEN"

        log_execution_feedback({
            "signal_id": sig["signal_id"],
            "symbol": "EURUSD",
            "direction": "BUY",
            "status": "REJECTED",
            "reason": "BROKER_CONSTRAINT_FAILED",
        })
        rows = _read_csv(t.SIGNALS_LOG)
        assert rows[0]["result"] == "REJECTED"


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStatistics:
    def test_empty_returns_zero(self, tmp_path):
        import python.performance.tracker as t
        t.SIGNALS_LOG = tmp_path / "signals.csv"
        t.PERF_LOG    = tmp_path / "performance.csv"
        stats = statistics()
        assert stats["total"] == 0
        assert stats["win_rate"] == 0.0

    def test_win_loss_counting(self, tmp_path):
        import python.performance.tracker as t
        t.SIGNALS_LOG    = tmp_path / "signals.csv"
        t.PERF_LOG       = tmp_path / "performance.csv"
        t.REJECTIONS_LOG = tmp_path / "rejections.csv"
        t.EXECUTIONS_LOG = tmp_path / "executions.csv"

        for i in range(3):
            sig = _make_signal(signal_id=f"SIG-{i}", entry_price=1.08500)
            log_signal(sig)
        update_trade_result("EURUSD", 1.08500, 1.09000, "BUY")  # WIN
        update_trade_result("EURUSD", 1.08500, 1.08200, "BUY")  # LOSS

        stats = statistics()
        assert stats["wins"] == 1
        assert stats["losses"] == 1

    def test_rejected_not_counted_as_loss(self, tmp_path):
        import python.performance.tracker as t
        t.SIGNALS_LOG    = tmp_path / "signals.csv"
        t.PERF_LOG       = tmp_path / "performance.csv"
        t.REJECTIONS_LOG = tmp_path / "rejections.csv"
        t.EXECUTIONS_LOG = tmp_path / "executions.csv"

        sig = _make_signal()
        log_signal(sig)
        log_execution_feedback({
            "signal_id": sig["signal_id"],
            "symbol": "EURUSD",
            "direction": "BUY",
            "status": "REJECTED",
            "reason": "SPREAD_TOO_HIGH",
        })
        stats = statistics()
        assert stats["total"] == 0          # REJECTED is not a closed trade
        assert stats["rejected_by_ea"] == 1

    def test_rejected_by_ea_count(self, tmp_path):
        import python.performance.tracker as t
        t.SIGNALS_LOG    = tmp_path / "signals.csv"
        t.PERF_LOG       = tmp_path / "performance.csv"
        t.REJECTIONS_LOG = tmp_path / "rejections.csv"
        t.EXECUTIONS_LOG = tmp_path / "executions.csv"

        for i in range(3):
            s = _make_signal(signal_id=f"SIG-{i}")
            log_signal(s)
            log_execution_feedback({
                "signal_id": s["signal_id"],
                "symbol": "EURUSD",
                "direction": "BUY",
                "status": "REJECTED",
                "reason": "SPREAD_TOO_HIGH",
            })
        stats = statistics()
        assert stats["rejected_by_ea"] == 3
