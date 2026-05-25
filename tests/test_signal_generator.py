"""
Unit tests for python.signal_generator.signal_generator

Covers:
  - generate_signal() rejects invalid/low-confidence/missing-price analyses
  - generate_signal() rejects setups with R:R below minimum
  - generate_signal() returns a correctly shaped signal dict for valid inputs
  - New required fields: signal_id, created_at, expires_at, valid_from,
    engine_cycle_id, status, payload_hash
  - save_signal() writes per-symbol JSON atomically
  - load_latest_signal(symbol) round-trip
  - load_latest_signal(symbol) returns None for expired signals
  - String fields in the output signal contain no characters that would
    break the MQL5 simple JSON parser
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

import pytest

from python.strategy_engine.market_structure import TrendDirection
from python.strategy_engine.mtf_engine import MTFAnalysis
from python.signal_generator.signal_generator import (
    _risk_reward,
    generate_signal,
    load_latest_signal,
    save_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_SIGNAL_KEYS = {
    "signal_id", "created_at", "expires_at", "valid_from",
    "engine_cycle_id", "status", "payload_hash",
    "symbol", "direction", "entry_price", "stop_loss", "take_profit",
    "risk_reward", "risk_percent", "confidence_score", "timestamp",
    "setup_type",
}


def _make_analysis(**overrides) -> MTFAnalysis:
    defaults = dict(
        symbol="EURUSD",
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
        stop_loss=1.09000,    # 100 pip risk
        take_profit=1.12200,  # 220 pip reward → 2.2:1 raw RR;
                              # spread-adjusted (1 pip BUY): adj_entry=1.10010, adj_sl=1.08990
                              # adj_rr = (1.122 - 1.10010) / (1.10010 - 1.08990) ≈ 2.18 > 2.0
        confluence_score=0.80,
        valid=True,
        reasons=["D1 bullish + H1 bullish BOS"],
    )
    defaults.update(overrides)
    return MTFAnalysis(**defaults)


# ---------------------------------------------------------------------------
# _risk_reward helper
# ---------------------------------------------------------------------------

class TestRiskReward:
    def test_basic_2_to_1(self):
        assert _risk_reward(1.1000, 1.0900, 1.1200) == pytest.approx(2.0)

    def test_zero_risk_returns_zero(self):
        assert _risk_reward(1.1000, 1.1000, 1.1200) == 0.0

    def test_sell_rr(self):
        assert _risk_reward(1.1000, 1.1100, 1.0800) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# generate_signal rejections
# ---------------------------------------------------------------------------

class TestGenerateSignalRejections:
    def test_none_when_analysis_not_valid(self):
        a = _make_analysis(valid=False)
        assert generate_signal(a) is None

    def test_none_when_low_confidence(self):
        a = _make_analysis(confluence_score=0.10)
        assert generate_signal(a) is None

    def test_none_when_entry_price_none(self):
        a = _make_analysis(entry_price=None, stop_loss=None, take_profit=None)
        assert generate_signal(a) is None

    def test_none_when_stop_loss_none(self):
        a = _make_analysis(stop_loss=None, take_profit=None)
        assert generate_signal(a) is None

    def test_none_when_take_profit_none(self):
        a = _make_analysis(take_profit=None)
        assert generate_signal(a) is None

    def test_none_when_rr_below_minimum(self):
        # entry=1.1, sl=1.099 (1 pip risk), tp=1.1005 (0.5 pip reward) → RR≈0.5
        a = _make_analysis(
            entry_price=1.1000,
            stop_loss=1.0990,
            take_profit=1.1005,
        )
        assert generate_signal(a) is None

    def test_none_when_direction_is_none(self):
        a = _make_analysis(signal_direction=None, valid=False)
        assert generate_signal(a) is None


# ---------------------------------------------------------------------------
# generate_signal valid path
# ---------------------------------------------------------------------------

class TestGenerateSignalValid:
    def _valid_signal(self, cycle_id=""):
        return generate_signal(_make_analysis(), cycle_id=cycle_id)

    def test_returns_dict(self):
        assert isinstance(self._valid_signal(), dict)

    def test_required_keys_present(self):
        sig = self._valid_signal()
        for key in _REQUIRED_SIGNAL_KEYS:
            assert key in sig, f"Missing key: {key}"

    def test_direction_buy(self):
        assert self._valid_signal()["direction"] == "BUY"

    def test_prices_rounded(self):
        sig = self._valid_signal()
        assert isinstance(sig["entry_price"], float)

    def test_setup_type_is_ict_prefixed(self):
        assert self._valid_signal()["setup_type"].startswith("ICT")

    def test_confidence_score_between_0_and_1(self):
        sig = self._valid_signal()
        assert 0.0 <= sig["confidence_score"] <= 1.0

    def test_status_is_active(self):
        sig = self._valid_signal()
        assert sig["status"] == "ACTIVE"

    def test_signal_id_contains_symbol(self):
        sig = self._valid_signal()
        assert "EURUSD" in sig["signal_id"]

    def test_expires_at_after_created_at(self):
        sig = self._valid_signal()
        created = datetime.fromisoformat(sig["created_at"].replace("Z", "+00:00"))
        expires = datetime.fromisoformat(sig["expires_at"].replace("Z", "+00:00"))
        assert expires > created

    def test_engine_cycle_id_propagated(self):
        sig = self._valid_signal(cycle_id="cycle-000042")
        assert sig["engine_cycle_id"] == "cycle-000042"

    def test_payload_hash_format(self):
        sig = self._valid_signal()
        assert sig["payload_hash"].startswith("sha256:")

    def test_payload_hash_covers_payload(self):
        """Hash computed on the same payload should be deterministic."""
        from python.signal_generator.signal_generator import _compute_payload_hash
        sig = self._valid_signal()
        expected = _compute_payload_hash({k: v for k, v in sig.items() if k != "payload_hash"})
        assert sig["payload_hash"] == expected


# ---------------------------------------------------------------------------
# Signal string safety (Issue #12)
# ---------------------------------------------------------------------------

class TestSignalStringSafety:
    """
    Ensure no string field in the emitted signal contains characters that
    break the MQL5 simple JSON parser (unescaped quotes, backslashes, or
    control characters).
    """

    def _string_fields(self, sig: dict) -> dict[str, str]:
        return {k: v for k, v in sig.items() if isinstance(v, str)}

    def test_no_raw_double_quotes_in_string_values(self):
        sig = generate_signal(_make_analysis())
        for k, v in self._string_fields(sig).items():
            assert '"' not in v, f"Field {k!r} contains a raw double-quote"

    def test_no_backslashes_in_string_values(self):
        sig = generate_signal(_make_analysis())
        for k, v in self._string_fields(sig).items():
            assert '\\' not in v, f"Field {k!r} contains a backslash"

    def test_signal_is_valid_json(self):
        sig = generate_signal(_make_analysis())
        dumped = json.dumps(sig)
        reloaded = json.loads(dumped)
        assert reloaded["symbol"] == sig["symbol"]


# ---------------------------------------------------------------------------
# save_signal / load_latest_signal round-trip
# ---------------------------------------------------------------------------

class TestSaveLoadRoundTrip:
    def test_round_trip(self, tmp_path, monkeypatch):
        import python.config as cfg_mod

        active_dir = tmp_path / "active"
        active_dir.mkdir()
        monkeypatch.setattr(cfg_mod, "SIGNAL_ACTIVE_DIR", active_dir)

        sig = generate_signal(_make_analysis())
        save_signal(sig)

        expected = active_dir / "EURUSD.json"
        assert expected.exists()
        data = json.loads(expected.read_text())
        assert data["symbol"] == "EURUSD"

    def test_load_after_save(self, tmp_path, monkeypatch):
        import python.config as cfg_mod

        active_dir = tmp_path / "active"
        active_dir.mkdir()
        monkeypatch.setattr(cfg_mod, "SIGNAL_ACTIVE_DIR", active_dir)

        sig = generate_signal(_make_analysis())
        save_signal(sig)

        loaded = load_latest_signal("EURUSD")
        assert loaded is not None
        assert loaded["symbol"] == "EURUSD"
        assert loaded["signal_id"] == sig["signal_id"]

    def test_load_returns_none_for_unknown_symbol(self, tmp_path, monkeypatch):
        import python.config as cfg_mod

        active_dir = tmp_path / "active"
        active_dir.mkdir()
        monkeypatch.setattr(cfg_mod, "SIGNAL_ACTIVE_DIR", active_dir)

        assert load_latest_signal("GBPUSD") is None

    def test_load_returns_none_when_expired(self, tmp_path, monkeypatch):
        import python.config as cfg_mod

        active_dir = tmp_path / "active"
        active_dir.mkdir()
        monkeypatch.setattr(cfg_mod, "SIGNAL_ACTIVE_DIR", active_dir)

        sig = generate_signal(_make_analysis())
        # Manually override expires_at to a past time
        past = datetime.now(timezone.utc) - timedelta(seconds=10)
        sig["expires_at"] = past.strftime("%Y-%m-%dT%H:%M:%SZ")
        save_signal(sig)

        assert load_latest_signal("EURUSD") is None
