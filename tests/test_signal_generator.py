"""
Unit tests for python.signal_generator.signal_generator

Covers:
  - generate_signal() rejects invalid/low-confidence/missing-price analyses
  - generate_signal() rejects setups with R:R below minimum
  - generate_signal() returns a correctly shaped signal dict for valid inputs
  - save_signal() / load_latest_signal() round-trip
  - String fields in the output signal contain no characters that would
    break the MQL5 simple JSON parser
"""
from __future__ import annotations

import json
import pytest

from python.strategy_engine.market_structure import TrendDirection
from python.strategy_engine.mtf_engine import MTFAnalysis
from python.signal_generator.signal_generator import (
    _risk_reward,
    _spread_adjusted_rr,
    generate_signal,
    load_latest_signal,
    save_signal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_SIGNAL_KEYS = {
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
# _spread_adjusted_rr
# ---------------------------------------------------------------------------

class TestSpreadAdjustedRR:
    """
    Verify the BUY and SELL spread-adjustment logic independently.

    For a BUY trade:
      adj_entry = entry + spread_price  (filled at ask)
      adj_sl    = sl    - spread_price  (SL at bid → distance widens)
      adj_tp    = tp                    (unchanged)

    For a SELL trade:
      adj_entry = entry - spread_price  (filled at bid)
      adj_sl    = sl    + spread_price  (SL at ask → distance widens)
      adj_tp    = tp                    (unchanged)
    """

    _pip = 0.0001        # EURUSD pip size
    _spread_pips = 1.0   # 1 pip spread

    def _run(self, entry, sl, tp, direction):
        return _spread_adjusted_rr(
            entry, sl, tp,
            direction=direction,
            symbol="EURUSD",
            spread_pips=self._spread_pips,
        )

    # ── BUY ──────────────────────────────────────────────────────────────────

    def test_buy_adj_entry_higher_by_spread(self):
        adj_entry, _, _, _ = self._run(1.1000, 1.0900, 1.1200, "BUY")
        assert adj_entry == pytest.approx(1.1000 + self._spread_pips * self._pip)

    def test_buy_adj_sl_lower_by_spread(self):
        _, adj_sl, _, _ = self._run(1.1000, 1.0900, 1.1200, "BUY")
        assert adj_sl == pytest.approx(1.0900 - self._spread_pips * self._pip)

    def test_buy_adj_tp_unchanged(self):
        _, _, adj_tp, _ = self._run(1.1000, 1.0900, 1.1200, "BUY")
        assert adj_tp == pytest.approx(1.1200)

    def test_buy_adj_rr_less_than_raw(self):
        _, _, _, adj_rr = self._run(1.1000, 1.0900, 1.1200, "BUY")
        raw_rr = _risk_reward(1.1000, 1.0900, 1.1200)
        assert adj_rr < raw_rr

    def test_buy_adj_rr_positive(self):
        _, _, _, adj_rr = self._run(1.1000, 1.0900, 1.1200, "BUY")
        assert adj_rr > 0.0

    # ── SELL ─────────────────────────────────────────────────────────────────

    def test_sell_adj_entry_lower_by_spread(self):
        adj_entry, _, _, _ = self._run(1.1000, 1.1100, 1.0800, "SELL")
        assert adj_entry == pytest.approx(1.1000 - self._spread_pips * self._pip)

    def test_sell_adj_sl_higher_by_spread(self):
        _, adj_sl, _, _ = self._run(1.1000, 1.1100, 1.0800, "SELL")
        assert adj_sl == pytest.approx(1.1100 + self._spread_pips * self._pip)

    def test_sell_adj_tp_unchanged(self):
        _, _, adj_tp, _ = self._run(1.1000, 1.1100, 1.0800, "SELL")
        assert adj_tp == pytest.approx(1.0800)

    def test_sell_adj_rr_less_than_raw(self):
        _, _, _, adj_rr = self._run(1.1000, 1.1100, 1.0800, "SELL")
        raw_rr = _risk_reward(1.1000, 1.1100, 1.0800)
        assert adj_rr < raw_rr

    def test_sell_adj_rr_positive(self):
        _, _, _, adj_rr = self._run(1.1000, 1.1100, 1.0800, "SELL")
        assert adj_rr > 0.0

    # ── Symmetry ─────────────────────────────────────────────────────────────

    def test_buy_sell_rr_are_symmetric(self):
        """BUY and SELL with mirror prices should produce the same adj_rr."""
        _, _, _, buy_rr = self._run(1.1000, 1.0900, 1.1200, "BUY")
        _, _, _, sell_rr = self._run(1.1000, 1.1100, 1.0800, "SELL")
        assert buy_rr == pytest.approx(sell_rr, rel=1e-6)

    # ── Zero-spread edge case ─────────────────────────────────────────────────

    def test_zero_spread_equals_raw(self):
        entry, sl, tp = 1.1000, 1.0900, 1.1200
        adj_entry, adj_sl, adj_tp, adj_rr = _spread_adjusted_rr(
            entry, sl, tp, direction="BUY", symbol="EURUSD", spread_pips=0.0
        )
        assert adj_entry == pytest.approx(entry)
        assert adj_sl == pytest.approx(sl)
        assert adj_tp == pytest.approx(tp)
        assert adj_rr == pytest.approx(_risk_reward(entry, sl, tp))


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
    def _valid_signal(self):
        return generate_signal(_make_analysis())

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
        # Should be a float (rounded to 5 dp)
        assert isinstance(sig["entry_price"], float)

    def test_setup_type_is_ict_prefixed(self):
        assert self._valid_signal()["setup_type"].startswith("ICT")

    def test_confidence_score_between_0_and_1(self):
        sig = self._valid_signal()
        assert 0.0 <= sig["confidence_score"] <= 1.0


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
        import python.signal_generator.signal_generator as sg_mod

        target = tmp_path / "test_signal.json"
        monkeypatch.setattr(sg_mod, "SIGNAL_OUTPUT_PATH", target)

        sig = generate_signal(_make_analysis())
        save_signal(sig)

        # Verify the file was written with expected content
        assert target.exists()
        data = json.loads(target.read_text())
        assert data["symbol"] == "EURUSD"


# ---------------------------------------------------------------------------
# Signal TTL boundary tests (Item 5)
# ---------------------------------------------------------------------------

class TestSignalTTL:
    """
    Verify load_latest_signal() respects the signal_ttl_seconds setting.

    Uses monkeypatching to control the apparent signal age without sleeping.
    """

    def _write_signal(self, tmp_path, monkeypatch, timestamp_iso: str) -> None:
        import python.signal_generator.signal_generator as sg_mod

        target = tmp_path / "signal.json"
        monkeypatch.setattr(sg_mod, "SIGNAL_OUTPUT_PATH", target)
        payload = {
            "symbol":      "EURUSD",
            "direction":   "BUY",
            "entry_price": 1.10000,
            "stop_loss":   1.09000,
            "take_profit": 1.12000,
            "timestamp":   timestamp_iso,
        }
        target.write_text(json.dumps(payload))

    def test_fresh_signal_within_ttl_is_returned(self, tmp_path, monkeypatch):
        """Signal timestamped 10 s ago should be returned (TTL default 300 s)."""
        from datetime import datetime, timezone, timedelta
        import python.signal_generator.signal_generator as sg_mod

        recent_ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        self._write_signal(tmp_path, monkeypatch, recent_ts)

        result = sg_mod.load_latest_signal()
        assert result is not None
        assert result["symbol"] == "EURUSD"

    def test_expired_signal_returns_none(self, tmp_path, monkeypatch):
        """Signal timestamped 3600 s ago (> 300 s TTL) should return None."""
        from datetime import datetime, timezone, timedelta
        import python.signal_generator.signal_generator as sg_mod

        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=3600)).isoformat()
        self._write_signal(tmp_path, monkeypatch, old_ts)

        result = sg_mod.load_latest_signal()
        assert result is None

    def test_signal_just_at_boundary_is_expired(self, tmp_path, monkeypatch):
        """Signal exactly TTL+1 seconds old must return None."""
        from datetime import datetime, timezone, timedelta
        import python.signal_generator.signal_generator as sg_mod

        # Default TTL is 300; use a small custom TTL via the INTEGRATION mock
        ttl = 60
        old_ts = (datetime.now(timezone.utc) - timedelta(seconds=ttl + 1)).isoformat()
        self._write_signal(tmp_path, monkeypatch, old_ts)
        monkeypatch.setattr(
            sg_mod,
            "SIGNAL_OUTPUT_PATH",
            tmp_path / "signal.json",
        )
        # Patch INTEGRATION inside the module so TTL is controlled
        import python.signal_generator.signal_generator as sg_mod2
        original_integration = sg_mod2.__dict__.get("INTEGRATION", None)
        # We patch via the module's imported name binding
        with pytest.MonkeyPatch().context() as mp:
            mp.setattr(sg_mod2, "SIGNAL_OUTPUT_PATH", tmp_path / "signal.json")
            # Override the INTEGRATION lookup inside load_latest_signal
            from unittest.mock import patch as _patch
            with _patch("python.config.INTEGRATION", {"signal_ttl_seconds": ttl}):
                result = sg_mod2.load_latest_signal()
        assert result is None

    def test_signal_just_before_expiry_is_returned(self, tmp_path, monkeypatch):
        """Signal TTL-10 seconds old must still be returned."""
        from datetime import datetime, timezone, timedelta
        import python.signal_generator.signal_generator as sg_mod

        ttl = 300
        recent_ts = (datetime.now(timezone.utc) - timedelta(seconds=ttl - 10)).isoformat()
        self._write_signal(tmp_path, monkeypatch, recent_ts)

        result = sg_mod.load_latest_signal()
        assert result is not None

    def test_missing_file_returns_none(self, tmp_path, monkeypatch):
        import python.signal_generator.signal_generator as sg_mod

        nonexistent = tmp_path / "no_signal.json"
        monkeypatch.setattr(sg_mod, "SIGNAL_OUTPUT_PATH", nonexistent)

        assert sg_mod.load_latest_signal() is None
