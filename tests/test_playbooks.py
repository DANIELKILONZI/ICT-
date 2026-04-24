"""
Unit tests for python.strategy_engine.playbooks

Covers:
  - PlaybookResult dataclass construction
  - setup1 returns no-match on flat/neutral data (no sweep, no FVG)
  - setup2 returns no-match when no D1 OB is present
  - setup3 time-gate: returns no-match when outside London Killzone
  - setup3 returns no-match inside Killzone when there is no recent H1 BOS
  - setup3 returns no-match inside Killzone when there is no M5 FVG
  - setup_name field is present on MTFAnalysis after analyse()
  - analyse() result is invalid and has no setup_name on flat data
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from python.strategy_engine.playbooks import (
    PlaybookResult,
    setup1_sweep_fvg_continuation,
    setup2_htf_ob_reversal,
    setup3_london_killzone,
    setup4_ny_killzone,
)
from python.strategy_engine.mtf_engine import _analyse_tf, MTFAnalysis, analyse


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _flat(n: int = 100, price: float = 1.10000) -> pd.DataFrame:
    """Return a flat OHLCV DataFrame (no swing structure, no FVG, no sweep)."""
    return pd.DataFrame({
        "time":  pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open":  [price] * n,
        "high":  [price + 0.00005] * n,
        "low":   [price - 0.00005] * n,
        "close": [price] * n,
    })


def _rising(n: int = 100, start: float = 1.09000, step: float = 0.00010) -> pd.DataFrame:
    """Return a steadily rising OHLCV DataFrame."""
    prices = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "time":  pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open":  [p - 0.00005 for p in prices],
        "high":  [p + 0.00020 for p in prices],
        "low":   [p - 0.00020 for p in prices],
        "close": prices,
    })


def _falling(n: int = 100, start: float = 1.11000, step: float = 0.00010) -> pd.DataFrame:
    """Return a steadily falling OHLCV DataFrame."""
    prices = [start - i * step for i in range(n)]
    return pd.DataFrame({
        "time":  pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open":  [p + 0.00005 for p in prices],
        "high":  [p + 0.00020 for p in prices],
        "low":   [p - 0.00020 for p in prices],
        "close": prices,
    })


# ---------------------------------------------------------------------------
# PlaybookResult dataclass
# ---------------------------------------------------------------------------

class TestPlaybookResult:
    def test_no_match_default(self):
        r = PlaybookResult(matched=False)
        assert not r.matched
        assert r.name == ""
        assert r.direction is None
        assert r.entry_price is None

    def test_matched_result_fields(self):
        r = PlaybookResult(
            matched=True,
            name="SWEEP_FVG_CONTINUATION",
            direction="BUY",
            entry_price=1.1000,
            stop_loss=1.0900,
            take_profit=1.1200,
            confluence_score=0.80,
            reasons=["reason A"],
        )
        assert r.matched
        assert r.name == "SWEEP_FVG_CONTINUATION"
        assert r.direction == "BUY"
        assert r.confluence_score == 0.80
        assert len(r.reasons) == 1

    def test_reasons_default_is_empty_list(self):
        r = PlaybookResult(matched=False)
        assert isinstance(r.reasons, list)
        assert r.reasons == []

    def test_component_fields_default_none(self):
        r = PlaybookResult(matched=False)
        assert r.m5_sweep is None
        assert r.m5_fvg is None
        assert r.h1_ob is None
        assert r.h1_fvg is None


# ---------------------------------------------------------------------------
# Setup 1 – SWEEP_FVG_CONTINUATION
# ---------------------------------------------------------------------------

class TestSetup1FlatData:
    """Flat / neutral data should never trigger Setup 1."""

    def _run(self, d1, h1, m5):
        return setup1_sweep_fvg_continuation(
            d1_tf=_analyse_tf(d1),
            h1_tf=_analyse_tf(h1),
            m5_tf=_analyse_tf(m5),
            df_m5=m5,
            symbol="EURUSD",
            current_price=float(m5["close"].iloc[-1]),
        )

    def test_flat_data_no_match(self):
        df = _flat()
        assert not self._run(df, df, df).matched

    def test_misaligned_trends_no_match(self):
        # D1 rising but H1 falling → no trend alignment
        assert not self._run(_rising(), _falling(), _flat()).matched

    def test_rising_d1_h1_but_no_sweep_no_match(self):
        # Even with aligned D1/H1 trend there is no M5 sweep on flat M5 data
        df_r = _rising()
        assert not self._run(df_r, df_r, _flat()).matched


# ---------------------------------------------------------------------------
# Setup 2 – HTF_OB_REVERSAL
# ---------------------------------------------------------------------------

class TestSetup2FlatData:
    """Flat data has no D1 OBs; Setup 2 must not match."""

    def _run(self, d1, h1, m5):
        return setup2_htf_ob_reversal(
            d1_tf=_analyse_tf(d1),
            h1_tf=_analyse_tf(h1),
            m5_tf=_analyse_tf(m5),
            df_m5=m5,
            symbol="EURUSD",
            current_price=float(m5["close"].iloc[-1]),
        )

    def test_flat_all_timeframes_no_match(self):
        df = _flat()
        assert not self._run(df, df, df).matched

    def test_rising_data_no_d1_ob_no_match(self):
        # Rising data produces BOS events but BOS alone does not create OBs
        # without a qualifying displacement candle; flat M5 ensures no OB is formed
        df = _rising()
        assert not self._run(df, df, _flat()).matched


# ---------------------------------------------------------------------------
# Setup 3 – LONDON_KILLZONE_EXPANSION (time-gate)
# ---------------------------------------------------------------------------

class TestSetup3TimeGate:
    """The _now_hour parameter allows deterministic testing of the time gate."""

    def _run(self, h1, m5, now_h):
        return setup3_london_killzone(
            h1_tf=_analyse_tf(h1),
            m5_tf=_analyse_tf(m5),
            df_h1=h1,
            df_m5=m5,
            symbol="EURUSD",
            current_price=float(m5["close"].iloc[-1]),
            _now_hour=now_h,
        )

    def test_outside_killzone_no_match(self):
        df = _flat()
        assert not self._run(df, df, now_h=12).matched  # noon UTC – outside 07-10

    def test_before_killzone_no_match(self):
        df = _flat()
        assert not self._run(df, df, now_h=6).matched   # 06:00 UTC

    def test_at_killzone_close_no_match(self):
        # The close boundary is exclusive
        df = _flat()
        assert not self._run(df, df, now_h=10).matched  # 10:00 UTC – not included

    def test_inside_killzone_flat_no_bos_no_match(self):
        # Inside killzone but flat data → no recent H1 BOS → no match
        df = _flat()
        assert not self._run(df, df, now_h=8).matched

    def test_inside_killzone_rising_but_no_m5_fvg_no_match(self):
        # Rising H1 may produce a BOS, but flat M5 has no FVG
        df_r = _rising()
        df_flat = _flat()
        result = self._run(h1=df_r, m5=df_flat, now_h=8)
        # Either it found no BOS at all (lookback short) or no FVG; either way: no match
        assert not result.matched


# ---------------------------------------------------------------------------
# Integration: analyse() with playbook dispatch
# ---------------------------------------------------------------------------

class TestAnalysePlaybookIntegration:
    """Integration tests for the analyse() function after the playbook refactor."""

    def _flat_df(self, n=100, price=1.1):
        return _flat(n, price)

    def test_flat_returns_mtfanalysis(self):
        df = self._flat_df()
        result = analyse("EURUSD", df, df, df)
        assert isinstance(result, MTFAnalysis)

    def test_flat_returns_invalid(self):
        df = self._flat_df()
        result = analyse("EURUSD", df, df, df)
        assert result.valid is False

    def test_setup_name_field_exists_on_result(self):
        df = self._flat_df()
        result = analyse("EURUSD", df, df, df)
        assert hasattr(result, "setup_name")

    def test_no_playbook_setup_name_is_none(self):
        df = self._flat_df()
        result = analyse("EURUSD", df, df, df)
        assert result.setup_name is None

    def test_score_clamped(self):
        df = self._flat_df()
        result = analyse("EURUSD", df, df, df)
        assert 0.0 <= result.confluence_score <= 1.0

    def test_reasons_is_list(self):
        df = self._flat_df()
        result = analyse("EURUSD", df, df, df)
        assert isinstance(result.reasons, list)

    def test_no_match_reasons_non_empty(self):
        df = self._flat_df()
        result = analyse("EURUSD", df, df, df)
        assert len(result.reasons) > 0


# ---------------------------------------------------------------------------
# Setup 4 – NY_KILLZONE_EXPANSION (time-gate)
# ---------------------------------------------------------------------------

class TestSetup4TimeGate:
    """The _now_hour parameter allows deterministic testing of the NY time gate."""

    def _run(self, h1, m5, now_h):
        return setup4_ny_killzone(
            h1_tf=_analyse_tf(h1),
            m5_tf=_analyse_tf(m5),
            df_h1=h1,
            df_m5=m5,
            symbol="EURUSD",
            current_price=float(m5["close"].iloc[-1]),
            _now_hour=now_h,
        )

    def test_outside_killzone_no_match(self):
        # 12 UTC is before NY opens (13 UTC)
        df = _flat()
        assert not self._run(df, df, now_h=12).matched

    def test_before_killzone_no_match(self):
        df = _flat()
        assert not self._run(df, df, now_h=6).matched  # 06:00 UTC – well outside

    def test_at_killzone_close_no_match(self):
        # The close boundary is exclusive (22 is not included)
        df = _flat()
        assert not self._run(df, df, now_h=22).matched

    def test_inside_killzone_flat_no_bos_no_match(self):
        # Inside NY Killzone but flat data → no recent H1 BOS → no match
        df = _flat()
        assert not self._run(df, df, now_h=15).matched

    def test_inside_killzone_rising_but_no_m5_fvg_no_match(self):
        # Rising H1 may produce a BOS, but flat M5 has no FVG
        df_r = _rising()
        df_flat = _flat()
        result = self._run(h1=df_r, m5=df_flat, now_h=15)
        # Either no BOS found (short lookback) or no M5 FVG; either way: no match
        assert not result.matched

    def test_name_field_when_matched(self):
        # Verify the setup name is correct if a match ever occurs
        # Use a result built directly to confirm the name constant
        r = PlaybookResult(
            matched=True,
            name="NY_KILLZONE_EXPANSION",
            direction="BUY",
            entry_price=1.10,
            stop_loss=1.09,
            take_profit=1.12,
        )
        assert r.name == "NY_KILLZONE_EXPANSION"


# ---------------------------------------------------------------------------
# MTFAnalysis.setup_name field is present in named-playbook results
# ---------------------------------------------------------------------------

class TestMTFAnalysisSetupName:
    def test_setup_name_default_is_none(self):
        from python.strategy_engine.market_structure import TrendDirection
        a = MTFAnalysis(
            symbol="EURUSD",
            d1_trend=TrendDirection.BULLISH,
            h1_trend=TrendDirection.BULLISH,
            h1_bos_direction=None,
            m5_sweep=None,
            m5_fvg=None,
            h1_ob=None,
            h1_fvg=None,
            fib_range=None,
            price_zone="UNKNOWN",
            signal_direction=None,
            entry_price=None,
            stop_loss=None,
            take_profit=None,
        )
        assert a.setup_name is None

    def test_setup_name_can_be_set(self):
        from python.strategy_engine.market_structure import TrendDirection
        a = MTFAnalysis(
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
            entry_price=1.1,
            stop_loss=1.09,
            take_profit=1.12,
            setup_name="SWEEP_FVG_CONTINUATION",
            valid=True,
        )
        assert a.setup_name == "SWEEP_FVG_CONTINUATION"
