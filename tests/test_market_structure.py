"""
Unit tests for python.strategy_engine.market_structure

Covers:
  - Swing high / low detection with various lookback values
  - Trend classification: BULLISH, BEARISH, RANGING
  - Edge cases: empty frame, single candle, flat prices, insufficient swings
"""
from __future__ import annotations

import pandas as pd
import pytest

from python.strategy_engine.market_structure import (
    SwingPoint,
    TrendDirection,
    analyse,
    classify_trend,
    detect_swing_highs,
    detect_swing_lows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(highs, lows, opens=None, closes=None):
    n = len(highs)
    opens  = opens  or highs
    closes = closes or highs
    return pd.DataFrame({
        "time":  pd.date_range("2024-01-01", periods=n, freq="h"),
        "open":  opens,
        "high":  highs,
        "low":   lows,
        "close": closes,
    })


def _sp(idx, price, kind="HIGH"):
    return SwingPoint(
        index=idx,
        price=float(price),
        time=pd.Timestamp("2024-01-01") + pd.Timedelta(hours=idx),
        kind=kind,
    )


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------

class TestDetectSwingHighs:
    def test_single_peak_detected(self):
        df = _make_df(highs=[1.0, 2.0, 3.0, 2.0, 1.0], lows=[0.5] * 5)
        swings = detect_swing_highs(df, lookback=1)
        assert any(s.index == 2 for s in swings)

    def test_plateau_not_double_counted(self):
        """Two equal highs should be treated as two candidates, not crash."""
        df = _make_df(highs=[1.0, 3.0, 3.0, 1.0], lows=[0.5] * 4)
        # Should not raise; result may vary but must be a list
        result = detect_swing_highs(df, lookback=1)
        assert isinstance(result, list)

    def test_monotone_rising_no_swing_highs(self):
        df = _make_df(highs=[1.0, 2.0, 3.0, 4.0, 5.0], lows=[0.5] * 5)
        swings = detect_swing_highs(df, lookback=1)
        assert swings == []

    def test_too_short_for_lookback(self):
        """Frame shorter than 2×lookback+1 produces no swings."""
        df = _make_df(highs=[1.0, 2.0], lows=[0.5, 0.5])
        assert detect_swing_highs(df, lookback=2) == []

    def test_multiple_peaks(self):
        df = _make_df(highs=[1, 3, 1, 4, 1, 2, 1], lows=[0.5] * 7)
        swings = detect_swing_highs(df, lookback=1)
        indices = {s.index for s in swings}
        assert 1 in indices
        assert 3 in indices

    def test_all_swing_kinds_are_high(self):
        df = _make_df(highs=[1, 3, 1, 4, 1], lows=[0.5] * 5)
        swings = detect_swing_highs(df, lookback=1)
        assert all(s.kind == "HIGH" for s in swings)


class TestDetectSwingLows:
    def test_single_trough_detected(self):
        df = _make_df(highs=[3.0] * 5, lows=[3.0, 2.0, 1.0, 2.0, 3.0])
        swings = detect_swing_lows(df, lookback=1)
        assert any(s.index == 2 for s in swings)

    def test_monotone_falling_no_swing_lows(self):
        df = _make_df(highs=[5.0] * 5, lows=[5.0, 4.0, 3.0, 2.0, 1.0])
        swings = detect_swing_lows(df, lookback=1)
        assert swings == []

    def test_all_swing_kinds_are_low(self):
        df = _make_df(highs=[3.0] * 5, lows=[3.0, 1.0, 3.0, 2.0, 3.0])
        swings = detect_swing_lows(df, lookback=1)
        assert all(s.kind == "LOW" for s in swings)


# ---------------------------------------------------------------------------
# Trend classification
# ---------------------------------------------------------------------------

class TestClassifyTrend:
    def test_bullish_hh_hl(self):
        highs = [_sp(0, 1.0), _sp(2, 1.1), _sp(4, 1.2)]
        lows  = [_sp(1, 0.9), _sp(3, 1.0), _sp(5, 1.05)]
        assert classify_trend(highs, lows) == TrendDirection.BULLISH

    def test_bearish_ll_lh(self):
        highs = [_sp(0, 1.2), _sp(2, 1.1), _sp(4, 1.0)]
        lows  = [_sp(1, 1.05), _sp(3, 0.95), _sp(5, 0.85)]
        assert classify_trend(highs, lows) == TrendDirection.BEARISH

    def test_ranging_mixed(self):
        highs = [_sp(0, 1.0), _sp(2, 1.1), _sp(4, 1.05)]  # not pure HH
        lows  = [_sp(1, 0.9), _sp(3, 1.0), _sp(5, 0.95)]
        assert classify_trend(highs, lows) == TrendDirection.RANGING

    def test_ranging_no_swings(self):
        assert classify_trend([], []) == TrendDirection.RANGING

    def test_ranging_single_swing(self):
        highs = [_sp(0, 1.0)]
        lows  = [_sp(1, 0.9)]
        assert classify_trend(highs, lows) == TrendDirection.RANGING

    def test_only_one_swing_each_not_enough(self):
        """Need at least 2 swings of each type to classify trend."""
        highs = [_sp(0, 1.0), _sp(2, 1.1)]
        lows  = [_sp(1, 0.9)]
        assert classify_trend(highs, lows) == TrendDirection.RANGING


# ---------------------------------------------------------------------------
# Full analyse()
# ---------------------------------------------------------------------------

class TestAnalyse:
    def _bullish_df(self, n=30):
        """Generates a simple step-up price series (clear uptrend)."""
        prices = [1.0 + i * 0.001 for i in range(n)]
        # Introduce alternating highs/lows so swings are detectable
        highs = [prices[i] + (0.002 if i % 4 == 2 else 0.0) for i in range(n)]
        lows  = [prices[i] - (0.002 if i % 4 == 0 else 0.0) for i in range(n)]
        return _make_df(highs=highs, lows=lows, opens=prices, closes=prices)

    def test_returns_market_structure_result(self):
        from python.strategy_engine.market_structure import MarketStructureResult
        df = self._bullish_df()
        result = analyse(df)
        assert isinstance(result, MarketStructureResult)

    def test_trend_is_enum(self):
        df = self._bullish_df()
        result = analyse(df)
        assert isinstance(result.trend, TrendDirection)

    def test_swing_lists_are_lists(self):
        df = self._bullish_df()
        result = analyse(df)
        assert isinstance(result.swing_highs, list)
        assert isinstance(result.swing_lows, list)

    def test_last_swing_none_or_swing_point(self):
        df = self._bullish_df()
        result = analyse(df)
        # last_swing_high/low is either None or a SwingPoint
        if result.last_swing_high is not None:
            assert isinstance(result.last_swing_high, SwingPoint)
        if result.last_swing_low is not None:
            assert isinstance(result.last_swing_low, SwingPoint)

    def test_empty_df_does_not_raise(self):
        df = _make_df([], [])
        result = analyse(df)
        assert result.trend == TrendDirection.RANGING
        assert result.swing_highs == []
        assert result.swing_lows == []
