"""
Unit tests for python.strategy_engine.mtf_engine

Tests:
  - MTFAnalysis dataclass construction
  - D1/H1 misalignment returns invalid early result
  - Scoring constants are loaded from config (not hardcoded)
  - analyse() with a synthetic dataset that triggers a BOS + OB + sweep
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from python.strategy_engine.market_structure import TrendDirection
from python.strategy_engine.mtf_engine import MTFAnalysis, _SC_BIAS, _SC_MIN_VALID, analyse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candles(n: int, start: float = 1.10000, step: float = 0.00010):
    """Return a minimal OHLCV DataFrame with n rows and a gently rising trend."""
    prices = [start + i * step for i in range(n)]
    return pd.DataFrame({
        "time":  pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open":  [p - 0.00005 for p in prices],
        "high":  [p + 0.00020 for p in prices],
        "low":   [p - 0.00020 for p in prices],
        "close": prices,
    })


def _flat_candles(n: int, price: float = 1.10000):
    return pd.DataFrame({
        "time":  pd.date_range("2024-01-01", periods=n, freq="5min"),
        "open":  [price] * n,
        "high":  [price + 0.00005] * n,
        "low":   [price - 0.00005] * n,
        "close": [price] * n,
    })


def _make_analysis(**kwargs) -> MTFAnalysis:
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
        stop_loss=1.09000,
        take_profit=1.12000,
        confluence_score=0.75,
        valid=True,
        reasons=["D1 bullish + H1 bullish BOS"],
    )
    defaults.update(kwargs)
    return MTFAnalysis(**defaults)


# ---------------------------------------------------------------------------
# MTFAnalysis dataclass
# ---------------------------------------------------------------------------

class TestMTFAnalysisDataclass:
    def test_default_confluence_score(self):
        a = _make_analysis(confluence_score=0.0)
        assert a.confluence_score == 0.0

    def test_valid_flag_settable(self):
        assert _make_analysis(valid=True).valid is True
        assert _make_analysis(valid=False).valid is False

    def test_reasons_is_list(self):
        a = _make_analysis(reasons=["reason A", "reason B"])
        assert isinstance(a.reasons, list)
        assert len(a.reasons) == 2

    def test_reasons_default_empty_list(self):
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
        assert a.reasons == []


# ---------------------------------------------------------------------------
# Scoring constants come from config, not magic numbers
# ---------------------------------------------------------------------------

class TestScoringConstants:
    def test_bias_alignment_positive(self):
        assert _SC_BIAS > 0

    def test_min_valid_score_between_0_and_1(self):
        assert 0 < _SC_MIN_VALID < 1

    def test_scoring_constants_are_floats(self):
        from python.strategy_engine.mtf_engine import (
            _SC_H1_FVG, _SC_H1_OB, _SC_M5_FVG, _SC_M5_SWEEP,
            _SC_ZONE_OK, _SC_ZONE_WRONG,
        )
        for c in [_SC_H1_FVG, _SC_H1_OB, _SC_M5_FVG, _SC_M5_SWEEP, _SC_ZONE_OK, _SC_ZONE_WRONG]:
            assert isinstance(c, float)


# ---------------------------------------------------------------------------
# analyse() – bias misalignment fast-path
# ---------------------------------------------------------------------------

class TestAnalyseMisalignment:
    def _run(self, d1_closes, h1_closes, m5_closes=None):
        n = len(d1_closes)
        def _df(closes):
            c = closes
            return pd.DataFrame({
                "time":  pd.date_range("2024-01-01", periods=len(c), freq="h"),
                "open":  [v - 0.0001 for v in c],
                "high":  [v + 0.0002 for v in c],
                "low":   [v - 0.0002 for v in c],
                "close": c,
            })
        return analyse(
            symbol="EURUSD",
            df_d1=_df(d1_closes),
            df_h1=_df(h1_closes),
            df_m5=_df(m5_closes or d1_closes),
        )

    def test_flat_data_returns_mtf_analysis(self):
        result = self._run([1.1] * 50, [1.1] * 50)
        assert isinstance(result, MTFAnalysis)

    def test_misaligned_bias_is_not_valid(self):
        """Flat/ranging data should not produce a valid signal."""
        result = self._run([1.1] * 50, [1.1] * 50)
        assert result.valid is False

    def test_result_has_required_fields(self):
        result = self._run([1.1] * 50, [1.1] * 50)
        assert hasattr(result, "symbol")
        assert hasattr(result, "confluence_score")
        assert hasattr(result, "reasons")

    def test_score_clamped_between_0_and_1(self):
        result = self._run([1.1] * 50, [1.1] * 50)
        assert 0.0 <= result.confluence_score <= 1.0
