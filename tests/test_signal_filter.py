"""
Unit tests for python.ml.signal_filter

Covers:
  - build_feature_vector() shape and dtype
  - Feature values map correctly from analysis attributes
  - passes_ml_filter() returns True when ML is disabled in config
  - passes_ml_filter() rejects (fail-closed) when model file is absent and
    ML is enabled
  - ml_confidence() returns -1.0 when model is unavailable
"""
from __future__ import annotations

import pytest
import numpy as np

from python.strategy_engine.market_structure import TrendDirection
from python.strategy_engine.mtf_engine import MTFAnalysis
from python.ml.signal_filter import (
    _trend_int,
    _zone_int,
    build_feature_vector,
    ml_confidence,
    passes_ml_filter,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analysis(**overrides) -> MTFAnalysis:
    defaults = dict(
        symbol="EURUSD",
        d1_trend=TrendDirection.BULLISH,
        h1_trend=TrendDirection.BEARISH,
        h1_bos_direction="BULLISH",
        m5_sweep=object(),   # truthy non-None sentinel
        m5_fvg=None,
        h1_ob=object(),
        h1_fvg=None,
        fib_range=None,
        price_zone="DISCOUNT",
        signal_direction="BUY",
        entry_price=1.10000,
        stop_loss=1.09000,
        take_profit=1.12000,
        confluence_score=0.72,
        valid=True,
        reasons=["D1 bullish + H1 bullish BOS"],
    )
    defaults.update(overrides)
    return MTFAnalysis(**defaults)


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

class TestEncodingHelpers:
    def test_trend_bullish_is_1(self):
        assert _trend_int(TrendDirection.BULLISH) == 1

    def test_trend_bearish_is_2(self):
        assert _trend_int(TrendDirection.BEARISH) == 2

    def test_trend_ranging_is_0(self):
        assert _trend_int(TrendDirection.RANGING) == 0

    def test_zone_discount_is_0(self):
        assert _zone_int("DISCOUNT") == 0

    def test_zone_premium_is_1(self):
        assert _zone_int("PREMIUM") == 1

    def test_zone_equilibrium_is_2(self):
        assert _zone_int("EQUILIBRIUM") == 2

    def test_zone_unknown_is_2(self):
        assert _zone_int("UNKNOWN") == 2


# ---------------------------------------------------------------------------
# build_feature_vector
# ---------------------------------------------------------------------------

class TestBuildFeatureVector:
    def test_shape_is_1_by_10(self):
        a = _make_analysis()
        X = build_feature_vector(a, atr_m5=0.0005, hour_utc=10)
        assert X.shape == (1, 10)

    def test_dtype_is_float(self):
        a = _make_analysis()
        X = build_feature_vector(a, atr_m5=0.0005, hour_utc=10)
        assert X.dtype == float

    def test_d1_trend_encoded_correctly(self):
        a_bull = _make_analysis(d1_trend=TrendDirection.BULLISH)
        a_bear = _make_analysis(d1_trend=TrendDirection.BEARISH)
        assert build_feature_vector(a_bull, 0, 0)[0][0] == 1.0
        assert build_feature_vector(a_bear, 0, 0)[0][0] == 2.0

    def test_sweep_present_is_1(self):
        a = _make_analysis(m5_sweep=object())
        X = build_feature_vector(a, 0, 0)
        assert X[0][4] == 1.0

    def test_sweep_absent_is_0(self):
        a = _make_analysis(m5_sweep=None)
        X = build_feature_vector(a, 0, 0)
        assert X[0][4] == 0.0

    def test_h1_ob_present_is_1(self):
        a = _make_analysis(h1_ob=object())
        X = build_feature_vector(a, 0, 0)
        assert X[0][6] == 1.0

    def test_h1_ob_absent_is_0(self):
        a = _make_analysis(h1_ob=None)
        X = build_feature_vector(a, 0, 0)
        assert X[0][6] == 0.0

    def test_confluence_score_in_last_position(self):
        a = _make_analysis(confluence_score=0.55)
        X = build_feature_vector(a, 0, 0)
        assert X[0][9] == pytest.approx(0.55)

    def test_atr_is_second_last(self):
        a = _make_analysis()
        X = build_feature_vector(a, atr_m5=0.0012, hour_utc=0)
        assert X[0][2] == pytest.approx(0.0012)

    def test_hour_utc_is_at_index_3(self):
        a = _make_analysis()
        X = build_feature_vector(a, atr_m5=0.0, hour_utc=14)
        assert X[0][3] == pytest.approx(14.0)

    def test_no_nan_in_feature_vector(self):
        a = _make_analysis()
        X = build_feature_vector(a, 0.0005, 9)
        assert not np.isnan(X).any()


# ---------------------------------------------------------------------------
# passes_ml_filter with ML disabled
# ---------------------------------------------------------------------------

class TestPassesMlFilterDisabled:
    def test_passes_when_ml_disabled(self):
        """When ml.enabled is False (default), filter always passes."""
        a = _make_analysis()
        assert passes_ml_filter(a) is True

    def test_passes_regardless_of_low_score(self):
        a = _make_analysis(confluence_score=0.0)
        assert passes_ml_filter(a) is True


# ---------------------------------------------------------------------------
# passes_ml_filter with ML enabled but model missing
# ---------------------------------------------------------------------------

class TestPassesMlFilterEnabled:
    def test_fail_closed_when_model_missing(self, monkeypatch):
        """
        When ml.enabled=True but the model file doesn't exist,
        the filter must reject (fail-closed) rather than silently pass.
        """
        import python.ml.signal_filter as sf
        # Enable ML and reset the cached model
        monkeypatch.setitem(sf.ML_CFG, "enabled", True)
        monkeypatch.setattr(sf, "_model", None)

        a = _make_analysis()
        result = passes_ml_filter(a)
        assert result is False

    def test_ml_confidence_minus_one_when_no_model(self, monkeypatch):
        import python.ml.signal_filter as sf
        monkeypatch.setattr(sf, "_model", None)
        monkeypatch.setattr(sf, "_MODEL_PATH", sf._MODEL_PATH.parent / "nonexistent.pkl")

        a = _make_analysis()
        conf = ml_confidence(a, 0.0, 12)
        assert conf == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Feature-contract: build_feature_vector length must match FEATURE_COLS
# ---------------------------------------------------------------------------

class TestFeatureContract:
    def test_feature_vector_length_matches_feature_cols(self):
        """
        The feature vector produced by build_feature_vector() must have exactly
        as many columns as FEATURE_COLS in scripts/train_ml.py.

        These two lists are defined independently; this test prevents silent
        drift between them.
        """
        import sys
        from pathlib import Path
        _repo = Path(__file__).resolve().parent.parent
        if str(_repo) not in sys.path:
            sys.path.insert(0, str(_repo))
        from scripts.train_ml import FEATURE_COLS  # noqa: PLC0415

        a = _make_analysis()
        X = build_feature_vector(a, atr_m5=0.0005, hour_utc=10)
        assert X.shape[1] == len(FEATURE_COLS), (
            f"build_feature_vector() produces {X.shape[1]} features but "
            f"scripts/train_ml.FEATURE_COLS has {len(FEATURE_COLS)} entries: {FEATURE_COLS}"
        )
