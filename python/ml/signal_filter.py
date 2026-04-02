"""
Optional ML Signal Filter

Uses XGBoost (or Random Forest) to add a probability-of-success filter on top
of the rule-based ICT signals.

Features used:
  - Market structure state (0=ranging, 1=bullish, 2=bearish) per timeframe
  - ATR on M5 (volatility proxy)
  - Session timing (hours since London open)
  - Liquidity sweep present (0/1)
  - Price zone (0=discount, 1=premium, 2=equilibrium)
  - H1 OB present (0/1)
  - H1 FVG present (0/1)
  - M5 FVG present (0/1)
  - Confluence score from rule engine

Model is trained offline on historical labelled trades and saved to disk.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from python.config import ML_CFG
from python.strategy_engine.market_structure import TrendDirection
from python.strategy_engine.mtf_engine import MTFAnalysis

logger = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / ML_CFG.get(
    "model_path", "ml/models/xgb_filter.pkl"
)
_THRESHOLD = ML_CFG.get("confidence_threshold", 0.65)

_model = None


def _load_model():
    global _model
    if _model is not None:
        return _model
    try:
        import joblib  # type: ignore

        _model = joblib.load(_MODEL_PATH)
        logger.info("ML model loaded from %s", _MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load ML model: %s", exc)
        _model = None
    return _model


def _trend_int(t: TrendDirection) -> int:
    if t == TrendDirection.BULLISH:
        return 1
    if t == TrendDirection.BEARISH:
        return 2
    return 0


def _zone_int(zone: str) -> int:
    if zone == "DISCOUNT":
        return 0
    if zone == "PREMIUM":
        return 1
    return 2


def build_feature_vector(analysis: MTFAnalysis, atr_m5: float, hour_utc: int) -> np.ndarray:
    """Return a 1-D numpy array of features for the ML model."""
    features = [
        _trend_int(analysis.d1_trend),
        _trend_int(analysis.h1_trend),
        atr_m5,
        hour_utc,
        1 if analysis.m5_sweep is not None else 0,
        _zone_int(analysis.price_zone),
        1 if analysis.h1_ob is not None else 0,
        1 if analysis.h1_fvg is not None else 0,
        1 if analysis.m5_fvg is not None else 0,
        analysis.confluence_score,
    ]
    return np.array(features, dtype=float).reshape(1, -1)


def ml_confidence(analysis: MTFAnalysis, atr_m5: float = 0.0, hour_utc: int = 12) -> float:
    """
    Return the ML model's probability of success for this setup.
    Returns -1.0 if model is unavailable (caller should ignore ML filter).
    """
    model = _load_model()
    if model is None:
        return -1.0
    try:
        X = build_feature_vector(analysis, atr_m5, hour_utc)
        prob = float(model.predict_proba(X)[0][1])
        return prob
    except Exception as exc:  # noqa: BLE001
        logger.warning("ML prediction error: %s", exc)
        return -1.0


def passes_ml_filter(
    analysis: MTFAnalysis,
    atr_m5: float = 0.0,
    hour_utc: int = 12,
    threshold: float = None,
) -> bool:
    """
    Returns True if ML model confidence is above threshold, or if model is unavailable.
    """
    if not ML_CFG.get("enabled", False):
        return True  # ML disabled → pass everything
    prob = ml_confidence(analysis, atr_m5, hour_utc)
    if prob < 0:
        return True  # model unavailable → pass
    thr = threshold or _THRESHOLD
    result = prob >= thr
    if not result:
        logger.debug("ML filter rejected signal (prob=%.3f < threshold=%.2f)", prob, thr)
    return result


# ── Training helper ──────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray, save_path: Optional[Path] = None) -> None:
    """
    Train an XGBoost classifier on (X, y) and save to disk.

    X: (n_samples, 10) feature matrix
    y: (n_samples,) binary labels (1=win, 0=loss)
    """
    try:
        import joblib  # type: ignore
        from xgboost import XGBClassifier  # type: ignore

        clf = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )
        clf.fit(X, y)
        out = save_path or _MODEL_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, out)
        logger.info("ML model saved to %s", out)
    except ImportError as exc:
        logger.error("XGBoost/joblib not installed: %s", exc)
