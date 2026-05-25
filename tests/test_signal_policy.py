"""
Unit tests for python.ml.signal_policy

Covers:
  - should_publish() logic for all combinations of ict_valid, risk_gate, ML result
  - build_signal_ml_metadata() output structure
"""
from __future__ import annotations

import pytest

from python.ml.signal_policy import build_signal_ml_metadata, should_publish


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ml_disabled():
    return {
        "ml_enabled": False,
        "ml_score": -1.0,
        "ml_decision": "DISABLED",
        "ml_quality": "UNKNOWN",
        "ml_features": {},
    }


def _ml_approved(score=0.75, quality="MEDIUM"):
    return {
        "ml_enabled": True,
        "ml_score": score,
        "ml_decision": "APPROVED",
        "ml_quality": quality,
        "ml_features": {"confluence_score": 0.8},
    }


def _ml_filtered(score=0.40, quality="LOW"):
    return {
        "ml_enabled": True,
        "ml_score": score,
        "ml_decision": "FILTERED",
        "ml_quality": quality,
        "ml_features": {"confluence_score": 0.5},
    }


def _ml_unavailable():
    return {
        "ml_enabled": True,
        "ml_score": -1.0,
        "ml_decision": "UNAVAILABLE",
        "ml_quality": "UNKNOWN",
        "ml_features": {},
    }


# ---------------------------------------------------------------------------
# should_publish
# ---------------------------------------------------------------------------

class TestShouldPublish:
    def test_ict_invalid_blocks(self):
        publish, reason = should_publish(False, True, _ml_disabled())
        assert publish is False
        assert "ICT" in reason

    def test_risk_gate_blocks(self):
        publish, reason = should_publish(True, False, _ml_disabled())
        assert publish is False
        assert "Risk" in reason

    def test_ml_disabled_publishes(self):
        publish, reason = should_publish(True, True, _ml_disabled())
        assert publish is True

    def test_ml_approved_publishes(self):
        publish, reason = should_publish(True, True, _ml_approved())
        assert publish is True

    def test_ml_filtered_low_publishes_with_default_min(self):
        """Default min_quality is LOW, so even LOW quality publishes."""
        publish, reason = should_publish(True, True, _ml_filtered())
        assert publish is True

    def test_ml_filtered_below_min_blocks(self, monkeypatch):
        """When min_quality is MEDIUM, LOW quality is blocked."""
        import python.ml.signal_policy as sp
        monkeypatch.setitem(sp.ML_CFG, "publish_min_quality", "MEDIUM")
        publish, reason = should_publish(True, True, _ml_filtered())
        assert publish is False
        assert "below minimum" in reason

    def test_ml_unavailable_fail_open(self, monkeypatch):
        """Default unavailable_action is publish (fail-open)."""
        import python.ml.signal_policy as sp
        monkeypatch.setitem(sp.ML_CFG, "unavailable_action", "publish")
        publish, reason = should_publish(True, True, _ml_unavailable())
        assert publish is True

    def test_ml_unavailable_fail_closed(self, monkeypatch):
        import python.ml.signal_policy as sp
        monkeypatch.setitem(sp.ML_CFG, "unavailable_action", "block")
        publish, reason = should_publish(True, True, _ml_unavailable())
        assert publish is False


# ---------------------------------------------------------------------------
# build_signal_ml_metadata
# ---------------------------------------------------------------------------

class TestBuildSignalMlMetadata:
    def test_contains_required_keys(self):
        meta = build_signal_ml_metadata(_ml_approved())
        assert "ict_valid" in meta
        assert "ml_enabled" in meta
        assert "ml_score" in meta
        assert "ml_decision" in meta
        assert "ml_quality" in meta
        assert "ml_features" in meta

    def test_ict_valid_defaults_true(self):
        meta = build_signal_ml_metadata(_ml_approved())
        assert meta["ict_valid"] is True

    def test_preserves_ml_score(self):
        meta = build_signal_ml_metadata(_ml_approved(score=0.88))
        assert meta["ml_score"] == 0.88
