"""
Signal Policy – Advisory-based publication decision.

The ML filter is advisory, not authoritative.  This module decides whether a
signal should be published based on the structured pipeline:

    ICT Rule Engine → produces valid candidate (ict_valid=True)
    Risk Gate       → confirms tradability
    ML Filter       → labels quality as HIGH / MEDIUM / LOW
    Signal Policy   → decides whether to publish

The policy NEVER mutates entry, SL, or TP.  It only labels and gates.

Configuration keys (config.yaml → ml section):
    ml.mode: "advisory" | "gate"
        - "advisory" (default): ML metadata attached, publication decided by policy
        - "gate": legacy boolean filter (backward compat)
    ml.publish_min_quality: "LOW" | "MEDIUM" | "HIGH"
        - Minimum ML quality label required to publish. Default: "LOW" (publish all).
"""
from __future__ import annotations

import logging
from typing import Optional

from python.config import ML_CFG

logger = logging.getLogger(__name__)

_QUALITY_RANK = {"UNKNOWN": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3}


def should_publish(
    ict_valid: bool,
    risk_gate_passed: bool,
    ml_result: dict,
) -> tuple[bool, str]:
    """
    Decide whether a signal should be published based on all pipeline stages.

    Parameters
    ----------
    ict_valid:
        True if the ICT rule engine produced a valid candidate.
    risk_gate_passed:
        True if all risk guards (daily loss, trade count, etc.) allow trading.
    ml_result:
        The dict returned by :func:`python.ml.signal_filter.ml_evaluate`.

    Returns
    -------
    (publish: bool, reason: str)
        Whether to publish and a human-readable reason for the decision.
    """
    if not ict_valid:
        return False, "ICT rule engine did not produce a valid candidate"

    if not risk_gate_passed:
        return False, "Risk gate blocked the signal"

    ml_enabled = ml_result.get("ml_enabled", False)
    ml_decision = ml_result.get("ml_decision", "DISABLED")
    ml_quality = ml_result.get("ml_quality", "UNKNOWN")

    # If ML is disabled, always publish (ICT + risk is sufficient)
    if not ml_enabled or ml_decision == "DISABLED":
        return True, "ML disabled – publishing on ICT + risk gate alone"

    # If ML model is unavailable, decide based on config
    if ml_decision == "UNAVAILABLE":
        fail_open = ML_CFG.get("unavailable_action", "publish") == "publish"
        if fail_open:
            return True, "ML model unavailable – fail-open (publishing)"
        return False, "ML model unavailable – fail-closed (blocking)"

    # ML is active and returned a quality label – apply minimum quality policy
    min_quality = ML_CFG.get("publish_min_quality", "LOW").upper()
    min_rank = _QUALITY_RANK.get(min_quality, 1)
    actual_rank = _QUALITY_RANK.get(ml_quality, 0)

    if actual_rank >= min_rank:
        return True, f"ML quality={ml_quality} meets minimum={min_quality}"
    else:
        logger.info(
            "Signal policy: ML quality %s below minimum %s – not publishing",
            ml_quality, min_quality,
        )
        return False, f"ML quality={ml_quality} below minimum={min_quality}"


def build_signal_ml_metadata(
    ml_result: dict,
    ict_valid: bool = True,
) -> dict:
    """
    Build the ML metadata fields to embed in the signal JSON.

    This ensures every signal carries full explainability about the ML
    decision, regardless of whether ML approved or filtered it.
    """
    return {
        "ict_valid": ict_valid,
        "ml_enabled": ml_result.get("ml_enabled", False),
        "ml_score": ml_result.get("ml_score", -1.0),
        "ml_decision": ml_result.get("ml_decision", "DISABLED"),
        "ml_quality": ml_result.get("ml_quality", "UNKNOWN"),
        "ml_features": ml_result.get("ml_features", {}),
    }
