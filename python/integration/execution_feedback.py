"""
Execution Feedback Poller

The MT5 EA writes a small JSON file to ``signals/feedback/`` each time it
executes or rejects a signal.  This module polls that directory and passes
each new feedback file to the performance tracker so that Layer 3 statistics
are kept in sync.

Feedback file format (written by the EA's WriteFeedback() MQL5 function):

.. code-block:: json

    {
      "signal_id": "EURUSD-20260525-083000-ICT_FVG_OB_SWEEP",
      "status": "REJECTED",
      "reason": "SPREAD_TOO_HIGH",
      "spread_pips": 4.2,
      "symbol": "EURUSD",
      "direction": "BUY",
      "timestamp": "2026-05-25T08:31:10Z",
      "detail": "spread 4.2 > max 3.0 pips"
    }

Usage (called from ``python/main.py``)::

    from python.integration.execution_feedback import ingest_pending

    # Call once per scan cycle; processes and archives new feedback files.
    ingest_pending()
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from python.config import EXECUTION_FEEDBACK_DIR
from python.performance.tracker import log_execution_feedback

logger = logging.getLogger(__name__)

# Processed files are moved here so they are not ingested twice.
_DONE_DIR: Path = EXECUTION_FEEDBACK_DIR / "processed"


def ingest_pending() -> int:
    """
    Read every unprocessed JSON file in ``EXECUTION_FEEDBACK_DIR``, call
    ``log_execution_feedback()`` for each, and move the file to the
    ``processed/`` sub-directory to prevent double-processing.

    Returns the number of feedback files ingested.
    """
    _DONE_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    for path in sorted(EXECUTION_FEEDBACK_DIR.glob("*.json")):
        try:
            feedback = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read feedback file %s: %s", path.name, exc)
            continue

        try:
            log_execution_feedback(feedback)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to ingest feedback %s: %s", path.name, exc)
            continue

        # Archive successfully processed file
        dest = _DONE_DIR / path.name
        try:
            path.rename(dest)
        except OSError as exc:
            logger.warning("Could not archive feedback file %s: %s", path.name, exc)

        count += 1

    if count:
        logger.info("Ingested %d EA feedback file(s).", count)
    return count
