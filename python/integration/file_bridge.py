"""
Integration – File Bridge
Watches the signal JSON file and notifies subscribers via a simple callback.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable, Optional

from python.config import SIGNAL_OUTPUT_PATH

logger = logging.getLogger(__name__)


def poll_signal_file(
    callback: Callable[[dict], None],
    poll_interval: float = 1.0,
) -> None:
    """
    Blocking loop that polls the signal JSON file.
    Calls *callback* whenever a new signal is detected.
    """
    last_mtime: float = 0.0

    logger.info("File bridge watching %s", SIGNAL_OUTPUT_PATH)
    while True:
        try:
            if SIGNAL_OUTPUT_PATH.exists():
                mtime = SIGNAL_OUTPUT_PATH.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    data = json.loads(SIGNAL_OUTPUT_PATH.read_text())
                    callback(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("File bridge error: %s", exc)
        time.sleep(poll_interval)
