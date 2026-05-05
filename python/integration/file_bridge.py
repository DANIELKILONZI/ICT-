"""
Integration – File Bridge
Watches the signal JSON file and notifies subscribers via a simple callback.

Uses the ``watchdog`` library for OS-native file-change notifications
(inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on Windows)
so the process does not busy-wait between checks.

Falls back to polling every *poll_interval* seconds when ``watchdog`` is not
installed.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from python.config import INTEGRATION, SIGNAL_OUTPUT_PATH

logger = logging.getLogger(__name__)

_SIGNAL_TTL: int = int(INTEGRATION.get("signal_ttl_seconds", 300))


def _is_signal_fresh(data: dict) -> bool:
    """Return True when *data* was emitted within the configured TTL window."""
    ts = data.get("timestamp")
    if not ts:
        return True  # no timestamp → let the callback decide
    try:
        signal_time = datetime.fromisoformat(ts)
        age = (datetime.now(timezone.utc) - signal_time).total_seconds()
        if age > _SIGNAL_TTL:
            logger.debug("File bridge: signal expired (age %.0fs > TTL %ds) – skipping.", age, _SIGNAL_TTL)
            return False
    except (ValueError, TypeError):
        pass
    return True


# ---------------------------------------------------------------------------
# Watchdog-based implementation
# ---------------------------------------------------------------------------

def _watch_with_watchdog(
    callback: Callable[[dict], None],
    poll_interval: float,
) -> None:
    """Use OS-native file-system events via the watchdog library."""
    from watchdog.events import FileSystemEventHandler  # type: ignore
    from watchdog.observers import Observer  # type: ignore

    watched_dir = SIGNAL_OUTPUT_PATH.parent
    watched_dir.mkdir(parents=True, exist_ok=True)

    last_mtime: float = 0.0

    class _Handler(FileSystemEventHandler):
        def on_modified(self, event):
            nonlocal last_mtime
            if Path(event.src_path).resolve() != SIGNAL_OUTPUT_PATH.resolve():
                return
            try:
                mtime = SIGNAL_OUTPUT_PATH.stat().st_mtime
                if mtime == last_mtime:
                    return  # spurious duplicate event
                last_mtime = mtime
                data = json.loads(SIGNAL_OUTPUT_PATH.read_text())
                if _is_signal_fresh(data):
                    callback(data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("File bridge error: %s", exc)

        # Also handle CREATE so we catch the very first write
        on_created = on_modified

    observer = Observer()
    observer.schedule(_Handler(), str(watched_dir), recursive=False)
    observer.start()
    logger.info("File bridge (watchdog) watching %s", SIGNAL_OUTPUT_PATH)

    try:
        while True:
            time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        observer.stop()

    observer.join()


# ---------------------------------------------------------------------------
# Polling fallback
# ---------------------------------------------------------------------------

def _watch_with_polling(
    callback: Callable[[dict], None],
    poll_interval: float,
) -> None:
    """Busy-wait polling fallback used when watchdog is unavailable."""
    last_mtime: float = 0.0

    logger.info("File bridge (polling, %.1fs) watching %s", poll_interval, SIGNAL_OUTPUT_PATH)
    while True:
        try:
            if SIGNAL_OUTPUT_PATH.exists():
                mtime = SIGNAL_OUTPUT_PATH.stat().st_mtime
                if mtime != last_mtime:
                    last_mtime = mtime
                    data = json.loads(SIGNAL_OUTPUT_PATH.read_text())
                    if _is_signal_fresh(data):
                        callback(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("File bridge error: %s", exc)
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def poll_signal_file(
    callback: Callable[[dict], None],
    poll_interval: float = 1.0,
) -> None:
    """
    Blocking loop that calls *callback* whenever the signal JSON file changes.

    Prefers OS-native notifications (watchdog) over polling.
    """
    try:
        import watchdog  # noqa: F401 - check availability
        _watch_with_watchdog(callback, poll_interval)
    except ImportError:
        logger.warning("watchdog not installed – falling back to polling. "
                       "Install watchdog for more efficient file monitoring.")
        _watch_with_polling(callback, poll_interval)
