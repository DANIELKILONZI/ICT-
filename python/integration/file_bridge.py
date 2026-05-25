"""
Integration – File Bridge
Watches the signals/active/ directory and notifies subscribers via a callback
whenever any symbol's signal JSON file changes.

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
from pathlib import Path
from typing import Callable, Optional

from python.config import SIGNAL_ACTIVE_DIR

logger = logging.getLogger(__name__)

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

    SIGNAL_ACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    last_mtimes: dict[Path, float] = {}

    class _Handler(FileSystemEventHandler):
        def _handle(self, src_path: str) -> None:
            path = Path(src_path).resolve()
            if path.suffix != ".json":
                return
            try:
                mtime = path.stat().st_mtime
                if last_mtimes.get(path) == mtime:
                    return  # spurious duplicate event
                last_mtimes[path] = mtime
                data = json.loads(path.read_text())
                callback(data)
            except Exception as exc:  # noqa: BLE001
                logger.warning("File bridge error: %s", exc)

        def on_modified(self, event):
            self._handle(event.src_path)

        on_created = on_modified

    observer = Observer()
    observer.schedule(_Handler(), str(SIGNAL_ACTIVE_DIR), recursive=False)
    observer.start()
    logger.info("File bridge (watchdog) watching %s", SIGNAL_ACTIVE_DIR)

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
    last_mtimes: dict[Path, float] = {}

    logger.info("File bridge (polling, %.1fs) watching %s", poll_interval, SIGNAL_ACTIVE_DIR)
    while True:
        try:
            SIGNAL_ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
            for path in SIGNAL_ACTIVE_DIR.glob("*.json"):
                mtime = path.stat().st_mtime
                if last_mtimes.get(path) != mtime:
                    last_mtimes[path] = mtime
                    try:
                        data = json.loads(path.read_text())
                        callback(data)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("File bridge error reading %s: %s", path, exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("File bridge poll error: %s", exc)
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def poll_signal_file(
    callback: Callable[[dict], None],
    poll_interval: float = 1.0,
) -> None:
    """
    Blocking loop that calls *callback* whenever any symbol's signal JSON
    file in the active signals directory changes.

    Prefers OS-native notifications (watchdog) over polling.
    """
    try:
        import watchdog  # noqa: F401 - check availability
        _watch_with_watchdog(callback, poll_interval)
    except ImportError:
        logger.warning("watchdog not installed – falling back to polling. "
                       "Install watchdog for more efficient file monitoring.")
        _watch_with_polling(callback, poll_interval)
