"""
Integration – ZeroMQ PUB Socket Bridge

Broadcasts every signal as a JSON message on a ZeroMQ PUB socket so that
any number of downstream subscribers (EAs, dashboards, other bots) can
receive signals without polling.

Topic frame: ``b"signal"``
Data  frame: UTF-8 encoded JSON of the signal dict.

Configuration (config.yaml → integration):
    socket_host: "0.0.0.0"   # bind address
    socket_port: 9999         # TCP port
"""
from __future__ import annotations

import json
import logging
import threading

from python.config import INTEGRATION

logger = logging.getLogger(__name__)

_socket = None
_socket_lock = threading.Lock()

_TOPIC = b"signal"


def _get_socket():
    """Return a lazy-initialised ZeroMQ PUB socket (singleton)."""
    global _socket
    with _socket_lock:
        if _socket is not None:
            return _socket
        try:
            import zmq  # type: ignore

            host = INTEGRATION.get("socket_host", "0.0.0.0")
            port = int(INTEGRATION.get("socket_port", 9999))
            ctx = zmq.Context.instance()
            sock = ctx.socket(zmq.PUB)
            sock.bind(f"tcp://{host}:{port}")
            logger.info("ZeroMQ PUB socket bound on tcp://%s:%d", host, port)
            _socket = sock
        except ImportError:
            logger.error("pyzmq is not installed – socket integration unavailable.")
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to bind ZeroMQ socket: %s", exc)
    return _socket


def publish_signal(signal: dict) -> None:
    """Publish *signal* to all ZeroMQ subscribers."""
    sock = _get_socket()
    if sock is None:
        return
    try:
        payload = json.dumps(signal).encode()
        sock.send_multipart([_TOPIC, payload])
        logger.debug("Signal published via ZeroMQ: %s", signal.get("symbol"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("ZeroMQ publish error: %s", exc)


def close() -> None:
    """Close the socket (call on shutdown)."""
    global _socket
    with _socket_lock:
        if _socket is not None:
            try:
                _socket.close(linger=0)
            except Exception:  # noqa: BLE001
                pass
            _socket = None
            logger.info("ZeroMQ PUB socket closed.")
