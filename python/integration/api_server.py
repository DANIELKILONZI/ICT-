"""
Integration – HTTP API Server (Flask)

Endpoints:
  GET  /signal      → latest signal JSON for a given ?symbol=
  POST /signal      → receive a signal (from EA callback / external sender)
  GET  /health      → health check

Security (Issue 12):
  - Binds to 127.0.0.1 by default (localhost-only).
  - API key: all /signal requests must include ``X-API-Key`` when configured.
  - IP allowlist: requests from unlisted IPs are rejected with HTTP 403.
  - HMAC-SHA256 signature: POST /signal validates ``X-Signature: sha256=<hex>``
    against the request body when ``integration.hmac_secret`` is configured.
  - Nonce replay protection: the JSON body's ``nonce`` field is tracked;
    duplicate nonces within ``nonce_window_seconds`` are rejected with HTTP 409.
  - Signal schema validation: POST body must include required fields.

Production deployment:
  The server is started via Gunicorn when available (2 workers by default,
  configurable with ``integration.http_workers``).  Flask's built-in dev server
  is used as a fallback when Gunicorn is not installed.
  Rate limiting (60 req/min per IP) is applied via Flask-Limiter when available.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import threading
import time
from typing import Any

from python.config import INTEGRATION
from python.signal_generator.signal_generator import load_latest_signal, save_signal

logger = logging.getLogger(__name__)

_API_KEY: str = INTEGRATION.get("api_key", "") or ""
_HMAC_SECRET: str = INTEGRATION.get("hmac_secret", "") or ""
_IP_ALLOWLIST: list[str] = INTEGRATION.get("ip_allowlist", []) or []
_NONCE_WINDOW: int = int(INTEGRATION.get("nonce_window_seconds", 300))

# In-memory nonce store: {nonce: received_at_epoch_seconds}
_seen_nonces: dict[str, float] = {}
_nonce_lock = threading.Lock()

# Required fields for POST /signal
_REQUIRED_SIGNAL_FIELDS = {
    "symbol", "direction", "entry_price", "stop_loss", "take_profit",
}


def _evict_old_nonces() -> None:
    """Remove nonces older than the replay window."""
    cutoff = time.time() - _NONCE_WINDOW
    stale = [k for k, v in _seen_nonces.items() if v < cutoff]
    for k in stale:
        del _seen_nonces[k]


def _check_api_key() -> "tuple[Any, int] | None":
    """Return a 401 response tuple if the key is wrong, or None if auth passes."""
    if not _API_KEY:
        return None
    from flask import request  # type: ignore

    provided = request.headers.get("X-API-Key", "")
    if provided != _API_KEY:
        return {"error": "unauthorized"}, 401
    return None


def _check_ip_allowlist() -> "tuple[Any, int] | None":
    """Return 403 if the caller's IP is not in the allowlist (when configured)."""
    if not _IP_ALLOWLIST:
        return None
    from flask import request  # type: ignore

    remote_ip = request.remote_addr or ""
    if remote_ip not in _IP_ALLOWLIST:
        logger.warning("Rejected request from unlisted IP: %s", remote_ip)
        return {"error": "forbidden"}, 403
    return None


def _check_hmac_signature(body: bytes) -> "tuple[Any, int] | None":
    """
    Validate the ``X-Signature: sha256=<hex>`` header against *body*.

    Returns None when the signature is valid (or when HMAC is not configured).
    Returns a 401 error tuple when the signature is missing or invalid.
    """
    if not _HMAC_SECRET:
        return None
    from flask import request  # type: ignore

    provided = request.headers.get("X-Signature", "")
    if not provided.startswith("sha256="):
        return {"error": "missing_signature"}, 401

    expected = hmac.new(
        _HMAC_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    provided_hex = provided[len("sha256="):]

    if not hmac.compare_digest(provided_hex, expected):
        logger.warning("HMAC signature mismatch on POST /signal")
        return {"error": "invalid_signature"}, 401
    return None


def _check_nonce(data: dict) -> "tuple[Any, int] | None":
    """
    Check the ``nonce`` field in *data* for replay attacks.

    Returns None if nonce is absent (nonce is optional) or is fresh.
    Returns a 409 error tuple if the nonce was already seen within the window.
    """
    nonce = data.get("nonce")
    if not nonce:
        return None  # Nonce is optional; only checked when present

    with _nonce_lock:
        _evict_old_nonces()
        if nonce in _seen_nonces:
            logger.warning("Replay detected: nonce %r already seen", nonce)
            return {"error": "replay_detected", "nonce": nonce}, 409
        _seen_nonces[nonce] = time.time()
    return None


def _validate_signal_schema(data: dict) -> "tuple[Any, int] | None":
    """Return 400 if any required signal field is missing."""
    missing = [f for f in _REQUIRED_SIGNAL_FIELDS if f not in data]
    if missing:
        return {"error": "missing_fields", "fields": missing}, 400
    return None


try:
    from flask import Flask, jsonify, request  # type: ignore

    app = Flask(__name__)

    # ── Rate limiting ──────────────────────────────────────────────────────
    try:
        from flask_limiter import Limiter  # type: ignore
        from flask_limiter.util import get_remote_address  # type: ignore

        limiter = Limiter(
            get_remote_address,
            app=app,
            default_limits=["60 per minute"],
            storage_uri="memory://",
        )
        logger.debug("Flask-Limiter enabled (60 req/min per IP)")
    except ImportError:
        logger.warning("flask-limiter not installed – rate limiting disabled")
        limiter = None  # type: ignore

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.route("/signal", methods=["GET"])
    def get_signal() -> Any:
        ip_err = _check_ip_allowlist()
        if ip_err is not None:
            body, status = ip_err
            return jsonify(body), status

        auth_err = _check_api_key()
        if auth_err is not None:
            body, status = auth_err
            return jsonify(body), status

        raw_sym = request.args.get("symbol", "").strip().upper()
        if not raw_sym:
            return jsonify({"error": "missing_symbol_param"}), 400

        # Validate symbol against the trusted configured list to break the taint
        # chain before using it for any file lookup or response data.
        from python.config import SYMBOLS
        _valid_symbols = {s.upper() for s in SYMBOLS}
        if raw_sym not in _valid_symbols:
            return jsonify({"error": "unknown_symbol"}), 404

        # Use the validated (trusted-source) symbol value from here on.
        symbol = raw_sym
        signal = load_latest_signal(symbol)
        if signal is None:
            return jsonify({"error": "no_active_signal"}), 404
        return jsonify(signal)

    @app.route("/signal", methods=["POST"])
    def post_signal() -> Any:
        ip_err = _check_ip_allowlist()
        if ip_err is not None:
            body, status = ip_err
            return jsonify(body), status

        auth_err = _check_api_key()
        if auth_err is not None:
            body, status = auth_err
            return jsonify(body), status

        raw_body = request.get_data()

        hmac_err = _check_hmac_signature(raw_body)
        if hmac_err is not None:
            body, status = hmac_err
            return jsonify(body), status

        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "empty_body"}), 400

        nonce_err = _check_nonce(data)
        if nonce_err is not None:
            body, status = nonce_err
            return jsonify(body), status

        schema_err = _validate_signal_schema(data)
        if schema_err is not None:
            body, status = schema_err
            return jsonify(body), status

        save_signal(data)
        logger.info("Signal received via HTTP POST: %s", data.get("symbol"))
        return jsonify({"status": "accepted"})

    def run_server() -> None:
        host    = INTEGRATION.get("http_host",    "127.0.0.1")
        port    = int(INTEGRATION.get("http_port",    5000))
        workers = int(INTEGRATION.get("http_workers", 2))

        # ── Try Gunicorn (production WSGI server) ──────────────────────────
        try:
            from gunicorn.app.base import BaseApplication  # type: ignore

            class _StandaloneApp(BaseApplication):
                def __init__(self, application, options=None):
                    self.options = options or {}
                    self.application = application
                    super().__init__()

                def load_config(self):
                    for key, value in self.options.items():
                        if key in self.cfg.settings and value is not None:
                            self.cfg.set(key.lower(), value)

                def load(self):
                    return self.application

            options = {
                "bind":    f"{host}:{port}",
                "workers": workers,
                "loglevel": "info",
            }
            logger.info(
                "Starting HTTP API via Gunicorn on %s:%d (%d workers)", host, port, workers
            )
            _StandaloneApp(app, options).run()

        except ImportError:
            logger.warning(
                "Gunicorn not installed – falling back to Flask dev server. "
                "Install gunicorn for production use."
            )
            app.run(host=host, port=port, debug=False, use_reloader=False)

except ImportError:
    logger.warning("Flask not installed – HTTP integration unavailable.")

    def run_server() -> None:
        raise RuntimeError("Flask is required for HTTP integration mode.")
