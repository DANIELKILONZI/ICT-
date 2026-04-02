"""
Integration – HTTP API Server (Flask)

Endpoints:
  GET  /signal      → latest signal JSON
  POST /signal      → receive a signal (from EA callback)
  GET  /health      → health check

Authentication:
  If `integration.api_key` is set in config.yaml, all /signal requests must
  include an `X-API-Key` header matching that value.  Requests with a missing
  or wrong key are rejected with HTTP 401.  The /health endpoint is public.

Production deployment:
  The server is started via Gunicorn when available (2 workers by default,
  configurable with `integration.http_workers`).  Flask's built-in dev server
  is used as a fallback when Gunicorn is not installed.
  Rate limiting (60 req/min per IP) is applied via Flask-Limiter when available.
"""
from __future__ import annotations

import logging
from typing import Any

from python.config import INTEGRATION
from python.signal_generator.signal_generator import load_latest_signal, save_signal

logger = logging.getLogger(__name__)

_API_KEY: str = INTEGRATION.get("api_key", "") or ""


def _check_api_key() -> "tuple[Any, int] | None":
    """Return a 401 response tuple if the key is wrong, or None if auth passes."""
    if not _API_KEY:
        return None  # No key configured – auth disabled
    from flask import request  # type: ignore

    provided = request.headers.get("X-API-Key", "")
    if provided != _API_KEY:
        return {"error": "unauthorized"}, 401
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
        auth_err = _check_api_key()
        if auth_err is not None:
            body, status = auth_err
            return jsonify(body), status
        signal = load_latest_signal()
        if signal is None:
            return jsonify({"error": "no_active_signal"}), 404
        return jsonify(signal)

    @app.route("/signal", methods=["POST"])
    def post_signal() -> Any:
        auth_err = _check_api_key()
        if auth_err is not None:
            body, status = auth_err
            return jsonify(body), status
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "empty_body"}), 400
        save_signal(data)
        logger.info("Signal received via HTTP POST: %s", data.get("symbol"))
        return jsonify({"status": "accepted"})

    def run_server() -> None:
        host    = INTEGRATION.get("http_host",    "0.0.0.0")
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
