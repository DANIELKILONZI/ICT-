"""
Integration – HTTP API Server (Flask)

Endpoints:
  GET  /signal      → latest signal JSON
  POST /signal      → receive a signal (from EA callback)
  GET  /health      → health check
"""
from __future__ import annotations

import logging
from typing import Any

from python.config import INTEGRATION
from python.signal_generator.signal_generator import load_latest_signal, save_signal

logger = logging.getLogger(__name__)

try:
    from flask import Flask, jsonify, request  # type: ignore

    app = Flask(__name__)

    @app.route("/health", methods=["GET"])
    def health() -> Any:
        return jsonify({"status": "ok"})

    @app.route("/signal", methods=["GET"])
    def get_signal() -> Any:
        signal = load_latest_signal()
        if signal is None:
            return jsonify({"error": "no_active_signal"}), 404
        return jsonify(signal)

    @app.route("/signal", methods=["POST"])
    def post_signal() -> Any:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "empty_body"}), 400
        save_signal(data)
        logger.info("Signal received via HTTP POST: %s", data.get("symbol"))
        return jsonify({"status": "accepted"})

    def run_server() -> None:
        host = INTEGRATION.get("http_host", "0.0.0.0")
        port = INTEGRATION.get("http_port", 5000)
        logger.info("Starting HTTP API on %s:%d", host, port)
        app.run(host=host, port=port, debug=False, use_reloader=False)

except ImportError:
    logger.warning("Flask not installed – HTTP integration unavailable.")

    def run_server() -> None:
        raise RuntimeError("Flask is required for HTTP integration mode.")
