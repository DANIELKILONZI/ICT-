"""
Unit tests for python.integration.api_server

Covers:
  - GET /health → 200 {"status": "ok"} (public endpoint, no auth required)
  - GET /signal  → 404 when no signal on disk
  - GET /signal  → 200 with signal payload when signal is available
  - GET /signal  → 401 when wrong API key is supplied
  - GET /signal  → 200 when correct API key is supplied
  - POST /signal → 400 on empty body
  - POST /signal → 400 with "missing_fields" listing when required keys absent
  - POST /signal → 200 {"status": "accepted"} on valid payload
  - POST /signal → 200 when no API key is configured (auth disabled)
  - POST /signal → 401 when wrong API key is supplied
  - POST /signal → 200 when correct API key is supplied
  - POST /result → 400 on empty body
  - POST /result → 400 with "missing_fields" when required keys absent
  - POST /result → 200 {"status": "updated"} on valid payload
  - POST /result → calls update_trade_result with correct arguments
  - POST /result → 401 when wrong API key is supplied
  - POST /result → 200 when correct API key is supplied
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

pytest.importorskip("flask", reason="Flask is required for api_server tests")

from python.integration import api_server  # noqa: E402  (after importorskip)
from python.performance.tracker import statistics as _statistics  # noqa: E402


@pytest.fixture()
def client():
    """Flask test client with testing mode enabled."""
    api_server.app.config["TESTING"] = True
    with api_server.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health  (public)
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.get_json() == {"status": "ok"}

    def test_no_api_key_required(self, client):
        """Health endpoint must be public even when an API key is configured."""
        with patch.object(api_server, "_API_KEY", "supersecret"):
            r = client.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# GET /signal
# ---------------------------------------------------------------------------

class TestGetSignal:
    def test_returns_404_when_no_signal(self, client):
        with patch("python.integration.api_server.load_latest_signal", return_value=None):
            r = client.get("/signal")
        assert r.status_code == 404
        assert r.get_json()["error"] == "no_active_signal"

    def test_returns_signal_dict_when_available(self, client):
        sig = {"symbol": "EURUSD", "direction": "BUY", "entry_price": 1.1}
        with patch("python.integration.api_server.load_latest_signal", return_value=sig):
            r = client.get("/signal")
        assert r.status_code == 200
        body = r.get_json()
        assert body["symbol"] == "EURUSD"
        assert body["direction"] == "BUY"

    def test_unauthorized_with_wrong_key(self, client):
        with patch.object(api_server, "_API_KEY", "correctkey"):
            r = client.get("/signal", headers={"X-API-Key": "wrongkey"})
        assert r.status_code == 401
        assert r.get_json()["error"] == "unauthorized"

    def test_authorized_with_correct_key(self, client):
        sig = {"symbol": "GBPUSD", "direction": "SELL"}
        with patch.object(api_server, "_API_KEY", "correctkey"):
            with patch("python.integration.api_server.load_latest_signal", return_value=sig):
                r = client.get("/signal", headers={"X-API-Key": "correctkey"})
        assert r.status_code == 200
        assert r.get_json()["symbol"] == "GBPUSD"

    def test_passes_without_key_when_auth_disabled(self, client):
        """When _API_KEY is empty, auth is disabled and requests pass through."""
        sig = {"symbol": "USDJPY", "direction": "BUY"}
        with patch.object(api_server, "_API_KEY", ""):
            with patch("python.integration.api_server.load_latest_signal", return_value=sig):
                r = client.get("/signal")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /signal
# ---------------------------------------------------------------------------

class TestPostSignal:
    _VALID = {
        "symbol": "EURUSD",
        "direction": "BUY",
        "entry_price": 1.10000,
        "stop_loss": 1.09000,
        "take_profit": 1.12000,
    }

    def test_returns_400_on_completely_empty_body(self, client):
        r = client.post(
            "/signal",
            data="",
            content_type="application/json",
        )
        assert r.status_code == 400

    def test_returns_400_on_missing_required_fields(self, client):
        payload = {"symbol": "EURUSD"}  # missing direction, entry_price, stop_loss, take_profit
        r = client.post("/signal", json=payload)
        assert r.status_code == 400
        body = r.get_json()
        assert body["error"] == "missing_fields"
        missing = body["fields"]
        assert "direction" in missing
        assert "entry_price" in missing
        assert "stop_loss" in missing
        assert "take_profit" in missing

    def test_returns_400_with_single_missing_field(self, client):
        payload = {k: v for k, v in self._VALID.items() if k != "stop_loss"}
        r = client.post("/signal", json=payload)
        assert r.status_code == 400
        body = r.get_json()
        assert body["fields"] == ["stop_loss"]

    def test_returns_accepted_on_valid_payload(self, client):
        with patch("python.integration.api_server.save_signal") as mock_save:
            r = client.post("/signal", json=self._VALID)
        assert r.status_code == 200
        assert r.get_json() == {"status": "accepted"}
        mock_save.assert_called_once_with(self._VALID)

    def test_valid_payload_calls_save_signal_with_full_dict(self, client):
        extra = {**self._VALID, "confidence_score": 0.85, "setup_type": "ICT_SWEEP"}
        with patch("python.integration.api_server.save_signal") as mock_save:
            client.post("/signal", json=extra)
        mock_save.assert_called_once_with(extra)

    def test_unauthorized_with_wrong_key(self, client):
        with patch.object(api_server, "_API_KEY", "secret"):
            r = client.post(
                "/signal",
                json=self._VALID,
                headers={"X-API-Key": "badsecret"},
            )
        assert r.status_code == 401
        assert r.get_json()["error"] == "unauthorized"

    def test_authorized_with_correct_key(self, client):
        with patch.object(api_server, "_API_KEY", "secret"):
            with patch("python.integration.api_server.save_signal"):
                r = client.post(
                    "/signal",
                    json=self._VALID,
                    headers={"X-API-Key": "secret"},
                )
        assert r.status_code == 200
        assert r.get_json() == {"status": "accepted"}

    def test_passes_without_key_when_auth_disabled(self, client):
        with patch.object(api_server, "_API_KEY", ""):
            with patch("python.integration.api_server.save_signal"):
                r = client.post("/signal", json=self._VALID)
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /result  (auth-gated, reports trade-close outcome)
# ---------------------------------------------------------------------------

class TestPostResult:
    _VALID = {
        "symbol": "EURUSD",
        "direction": "BUY",
        "exit_price": 1.11000,
    }

    def test_returns_400_on_empty_body(self, client):
        r = client.post("/result", data="", content_type="application/json")
        assert r.status_code == 400

    def test_returns_400_on_missing_fields(self, client):
        r = client.post("/result", json={"symbol": "EURUSD"})
        assert r.status_code == 400
        body = r.get_json()
        assert body["error"] == "missing_fields"
        assert "direction" in body["fields"]
        assert "exit_price" in body["fields"]

    def test_returns_updated_on_valid_payload(self, client):
        with patch("python.integration.api_server.update_trade_result"):
            r = client.post("/result", json=self._VALID)
        assert r.status_code == 200
        assert r.get_json() == {"status": "updated"}

    def test_calls_update_trade_result_with_correct_args(self, client):
        with patch("python.integration.api_server.update_trade_result") as mock_utr:
            client.post("/result", json=self._VALID)
        mock_utr.assert_called_once_with(
            symbol="EURUSD",
            entry_price=0.0,
            exit_price=1.11000,
            direction="BUY",
        )

    def test_optional_entry_price_forwarded(self, client):
        payload = {**self._VALID, "entry_price": 1.10000}
        with patch("python.integration.api_server.update_trade_result") as mock_utr:
            client.post("/result", json=payload)
        mock_utr.assert_called_once_with(
            symbol="EURUSD",
            entry_price=1.10000,
            exit_price=1.11000,
            direction="BUY",
        )

    def test_unauthorized_with_wrong_key(self, client):
        with patch.object(api_server, "_API_KEY", "secret"):
            r = client.post("/result", json=self._VALID,
                            headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    def test_authorized_with_correct_key(self, client):
        with patch.object(api_server, "_API_KEY", "secret"):
            with patch("python.integration.api_server.update_trade_result"):
                r = client.post("/result", json=self._VALID,
                                headers={"X-API-Key": "secret"})
        assert r.status_code == 200

class TestMetrics:
    def test_returns_200_with_stats_dict(self, client):
        fake_stats = {"total": 10, "wins": 7, "losses": 3, "win_rate": 0.7}
        with patch("python.integration.api_server.statistics", return_value=fake_stats):
            r = client.get("/metrics")
        assert r.status_code == 200
        body = r.get_json()
        assert body["total"] == 10
        assert body["win_rate"] == 0.7

    def test_no_api_key_required(self, client):
        """Metrics endpoint must be public even when an API key is configured."""
        fake_stats = {"total": 0}
        with patch.object(api_server, "_API_KEY", "supersecret"):
            with patch("python.integration.api_server.statistics", return_value=fake_stats):
                r = client.get("/metrics")
        assert r.status_code == 200
