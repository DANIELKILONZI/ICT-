"""
Unit tests for python.integration.api_server security features (Issue 12)

Covers:
  - IP allowlist enforcement (403 when not in list)
  - HMAC-SHA256 signature validation on POST /signal (401 when wrong)
  - Nonce replay detection on POST /signal (409 on duplicate)
  - Schema validation on POST /signal (400 when required fields missing)
  - Valid requests with all security checks configured pass through
  - /health is public (no IP/auth checks)
"""
from __future__ import annotations

import hashlib
import hmac
import json
import uuid

import pytest

flask = pytest.importorskip("flask", reason="Flask required for API security tests")

import python.integration.api_server as api_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sign(secret: str, body: bytes) -> str:
    """Compute the expected X-Signature header value."""
    digest = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _make_signal(**overrides) -> dict:
    base = {
        "signal_id": "EURUSD-20260101-080000-ICT_FVG",
        "symbol": "EURUSD",
        "direction": "BUY",
        "entry_price": 1.08500,
        "stop_loss": 1.08300,
        "take_profit": 1.09000,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(monkeypatch, tmp_path):
    """Flask test client with security features configured."""
    import python.config as cfg_mod
    # Point signal files to tmp dir and configure EURUSD as a valid symbol
    active_dir = tmp_path / "active"
    active_dir.mkdir()
    monkeypatch.setattr(cfg_mod, "SIGNAL_ACTIVE_DIR", active_dir)
    monkeypatch.setattr(cfg_mod, "SYMBOLS", ["EURUSD", "GBPUSD"])

    # Configure security
    monkeypatch.setattr(api_mod, "_API_KEY",       "test-api-key-123")
    monkeypatch.setattr(api_mod, "_HMAC_SECRET",   "my-hmac-secret")
    monkeypatch.setattr(api_mod, "_IP_ALLOWLIST",  ["127.0.0.1"])
    monkeypatch.setattr(api_mod, "_NONCE_WINDOW",  300)
    monkeypatch.setattr(api_mod, "_seen_nonces",   {})

    return api_mod.app.test_client()


@pytest.fixture()
def open_client(monkeypatch, tmp_path):
    """Flask test client with NO security configured (all checks disabled)."""
    import python.config as cfg_mod
    active_dir = tmp_path / "active"
    active_dir.mkdir()
    monkeypatch.setattr(cfg_mod, "SIGNAL_ACTIVE_DIR", active_dir)
    monkeypatch.setattr(cfg_mod, "SYMBOLS", ["EURUSD", "GBPUSD"])

    monkeypatch.setattr(api_mod, "_API_KEY",       "")
    monkeypatch.setattr(api_mod, "_HMAC_SECRET",   "")
    monkeypatch.setattr(api_mod, "_IP_ALLOWLIST",  [])
    monkeypatch.setattr(api_mod, "_seen_nonces",   {})

    return api_mod.app.test_client()


# ---------------------------------------------------------------------------
# /health is always public
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "ok"


# ---------------------------------------------------------------------------
# IP allowlist
# ---------------------------------------------------------------------------

class TestIpAllowlist:
    def test_allowed_ip_passes(self, client):
        """127.0.0.1 is in the allowlist → should reach auth layer (401 without key)."""
        resp = client.get("/signal?symbol=EURUSD",
                          environ_base={"REMOTE_ADDR": "127.0.0.1"})
        # 401 means IP was allowed, auth key was checked next
        assert resp.status_code in (401, 404, 200)

    def test_disallowed_ip_blocked(self, client):
        resp = client.get("/signal?symbol=EURUSD",
                          environ_base={"REMOTE_ADDR": "10.99.99.99"})
        assert resp.status_code == 403
        assert "forbidden" in resp.get_json()["error"]

    def test_empty_allowlist_allows_all(self, open_client):
        resp = open_client.get("/signal?symbol=EURUSD",
                               environ_base={"REMOTE_ADDR": "192.168.1.100"})
        # Should not be 403
        assert resp.status_code != 403


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------

class TestApiKey:
    def test_wrong_key_returns_401(self, client):
        resp = client.get("/signal?symbol=EURUSD",
                          headers={"X-API-Key": "wrong"},
                          environ_base={"REMOTE_ADDR": "127.0.0.1"})
        assert resp.status_code == 401

    def test_correct_key_passes(self, client):
        resp = client.get("/signal?symbol=EURUSD",
                          headers={"X-API-Key": "test-api-key-123"},
                          environ_base={"REMOTE_ADDR": "127.0.0.1"})
        assert resp.status_code in (404, 200)  # 404 = no signal file, but auth passed

    def test_no_key_configured_allows_all(self, open_client):
        resp = open_client.get("/signal?symbol=EURUSD")
        assert resp.status_code != 401


# ---------------------------------------------------------------------------
# HMAC signature validation on POST /signal
# ---------------------------------------------------------------------------

class TestHmacSignature:
    def _headers(self, body: bytes, key: str = "test-api-key-123",
                 sig_secret: str = "my-hmac-secret") -> dict:
        return {
            "X-API-Key": key,
            "X-Signature": _sign(sig_secret, body),
            "Content-Type": "application/json",
        }

    def test_valid_signature_accepted(self, client, monkeypatch, tmp_path):
        import python.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "SYMBOLS", ["EURUSD"])
        active = tmp_path / "active2"
        active.mkdir()
        monkeypatch.setattr(cfg_mod, "SIGNAL_ACTIVE_DIR", active)

        body = json.dumps(_make_signal()).encode()
        resp = client.post(
            "/signal",
            data=body,
            headers=self._headers(body),
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert resp.status_code == 200

    def test_wrong_signature_returns_401(self, client):
        body = json.dumps(_make_signal()).encode()
        resp = client.post(
            "/signal",
            data=body,
            headers={
                "X-API-Key": "test-api-key-123",
                "X-Signature": "sha256=deadbeef",
                "Content-Type": "application/json",
            },
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert resp.status_code == 401
        assert "invalid_signature" in resp.get_json()["error"]

    def test_missing_signature_returns_401(self, client):
        body = json.dumps(_make_signal()).encode()
        resp = client.post(
            "/signal",
            data=body,
            headers={
                "X-API-Key": "test-api-key-123",
                "Content-Type": "application/json",
            },
            environ_base={"REMOTE_ADDR": "127.0.0.1"},
        )
        assert resp.status_code == 401
        assert "missing_signature" in resp.get_json()["error"]

    def test_no_secret_configured_skips_hmac(self, open_client, monkeypatch, tmp_path):
        import python.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "SYMBOLS", ["EURUSD"])
        active = tmp_path / "active3"
        active.mkdir()
        monkeypatch.setattr(cfg_mod, "SIGNAL_ACTIVE_DIR", active)

        body = json.dumps(_make_signal()).encode()
        resp = open_client.post(
            "/signal",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Nonce replay protection
# ---------------------------------------------------------------------------

class TestNonceReplay:
    def _post(self, client, payload: dict, monkeypatch, tmp_path):
        import python.config as cfg_mod
        active = tmp_path / "active"
        active.mkdir(exist_ok=True)
        monkeypatch.setattr(cfg_mod, "SIGNAL_ACTIVE_DIR", active)
        monkeypatch.setattr(cfg_mod, "SYMBOLS", ["EURUSD"])
        monkeypatch.setattr(api_mod, "_HMAC_SECRET", "")   # disable HMAC for simplicity
        monkeypatch.setattr(api_mod, "_API_KEY",     "")
        monkeypatch.setattr(api_mod, "_IP_ALLOWLIST", [])

        body = json.dumps(payload).encode()
        return client.post(
            "/signal",
            data=body,
            headers={"Content-Type": "application/json"},
        )

    def test_first_nonce_accepted(self, open_client, monkeypatch, tmp_path):
        monkeypatch.setattr(api_mod, "_seen_nonces", {})
        payload = {**_make_signal(), "nonce": "unique-nonce-abc"}
        resp = self._post(open_client, payload, monkeypatch, tmp_path)
        assert resp.status_code == 200

    def test_duplicate_nonce_rejected(self, open_client, monkeypatch, tmp_path):
        monkeypatch.setattr(api_mod, "_seen_nonces", {})
        payload = {**_make_signal(), "nonce": "same-nonce-xyz"}
        self._post(open_client, payload, monkeypatch, tmp_path)
        resp = self._post(open_client, payload, monkeypatch, tmp_path)
        assert resp.status_code == 409
        assert "replay_detected" in resp.get_json()["error"]

    def test_no_nonce_field_accepted(self, open_client, monkeypatch, tmp_path):
        monkeypatch.setattr(api_mod, "_seen_nonces", {})
        payload = _make_signal()   # no nonce field
        resp = self._post(open_client, payload, monkeypatch, tmp_path)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Schema validation on POST /signal
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    def _post_no_auth(self, open_client, payload: dict, monkeypatch, tmp_path):
        import python.config as cfg_mod
        active = tmp_path / "av"
        active.mkdir(exist_ok=True)
        monkeypatch.setattr(cfg_mod, "SIGNAL_ACTIVE_DIR", active)
        monkeypatch.setattr(cfg_mod, "SYMBOLS", ["EURUSD"])
        monkeypatch.setattr(api_mod, "_seen_nonces", {})
        body = json.dumps(payload).encode()
        return open_client.post(
            "/signal",
            data=body,
            headers={"Content-Type": "application/json"},
        )

    def test_missing_symbol_returns_400(self, open_client, monkeypatch, tmp_path):
        payload = {"direction": "BUY", "entry_price": 1.1,
                   "stop_loss": 1.0, "take_profit": 1.2}
        resp = self._post_no_auth(open_client, payload, monkeypatch, tmp_path)
        assert resp.status_code == 400
        data = resp.get_json()
        assert "symbol" in data["fields"]

    def test_all_required_fields_present(self, open_client, monkeypatch, tmp_path):
        resp = self._post_no_auth(open_client, _make_signal(), monkeypatch, tmp_path)
        assert resp.status_code == 200
