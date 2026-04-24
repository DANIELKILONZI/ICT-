"""
Unit tests for python.integration.socket_bridge

Covers:
  - _get_socket() lazy initialisation (zmq available / unavailable)
  - _get_socket() returns cached socket on repeated calls
  - _get_socket() graceful degradation when bind raises
  - publish_signal() sends the correct multipart message
  - publish_signal() is a no-op when socket is None
  - close() calls socket.close(linger=0) and sets _socket to None
  - close() is idempotent (safe when socket is already None)
"""
from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, call, patch

import pytest

import python.integration.socket_bridge as socket_bridge


@pytest.fixture(autouse=True)
def reset_socket():
    """Reset the singleton socket state before and after every test."""
    socket_bridge._socket = None
    yield
    socket_bridge._socket = None


# ---------------------------------------------------------------------------
# _get_socket
# ---------------------------------------------------------------------------

class TestGetSocket:
    def test_returns_none_when_zmq_absent(self):
        with patch.dict(sys.modules, {"zmq": None}):
            result = socket_bridge._get_socket()
        assert result is None
        assert socket_bridge._socket is None

    def test_creates_socket_when_zmq_available(self):
        mock_sock = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.socket.return_value = mock_sock
        mock_zmq = MagicMock()
        mock_zmq.Context.instance.return_value = mock_ctx
        mock_zmq.PUB = 1

        with patch.dict(sys.modules, {"zmq": mock_zmq}):
            result = socket_bridge._get_socket()

        assert result is mock_sock
        assert socket_bridge._socket is mock_sock
        mock_sock.bind.assert_called_once()

    def test_bind_uses_configured_host_and_port(self):
        mock_sock = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.socket.return_value = mock_sock
        mock_zmq = MagicMock()
        mock_zmq.Context.instance.return_value = mock_ctx
        mock_zmq.PUB = 1

        with patch("python.integration.socket_bridge.INTEGRATION",
                   {"socket_host": "127.0.0.1", "socket_port": 5555}):
            with patch.dict(sys.modules, {"zmq": mock_zmq}):
                socket_bridge._get_socket()

        mock_sock.bind.assert_called_once_with("tcp://127.0.0.1:5555")

    def test_returns_cached_socket_on_second_call(self):
        mock_sock = MagicMock()
        socket_bridge._socket = mock_sock

        result = socket_bridge._get_socket()

        assert result is mock_sock

    def test_returns_none_when_bind_raises(self):
        mock_sock = MagicMock()
        mock_sock.bind.side_effect = OSError("address in use")
        mock_ctx = MagicMock()
        mock_ctx.socket.return_value = mock_sock
        mock_zmq = MagicMock()
        mock_zmq.Context.instance.return_value = mock_ctx
        mock_zmq.PUB = 1

        with patch.dict(sys.modules, {"zmq": mock_zmq}):
            result = socket_bridge._get_socket()

        assert result is None


# ---------------------------------------------------------------------------
# publish_signal
# ---------------------------------------------------------------------------

class TestPublishSignal:
    def test_publishes_correct_topic_and_json(self):
        mock_sock = MagicMock()
        socket_bridge._socket = mock_sock

        signal = {"symbol": "EURUSD", "direction": "BUY"}
        socket_bridge.publish_signal(signal)

        mock_sock.send_multipart.assert_called_once()
        frames = mock_sock.send_multipart.call_args[0][0]
        assert frames[0] == b"signal"
        assert json.loads(frames[1]) == signal

    def test_publishes_full_signal_dict(self):
        mock_sock = MagicMock()
        socket_bridge._socket = mock_sock

        signal = {
            "symbol": "XAUUSD",
            "direction": "SELL",
            "entry_price": 2000.0,
            "stop_loss": 2010.0,
            "take_profit": 1980.0,
        }
        socket_bridge.publish_signal(signal)

        frames = mock_sock.send_multipart.call_args[0][0]
        received = json.loads(frames[1])
        assert received["symbol"] == "XAUUSD"
        assert received["entry_price"] == 2000.0

    def test_no_op_when_socket_is_none(self):
        # _socket is None (reset by fixture); should not raise
        socket_bridge.publish_signal({"symbol": "EURUSD"})  # no exception

    def test_send_error_does_not_propagate(self):
        mock_sock = MagicMock()
        mock_sock.send_multipart.side_effect = RuntimeError("zmq send failed")
        socket_bridge._socket = mock_sock

        # Should log a warning but not raise
        socket_bridge.publish_signal({"symbol": "EURUSD"})


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_calls_socket_close_with_linger_zero(self):
        mock_sock = MagicMock()
        socket_bridge._socket = mock_sock

        socket_bridge.close()

        mock_sock.close.assert_called_once_with(linger=0)

    def test_close_sets_socket_to_none(self):
        mock_sock = MagicMock()
        socket_bridge._socket = mock_sock

        socket_bridge.close()

        assert socket_bridge._socket is None

    def test_close_when_socket_already_none_is_safe(self):
        # _socket is None; calling close() must not raise
        socket_bridge.close()

    def test_close_is_idempotent(self):
        mock_sock = MagicMock()
        socket_bridge._socket = mock_sock

        socket_bridge.close()
        socket_bridge.close()  # second call – _socket is already None

        assert socket_bridge._socket is None
