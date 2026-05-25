from __future__ import annotations

from datetime import datetime, timezone

from python.config import current_session_bounds, session_for_symbol, session_is_open


def test_session_for_symbol_uses_symbol_override():
    sess = session_for_symbol("XAUUSD")
    assert sess.get("london_open") == "00:00"
    assert sess.get("london_close") == "23:59"


def test_session_for_symbol_uses_default_when_no_override():
    sess = session_for_symbol("EURUSD")
    assert sess.get("london_open") in ("08:00", 8)


def test_session_is_open_for_xau_all_day():
    now = datetime(2026, 1, 1, 3, 0, tzinfo=timezone.utc)
    assert session_is_open("XAUUSD", now_utc=now)


def test_current_session_bounds_for_open_session():
    now = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc)
    bounds = current_session_bounds("EURUSD", now_utc=now)
    assert bounds is not None
    assert bounds[0] <= now <= bounds[1]
