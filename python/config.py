"""
ICT Trading System – Central Configuration Loader

Loads config.yaml, merges missing keys with built-in defaults, validates
required keys, and exposes typed convenience constants.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml


_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"

# ---------------------------------------------------------------------------
# Built-in defaults for every optional key.
# These are merged with the user-supplied config.yaml so that the system
# never crashes due to a missing key.
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "system": {
        "log_level": "INFO",
        "log_file": "logs/ict_system.log",
        "signal_output_path": "signals/active",
        "performance_log": "logs/performance.csv",
        "signals_log": "logs/signals.csv",
        "candidates_log": "logs/candidates.csv",
        "executions_log": "logs/executions.csv",
        "rejections_log": "logs/rejections.csv",
        "execution_feedback_dir": "signals/feedback",
        "scan_interval_seconds": 60,
    },
    "mt5": {
        "host": "localhost",
        "port": 18812,
        "login": 0,
        "password": "",
        "server": "",
    },
    "symbols": ["EURUSD"],
    "timeframes": {
        "macro": "D1",
        "structure": "H1",
        "entry": "M5",
        "supported": ["M1", "M5", "M15", "H1", "H4", "D1"],
    },
    "data": {
        "source": "csv",
        "csv_path": "data/csv/",
        "sqlite_path": "data/ict_data.db",
        "candle_history": 500,
    },
    "strategy": {
        "atr_period": 14,
        "atr_multiplier": 1.5,
        "swing_lookback": 2,
        "equal_level_pips": 3,
        "fib_swing_lookback": 50,
        "ob_search_window": 10,
        "disp_search_window": 6,
        "london_killzone_open": 7,    # UTC hour – London Killzone open
        "london_killzone_close": 10,  # UTC hour – London Killzone close
        "killzone_bos_lookback": 3,   # H1 BOS must be within this many candles
    },
    "scoring": {
        "bias_alignment": 0.20,
        "correct_zone": 0.15,
        "wrong_zone_penalty": 0.30,
        "h1_ob": 0.20,
        "h1_fvg": 0.15,
        "m5_sweep": 0.20,
        "m5_fvg": 0.10,
        "min_valid_score": 0.50,
    },
    "risk": {
        "risk_percent": 1.0,
        "max_daily_loss_percent": 3.0,
        "max_trades_per_day": 5,
        "max_spread_pips": 3.0,
        "max_slippage_pips": 2.0,
        "allow_multiple_positions_per_symbol": False,
        "trading_hours": {
            "london_open": 8,
            "london_close": 17,
            "ny_open": 13,
            "ny_close": 22,
        },
        "default_session": {},
        "sessions": {},
        "correlation_groups": {},
        "max_correlated_exposure_percent": 2.0,
    },
    "signal": {
        "min_confidence": 0.65,
        "min_risk_reward": 2.0,
        "spread_pips": 1.0,
    },
    "ml": {
        "enabled": False,
        "model_path": "ml/models/xgb_filter.pkl",
        "confidence_threshold": 0.65,
    },
    "integration": {
        "mode": "file",
        "http_host": "127.0.0.1",
        "http_port": 5000,
        "http_workers": 2,
        "socket_host": "0.0.0.0",
        "socket_port": 9999,
        "signal_ttl_seconds": 300,
        "api_key": "",
        "ip_allowlist": [],
        "hmac_secret": "",
        "nonce_window_seconds": 300,
    },
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "chat_id": "",
    },
    "backtest": {
        "output_dir": "backtests/signals",
        "data_source": "csv",
    },
    "pip_sizes": {
        "default": 0.0001,
        "JPY": 0.01,
        "XAU": 0.10,
        "XAG": 0.01,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Recursively merge *override* into *base*, returning a new dict.
    Keys present in *override* take precedence; keys absent from *override*
    fall back to *base* (i.e. the defaults).
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load() -> dict[str, Any]:
    """Load config.yaml (if it exists) and merge with built-in defaults."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH, "r") as fh:
            user_cfg = yaml.safe_load(fh) or {}
    else:
        user_cfg = {}
    return _deep_merge(_DEFAULTS, user_cfg)


# ---------------------------------------------------------------------------
# Singleton config dict – always fully populated thanks to _DEFAULTS merge.
# ---------------------------------------------------------------------------
CONFIG: dict[str, Any] = _load()


def get(key_path: str, default: Any = None) -> Any:
    """Dot-separated key lookup, e.g. get('risk.risk_percent')."""
    parts = key_path.split(".")
    node: Any = CONFIG
    for p in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(p, default)
    return node


# ---------------------------------------------------------------------------
# Convenience constants derived from config
# ---------------------------------------------------------------------------
SYMBOLS: list[str] = CONFIG.get("symbols", ["EURUSD"])
TIMEFRAMES: dict[str, str] = CONFIG.get("timeframes", {})
RISK: dict[str, Any] = CONFIG.get("risk", {})
STRATEGY: dict[str, Any] = CONFIG.get("strategy", {})
SIGNAL_CFG: dict[str, Any] = CONFIG.get("signal", {})
INTEGRATION: dict[str, Any] = CONFIG.get("integration", {})
ML_CFG: dict[str, Any] = CONFIG.get("ml", {})
TELEGRAM: dict[str, Any] = CONFIG.get("telegram", {})
SCORING: dict[str, Any] = CONFIG.get("scoring", {})
PIP_SIZES: dict[str, Any] = CONFIG.get("pip_sizes", {"default": 0.0001})


def _parse_session_time(value: Any, fallback: str | None = None) -> time | None:
    """Parse session times from int hour or HH:MM string."""
    if value is None:
        if fallback is None:
            return None
        value = fallback
    if isinstance(value, int):
        return time(hour=max(0, min(23, value)), minute=0)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        if ":" in raw:
            hh, mm = raw.split(":", 1)
            return time(hour=max(0, min(23, int(hh))), minute=max(0, min(59, int(mm))))
        return time(hour=max(0, min(23, int(raw))), minute=0)
    return None


def _session_windows(session_cfg: dict[str, Any]) -> list[tuple[time, time]]:
    windows: list[tuple[time, time]] = []
    for open_k, close_k in (("london_open", "london_close"), ("ny_open", "ny_close")):
        open_t = _parse_session_time(session_cfg.get(open_k))
        if open_t is None:
            continue
        close_t = _parse_session_time(session_cfg.get(close_k), fallback="23:59")
        if close_t is None:
            continue
        windows.append((open_t, close_t))
    return windows


def session_for_symbol(symbol: str) -> dict[str, Any]:
    """
    Return merged session config for *symbol*.

    Precedence:
      1) risk.sessions[{SYMBOL}]
      2) risk.default_session
      3) legacy risk.trading_hours
    """
    risk = CONFIG.get("risk", {})
    default_session = risk.get("default_session", {}) or {}
    legacy_hours = risk.get("trading_hours", {}) or {}
    base = default_session or legacy_hours
    symbol_overrides = (risk.get("sessions", {}) or {}).get(symbol.upper(), {}) or {}
    merged = {**base, **symbol_overrides}
    return merged


def session_is_open(symbol: str, now_utc: datetime | None = None) -> bool:
    """Return True when now is inside any configured trading window for symbol."""
    now = now_utc or datetime.now(timezone.utc)
    windows = _session_windows(session_for_symbol(symbol))
    if not windows:
        return True
    for start, end in windows:
        if start <= end:
            if start <= now.time() < end:
                return True
        else:
            if now.time() >= start or now.time() < end:
                return True
    return False


def current_session_bounds(symbol: str, now_utc: datetime | None = None) -> tuple[datetime, datetime] | None:
    """
    Return (valid_from, valid_to) UTC datetimes for the active session window.
    """
    now = now_utc or datetime.now(timezone.utc)
    windows = _session_windows(session_for_symbol(symbol))
    if not windows:
        return now, now + timedelta(hours=24)

    for start, end in windows:
        start_dt = now.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
        end_dt = now.replace(hour=end.hour, minute=end.minute, second=0, microsecond=0)

        if start <= end:
            if now.time() < start:
                continue
            if now.time() >= end:
                continue
            return start_dt, end_dt

        # Overnight window (e.g. 22:00-06:00)
        if now.time() >= start:
            return start_dt, end_dt + timedelta(days=1)
        if now.time() < end:
            return start_dt - timedelta(days=1), end_dt

    return None


def pip_size_for(symbol: str) -> float:
    """
    Return the pip size for *symbol* using the ``pip_sizes`` config map.

    Keys in the map are substring patterns matched (case-insensitively) against
    the symbol name.  The first matching key wins; ``default`` is the fallback.

    Examples:
        pip_size_for("USDJPY")  → 0.01   (matched by "JPY")
        pip_size_for("EURUSD")  → 0.0001 (no specific match → default)
        pip_size_for("XAUUSD")  → 0.10   (matched by "XAU")
    """
    sym = symbol.upper()
    for pattern, size in PIP_SIZES.items():
        if pattern == "default":
            continue
        if pattern.upper() in sym:
            return float(size)
    return float(PIP_SIZES.get("default", 0.0001))


SIGNAL_OUTPUT_PATH: Path = _ROOT / CONFIG["system"]["signal_output_path"]
SIGNAL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# Canonical active-signals directory.  One JSON file per symbol lives here.
SIGNAL_ACTIVE_DIR: Path = SIGNAL_OUTPUT_PATH


def signal_path_for(symbol: str) -> Path:
    """Return the Path for *symbol*'s active signal file.

    Example: signal_path_for("EURUSD") → .../signals/active/EURUSD.json
    """
    return SIGNAL_ACTIVE_DIR / f"{symbol}.json"

LOG_FILE: Path = _ROOT / CONFIG["system"]["log_file"]
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

PERF_LOG: Path = _ROOT / CONFIG["system"]["performance_log"]
PERF_LOG.parent.mkdir(parents=True, exist_ok=True)

# Multi-layer performance logs (Issue 11)
CANDIDATES_LOG: Path = _ROOT / CONFIG["system"]["candidates_log"]
SIGNALS_LOG: Path = _ROOT / CONFIG["system"]["signals_log"]
EXECUTIONS_LOG: Path = _ROOT / CONFIG["system"]["executions_log"]
REJECTIONS_LOG: Path = _ROOT / CONFIG["system"]["rejections_log"]
EXECUTION_FEEDBACK_DIR: Path = _ROOT / CONFIG["system"]["execution_feedback_dir"]
EXECUTION_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
for _p in (CANDIDATES_LOG, SIGNALS_LOG, EXECUTIONS_LOG, REJECTIONS_LOG):
    _p.parent.mkdir(parents=True, exist_ok=True)

# Backtest output directory
BACKTEST_OUTPUT_DIR: Path = _ROOT / CONFIG.get("backtest", {}).get("output_dir", "backtests/signals")
BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
