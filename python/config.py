"""
ICT Trading System – Central Configuration Loader
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _ROOT / "config.yaml"


def _load() -> dict[str, Any]:
    with open(_CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


# Singleton config dict
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

SIGNAL_OUTPUT_PATH: Path = _ROOT / CONFIG["system"]["signal_output_path"]
SIGNAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

LOG_FILE: Path = _ROOT / CONFIG["system"]["log_file"]
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

PERF_LOG: Path = _ROOT / CONFIG["system"]["performance_log"]
PERF_LOG.parent.mkdir(parents=True, exist_ok=True)
