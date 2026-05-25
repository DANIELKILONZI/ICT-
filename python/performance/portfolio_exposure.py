"""
Portfolio Correlation Exposure Guard

Prevents excessive aggregate risk in correlated symbol groups.

Configuration (config.yaml → risk):
  correlation_groups:
    USD_MAJORS: [EURUSD, GBPUSD, USDJPY]
    METALS: [XAUUSD]
  max_correlated_exposure_percent: 2.0
"""
from __future__ import annotations

import csv
import logging
import threading
from datetime import datetime, timezone

from python.config import CONFIG, RISK, SIGNALS_LOG, PERF_LOG

logger = logging.getLogger(__name__)

_csv_lock = threading.Lock()


def _correlation_group_for(symbol: str) -> str | None:
    """Return the correlation group name that *symbol* belongs to, or None."""
    groups: dict = RISK.get("correlation_groups", {}) or {}
    sym_upper = symbol.upper()
    for group_name, members in groups.items():
        if isinstance(members, list) and sym_upper in [m.upper() for m in members]:
            return group_name
    return None


def _group_members(group_name: str) -> list[str]:
    """Return all symbols in the given correlation group."""
    groups: dict = RISK.get("correlation_groups", {}) or {}
    members = groups.get(group_name, [])
    return [m.upper() for m in members] if isinstance(members, list) else []


def _open_risk_for_symbols(symbols: list[str]) -> float:
    """
    Sum the risk_percent of all currently OPEN signals for the given symbols.

    Reads from signals.csv (Layer 2). Only rows with result=="OPEN" are counted.
    """
    path = SIGNALS_LOG if SIGNALS_LOG.exists() else PERF_LOG
    if not path.exists():
        return 0.0

    target = {s.upper() for s in symbols}
    total_risk = 0.0

    with _csv_lock:
        with open(path, "r", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("result") != "OPEN":
                    continue
                if row.get("symbol", "").upper() not in target:
                    continue
                try:
                    total_risk += float(row.get("risk_percent", 0))
                except (ValueError, TypeError):
                    total_risk += 1.0  # conservative fallback
    return total_risk


def is_portfolio_exposure_allowed(symbol: str, risk_percent: float) -> bool:
    """
    Return True if adding a new trade for *symbol* at *risk_percent* would
    keep the correlation group's total exposure within the configured limit.

    When *symbol* does not belong to any correlation group, the check passes
    unconditionally (single-symbol risk is governed by other guards).
    """
    group = _correlation_group_for(symbol)
    if group is None:
        return True

    max_exposure = RISK.get("max_correlated_exposure_percent", 2.0)
    members = _group_members(group)
    current_risk = _open_risk_for_symbols(members)

    if current_risk + risk_percent > max_exposure:
        logger.info(
            "Portfolio exposure guard: %s group=%s current=%.2f%% + new=%.2f%% > max=%.2f%%",
            symbol, group, current_risk, risk_percent, max_exposure,
        )
        return False

    return True
