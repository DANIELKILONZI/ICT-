"""
ICT Trading System – Central Exception Hierarchy

All custom exceptions are defined here so that callers can import a specific
type and catch it precisely instead of using a bare ``except Exception``.

Hierarchy
---------
ICTBaseError
├── DataSourceError          – failure fetching or caching market data
├── SignalValidationError    – signal dict is malformed / fails validation
└── RiskViolation            – a risk-management guard was triggered
"""
from __future__ import annotations


class ICTBaseError(RuntimeError):
    """Base class for all ICT system errors."""


class DataSourceError(ICTBaseError):
    """
    Raised when a market data source (MT5, CSV, SQLite) fails to deliver data.

    Attributes
    ----------
    symbol : str | None
    timeframe : str | None
    source : str | None   – "mt5" | "csv" | "sqlite"
    """

    def __init__(
        self,
        message: str,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        source: str | None = None,
    ) -> None:
        super().__init__(message)
        self.symbol = symbol
        self.timeframe = timeframe
        self.source = source

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.symbol:
            parts.append(f"symbol={self.symbol}")
        if self.timeframe:
            parts.append(f"timeframe={self.timeframe}")
        if self.source:
            parts.append(f"source={self.source}")
        return " | ".join(parts)


class SignalValidationError(ICTBaseError):
    """
    Raised when a generated or received signal fails validation.

    Attributes
    ----------
    field : str | None   – the specific field that caused the failure
    value             – the offending value, if available
    """

    def __init__(
        self,
        message: str,
        *,
        field: str | None = None,
        value=None,
    ) -> None:
        super().__init__(message)
        self.field = field
        self.value = value

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.field:
            parts.append(f"field={self.field!r}")
        if self.value is not None:
            parts.append(f"value={self.value!r}")
        return " | ".join(parts)


class RiskViolation(ICTBaseError):
    """
    Raised when a risk-management constraint is breached.

    Attributes
    ----------
    rule : str | None   – name of the violated rule (e.g. "max_daily_loss")
    actual            – the measured value that triggered the violation
    limit             – the configured threshold
    """

    def __init__(
        self,
        message: str,
        *,
        rule: str | None = None,
        actual=None,
        limit=None,
    ) -> None:
        super().__init__(message)
        self.rule = rule
        self.actual = actual
        self.limit = limit

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.rule:
            parts.append(f"rule={self.rule!r}")
        if self.actual is not None:
            parts.append(f"actual={self.actual}")
        if self.limit is not None:
            parts.append(f"limit={self.limit}")
        return " | ".join(parts)
