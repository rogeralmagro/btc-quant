"""BacktestResult and CircuitBreakerEvent models: final backtest output."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from btc_quant.backtester.models.enums import CircuitBreakerState, StrategyTag
from btc_quant.backtester.models.portfolio import Portfolio, PortfolioSnapshot
from btc_quant.backtester.models.trade import Trade


def _assert_utc(dt: datetime, field_name: str) -> None:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be tz-aware (UTC)")
    if dt.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field_name} timezone must be UTC (offset must be 0)")


@dataclass(frozen=True)
class CircuitBreakerEvent:
    """Record of a single circuit-breaker state transition."""

    timestamp_utc: datetime
    trigger_name: str
    state_before: CircuitBreakerState
    state_after: CircuitBreakerState
    description: str

    def __post_init__(self) -> None:
        _assert_utc(self.timestamp_utc, "timestamp_utc")


@dataclass(frozen=True)
class BacktestResult:
    """Complete output of a backtest run. Immutable historical record.

    Contains everything needed for downstream analysis: the final portfolio
    state, full equity curve, all trades, per-strategy attribution, and any
    circuit-breaker events that occurred.

    Invariant enforced at construction:
    - sum(len(v) for v in trades_by_strategy.values()) == len(closed_trades)

    Note: frozen prevents field reassignment. Contained mutable objects
    (Portfolio, lists) can still be mutated; treat them as read-only after
    construction.
    """

    # Backtest configuration:
    start_date_utc: datetime
    end_date_utc: datetime
    initial_capital_eur: float
    strategies_used: list[StrategyTag]

    # Results:
    final_portfolio: Portfolio
    equity_curve: list[PortfolioSnapshot]  # chronological
    closed_trades: list[Trade]

    # Per-strategy attribution:
    trades_by_strategy: dict[StrategyTag, list[Trade]]

    # Circuit breaker events (empty if none triggered):
    circuit_breaker_events: list[CircuitBreakerEvent] = field(default_factory=list)

    # Metadata:
    backtest_metadata: dict[str, Any] = field(default_factory=dict)
    # e.g. {"engine_version": "0.1", "execution_model": "open_next_bar",
    #        "slippage_model": "regime_dependent"}

    def __post_init__(self) -> None:
        _assert_utc(self.start_date_utc, "start_date_utc")
        _assert_utc(self.end_date_utc, "end_date_utc")

        if self.end_date_utc < self.start_date_utc:
            raise ValueError(
                f"end_date_utc ({self.end_date_utc}) must be >= "
                f"start_date_utc ({self.start_date_utc})"
            )

        if self.initial_capital_eur <= 0:
            raise ValueError(
                f"initial_capital_eur must be > 0, got {self.initial_capital_eur}"
            )

        total_by_strategy = sum(
            len(trades) for trades in self.trades_by_strategy.values()
        )
        if total_by_strategy != len(self.closed_trades):
            raise ValueError(
                f"trades_by_strategy total ({total_by_strategy}) must equal "
                f"len(closed_trades) ({len(self.closed_trades)})"
            )
