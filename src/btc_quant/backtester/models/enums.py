"""Core enums for the backtester: orders, status, strategy tags, circuit breaker."""

from __future__ import annotations

from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    # V1 only uses MARKET. LIMIT/STOP/STOP_LIMIT are reserved for future use;
    # no execution logic is implemented for them in this version.


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class TradeOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    OPEN = "open"  # trade not yet closed


class StrategyTag(str, Enum):
    """Canonical strategy IDs. Centralised to prevent typos across the system."""

    STRAT_06_BASELINE = "strat06_baseline"
    STRAT_06_MODULATION = "strat06_modulation"
    STRAT_06_DEEP_VALUE = "strat06_deep_value"
    STRAT_07_TACTICAL = "strat07_tactical"
    DCA_BENCHMARK = "dca_benchmark"
    BUY_AND_HOLD_BENCHMARK = "buy_and_hold_benchmark"


class CircuitBreakerState(str, Enum):
    NORMAL = "normal"
    PAUSED = "paused"
    HALTED = "halted"
    KILLED = "killed"
