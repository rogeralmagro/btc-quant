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
    STRAT_06_BUFFER = "strat06_buffer"
    STRAT_06_RESERVE = "strat06_reserve"
    STRAT_07_TACTICAL = "strat07_tactical"
    DCA_BENCHMARK = "dca_benchmark"
    BUY_AND_HOLD_BENCHMARK = "buy_and_hold_benchmark"


class MarketRegime(str, Enum):
    """Volatility regime used to select cost parameters in ExecutionSimulatorV2.

    NORMAL:   ATR(14) < 1.5 × annual ATR average  → slippage 0.05 %
    VOLATILE: 1.5 × avg ≤ ATR < 3.0 × avg          → slippage 0.15 %
    STRESS:   ATR(14) ≥ 3.0 × annual ATR average   → slippage 0.50 %
    CASCADE:  intra-bar move > 10 % of open        → slippage 1.00 %
              (overrides NORMAL/VOLATILE/STRESS on the bar being executed)
    """

    NORMAL = "normal"
    VOLATILE = "volatile"
    STRESS = "stress"
    CASCADE = "cascade"


class CircuitBreakerState(str, Enum):
    NORMAL = "normal"
    PAUSED = "paused"
    HALTED = "halted"
    KILLED = "killed"
