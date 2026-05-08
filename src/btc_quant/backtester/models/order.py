"""Order and ExecutedOrder models: from intent to execution record."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from btc_quant.backtester.models.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    StrategyTag,
)

_UTC_REQUIRED = "must be tz-aware (UTC)"
_UTC_OFFSET_REQUIRED = "timezone must be UTC (offset must be 0)"


def _assert_utc(dt: datetime, field_name: str) -> None:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} {_UTC_REQUIRED}")
    if dt.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field_name} {_UTC_OFFSET_REQUIRED}")


@dataclass(frozen=True)
class Order:
    """Order sent to the executor.

    In backtest: simulated by ExecutionSimulator.
    In live: sent to Binance API.

    V1 uses only MARKET orders. LIMIT / STOP / STOP_LIMIT fields are present
    for forward-compatibility but carry no execution logic in this version.
    """

    order_id: UUID
    strategy_tag: StrategyTag
    timestamp_utc: datetime  # when the order was submitted
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity_btc: float

    # Only for LIMIT / STOP / STOP_LIMIT:
    limit_price: float | None = None
    stop_price: float | None = None

    # Traceability:
    parent_signal_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _assert_utc(self.timestamp_utc, "timestamp_utc")

        if self.quantity_btc <= 0:
            raise ValueError(f"quantity_btc must be > 0, got {self.quantity_btc}")

        match self.order_type:
            case OrderType.MARKET:
                if self.limit_price is not None:
                    raise ValueError(
                        "MARKET order must not have limit_price"
                    )
                if self.stop_price is not None:
                    raise ValueError(
                        "MARKET order must not have stop_price"
                    )
            case OrderType.LIMIT:
                if self.limit_price is None or self.limit_price <= 0:
                    raise ValueError(
                        f"LIMIT order requires limit_price > 0, got {self.limit_price}"
                    )
            case OrderType.STOP:
                if self.stop_price is None or self.stop_price <= 0:
                    raise ValueError(
                        f"STOP order requires stop_price > 0, got {self.stop_price}"
                    )
            case OrderType.STOP_LIMIT:
                if self.stop_price is None or self.stop_price <= 0:
                    raise ValueError(
                        f"STOP_LIMIT order requires stop_price > 0, got {self.stop_price}"
                    )
                if self.limit_price is None or self.limit_price <= 0:
                    raise ValueError(
                        f"STOP_LIMIT order requires limit_price > 0, got {self.limit_price}"
                    )


@dataclass(frozen=True)
class ExecutedOrder:
    """Filled order — result of the executor.

    Records the actual execution price (which includes slippage), executed
    quantity (which may be partial), and fees paid.
    """

    order: Order
    executed_at_utc: datetime
    executed_price: float       # includes slippage
    executed_quantity_btc: float  # may be less than order.quantity_btc for partial fills
    fees_eur: float
    slippage_eur: float         # difference between theoretical and executed price value
    status: OrderStatus         # FILLED or PARTIALLY_FILLED
    notes: str = ""             # e.g. "executed at gap open"

    def __post_init__(self) -> None:
        _assert_utc(self.executed_at_utc, "executed_at_utc")

        if self.executed_at_utc < self.order.timestamp_utc:
            raise ValueError(
                f"executed_at_utc ({self.executed_at_utc}) must be >= "
                f"order.timestamp_utc ({self.order.timestamp_utc})"
            )

        if self.executed_price <= 0:
            raise ValueError(f"executed_price must be > 0, got {self.executed_price}")

        if not (0 < self.executed_quantity_btc <= self.order.quantity_btc):
            raise ValueError(
                f"executed_quantity_btc must be in (0, {self.order.quantity_btc}], "
                f"got {self.executed_quantity_btc}"
            )

        if self.fees_eur < 0:
            raise ValueError(f"fees_eur must be >= 0, got {self.fees_eur}")

        if self.status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            raise ValueError(
                f"ExecutedOrder status must be FILLED or PARTIALLY_FILLED, got {self.status!r}"
            )


def make_order(
    strategy_tag: StrategyTag,
    timestamp_utc: datetime,
    symbol: str,
    side: OrderSide,
    order_type: OrderType,
    quantity_btc: float,
    **kwargs: Any,
) -> Order:
    """Convenience constructor that auto-generates order_id."""
    return Order(
        order_id=uuid4(),
        strategy_tag=strategy_tag,
        timestamp_utc=timestamp_utc,
        symbol=symbol,
        side=side,
        order_type=order_type,
        quantity_btc=quantity_btc,
        **kwargs,
    )
