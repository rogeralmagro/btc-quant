"""Position model: open trade state with risk management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from btc_quant.backtester.models.enums import OrderSide, StrategyTag
from btc_quant.backtester.models.order import ExecutedOrder


@dataclass
class TakeProfitLevel:
    """A single take-profit target within a position."""

    price: float
    quantity_pct: float          # fraction of original position to close at this level
    executed: bool = False
    executed_at_utc: datetime | None = None
    executed_quantity_btc: float | None = None


@dataclass
class Position:
    """Open position in the system.

    Mutable: changes as partial TPs execute, stops tighten, and quantity
    reduces. Belongs to exactly one strategy; capital never mixes across pools.

    V1 constraint: long-only (side must be BUY).
    """

    position_id: UUID
    strategy_tag: StrategyTag
    symbol: str
    side: OrderSide              # BUY = long position (V1: long-only)
    opened_at_utc: datetime
    quantity_btc: float          # current size (reduced by partial TP exits)
    entry_price_avg: float       # weighted average entry price

    # Accumulated costs:
    fees_paid_eur: float = 0.0

    # Risk management state:
    stop_loss_price: float | None = None
    take_profit_levels: list[TakeProfitLevel] = field(default_factory=list)

    # Traceability:
    entry_orders: list[ExecutedOrder] = field(default_factory=list)
    partial_exit_orders: list[ExecutedOrder] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.quantity_btc <= 0:
            raise ValueError(f"quantity_btc must be > 0, got {self.quantity_btc}")
        if self.entry_price_avg <= 0:
            raise ValueError(f"entry_price_avg must be > 0, got {self.entry_price_avg}")
        if self.side != OrderSide.BUY:
            raise ValueError(
                f"V1 supports long-only positions (side must be BUY); got {self.side!r}"
            )

    # ── P&L metrics ─────────────────────────────────────────────────────────

    def unrealized_pnl_eur(self, current_price: float) -> float:
        """Unrealized P&L in EUR. Positive when in profit."""
        return (current_price - self.entry_price_avg) * self.quantity_btc

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Unrealized return as a fraction of entry cost. 0.05 = +5%."""
        return (current_price - self.entry_price_avg) / self.entry_price_avg

    def unrealized_r_multiple(self, current_price: float) -> float | None:
        """P&L expressed as multiples of the initial risk (R).

        R = entry_price_avg − stop_loss_price for a LONG position.

        Returns None if no stop_loss_price is set.
        """
        if self.stop_loss_price is None:
            return None
        risk = self.entry_price_avg - self.stop_loss_price
        if risk <= 0:
            return None  # degenerate: stop at or above entry
        return (current_price - self.entry_price_avg) / risk

    # ── Risk management mutations ────────────────────────────────────────────

    def update_stop_loss(self, new_price: float) -> None:
        """Move stop-loss to new_price. Only tightening is allowed (up for LONG).

        Raises ValueError if:
        - Position has no initial stop — the strategy is misbehaving. If a stop
          must be added after open, refactor to a separate set_stop_loss method.
        - new_price <= 0.
        - new_price <= current stop — would loosen the stop.
        """
        if self.stop_loss_price is None:
            raise ValueError(
                "Cannot update stop_loss on position without initial stop. "
                "Position was created without stop_loss_price; updating implies "
                "the strategy is misbehaving. If you genuinely need to add a stop "
                "after position open, refactor to use a separate set_stop_loss method."
            )
        if new_price <= 0:
            raise ValueError(f"Stop must be positive, got {new_price}")
        if new_price <= self.stop_loss_price:
            raise ValueError(
                "Cannot loosen stop_loss. Stop can only be tightened "
                f"(moved up for LONG). Current: {self.stop_loss_price}, "
                f"proposed: {new_price}"
            )
        self.stop_loss_price = new_price

    def mark_tp_executed(
        self, level_index: int, executed_at: datetime, qty: float
    ) -> None:
        """Mark a take-profit level as executed.

        Raises ValueError if index is out of range or level is already executed.
        """
        n = len(self.take_profit_levels)
        if n == 0:
            raise ValueError("Position has no take_profit_levels")
        if not (0 <= level_index < n):
            raise ValueError(
                f"level_index {level_index} out of range (0..{n - 1})"
            )
        level = self.take_profit_levels[level_index]
        if level.executed:
            raise ValueError(
                f"TakeProfitLevel at index {level_index} is already executed"
            )
        level.executed = True
        level.executed_at_utc = executed_at
        level.executed_quantity_btc = qty

    # ── Quantity management ──────────────────────────────────────────────────

    def reduce_quantity(self, amount: float, executed_at: datetime) -> None:  # noqa: ARG002
        """Reduce position size by amount BTC (e.g. after a partial TP exit).

        Raises ValueError if amount is not in (0, quantity_btc].
        """
        if amount <= 0:
            raise ValueError(f"amount must be > 0, got {amount}")
        if amount > self.quantity_btc:
            raise ValueError(
                f"Cannot reduce by {amount} BTC: position only holds "
                f"{self.quantity_btc} BTC"
            )
        self.quantity_btc -= amount

    # ── Accumulation ─────────────────────────────────────────────────────────

    def add_to_position(self, executed_order: ExecutedOrder) -> None:
        """Add a follow-on BUY execution. Recalculates the weighted-average entry price.

        Raises:
            ValueError: if executed_order.order.side is not BUY, or symbol mismatches.
        """
        if executed_order.order.side != OrderSide.BUY:
            raise ValueError(
                f"add_to_position requires a BUY order; "
                f"got {executed_order.order.side!r}"
            )
        if executed_order.order.symbol != self.symbol:
            raise ValueError(
                f"Symbol mismatch: position is {self.symbol!r}, "
                f"order is {executed_order.order.symbol!r}"
            )
        new_qty = executed_order.executed_quantity_btc
        new_price = executed_order.executed_price
        total_cost = self.quantity_btc * self.entry_price_avg + new_qty * new_price
        new_total_qty = self.quantity_btc + new_qty
        self.entry_price_avg = total_cost / new_total_qty
        self.quantity_btc = new_total_qty
        self.fees_paid_eur += executed_order.fees_eur
        self.entry_orders.append(executed_order)
