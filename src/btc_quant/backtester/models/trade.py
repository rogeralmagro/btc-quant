"""Trade model: completed entry+exit historical record."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID, uuid4

from btc_quant.backtester.models.enums import OrderSide, StrategyTag, TradeOutcome
from btc_quant.backtester.models.order import ExecutedOrder
from btc_quant.backtester.models.position import Position

_BREAKEVEN_THRESHOLD_EUR = 0.01  # net P&L within ±1 cent is treated as breakeven


@dataclass(frozen=True)
class Trade:
    """Completed trade (full entry + exit lifecycle). Immutable historical record.

    A Position generates exactly one Trade when fully closed.
    The trade captures the weighted average exit price across all exits
    (partial TPs + final close), so gross_pnl_eur reflects the entire
    lifecycle rather than only the final close leg.

    Note on initial_stop_loss: populated from position.stop_loss_price at
    close time, which may differ from the original stop if it was tightened
    via update_stop_loss. Position does not record the initial stop separately.
    """

    trade_id: UUID
    strategy_tag: StrategyTag
    symbol: str
    side: OrderSide

    # Entry:
    opened_at_utc: datetime
    entry_price_avg: float
    quantity_btc: float  # total quantity across all exits (original position size)

    # Exit:
    closed_at_utc: datetime
    exit_price_avg: float  # weighted average across partial + final exits

    # Result:
    gross_pnl_eur: float   # (exit_price_avg - entry_price_avg) * quantity_btc
    fees_eur: float        # all fees: entry (position.fees_paid_eur) + all exit fees
    net_pnl_eur: float     # gross_pnl_eur - fees_eur
    return_pct: float      # (exit_price_avg - entry_price_avg) / entry_price_avg

    # Original risk levels:
    initial_stop_loss: float | None = None
    initial_take_profits: list[float] = field(default_factory=list)

    # Outcome:
    outcome: TradeOutcome = TradeOutcome.OPEN
    exit_reason: str = ""  # e.g. "stop_loss_hit", "tp2_hit", "regime_flip"

    # Additional metrics:
    duration_hours: float = 0.0
    r_multiple_realized: float | None = None

    # Full traceability:
    entry_orders: list[ExecutedOrder] = field(default_factory=list)
    exit_orders: list[ExecutedOrder] = field(default_factory=list)  # partial TPs + final

    @classmethod
    def from_closed_position(
        cls,
        position: Position,
        exit_orders: list[ExecutedOrder],
        exit_reason: str,
    ) -> Trade:
        """Build a Trade from a fully closed Position.

        exit_orders covers only the FINAL close leg. Partial exits already
        recorded in position.partial_exit_orders are included automatically.

        Raises:
            ValueError: if exit_orders is empty.
        """
        if not exit_orders:
            raise ValueError("exit_orders must not be empty to close a position")

        all_exits = list(position.partial_exit_orders) + list(exit_orders)

        total_qty = sum(ex.executed_quantity_btc for ex in all_exits)
        exit_price_avg = (
            sum(ex.executed_quantity_btc * ex.executed_price for ex in all_exits)
            / total_qty
        )
        fees_eur = position.fees_paid_eur + sum(ex.fees_eur for ex in all_exits)
        gross_pnl_eur = (exit_price_avg - position.entry_price_avg) * total_qty
        net_pnl_eur = gross_pnl_eur - fees_eur
        return_pct = (exit_price_avg - position.entry_price_avg) / position.entry_price_avg
        closed_at_utc = max(ex.executed_at_utc for ex in all_exits)
        duration_hours = (
            (closed_at_utc - position.opened_at_utc).total_seconds() / 3600
        )

        if abs(net_pnl_eur) < _BREAKEVEN_THRESHOLD_EUR:
            outcome = TradeOutcome.BREAKEVEN
        elif net_pnl_eur > 0:
            outcome = TradeOutcome.WIN
        else:
            outcome = TradeOutcome.LOSS

        r_multiple_realized: float | None = None
        if position.stop_loss_price is not None:
            risk = position.entry_price_avg - position.stop_loss_price
            if risk > 0:
                r_multiple_realized = (
                    (exit_price_avg - position.entry_price_avg) / risk
                )

        return cls(
            trade_id=uuid4(),
            strategy_tag=position.strategy_tag,
            symbol=position.symbol,
            side=position.side,
            opened_at_utc=position.opened_at_utc,
            entry_price_avg=position.entry_price_avg,
            quantity_btc=total_qty,
            closed_at_utc=closed_at_utc,
            exit_price_avg=exit_price_avg,
            gross_pnl_eur=gross_pnl_eur,
            fees_eur=fees_eur,
            net_pnl_eur=net_pnl_eur,
            return_pct=return_pct,
            initial_stop_loss=position.stop_loss_price,
            initial_take_profits=[tp.price for tp in position.take_profit_levels],
            outcome=outcome,
            exit_reason=exit_reason,
            duration_hours=duration_hours,
            r_multiple_realized=r_multiple_realized,
            entry_orders=list(position.entry_orders),
            exit_orders=all_exits,
        )
