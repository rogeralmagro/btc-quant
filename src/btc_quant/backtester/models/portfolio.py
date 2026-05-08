"""Portfolio and PortfolioSnapshot models: full system state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.backtester.models.order import ExecutedOrder
from btc_quant.backtester.models.pool import CapitalPool
from btc_quant.backtester.models.position import Position
from btc_quant.backtester.models.trade import Trade


@dataclass
class Portfolio:
    """Complete system state at any point in time.

    Holds one CapitalPool per strategy and tracks all open positions and
    closed trades. The engine keeps one Portfolio instance and mutates it
    bar by bar.

    Pools are the source of truth for capital; open_positions track active
    risk; closed_trades form the historical record.
    """

    pools: dict[StrategyTag, CapitalPool]
    open_positions: dict[UUID, Position] = field(default_factory=dict)
    closed_trades: list[Trade] = field(default_factory=list)

    # ── Global metrics ───────────────────────────────────────────────────────

    def total_btc_held(self) -> float:
        """Sum of BTC held across all strategy pools."""
        return sum(p.btc_held for p in self.pools.values())

    def total_value_eur(self, current_btc_price: float) -> float:
        """Sum of total_value_eur across all pools at current_btc_price."""
        return sum(p.total_value_eur(current_btc_price) for p in self.pools.values())

    def total_invested_eur(self) -> float:
        """Sum of total_invested_eur across all pools (cumulative capital in)."""
        return sum(p.total_invested_eur for p in self.pools.values())

    # ── Pool access ──────────────────────────────────────────────────────────

    def get_pool(self, tag: StrategyTag) -> CapitalPool:
        """Return the CapitalPool for tag.

        Raises:
            ValueError: if no pool is registered for tag.
        """
        if tag not in self.pools:
            raise ValueError(f"No pool registered for {tag!r}")
        return self.pools[tag]

    # ── Position lifecycle ───────────────────────────────────────────────────

    def open_position(self, position: Position) -> None:
        """Register a new open position.

        Raises:
            ValueError: if position_id already exists or its strategy has no pool.
        """
        if position.position_id in self.open_positions:
            raise ValueError(
                f"Position {position.position_id} is already open"
            )
        if position.strategy_tag not in self.pools:
            raise ValueError(
                f"No pool for strategy {position.strategy_tag!r}; "
                "register the pool before opening positions"
            )
        self.open_positions[position.position_id] = position

    def close_position(
        self,
        position_id: UUID,
        exit_orders: list[ExecutedOrder],
        exit_reason: str,
    ) -> Trade:
        """Close an open position, build a Trade, and append it to closed_trades.

        Raises:
            ValueError: if no open position with position_id exists.
        """
        if position_id not in self.open_positions:
            raise ValueError(f"No open position with id {position_id}")
        position = self.open_positions.pop(position_id)
        trade = Trade.from_closed_position(position, exit_orders, exit_reason)
        self.closed_trades.append(trade)
        return trade

    # ── Inter-pool transfers ─────────────────────────────────────────────────

    def transfer_cash(
        self,
        from_tag: StrategyTag,
        to_tag: StrategyTag,
        amount: float,
        reason: str,
    ) -> None:
        """Move cash between two pools (e.g. profit recycling from STRAT-07 to STRAT-06).

        Uses remove_cash on the source (validates no overdraft) and directly
        credits cash_eur on the destination so that total_invested_eur is not
        inflated by internal transfers.

        Raises:
            ValueError: if either pool does not exist, or source has insufficient funds.
        """
        from_pool = self.get_pool(from_tag)
        to_pool = self.get_pool(to_tag)
        from_pool.remove_cash(amount, reason=reason)
        to_pool.cash_eur += amount  # internal: bypass add_cash to not inflate invested_eur

    # ── Snapshot ─────────────────────────────────────────────────────────────

    def snapshot(self, timestamp_utc: datetime, btc_price: float) -> PortfolioSnapshot:
        """Capture an immutable point-in-time snapshot for the equity curve."""
        return PortfolioSnapshot(
            timestamp_utc=timestamp_utc,
            btc_price=btc_price,
            total_value_eur=self.total_value_eur(btc_price),
            total_btc=self.total_btc_held(),
            cash_by_pool={tag: pool.cash_eur for tag, pool in self.pools.items()},
            btc_by_pool={tag: pool.btc_held for tag, pool in self.pools.items()},
            open_positions_count=len(self.open_positions),
        )


@dataclass(frozen=True)
class PortfolioSnapshot:
    """Immutable point-in-time view of the portfolio. Used to build the equity curve.

    Note: cash_by_pool and btc_by_pool are dicts (mutable containers inside a
    frozen dataclass). Frozen prevents field reassignment, not dict mutation;
    callers should treat these dicts as read-only.
    """

    timestamp_utc: datetime
    btc_price: float
    total_value_eur: float
    total_btc: float
    cash_by_pool: dict[StrategyTag, float]
    btc_by_pool: dict[StrategyTag, float]
    open_positions_count: int
