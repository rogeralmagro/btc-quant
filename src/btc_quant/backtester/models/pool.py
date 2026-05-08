"""CapitalPool model: per-strategy capital account."""

from __future__ import annotations

from dataclasses import dataclass, field

from btc_quant.backtester.models.enums import StrategyTag


@dataclass
class CapitalPool:
    """Isolated capital account for a single strategy.

    Allows STRAT-06 and STRAT-07 to hold independent cash and BTC balances.
    Cross-pool transfers (profit recycling) are handled by Portfolio, not here.

    Invariants maintained at all times:
    - cash_eur >= 0  (no overdraft)
    - btc_held >= 0
    - cost_basis_eur >= 0
    """

    strategy_tag: StrategyTag
    cash_eur: float              # available cash (EUR / USDT equivalent)
    btc_held: float = 0.0       # BTC held by this strategy
    cost_basis_eur: float = 0.0 # total EUR spent on btc_held (for avg-entry tracking)
    total_invested_eur: float = 0.0  # cumulative cash ever added to this pool
    realized_pnl_eur: float = 0.0   # closed P&L from completed trades/sales

    def __post_init__(self) -> None:
        if self.cash_eur < 0:
            raise ValueError(f"cash_eur must be >= 0, got {self.cash_eur}")
        if self.btc_held < 0:
            raise ValueError(f"btc_held must be >= 0, got {self.btc_held}")
        if self.cost_basis_eur < 0:
            raise ValueError(f"cost_basis_eur must be >= 0, got {self.cost_basis_eur}")
        if self.total_invested_eur < 0:
            raise ValueError(
                f"total_invested_eur must be >= 0, got {self.total_invested_eur}"
            )

    # ── Read-only metrics ────────────────────────────────────────────────────

    def total_value_eur(self, current_btc_price: float) -> float:
        """Total pool value: cash + BTC at current_btc_price."""
        return self.cash_eur + self.btc_held * current_btc_price

    def average_entry_price(self) -> float | None:
        """Weighted average price paid per BTC held. None if no BTC held."""
        if self.btc_held > 0:
            return self.cost_basis_eur / self.btc_held
        return None

    # ── Cash mutations ───────────────────────────────────────────────────────

    def add_cash(self, amount: float, source: str) -> None:  # noqa: ARG002
        """Credit cash to the pool (e.g. initial allocation, profit recycle).

        Raises:
            ValueError: if amount <= 0.
        """
        if amount <= 0:
            raise ValueError(f"add_cash amount must be > 0, got {amount}")
        self.cash_eur += amount
        self.total_invested_eur += amount

    def remove_cash(self, amount: float, reason: str) -> None:  # noqa: ARG002
        """Debit cash from the pool (e.g. to fund a BTC purchase).

        Raises:
            ValueError: if amount <= 0 or amount > cash_eur (no overdraft).
        """
        if amount <= 0:
            raise ValueError(f"remove_cash amount must be > 0, got {amount}")
        if amount > self.cash_eur:
            raise ValueError(
                f"Overdraft denied: requested {amount:.2f} EUR but pool only holds "
                f"{self.cash_eur:.2f} EUR"
            )
        self.cash_eur -= amount

    # ── BTC mutations ────────────────────────────────────────────────────────

    def add_btc(self, qty: float, cost_eur: float) -> None:
        """Credit BTC to the pool and record its EUR cost for avg-entry tracking.

        Raises:
            ValueError: if qty <= 0 or cost_eur <= 0.
        """
        if qty <= 0:
            raise ValueError(f"add_btc qty must be > 0, got {qty}")
        if cost_eur <= 0:
            raise ValueError(f"add_btc cost_eur must be > 0, got {cost_eur}")
        self.btc_held += qty
        self.cost_basis_eur += cost_eur

    def remove_btc(self, qty: float, revenue_eur: float) -> None:
        """Debit BTC from the pool and realise P&L against the avg cost basis.

        Proportionally reduces cost_basis_eur based on the fraction of BTC sold.
        Updates realized_pnl_eur with (revenue - cost_of_sold_btc).

        Raises:
            ValueError: if qty <= 0 or qty > btc_held (no oversell).
        """
        if qty <= 0:
            raise ValueError(f"remove_btc qty must be > 0, got {qty}")
        if qty > self.btc_held:
            raise ValueError(
                f"Oversell denied: requested {qty} BTC but pool only holds "
                f"{self.btc_held} BTC"
            )
        avg_cost_per_btc = self.cost_basis_eur / self.btc_held
        cost_of_sold = qty * avg_cost_per_btc
        self.realized_pnl_eur += revenue_eur - cost_of_sold
        self.btc_held = max(0.0, self.btc_held - qty)
        self.cost_basis_eur = max(0.0, self.cost_basis_eur - cost_of_sold)
