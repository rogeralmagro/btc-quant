"""Unit tests for CapitalPool."""

from __future__ import annotations

import pytest

from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.backtester.models.pool import CapitalPool

TAG = StrategyTag.STRAT_06_BASELINE


def _pool(**overrides) -> CapitalPool:
    defaults = dict(strategy_tag=TAG, cash_eur=10_000.0)
    defaults.update(overrides)
    return CapitalPool(**defaults)


# ---------------------------------------------------------------------------
# Creation & validation
# ---------------------------------------------------------------------------


class TestPoolCreation:
    def test_pool_creation(self) -> None:
        p = _pool()
        assert p.strategy_tag == TAG
        assert p.cash_eur == 10_000.0
        assert p.btc_held == 0.0
        assert p.cost_basis_eur == 0.0
        assert p.total_invested_eur == 0.0
        assert p.realized_pnl_eur == 0.0

    def test_pool_with_btc(self) -> None:
        p = _pool(btc_held=0.5, cost_basis_eur=22_500.0)
        assert p.btc_held == 0.5
        assert p.cost_basis_eur == 22_500.0

    def test_negative_cash_raises(self) -> None:
        with pytest.raises(ValueError, match="cash_eur must be >= 0"):
            _pool(cash_eur=-1.0)

    def test_negative_btc_raises(self) -> None:
        with pytest.raises(ValueError, match="btc_held must be >= 0"):
            _pool(btc_held=-0.1)

    def test_negative_cost_basis_raises(self) -> None:
        with pytest.raises(ValueError, match="cost_basis_eur must be >= 0"):
            _pool(cost_basis_eur=-100.0)

    def test_zero_cash_is_valid(self) -> None:
        p = _pool(cash_eur=0.0)
        assert p.cash_eur == 0.0


# ---------------------------------------------------------------------------
# Read-only metrics
# ---------------------------------------------------------------------------


class TestPoolMetrics:
    def test_total_value_calculation(self) -> None:
        p = _pool(cash_eur=5_000.0, btc_held=0.1, cost_basis_eur=4_000.0)
        assert p.total_value_eur(45_000.0) == pytest.approx(5_000.0 + 0.1 * 45_000.0)

    def test_total_value_no_btc(self) -> None:
        p = _pool(cash_eur=10_000.0)
        assert p.total_value_eur(50_000.0) == pytest.approx(10_000.0)

    def test_average_entry_price(self) -> None:
        # Bought 0.5 BTC for 22 500 EUR → avg = 45 000
        p = _pool(btc_held=0.5, cost_basis_eur=22_500.0)
        assert p.average_entry_price() == pytest.approx(45_000.0)

    def test_average_entry_price_none_when_empty(self) -> None:
        p = _pool()
        assert p.average_entry_price() is None

    def test_average_entry_price_none_after_full_sale(self) -> None:
        p = _pool(btc_held=0.5, cost_basis_eur=22_500.0)
        p.remove_btc(0.5, revenue_eur=25_000.0)
        assert p.average_entry_price() is None


# ---------------------------------------------------------------------------
# Cash mutations
# ---------------------------------------------------------------------------


class TestCashMutations:
    def test_add_cash_updates_balance(self) -> None:
        p = _pool(cash_eur=1_000.0)
        p.add_cash(500.0, source="initial_allocation")
        assert p.cash_eur == pytest.approx(1_500.0)

    def test_add_cash_updates_total_invested(self) -> None:
        p = _pool(cash_eur=1_000.0)
        p.add_cash(500.0, source="profit_recycle")
        assert p.total_invested_eur == pytest.approx(500.0)

    def test_add_cash_cumulative(self) -> None:
        p = _pool(cash_eur=0.0)
        p.add_cash(1_000.0, source="week1")
        p.add_cash(500.0, source="week2")
        assert p.cash_eur == pytest.approx(1_500.0)
        assert p.total_invested_eur == pytest.approx(1_500.0)

    def test_add_cash_zero_raises(self) -> None:
        p = _pool()
        with pytest.raises(ValueError, match="must be > 0"):
            p.add_cash(0.0, source="x")

    def test_add_cash_negative_raises(self) -> None:
        p = _pool()
        with pytest.raises(ValueError, match="must be > 0"):
            p.add_cash(-100.0, source="x")

    def test_remove_cash_updates_balance(self) -> None:
        p = _pool(cash_eur=10_000.0)
        p.remove_cash(3_000.0, reason="btc_purchase")
        assert p.cash_eur == pytest.approx(7_000.0)

    def test_remove_cash_exact_balance(self) -> None:
        p = _pool(cash_eur=10_000.0)
        p.remove_cash(10_000.0, reason="full_deployment")
        assert p.cash_eur == pytest.approx(0.0)

    def test_overdraft_raises(self) -> None:
        p = _pool(cash_eur=1_000.0)
        with pytest.raises(ValueError, match="Overdraft denied"):
            p.remove_cash(1_001.0, reason="too_much")

    def test_remove_cash_zero_raises(self) -> None:
        p = _pool()
        with pytest.raises(ValueError, match="must be > 0"):
            p.remove_cash(0.0, reason="x")

    def test_add_remove_cash_updates_balance(self) -> None:
        p = _pool(cash_eur=0.0)
        p.add_cash(5_000.0, source="alloc")
        p.remove_cash(2_000.0, reason="purchase")
        p.add_cash(1_000.0, source="recycle")
        assert p.cash_eur == pytest.approx(4_000.0)
        assert p.total_invested_eur == pytest.approx(6_000.0)


# ---------------------------------------------------------------------------
# BTC mutations
# ---------------------------------------------------------------------------


class TestBtcMutations:
    def test_add_btc_updates_holdings_and_cost_basis(self) -> None:
        p = _pool()
        p.add_btc(qty=0.2, cost_eur=9_000.0)
        assert p.btc_held == pytest.approx(0.2)
        assert p.cost_basis_eur == pytest.approx(9_000.0)

    def test_add_btc_cumulative_avg(self) -> None:
        """Two buys at different prices → correct weighted average."""
        p = _pool()
        p.add_btc(qty=0.5, cost_eur=22_500.0)   # avg = 45 000
        p.add_btc(qty=0.5, cost_eur=25_000.0)   # avg = 47 500 (DCA up)
        assert p.average_entry_price() == pytest.approx(47_500.0)

    def test_add_btc_zero_qty_raises(self) -> None:
        p = _pool()
        with pytest.raises(ValueError, match="qty must be > 0"):
            p.add_btc(qty=0.0, cost_eur=100.0)

    def test_add_btc_zero_cost_raises(self) -> None:
        p = _pool()
        with pytest.raises(ValueError, match="cost_eur must be > 0"):
            p.add_btc(qty=0.1, cost_eur=0.0)

    def test_remove_btc_updates_holdings_and_cost_basis(self) -> None:
        p = _pool(btc_held=1.0, cost_basis_eur=45_000.0)
        p.remove_btc(qty=0.5, revenue_eur=25_000.0)
        assert p.btc_held == pytest.approx(0.5)
        # cost_basis reduced proportionally: 45 000 * 0.5 = 22 500 removed
        assert p.cost_basis_eur == pytest.approx(22_500.0)

    def test_remove_btc_realizes_pnl_profit(self) -> None:
        """Sell 0.5 BTC bought at avg 45 000 for 25 000 → realise +2 500."""
        p = _pool(btc_held=1.0, cost_basis_eur=45_000.0)
        p.remove_btc(qty=0.5, revenue_eur=25_000.0)
        # cost_of_sold = 0.5 * 45 000 = 22 500; realized = 25 000 - 22 500 = 2 500
        assert p.realized_pnl_eur == pytest.approx(2_500.0)

    def test_remove_btc_realizes_pnl_loss(self) -> None:
        """Sell at a loss → negative realized P&L."""
        p = _pool(btc_held=1.0, cost_basis_eur=45_000.0)
        p.remove_btc(qty=0.5, revenue_eur=20_000.0)
        # cost = 22 500; realized = 20 000 - 22 500 = -2 500
        assert p.realized_pnl_eur == pytest.approx(-2_500.0)

    def test_oversell_btc_raises(self) -> None:
        p = _pool(btc_held=0.5, cost_basis_eur=22_500.0)
        with pytest.raises(ValueError, match="Oversell denied"):
            p.remove_btc(qty=0.6, revenue_eur=27_000.0)

    def test_remove_btc_zero_qty_raises(self) -> None:
        p = _pool(btc_held=0.5, cost_basis_eur=22_500.0)
        with pytest.raises(ValueError, match="qty must be > 0"):
            p.remove_btc(qty=0.0, revenue_eur=0.0)

    def test_remove_btc_full_position(self) -> None:
        p = _pool(btc_held=1.0, cost_basis_eur=45_000.0)
        p.remove_btc(qty=1.0, revenue_eur=50_000.0)
        assert p.btc_held == pytest.approx(0.0)
        assert p.cost_basis_eur == pytest.approx(0.0)
        assert p.realized_pnl_eur == pytest.approx(5_000.0)
