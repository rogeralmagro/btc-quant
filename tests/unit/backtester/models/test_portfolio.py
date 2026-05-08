"""Unit tests for Portfolio and PortfolioSnapshot."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from btc_quant.backtester.models.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    StrategyTag,
    TradeOutcome,
)
from btc_quant.backtester.models.order import ExecutedOrder, Order
from btc_quant.backtester.models.pool import CapitalPool
from btc_quant.backtester.models.portfolio import Portfolio, PortfolioSnapshot
from btc_quant.backtester.models.position import Position

UTC = timezone.utc
TS = datetime(2024, 9, 1, 0, 0, 0, tzinfo=UTC)
CLOSE_TS = datetime(2024, 9, 5, 0, 0, 0, tzinfo=UTC)

TAG06 = StrategyTag.STRAT_06_BASELINE
TAG07 = StrategyTag.STRAT_07_TACTICAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pool(tag: StrategyTag, cash: float = 10_000.0) -> CapitalPool:
    return CapitalPool(strategy_tag=tag, cash_eur=cash)


def _portfolio(
    cash06: float = 10_000.0,
    cash07: float = 5_000.0,
) -> Portfolio:
    return Portfolio(
        pools={
            TAG06: _pool(TAG06, cash06),
            TAG07: _pool(TAG07, cash07),
        }
    )


def _position(
    tag: StrategyTag = TAG07,
    qty: float = 0.2,
    entry: float = 45_000.0,
) -> Position:
    return Position(
        position_id=uuid4(),
        strategy_tag=tag,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        opened_at_utc=TS,
        quantity_btc=qty,
        entry_price_avg=entry,
        stop_loss_price=43_000.0,
    )


def _exit_order(qty: float = 0.2, price: float = 50_000.0) -> ExecutedOrder:
    order = Order(
        order_id=uuid4(),
        strategy_tag=TAG07,
        timestamp_utc=TS,
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity_btc=qty,
    )
    return ExecutedOrder(
        order=order,
        executed_at_utc=CLOSE_TS,
        executed_price=price,
        executed_quantity_btc=qty,
        fees_eur=2.0,
        slippage_eur=0.0,
        status=OrderStatus.FILLED,
    )


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


class TestPortfolioCreation:
    def test_portfolio_creation(self) -> None:
        pf = _portfolio()
        assert len(pf.pools) == 2
        assert pf.open_positions == {}
        assert pf.closed_trades == []

    def test_portfolio_empty_pools_is_valid(self) -> None:
        pf = Portfolio(pools={})
        assert pf.total_btc_held() == 0.0


# ---------------------------------------------------------------------------
# Global metrics
# ---------------------------------------------------------------------------


class TestPortfolioMetrics:
    def test_total_btc_held(self) -> None:
        pf = _portfolio()
        pf.pools[TAG06].add_btc(qty=0.3, cost_eur=13_500.0)
        pf.pools[TAG07].add_btc(qty=0.1, cost_eur=4_500.0)
        assert pf.total_btc_held() == pytest.approx(0.4)

    def test_total_btc_held_zero(self) -> None:
        pf = _portfolio()
        assert pf.total_btc_held() == 0.0

    def test_total_value_eur(self) -> None:
        pf = _portfolio(cash06=5_000.0, cash07=2_000.0)
        pf.pools[TAG06].add_btc(qty=0.2, cost_eur=9_000.0)
        # value = (5000 + 0.2 * 50000) + (2000 + 0 * 50000) = 15000 + 2000 = 17000
        assert pf.total_value_eur(50_000.0) == pytest.approx(17_000.0)

    def test_total_invested_eur(self) -> None:
        pf = _portfolio()
        pf.pools[TAG06].add_cash(1_000.0, source="extra")
        pf.pools[TAG07].add_cash(500.0, source="extra")
        # total_invested = 1000 (from add_cash) + 500 (from add_cash)
        # NB: initial cash_eur at pool creation does NOT set total_invested_eur
        assert pf.total_invested_eur() == pytest.approx(1_500.0)


# ---------------------------------------------------------------------------
# Pool access
# ---------------------------------------------------------------------------


class TestGetPool:
    def test_get_pool_existing(self) -> None:
        pf = _portfolio()
        pool = pf.get_pool(TAG06)
        assert pool.strategy_tag == TAG06
        assert pool.cash_eur == 10_000.0

    def test_get_pool_missing_raises(self) -> None:
        pf = _portfolio()
        with pytest.raises(ValueError, match="No pool registered"):
            pf.get_pool(StrategyTag.DCA_BENCHMARK)

    def test_get_pool_returns_same_object(self) -> None:
        pf = _portfolio()
        p1 = pf.get_pool(TAG06)
        p2 = pf.get_pool(TAG06)
        assert p1 is p2


# ---------------------------------------------------------------------------
# Position lifecycle
# ---------------------------------------------------------------------------


class TestOpenPosition:
    def test_open_position(self) -> None:
        pf = _portfolio()
        pos = _position()
        pf.open_position(pos)
        assert pos.position_id in pf.open_positions
        assert pf.open_positions[pos.position_id] is pos

    def test_open_position_duplicate_raises(self) -> None:
        pf = _portfolio()
        pos = _position()
        pf.open_position(pos)
        with pytest.raises(ValueError, match="already open"):
            pf.open_position(pos)

    def test_open_position_no_pool_raises(self) -> None:
        pf = Portfolio(pools={TAG06: _pool(TAG06)})
        pos = _position(tag=TAG07)  # TAG07 has no pool
        with pytest.raises(ValueError, match="No pool for strategy"):
            pf.open_position(pos)

    def test_open_multiple_positions(self) -> None:
        pf = _portfolio()
        p1, p2 = _position(), _position()
        pf.open_position(p1)
        pf.open_position(p2)
        assert len(pf.open_positions) == 2


class TestClosePosition:
    def test_close_position_creates_trade(self) -> None:
        pf = _portfolio()
        pos = _position()
        pf.open_position(pos)
        ex = _exit_order(qty=pos.quantity_btc, price=50_000.0)
        trade = pf.close_position(pos.position_id, [ex], exit_reason="tp1_hit")
        assert trade.outcome == TradeOutcome.WIN
        assert trade.exit_reason == "tp1_hit"

    def test_close_position_removes_from_open(self) -> None:
        pf = _portfolio()
        pos = _position()
        pf.open_position(pos)
        ex = _exit_order(qty=pos.quantity_btc)
        pf.close_position(pos.position_id, [ex], exit_reason="tp1_hit")
        assert pos.position_id not in pf.open_positions

    def test_close_position_adds_to_closed_trades(self) -> None:
        pf = _portfolio()
        pos = _position()
        pf.open_position(pos)
        ex = _exit_order(qty=pos.quantity_btc)
        trade = pf.close_position(pos.position_id, [ex], exit_reason="tp1_hit")
        assert len(pf.closed_trades) == 1
        assert pf.closed_trades[0] is trade

    def test_close_position_missing_raises(self) -> None:
        pf = _portfolio()
        with pytest.raises(ValueError, match="No open position"):
            pf.close_position(uuid4(), [_exit_order()], exit_reason="tp1_hit")

    def test_close_position_accumulates_trades(self) -> None:
        pf = _portfolio()
        for _ in range(3):
            pos = _position()
            pf.open_position(pos)
            ex = _exit_order(qty=pos.quantity_btc)
            pf.close_position(pos.position_id, [ex], exit_reason="tp1_hit")
        assert len(pf.closed_trades) == 3
        assert len(pf.open_positions) == 0


# ---------------------------------------------------------------------------
# Inter-pool transfer
# ---------------------------------------------------------------------------


class TestTransferCash:
    def test_transfer_cash_moves_funds(self) -> None:
        pf = _portfolio(cash06=10_000.0, cash07=5_000.0)
        pf.transfer_cash(TAG07, TAG06, amount=2_000.0, reason="profit_recycle")
        assert pf.pools[TAG07].cash_eur == pytest.approx(3_000.0)
        assert pf.pools[TAG06].cash_eur == pytest.approx(12_000.0)

    def test_transfer_does_not_inflate_total_invested(self) -> None:
        pf = _portfolio(cash06=10_000.0, cash07=5_000.0)
        invested_before = pf.total_invested_eur()
        pf.transfer_cash(TAG07, TAG06, amount=2_000.0, reason="recycle")
        assert pf.total_invested_eur() == pytest.approx(invested_before)

    def test_transfer_cash_overdraft_raises(self) -> None:
        pf = _portfolio(cash06=10_000.0, cash07=500.0)
        with pytest.raises(ValueError, match="Overdraft denied"):
            pf.transfer_cash(TAG07, TAG06, amount=1_000.0, reason="recycle")

    def test_transfer_cash_missing_source_raises(self) -> None:
        pf = _portfolio()
        with pytest.raises(ValueError, match="No pool registered"):
            pf.transfer_cash(
                StrategyTag.DCA_BENCHMARK, TAG06, amount=100.0, reason="x"
            )

    def test_transfer_cash_missing_dest_raises(self) -> None:
        pf = _portfolio()
        with pytest.raises(ValueError, match="No pool registered"):
            pf.transfer_cash(
                TAG07, StrategyTag.DCA_BENCHMARK, amount=100.0, reason="x"
            )


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestPortfolioSnapshot:
    def test_snapshot_captures_state(self) -> None:
        pf = _portfolio(cash06=8_000.0, cash07=3_000.0)
        pf.pools[TAG06].add_btc(qty=0.2, cost_eur=9_000.0)
        pos = _position()
        pf.open_position(pos)

        snap = pf.snapshot(TS, btc_price=45_000.0)

        assert snap.timestamp_utc == TS
        assert snap.btc_price == 45_000.0
        assert snap.total_btc == pytest.approx(0.2)
        # total_value = (8000 + 0.2*45000) + (3000 + 0) = 17000 + 3000 = 20000
        assert snap.total_value_eur == pytest.approx(20_000.0)
        assert snap.open_positions_count == 1
        assert snap.cash_by_pool[TAG06] == pytest.approx(8_000.0)
        assert snap.btc_by_pool[TAG06] == pytest.approx(0.2)
        assert snap.cash_by_pool[TAG07] == pytest.approx(3_000.0)

    def test_snapshot_immutable(self) -> None:
        pf = _portfolio()
        snap = pf.snapshot(TS, btc_price=45_000.0)
        with pytest.raises(FrozenInstanceError):
            snap.btc_price = 50_000.0  # type: ignore[misc]

    def test_snapshot_open_positions_count(self) -> None:
        pf = _portfolio()
        assert pf.snapshot(TS, 45_000.0).open_positions_count == 0
        pf.open_position(_position())
        assert pf.snapshot(TS, 45_000.0).open_positions_count == 1
