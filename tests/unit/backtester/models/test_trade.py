"""Unit tests for Trade model and Trade.from_closed_position."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from btc_quant.backtester.models.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    StrategyTag,
    TradeOutcome,
)
from btc_quant.backtester.models.order import ExecutedOrder, Order
from btc_quant.backtester.models.position import Position, TakeProfitLevel
from btc_quant.backtester.models.trade import Trade, _BREAKEVEN_THRESHOLD_EUR

UTC = timezone.utc
OPEN_TS = datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC)
CLOSE_TS = datetime(2024, 6, 4, 12, 0, 0, tzinfo=UTC)  # 84 hours later

TAG = StrategyTag.STRAT_07_TACTICAL
ENTRY = 45_000.0
STOP = 43_500.0  # risk = 1 500 / BTC
QTY = 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _order(qty: float = QTY, ts: datetime = OPEN_TS) -> Order:
    return Order(
        order_id=uuid4(),
        strategy_tag=TAG,
        timestamp_utc=ts,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity_btc=qty,
    )


def _exec(
    order: Order,
    price: float,
    ts: datetime,
    qty: float | None = None,
    fees: float = 5.0,
) -> ExecutedOrder:
    return ExecutedOrder(
        order=order,
        executed_at_utc=ts,
        executed_price=price,
        executed_quantity_btc=qty if qty is not None else order.quantity_btc,
        fees_eur=fees,
        slippage_eur=0.0,
        status=OrderStatus.FILLED,
    )


def _position(
    qty: float = QTY,
    entry: float = ENTRY,
    stop: float | None = STOP,
    fees_paid: float = 5.0,
) -> Position:
    return Position(
        position_id=uuid4(),
        strategy_tag=TAG,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        opened_at_utc=OPEN_TS,
        quantity_btc=qty,
        entry_price_avg=entry,
        stop_loss_price=stop,
        fees_paid_eur=fees_paid,
    )


def _minimal_trade(**overrides) -> Trade:
    """Construct a Trade directly (not via classmethod)."""
    defaults = dict(
        trade_id=uuid4(),
        strategy_tag=TAG,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        opened_at_utc=OPEN_TS,
        entry_price_avg=ENTRY,
        quantity_btc=QTY,
        closed_at_utc=CLOSE_TS,
        exit_price_avg=50_000.0,
        gross_pnl_eur=2_500.0,
        fees_eur=10.0,
        net_pnl_eur=2_490.0,
        return_pct=5_000 / 45_000,
        outcome=TradeOutcome.WIN,
        exit_reason="tp1_hit",
        duration_hours=84.0,
    )
    defaults.update(overrides)
    return Trade(**defaults)


# ---------------------------------------------------------------------------
# Direct construction
# ---------------------------------------------------------------------------


class TestTradeCreation:
    def test_trade_winning(self) -> None:
        t = _minimal_trade()
        assert t.outcome == TradeOutcome.WIN
        assert t.net_pnl_eur == 2_490.0
        assert isinstance(t.trade_id, UUID)

    def test_trade_losing(self) -> None:
        t = _minimal_trade(
            exit_price_avg=43_000.0,
            gross_pnl_eur=-1_000.0,
            net_pnl_eur=-1_010.0,
            return_pct=(43_000 - 45_000) / 45_000,
            outcome=TradeOutcome.LOSS,
        )
        assert t.outcome == TradeOutcome.LOSS
        assert t.net_pnl_eur < 0

    def test_trade_defaults(self) -> None:
        t = _minimal_trade()
        assert t.initial_stop_loss is None
        assert t.initial_take_profits == []
        assert t.r_multiple_realized is None
        assert t.entry_orders == []
        assert t.exit_orders == []

    def test_trade_immutable(self) -> None:
        t = _minimal_trade()
        with pytest.raises(FrozenInstanceError):
            t.net_pnl_eur = 0.0  # type: ignore[misc]

    def test_trade_ids_are_unique(self) -> None:
        t1 = _minimal_trade()
        t2 = _minimal_trade()
        assert t1.trade_id != t2.trade_id


# ---------------------------------------------------------------------------
# from_closed_position — simple close (no partials)
# ---------------------------------------------------------------------------


class TestFromClosedPositionSimple:
    def test_trade_from_closed_position_winning(self) -> None:
        pos = _position(fees_paid=5.0)
        exit_ord = _order()
        ex = _exec(exit_ord, price=50_000.0, ts=CLOSE_TS, fees=5.0)

        t = Trade.from_closed_position(pos, [ex], exit_reason="tp1_hit")

        assert t.strategy_tag == TAG
        assert t.symbol == "BTCUSDT"
        assert t.entry_price_avg == ENTRY
        assert t.exit_price_avg == pytest.approx(50_000.0)
        assert t.quantity_btc == pytest.approx(QTY)
        # gross = (50000 - 45000) * 0.5 = 2 500
        assert t.gross_pnl_eur == pytest.approx(2_500.0)
        # fees = 5 (entry) + 5 (exit) = 10
        assert t.fees_eur == pytest.approx(10.0)
        assert t.net_pnl_eur == pytest.approx(2_490.0)
        assert t.outcome == TradeOutcome.WIN
        assert t.exit_reason == "tp1_hit"
        assert t.closed_at_utc == CLOSE_TS

    def test_trade_from_closed_position_losing(self) -> None:
        pos = _position(fees_paid=5.0)
        exit_ord = _order()
        ex = _exec(exit_ord, price=43_000.0, ts=CLOSE_TS, fees=5.0)

        t = Trade.from_closed_position(pos, [ex], exit_reason="stop_loss_hit")

        # gross = (43000 - 45000) * 0.5 = -1 000
        assert t.gross_pnl_eur == pytest.approx(-1_000.0)
        assert t.net_pnl_eur == pytest.approx(-1_010.0)
        assert t.outcome == TradeOutcome.LOSS
        assert t.exit_reason == "stop_loss_hit"

    def test_trade_breakeven_within_tolerance(self) -> None:
        """net_pnl within ±_BREAKEVEN_THRESHOLD_EUR is classified as BREAKEVEN."""
        pos = _position(fees_paid=0.0)
        exit_ord = _order()
        # Price exactly at entry → gross = 0, fees = 0, net = 0 → BREAKEVEN
        ex = _exec(exit_ord, price=ENTRY, ts=CLOSE_TS, fees=0.0)
        t = Trade.from_closed_position(pos, [ex], exit_reason="manual_close")
        assert t.outcome == TradeOutcome.BREAKEVEN
        assert abs(t.net_pnl_eur) < _BREAKEVEN_THRESHOLD_EUR

    def test_trade_duration_hours(self) -> None:
        pos = _position()
        exit_ord = _order()
        ex = _exec(exit_ord, price=50_000.0, ts=CLOSE_TS)
        t = Trade.from_closed_position(pos, [ex], exit_reason="tp1_hit")
        # OPEN_TS = 2024-06-01 00:00, CLOSE_TS = 2024-06-04 12:00 → 84h
        assert t.duration_hours == pytest.approx(84.0)

    def test_trade_return_pct(self) -> None:
        pos = _position()
        exit_ord = _order()
        ex = _exec(exit_ord, price=50_000.0, ts=CLOSE_TS, fees=0.0)
        t = Trade.from_closed_position(pos, [ex], exit_reason="tp1_hit")
        assert t.return_pct == pytest.approx(5_000 / 45_000)

    def test_trade_has_auto_uuid(self) -> None:
        pos = _position()
        ex = _exec(_order(), price=50_000.0, ts=CLOSE_TS)
        t = Trade.from_closed_position(pos, [ex], exit_reason="tp1_hit")
        assert isinstance(t.trade_id, UUID)

    def test_trade_ids_unique_across_calls(self) -> None:
        pos = _position()
        ex1 = _exec(_order(), price=50_000.0, ts=CLOSE_TS)
        ex2 = _exec(_order(), price=50_000.0, ts=CLOSE_TS)
        t1 = Trade.from_closed_position(pos, [ex1], exit_reason="tp1_hit")
        t2 = Trade.from_closed_position(pos, [ex2], exit_reason="tp1_hit")
        assert t1.trade_id != t2.trade_id

    def test_trade_stop_loss_recorded(self) -> None:
        pos = _position(stop=STOP)
        ex = _exec(_order(), price=50_000.0, ts=CLOSE_TS)
        t = Trade.from_closed_position(pos, [ex], exit_reason="tp1_hit")
        assert t.initial_stop_loss == STOP

    def test_trade_no_stop_loss(self) -> None:
        pos = _position(stop=None)
        ex = _exec(_order(), price=50_000.0, ts=CLOSE_TS)
        t = Trade.from_closed_position(pos, [ex], exit_reason="dca_close")
        assert t.initial_stop_loss is None
        assert t.r_multiple_realized is None

    def test_trade_tp_prices_recorded(self) -> None:
        pos = _position()
        pos.take_profit_levels = [
            TakeProfitLevel(price=48_000.0, quantity_pct=0.5),
            TakeProfitLevel(price=52_000.0, quantity_pct=0.5),
        ]
        ex = _exec(_order(), price=50_000.0, ts=CLOSE_TS)
        t = Trade.from_closed_position(pos, [ex], exit_reason="tp2_hit")
        assert t.initial_take_profits == [48_000.0, 52_000.0]

    def test_trade_empty_exit_orders_raises(self) -> None:
        pos = _position()
        with pytest.raises(ValueError, match="exit_orders must not be empty"):
            Trade.from_closed_position(pos, [], exit_reason="tp1_hit")


# ---------------------------------------------------------------------------
# from_closed_position — with partial exits
# ---------------------------------------------------------------------------


class TestFromClosedPositionWithPartials:
    def test_weighted_exit_price_with_partials(self) -> None:
        """TP1 closes 50% at 50 000, final close closes 50% at 48 000.
        Weighted exit = (0.25*50000 + 0.25*48000) / 0.5 = 49 000.
        gross = (49000 - 45000) * 0.5 = 2 000.
        """
        pos = _position(qty=0.5, fees_paid=5.0)

        # Partial exit (already in position.partial_exit_orders)
        partial_ord = _order(qty=0.25)
        partial_ex = _exec(partial_ord, price=50_000.0, ts=OPEN_TS + timedelta(days=2), fees=2.5)
        pos.partial_exit_orders.append(partial_ex)
        pos.quantity_btc = 0.25  # simulate position after partial

        # Final close
        final_ord = _order(qty=0.25)
        final_ex = _exec(final_ord, price=48_000.0, ts=CLOSE_TS, fees=2.5)

        t = Trade.from_closed_position(pos, [final_ex], exit_reason="regime_flip")

        assert t.quantity_btc == pytest.approx(0.5)
        assert t.exit_price_avg == pytest.approx(49_000.0)
        assert t.gross_pnl_eur == pytest.approx(2_000.0)
        # fees = 5 (entry) + 2.5 (partial) + 2.5 (final) = 10
        assert t.fees_eur == pytest.approx(10.0)
        assert t.net_pnl_eur == pytest.approx(1_990.0)
        assert t.outcome == TradeOutcome.WIN

    def test_closed_at_is_latest_exit(self) -> None:
        pos = _position(qty=0.5)
        ts_partial = OPEN_TS + timedelta(days=2)
        partial_ex = _exec(_order(qty=0.25), price=50_000.0, ts=ts_partial)
        pos.partial_exit_orders.append(partial_ex)
        pos.quantity_btc = 0.25

        final_ex = _exec(_order(qty=0.25), price=48_000.0, ts=CLOSE_TS)
        t = Trade.from_closed_position(pos, [final_ex], exit_reason="regime_flip")

        assert t.closed_at_utc == CLOSE_TS  # CLOSE_TS > ts_partial


# ---------------------------------------------------------------------------
# R-multiple
# ---------------------------------------------------------------------------


class TestTradeRMultiple:
    def test_trade_r_multiple_realized_profit(self) -> None:
        """entry=45000, stop=43500, exit=48000 → R = 3000/1500 = 2.0"""
        pos = _position(stop=STOP)
        ex = _exec(_order(), price=48_000.0, ts=CLOSE_TS, fees=0.0)
        t = Trade.from_closed_position(pos, [ex], exit_reason="tp2_hit")
        assert t.r_multiple_realized == pytest.approx(2.0)

    def test_trade_r_multiple_realized_loss(self) -> None:
        """exit=44250, loss=-750, R=-750/1500=-0.5"""
        pos = _position(stop=STOP)
        ex = _exec(_order(), price=44_250.0, ts=CLOSE_TS, fees=0.0)
        t = Trade.from_closed_position(pos, [ex], exit_reason="stop_loss_hit")
        assert t.r_multiple_realized == pytest.approx(-0.5)

    def test_trade_r_multiple_none_without_stop(self) -> None:
        pos = _position(stop=None)
        ex = _exec(_order(), price=50_000.0, ts=CLOSE_TS)
        t = Trade.from_closed_position(pos, [ex], exit_reason="manual")
        assert t.r_multiple_realized is None
