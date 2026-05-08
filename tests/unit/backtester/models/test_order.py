"""Unit tests for Order and ExecutedOrder models."""

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
)
from btc_quant.backtester.models.order import ExecutedOrder, Order, make_order

UTC = timezone.utc
TS = datetime(2024, 3, 1, 9, 0, 0, tzinfo=UTC)
TS_LATER = datetime(2024, 3, 1, 9, 5, 0, tzinfo=UTC)


def _market_order(**overrides) -> Order:
    defaults = dict(
        order_id=uuid4(),
        strategy_tag=StrategyTag.STRAT_07_TACTICAL,
        timestamp_utc=TS,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity_btc=0.1,
    )
    defaults.update(overrides)
    return Order(**defaults)


def _executed(order: Order | None = None, **overrides) -> ExecutedOrder:
    if order is None:
        order = _market_order()
    defaults = dict(
        order=order,
        executed_at_utc=TS_LATER,
        executed_price=45_000.0,
        executed_quantity_btc=order.quantity_btc,
        fees_eur=2.0,
        slippage_eur=0.5,
        status=OrderStatus.FILLED,
    )
    defaults.update(overrides)
    return ExecutedOrder(**defaults)


# ---------------------------------------------------------------------------
# Order
# ---------------------------------------------------------------------------


class TestOrderCreation:
    def test_market_order_creation(self) -> None:
        o = _market_order()
        assert isinstance(o.order_id, UUID)
        assert o.order_type == OrderType.MARKET
        assert o.quantity_btc == 0.1
        assert o.limit_price is None
        assert o.stop_price is None

    def test_limit_order_creation(self) -> None:
        o = _market_order(order_type=OrderType.LIMIT, limit_price=44_000.0)
        assert o.limit_price == 44_000.0

    def test_stop_order_creation(self) -> None:
        o = _market_order(order_type=OrderType.STOP, stop_price=40_000.0)
        assert o.stop_price == 40_000.0

    def test_stop_limit_order_creation(self) -> None:
        o = _market_order(
            order_type=OrderType.STOP_LIMIT,
            stop_price=40_000.0,
            limit_price=39_500.0,
        )
        assert o.stop_price == 40_000.0
        assert o.limit_price == 39_500.0

    def test_make_order_autogenerates_id(self) -> None:
        o = make_order(
            strategy_tag=StrategyTag.STRAT_07_TACTICAL,
            timestamp_utc=TS,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity_btc=0.05,
        )
        assert isinstance(o.order_id, UUID)

    def test_order_ids_are_unique(self) -> None:
        o1 = _market_order()
        o2 = _market_order()
        assert o1.order_id != o2.order_id

    def test_order_parent_signal_id_optional(self) -> None:
        sig_id = uuid4()
        o = _market_order(parent_signal_id=sig_id)
        assert o.parent_signal_id == sig_id

    def test_order_metadata_default_empty(self) -> None:
        o = _market_order()
        assert o.metadata == {}


class TestOrderValidation:
    def test_naive_datetime_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            _market_order(timestamp_utc=datetime(2024, 3, 1, 9, 0, 0))

    def test_non_utc_offset_raises(self) -> None:
        ts = datetime(2024, 3, 1, 9, 0, 0, tzinfo=timezone(timedelta(hours=2)))
        with pytest.raises(ValueError, match="UTC"):
            _market_order(timestamp_utc=ts)

    def test_zero_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity_btc must be > 0"):
            _market_order(quantity_btc=0.0)

    def test_negative_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity_btc must be > 0"):
            _market_order(quantity_btc=-0.5)

    def test_market_with_limit_price_raises(self) -> None:
        with pytest.raises(ValueError, match="MARKET order must not have limit_price"):
            _market_order(limit_price=44_000.0)

    def test_market_with_stop_price_raises(self) -> None:
        with pytest.raises(ValueError, match="MARKET order must not have stop_price"):
            _market_order(stop_price=40_000.0)

    def test_limit_without_price_raises(self) -> None:
        with pytest.raises(ValueError, match="LIMIT order requires limit_price"):
            _market_order(order_type=OrderType.LIMIT)

    def test_limit_with_zero_price_raises(self) -> None:
        with pytest.raises(ValueError, match="LIMIT order requires limit_price"):
            _market_order(order_type=OrderType.LIMIT, limit_price=0.0)

    def test_stop_without_stop_price_raises(self) -> None:
        with pytest.raises(ValueError, match="STOP order requires stop_price"):
            _market_order(order_type=OrderType.STOP)

    def test_stop_limit_without_stop_price_raises(self) -> None:
        with pytest.raises(ValueError, match="STOP_LIMIT order requires stop_price"):
            _market_order(order_type=OrderType.STOP_LIMIT, limit_price=39_500.0)

    def test_stop_limit_without_limit_price_raises(self) -> None:
        with pytest.raises(ValueError, match="STOP_LIMIT order requires limit_price"):
            _market_order(order_type=OrderType.STOP_LIMIT, stop_price=40_000.0)


class TestOrderImmutable:
    def test_order_immutable(self) -> None:
        o = _market_order()
        with pytest.raises(FrozenInstanceError):
            o.quantity_btc = 1.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExecutedOrder
# ---------------------------------------------------------------------------


class TestExecutedOrderCreation:
    def test_executed_order_filled(self) -> None:
        o = _market_order()
        ex = _executed(o)
        assert ex.order is o
        assert ex.executed_price == 45_000.0
        assert ex.executed_quantity_btc == 0.1
        assert ex.fees_eur == 2.0
        assert ex.slippage_eur == 0.5
        assert ex.status == OrderStatus.FILLED
        assert ex.notes == ""

    def test_executed_order_partially_filled(self) -> None:
        o = _market_order(quantity_btc=1.0)
        ex = _executed(o, executed_quantity_btc=0.6, status=OrderStatus.PARTIALLY_FILLED)
        assert ex.status == OrderStatus.PARTIALLY_FILLED
        assert ex.executed_quantity_btc == 0.6

    def test_executed_at_same_time_as_order_is_valid(self) -> None:
        o = _market_order()
        ex = _executed(o, executed_at_utc=TS)
        assert ex.executed_at_utc == TS

    def test_zero_slippage_is_valid(self) -> None:
        ex = _executed(slippage_eur=0.0)
        assert ex.slippage_eur == 0.0

    def test_negative_slippage_is_valid(self) -> None:
        """Favorable slippage (e.g. gap open in our favour) can be negative."""
        ex = _executed(slippage_eur=-1.5)
        assert ex.slippage_eur == -1.5

    def test_zero_fees_is_valid(self) -> None:
        ex = _executed(fees_eur=0.0)
        assert ex.fees_eur == 0.0

    def test_notes_field(self) -> None:
        ex = _executed(notes="executed at gap open")
        assert ex.notes == "executed at gap open"


class TestExecutedOrderValidation:
    def test_executed_before_order_raises(self) -> None:
        o = _market_order()
        before_ts = datetime(2024, 3, 1, 8, 59, 0, tzinfo=UTC)
        with pytest.raises(ValueError, match="executed_at_utc"):
            _executed(o, executed_at_utc=before_ts)

    def test_naive_executed_at_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            _executed(executed_at_utc=datetime(2024, 3, 1, 9, 5, 0))

    def test_zero_executed_price_raises(self) -> None:
        with pytest.raises(ValueError, match="executed_price must be > 0"):
            _executed(executed_price=0.0)

    def test_negative_executed_price_raises(self) -> None:
        with pytest.raises(ValueError, match="executed_price must be > 0"):
            _executed(executed_price=-1.0)

    def test_zero_executed_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="executed_quantity_btc"):
            _executed(executed_quantity_btc=0.0)

    def test_executed_quantity_exceeds_order_raises(self) -> None:
        o = _market_order(quantity_btc=0.1)
        with pytest.raises(ValueError, match="executed_quantity_btc"):
            _executed(o, executed_quantity_btc=0.2)

    def test_negative_fees_raises(self) -> None:
        with pytest.raises(ValueError, match="fees_eur must be >= 0"):
            _executed(fees_eur=-0.01)

    def test_status_pending_raises(self) -> None:
        with pytest.raises(ValueError, match="FILLED or PARTIALLY_FILLED"):
            _executed(status=OrderStatus.PENDING)

    def test_status_cancelled_raises(self) -> None:
        with pytest.raises(ValueError, match="FILLED or PARTIALLY_FILLED"):
            _executed(status=OrderStatus.CANCELLED)

    def test_status_rejected_raises(self) -> None:
        with pytest.raises(ValueError, match="FILLED or PARTIALLY_FILLED"):
            _executed(status=OrderStatus.REJECTED)


class TestExecutedOrderImmutable:
    def test_executed_order_immutable(self) -> None:
        ex = _executed()
        with pytest.raises(FrozenInstanceError):
            ex.executed_price = 50_000.0  # type: ignore[misc]
