"""Unit tests for Position and TakeProfitLevel models."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from btc_quant.backtester.models.enums import OrderSide, StrategyTag
from btc_quant.backtester.models.position import Position, TakeProfitLevel

UTC = timezone.utc
OPEN_TS = datetime(2024, 6, 1, 0, 0, 0, tzinfo=UTC)
LATER_TS = datetime(2024, 6, 2, 0, 0, 0, tzinfo=UTC)

ENTRY = 45_000.0
STOP = 43_500.0   # risk = 1 500 EUR/BTC
QTY = 0.2         # 0.2 BTC


def _position(**overrides) -> Position:
    defaults = dict(
        position_id=uuid4(),
        strategy_tag=StrategyTag.STRAT_07_TACTICAL,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        opened_at_utc=OPEN_TS,
        quantity_btc=QTY,
        entry_price_avg=ENTRY,
        stop_loss_price=STOP,
    )
    defaults.update(overrides)
    return Position(**defaults)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


class TestPositionCreation:
    def test_position_creation(self) -> None:
        p = _position()
        assert p.quantity_btc == QTY
        assert p.entry_price_avg == ENTRY
        assert p.stop_loss_price == STOP
        assert p.side == OrderSide.BUY
        assert p.fees_paid_eur == 0.0
        assert p.take_profit_levels == []
        assert p.entry_orders == []
        assert p.partial_exit_orders == []
        assert p.metadata == {}

    def test_position_without_stop_is_valid(self) -> None:
        p = _position(stop_loss_price=None)
        assert p.stop_loss_price is None

    def test_position_with_tp_levels(self) -> None:
        tps = [
            TakeProfitLevel(price=48_000.0, quantity_pct=0.5),
            TakeProfitLevel(price=52_000.0, quantity_pct=0.5),
        ]
        p = _position(take_profit_levels=tps)
        assert len(p.take_profit_levels) == 2
        assert p.take_profit_levels[0].price == 48_000.0


class TestPositionValidation:
    def test_zero_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity_btc must be > 0"):
            _position(quantity_btc=0.0)

    def test_negative_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity_btc must be > 0"):
            _position(quantity_btc=-0.1)

    def test_zero_entry_price_raises(self) -> None:
        with pytest.raises(ValueError, match="entry_price_avg must be > 0"):
            _position(entry_price_avg=0.0)

    def test_negative_entry_price_raises(self) -> None:
        with pytest.raises(ValueError, match="entry_price_avg must be > 0"):
            _position(entry_price_avg=-1.0)

    def test_position_validates_long_only(self) -> None:
        with pytest.raises(ValueError, match="long-only"):
            _position(side=OrderSide.SELL)


# ---------------------------------------------------------------------------
# P&L metrics
# ---------------------------------------------------------------------------


class TestUnrealizedPnL:
    def test_unrealized_pnl_long_in_profit(self) -> None:
        p = _position()
        # current = 48 000, gain = 3 000/BTC, qty = 0.2 → 600 EUR
        assert p.unrealized_pnl_eur(48_000.0) == pytest.approx(600.0)

    def test_unrealized_pnl_long_in_loss(self) -> None:
        p = _position()
        # current = 44 250, loss = -750/BTC, qty = 0.2 → -150 EUR
        assert p.unrealized_pnl_eur(44_250.0) == pytest.approx(-150.0)

    def test_unrealized_pnl_at_entry_is_zero(self) -> None:
        p = _position()
        assert p.unrealized_pnl_eur(ENTRY) == pytest.approx(0.0)

    def test_unrealized_pnl_pct_in_profit(self) -> None:
        p = _position()
        # (48 000 - 45 000) / 45 000 ≈ 0.06667
        assert p.unrealized_pnl_pct(48_000.0) == pytest.approx(3_000 / 45_000)

    def test_unrealized_pnl_pct_in_loss(self) -> None:
        p = _position()
        assert p.unrealized_pnl_pct(44_250.0) == pytest.approx(-750 / 45_000)


class TestRMultiple:
    def test_r_multiple_returns_none_without_stop(self) -> None:
        p = _position(stop_loss_price=None)
        assert p.unrealized_r_multiple(48_000.0) is None

    def test_r_multiple_calculation_profit(self) -> None:
        p = _position()
        # risk = 45 000 - 43 500 = 1 500; gain = 48 000 - 45 000 = 3 000
        # R = 3 000 / 1 500 = 2.0
        assert p.unrealized_r_multiple(48_000.0) == pytest.approx(2.0)

    def test_r_multiple_calculation_loss(self) -> None:
        p = _position()
        # loss = 44 250 - 45 000 = -750; R = -750 / 1 500 = -0.5
        assert p.unrealized_r_multiple(44_250.0) == pytest.approx(-0.5)

    def test_r_multiple_at_stop_is_minus_one(self) -> None:
        p = _position()
        # at stop: (43 500 - 45 000) / 1 500 = -1.0
        assert p.unrealized_r_multiple(STOP) == pytest.approx(-1.0)

    def test_r_multiple_at_entry_is_zero(self) -> None:
        p = _position()
        assert p.unrealized_r_multiple(ENTRY) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# update_stop_loss
# ---------------------------------------------------------------------------


class TestUpdateStopLoss:
    def test_update_stop_loss_on_position_without_stop_raises(self) -> None:
        p = _position(stop_loss_price=None)
        with pytest.raises(ValueError, match="without initial stop"):
            p.update_stop_loss(44_000.0)

    def test_update_stop_loss_loosening_raises(self) -> None:
        p = _position()  # stop = 43 500
        with pytest.raises(ValueError, match="Cannot loosen stop_loss"):
            p.update_stop_loss(43_000.0)

    def test_update_stop_loss_same_price_raises(self) -> None:
        p = _position()
        with pytest.raises(ValueError, match="Cannot loosen stop_loss"):
            p.update_stop_loss(STOP)

    def test_update_stop_loss_zero_raises(self) -> None:
        p = _position()
        with pytest.raises(ValueError, match="positive"):
            p.update_stop_loss(0.0)

    def test_update_stop_loss_negative_raises(self) -> None:
        p = _position()
        with pytest.raises(ValueError, match="positive"):
            p.update_stop_loss(-100.0)

    def test_update_stop_loss_tightening_succeeds(self) -> None:
        p = _position()  # stop = 43 500
        p.update_stop_loss(44_000.0)
        assert p.stop_loss_price == 44_000.0

    def test_update_stop_loss_only_tightens(self) -> None:
        """Multiple tightening steps; each must be strictly higher."""
        p = _position()  # stop = 43 500
        p.update_stop_loss(44_000.0)
        p.update_stop_loss(44_500.0)
        assert p.stop_loss_price == 44_500.0
        with pytest.raises(ValueError, match="Cannot loosen"):
            p.update_stop_loss(44_000.0)

    def test_update_stop_loss_to_breakeven(self) -> None:
        """Moving stop to entry price (breakeven) is valid tightening."""
        p = _position()  # entry = 45 000, stop = 43 500
        p.update_stop_loss(ENTRY)  # 45 000 > 43 500 → valid
        assert p.stop_loss_price == ENTRY


# ---------------------------------------------------------------------------
# mark_tp_executed
# ---------------------------------------------------------------------------


class TestMarkTpExecuted:
    def _position_with_tps(self) -> Position:
        tps = [
            TakeProfitLevel(price=48_000.0, quantity_pct=0.5),
            TakeProfitLevel(price=52_000.0, quantity_pct=0.5),
        ]
        return _position(take_profit_levels=tps)

    def test_mark_tp_executed_sets_fields(self) -> None:
        p = self._position_with_tps()
        p.mark_tp_executed(0, LATER_TS, qty=0.1)
        lvl = p.take_profit_levels[0]
        assert lvl.executed is True
        assert lvl.executed_at_utc == LATER_TS
        assert lvl.executed_quantity_btc == 0.1

    def test_mark_second_tp_independently(self) -> None:
        p = self._position_with_tps()
        p.mark_tp_executed(1, LATER_TS, qty=0.1)
        assert p.take_profit_levels[0].executed is False
        assert p.take_profit_levels[1].executed is True

    def test_mark_tp_executed_validates_index_too_high(self) -> None:
        p = self._position_with_tps()
        with pytest.raises(ValueError, match="out of range"):
            p.mark_tp_executed(2, LATER_TS, qty=0.1)

    def test_mark_tp_executed_validates_index_negative(self) -> None:
        p = self._position_with_tps()
        with pytest.raises(ValueError, match="out of range"):
            p.mark_tp_executed(-1, LATER_TS, qty=0.1)

    def test_mark_tp_executed_no_levels_raises(self) -> None:
        p = _position()  # no TP levels
        with pytest.raises(ValueError, match="no take_profit_levels"):
            p.mark_tp_executed(0, LATER_TS, qty=0.1)

    def test_mark_tp_executed_twice_raises(self) -> None:
        p = self._position_with_tps()
        p.mark_tp_executed(0, LATER_TS, qty=0.1)
        with pytest.raises(ValueError, match="already executed"):
            p.mark_tp_executed(0, LATER_TS, qty=0.1)


# ---------------------------------------------------------------------------
# reduce_quantity
# ---------------------------------------------------------------------------


class TestReduceQuantity:
    def test_reduce_quantity_updates_btc(self) -> None:
        p = _position()  # qty = 0.2
        p.reduce_quantity(0.1, LATER_TS)
        assert p.quantity_btc == pytest.approx(0.1)

    def test_reduce_quantity_full_close(self) -> None:
        p = _position()
        p.reduce_quantity(QTY, LATER_TS)
        assert p.quantity_btc == pytest.approx(0.0)

    def test_reduce_quantity_validates_amount_zero(self) -> None:
        p = _position()
        with pytest.raises(ValueError, match="amount must be > 0"):
            p.reduce_quantity(0.0, LATER_TS)

    def test_reduce_quantity_validates_amount_negative(self) -> None:
        p = _position()
        with pytest.raises(ValueError, match="amount must be > 0"):
            p.reduce_quantity(-0.05, LATER_TS)

    def test_reduce_quantity_validates_amount(self) -> None:
        p = _position()  # qty = 0.2
        with pytest.raises(ValueError, match="only holds"):
            p.reduce_quantity(0.3, LATER_TS)

    def test_reduce_quantity_sequential(self) -> None:
        p = _position()  # qty = 0.2
        p.reduce_quantity(0.05, LATER_TS)
        p.reduce_quantity(0.05, LATER_TS)
        assert p.quantity_btc == pytest.approx(0.1)
