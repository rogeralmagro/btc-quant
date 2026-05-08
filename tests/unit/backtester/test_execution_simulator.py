"""Unit tests for ExecutionSimulator."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd
import pytest

from btc_quant.backtester.execution_simulator import ExecutionSimulator
from btc_quant.backtester.models.enums import (
    OrderSide,
    OrderStatus,
    OrderType,
    StrategyTag,
)
from btc_quant.backtester.models.order import Order
from btc_quant.backtester.models.position import Position

UTC = timezone.utc
TS = datetime(2024, 4, 1, tzinfo=UTC)
TS_NEXT = datetime(2024, 4, 2, tzinfo=UTC)

TAG = StrategyTag.STRAT_07_TACTICAL
FEE = 0.001       # 0.1%
SLIP = 0.0005     # 0.05%


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sim(fee: float = FEE, slip: float = SLIP) -> ExecutionSimulator:
    return ExecutionSimulator(fee_rate=fee, slippage_rate=slip)


def _market_order(
    side: OrderSide = OrderSide.BUY,
    qty: float = 1.0,
    ts: datetime = TS,
) -> Order:
    return Order(
        order_id=uuid4(),
        strategy_tag=TAG,
        timestamp_utc=ts,
        symbol="BTCUSDT",
        side=side,
        order_type=OrderType.MARKET,
        quantity_btc=qty,
    )


def _bar(
    open_: float = 45_000.0,
    high: float = 46_000.0,
    low: float = 44_000.0,
    close: float = 45_500.0,
    ts: datetime = TS_NEXT,
) -> pd.Series:
    return pd.Series(
        {
            "timestamp_utc": pd.Timestamp(ts),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": 100.0,
        }
    )


def _long_position(
    entry: float = 45_000.0,
    qty: float = 1.0,
    stop: float | None = 43_000.0,
) -> Position:
    return Position(
        position_id=uuid4(),
        strategy_tag=TAG,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        opened_at_utc=TS,
        quantity_btc=qty,
        entry_price_avg=entry,
        stop_loss_price=stop,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestExecutionSimulatorCreation:
    def test_default_rates(self) -> None:
        sim = ExecutionSimulator()
        assert sim._fee_rate == FEE
        assert sim._slippage_rate == SLIP

    def test_custom_rates(self) -> None:
        sim = ExecutionSimulator(fee_rate=0.002, slippage_rate=0.001)
        assert sim._fee_rate == 0.002

    def test_invalid_fee_rate_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="fee_rate"):
            ExecutionSimulator(fee_rate=0.02)

    def test_negative_fee_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="fee_rate"):
            ExecutionSimulator(fee_rate=-0.001)

    def test_negative_slippage_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="slippage_rate"):
            ExecutionSimulator(slippage_rate=-0.001)

    def test_slippage_rate_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="slippage_rate"):
            ExecutionSimulator(slippage_rate=0.05)

    def test_zero_rates_valid(self) -> None:
        sim = ExecutionSimulator(fee_rate=0.0, slippage_rate=0.0)
        order = _market_order()
        bar = _bar(open_=45_000.0)
        ex = sim.execute(order, bar)
        assert ex.executed_price == pytest.approx(45_000.0)
        assert ex.fees_eur == pytest.approx(0.0)
        assert ex.slippage_eur == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# execute() — MARKET orders
# ---------------------------------------------------------------------------


class TestExecuteMarket:
    def test_market_buy_executes_at_next_open_with_slippage(self) -> None:
        sim = _sim()
        order = _market_order(side=OrderSide.BUY, qty=1.0)
        bar = _bar(open_=45_000.0)
        ex = sim.execute(order, bar)

        expected_price = 45_000.0 * (1 + SLIP)  # 45_022.5
        assert ex.executed_price == pytest.approx(expected_price)
        assert ex.executed_quantity_btc == pytest.approx(1.0)
        assert ex.status == OrderStatus.FILLED

    def test_market_sell_executes_at_next_open_with_slippage(self) -> None:
        sim = _sim()
        order = _market_order(side=OrderSide.SELL, qty=1.0)
        bar = _bar(open_=45_000.0)
        ex = sim.execute(order, bar)

        expected_price = 45_000.0 * (1 - SLIP)  # 44_977.5
        assert ex.executed_price == pytest.approx(expected_price)

    def test_fees_calculated_correctly(self) -> None:
        sim = _sim(fee=FEE, slip=0.0)
        order = _market_order(side=OrderSide.BUY, qty=2.0)
        bar = _bar(open_=50_000.0)
        ex = sim.execute(order, bar)

        # notional = 50_000 * 2 = 100_000; fees = 100_000 * 0.001 = 100
        assert ex.fees_eur == pytest.approx(100.0)

    def test_buy_slippage_is_positive(self) -> None:
        """BUY executes above open: slippage_eur > 0."""
        sim = _sim()
        ex = sim.execute(_market_order(side=OrderSide.BUY, qty=1.0), _bar(open_=45_000.0))
        assert ex.slippage_eur > 0

    def test_sell_slippage_is_negative(self) -> None:
        """SELL executes below open: slippage_eur < 0."""
        sim = _sim()
        ex = sim.execute(_market_order(side=OrderSide.SELL, qty=1.0), _bar(open_=45_000.0))
        assert ex.slippage_eur < 0

    def test_executed_quantity_matches_order(self) -> None:
        sim = _sim()
        order = _market_order(qty=0.25)
        ex = sim.execute(order, _bar())
        assert ex.executed_quantity_btc == pytest.approx(0.25)

    def test_limit_order_raises_not_implemented(self) -> None:
        sim = _sim()
        order = Order(
            order_id=uuid4(),
            strategy_tag=TAG,
            timestamp_utc=TS,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity_btc=1.0,
            limit_price=44_000.0,
        )
        with pytest.raises(NotImplementedError, match="MARKET"):
            sim.execute(order, _bar())

    def test_stop_order_raises_not_implemented(self) -> None:
        sim = _sim()
        order = Order(
            order_id=uuid4(),
            strategy_tag=TAG,
            timestamp_utc=TS,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity_btc=1.0,
            stop_price=46_000.0,
        )
        with pytest.raises(NotImplementedError, match="MARKET"):
            sim.execute(order, _bar())


# ---------------------------------------------------------------------------
# execute_stop_or_tp()
# ---------------------------------------------------------------------------


class TestExecuteStopOrTp:
    # --- Stop-loss ---

    def test_stop_not_triggered_when_low_above_stop_level(self) -> None:
        sim = _sim()
        pos = _long_position(stop=43_000.0)
        bar = _bar(open_=45_000.0, high=46_000.0, low=44_000.0)
        result = sim.execute_stop_or_tp(pos, bar, "stop", level_price=43_000.0)
        assert result is None

    def test_stop_triggered_when_low_touches_stop(self) -> None:
        sim = _sim(slip=0.0, fee=0.0)
        pos = _long_position(stop=43_000.0)
        bar = _bar(open_=45_000.0, high=46_000.0, low=43_000.0)
        result = sim.execute_stop_or_tp(pos, bar, "stop", level_price=43_000.0)
        assert result is not None
        assert result.executed_price == pytest.approx(43_000.0)

    def test_stop_triggered_intra_bar(self) -> None:
        """Low crosses stop level intra-bar: fill at level, not at open."""
        sim = _sim(slip=0.0, fee=0.0)
        pos = _long_position(stop=43_000.0)
        # open=45000 (above stop), low=42000 (below stop) → intra-bar touch
        bar = _bar(open_=45_000.0, high=46_000.0, low=42_000.0)
        result = sim.execute_stop_or_tp(pos, bar, "stop", level_price=43_000.0)
        assert result is not None
        assert result.executed_price == pytest.approx(43_000.0)

    def test_stop_triggered_at_open_when_gap_below(self) -> None:
        """Gap open below stop: fill at open (worse for the holder)."""
        sim = _sim(slip=0.0, fee=0.0)
        pos = _long_position(stop=43_000.0)
        # Gap: open=42_000 < stop=43_000
        bar = _bar(open_=42_000.0, high=43_500.0, low=41_500.0)
        result = sim.execute_stop_or_tp(pos, bar, "stop", level_price=43_000.0)
        assert result is not None
        assert result.executed_price == pytest.approx(42_000.0)

    def test_stop_gap_slippage_is_negative(self) -> None:
        """Gap fill below level: slippage = (exec - level) * qty < 0."""
        sim = _sim(slip=0.0, fee=0.0)
        pos = _long_position(stop=43_000.0, qty=1.0)
        bar = _bar(open_=42_000.0, high=43_500.0, low=41_500.0)
        result = sim.execute_stop_or_tp(pos, bar, "stop", level_price=43_000.0)
        # slippage = (42000 - 43000) * 1.0 = -1000
        assert result.slippage_eur == pytest.approx(-1_000.0)

    # --- Take-profit ---

    def test_tp_not_triggered_when_high_below_tp_level(self) -> None:
        sim = _sim()
        pos = _long_position()
        bar = _bar(open_=45_000.0, high=47_000.0, low=44_000.0)
        result = sim.execute_stop_or_tp(pos, bar, "tp1", level_price=48_000.0)
        assert result is None

    def test_tp_triggered_when_high_touches_tp(self) -> None:
        sim = _sim(slip=0.0, fee=0.0)
        pos = _long_position()
        bar = _bar(open_=45_000.0, high=48_000.0, low=44_000.0)
        result = sim.execute_stop_or_tp(pos, bar, "tp1", level_price=48_000.0)
        assert result is not None
        assert result.executed_price == pytest.approx(48_000.0)

    def test_tp_triggered_intra_bar(self) -> None:
        """High crosses TP intra-bar: fill at level."""
        sim = _sim(slip=0.0, fee=0.0)
        pos = _long_position()
        bar = _bar(open_=45_000.0, high=50_000.0, low=44_000.0)
        result = sim.execute_stop_or_tp(pos, bar, "tp1", level_price=48_000.0)
        assert result is not None
        assert result.executed_price == pytest.approx(48_000.0)

    def test_tp_triggered_at_open_when_gap_above(self) -> None:
        """Gap open above TP: fill at open (favorable)."""
        sim = _sim(slip=0.0, fee=0.0)
        pos = _long_position()
        bar = _bar(open_=50_000.0, high=51_000.0, low=49_000.0)
        result = sim.execute_stop_or_tp(pos, bar, "tp1", level_price=48_000.0)
        assert result is not None
        assert result.executed_price == pytest.approx(50_000.0)

    def test_tp_gap_slippage_is_positive(self) -> None:
        """Gap fill above level: slippage = (exec - level) * qty > 0."""
        sim = _sim(slip=0.0, fee=0.0)
        pos = _long_position(qty=1.0)
        bar = _bar(open_=50_000.0, high=51_000.0, low=49_000.0)
        result = sim.execute_stop_or_tp(pos, bar, "tp1", level_price=48_000.0)
        # slippage = (50000 - 48000) * 1.0 = 2000
        assert result.slippage_eur == pytest.approx(2_000.0)

    def test_exact_level_hit_zero_slippage(self) -> None:
        sim = _sim(slip=0.0, fee=0.0)
        pos = _long_position()
        bar = _bar(open_=45_000.0, high=48_000.0, low=44_000.0)
        result = sim.execute_stop_or_tp(pos, bar, "tp1", level_price=48_000.0)
        assert result.slippage_eur == pytest.approx(0.0)

    def test_fees_applied_on_stop_tp(self) -> None:
        sim = _sim(slip=0.0, fee=FEE)
        pos = _long_position(qty=1.0)
        bar = _bar(open_=45_000.0, high=48_000.0, low=44_000.0)
        result = sim.execute_stop_or_tp(pos, bar, "tp1", level_price=48_000.0)
        # notional = 48_000 * 1 = 48_000; fees = 48_000 * 0.001 = 48
        assert result.fees_eur == pytest.approx(48.0)

    def test_notes_contain_label(self) -> None:
        sim = _sim()
        pos = _long_position()
        bar = _bar(open_=42_000.0, high=43_500.0, low=41_500.0)
        result = sim.execute_stop_or_tp(pos, bar, "stop", level_price=43_000.0)
        assert "stop" in result.notes

    def test_tp2_label_in_notes(self) -> None:
        sim = _sim(slip=0.0, fee=0.0)
        pos = _long_position()
        bar = _bar(open_=45_000.0, high=52_000.0, low=44_000.0)
        result = sim.execute_stop_or_tp(pos, bar, "tp2", level_price=50_000.0)
        assert "tp2" in result.notes
