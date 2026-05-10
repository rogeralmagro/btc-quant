"""Unit tests for BacktestResult and CircuitBreakerEvent."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from btc_quant.backtester.models.enums import (
    CircuitBreakerState,
    OrderSide,
    OrderStatus,
    OrderType,
    StrategyTag,
    TradeOutcome,
)
from btc_quant.backtester.models.order import ExecutedOrder, Order
from btc_quant.backtester.models.pool import CapitalPool
from btc_quant.backtester.models.portfolio import Portfolio, PortfolioSnapshot
from btc_quant.backtester.models.result import BacktestResult, CircuitBreakerEvent
from btc_quant.backtester.models.trade import Trade

UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)
END = datetime(2024, 12, 31, tzinfo=UTC)
MID = datetime(2024, 6, 15, tzinfo=UTC)
CLOSE = datetime(2024, 6, 20, tzinfo=UTC)

TAG06 = StrategyTag.STRAT_06_BASELINE
TAG07 = StrategyTag.STRAT_07_TACTICAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _portfolio() -> Portfolio:
    return Portfolio(
        pools={
            TAG06: CapitalPool(strategy_tag=TAG06, cash_eur=8_000.0),
            TAG07: CapitalPool(strategy_tag=TAG07, cash_eur=3_000.0),
        }
    )


def _snapshot(ts: datetime = START) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        timestamp_utc=ts,
        btc_price=45_000.0,
        total_value_eur=10_000.0,
        total_btc=0.0,
        cash_by_pool={TAG06: 7_000.0, TAG07: 3_000.0},
        btc_by_pool={TAG06: 0.0, TAG07: 0.0},
        open_positions_count=0,
    )


def _trade(tag: StrategyTag = TAG07, pnl: float = 500.0) -> Trade:
    order = Order(
        order_id=uuid4(),
        strategy_tag=tag,
        timestamp_utc=MID,
        symbol="BTCUSDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity_btc=0.1,
    )
    ex = ExecutedOrder(
        order=order,
        executed_at_utc=CLOSE,
        executed_price=50_000.0,
        executed_quantity_btc=0.1,
        fees_eur=2.0,
        slippage_eur=0.0,
        status=OrderStatus.FILLED,
    )
    return Trade(
        trade_id=uuid4(),
        strategy_tag=tag,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        opened_at_utc=MID,
        entry_price_avg=45_000.0,
        quantity_btc=0.1,
        closed_at_utc=CLOSE,
        exit_price_avg=50_000.0,
        gross_pnl_eur=500.0,
        fees_eur=4.0,
        net_pnl_eur=pnl,
        return_pct=500 / 45_000,
        outcome=TradeOutcome.WIN if pnl > 0 else TradeOutcome.LOSS,
        exit_reason="tp1_hit",
        duration_hours=120.0,
        exit_orders=[ex],
    )


def _result(
    trades: list[Trade] | None = None,
    trades_by_strategy: dict[StrategyTag, list[Trade]] | None = None,
    **overrides,
) -> BacktestResult:
    trades = trades or []
    trades_by_strategy = trades_by_strategy or {TAG06: [], TAG07: trades}
    defaults = dict(
        start_date_utc=START,
        end_date_utc=END,
        initial_capital_eur=10_000.0,
        strategies_used=[TAG06, TAG07],
        final_portfolio=_portfolio(),
        equity_curve=[_snapshot(START), _snapshot(END)],
        closed_trades=trades,
        trades_by_strategy=trades_by_strategy,
    )
    defaults.update(overrides)
    return BacktestResult(**defaults)


# ---------------------------------------------------------------------------
# CircuitBreakerEvent
# ---------------------------------------------------------------------------


class TestCircuitBreakerEvent:
    def test_circuit_breaker_event_creation(self) -> None:
        ev = CircuitBreakerEvent(
            timestamp_utc=MID,
            trigger_name="max_drawdown_exceeded",
            state_before=CircuitBreakerState.NORMAL,
            state_after=CircuitBreakerState.PAUSED,
            description="Portfolio drawdown exceeded 20%",
        )
        assert ev.trigger_name == "max_drawdown_exceeded"
        assert ev.state_before == CircuitBreakerState.NORMAL
        assert ev.state_after == CircuitBreakerState.PAUSED

    def test_circuit_breaker_event_immutable(self) -> None:
        ev = CircuitBreakerEvent(
            timestamp_utc=MID,
            trigger_name="test",
            state_before=CircuitBreakerState.NORMAL,
            state_after=CircuitBreakerState.HALTED,
            description="test",
        )
        with pytest.raises(FrozenInstanceError):
            ev.trigger_name = "other"  # type: ignore[misc]

    def test_circuit_breaker_naive_datetime_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            CircuitBreakerEvent(
                timestamp_utc=datetime(2024, 6, 15),
                trigger_name="test",
                state_before=CircuitBreakerState.NORMAL,
                state_after=CircuitBreakerState.PAUSED,
                description="test",
            )

    def test_circuit_breaker_non_utc_raises(self) -> None:
        ts = datetime(2024, 6, 15, tzinfo=timezone(timedelta(hours=1)))
        with pytest.raises(ValueError, match="UTC"):
            CircuitBreakerEvent(
                timestamp_utc=ts,
                trigger_name="test",
                state_before=CircuitBreakerState.NORMAL,
                state_after=CircuitBreakerState.PAUSED,
                description="test",
            )


# ---------------------------------------------------------------------------
# BacktestResult — creation & validation
# ---------------------------------------------------------------------------


class TestBacktestResultCreation:
    def test_backtest_result_creation(self) -> None:
        r = _result()
        assert r.start_date_utc == START
        assert r.end_date_utc == END
        assert r.initial_capital_eur == 10_000.0
        assert r.closed_trades == []
        assert r.circuit_breaker_events == []
        assert r.backtest_metadata == {}

    def test_backtest_result_with_trades(self) -> None:
        t1 = _trade(TAG07, pnl=500.0)
        t2 = _trade(TAG07, pnl=-200.0)
        r = _result(
            trades=[t1, t2],
            trades_by_strategy={TAG06: [], TAG07: [t1, t2]},
        )
        assert len(r.closed_trades) == 2

    def test_backtest_result_with_circuit_breaker_events(self) -> None:
        ev = CircuitBreakerEvent(
            timestamp_utc=MID,
            trigger_name="dd_breach",
            state_before=CircuitBreakerState.NORMAL,
            state_after=CircuitBreakerState.PAUSED,
            description="20% drawdown",
        )
        r = _result(circuit_breaker_events=[ev])
        assert len(r.circuit_breaker_events) == 1

    def test_backtest_result_with_metadata(self) -> None:
        r = _result(backtest_metadata={"engine_version": "0.1"})
        assert r.backtest_metadata["engine_version"] == "0.1"

    def test_equity_curve_empty_is_valid(self) -> None:
        r = _result(equity_curve=[])
        assert r.equity_curve == []

    def test_same_start_end_date_is_valid(self) -> None:
        r = _result(start_date_utc=START, end_date_utc=START)
        assert r.start_date_utc == r.end_date_utc


class TestBacktestResultValidation:
    def test_naive_start_date_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            _result(start_date_utc=datetime(2024, 1, 1))

    def test_naive_end_date_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            _result(end_date_utc=datetime(2024, 12, 31))

    def test_end_before_start_raises(self) -> None:
        with pytest.raises(ValueError, match="end_date_utc"):
            _result(start_date_utc=END, end_date_utc=START)

    def test_zero_capital_is_valid(self) -> None:
        """Zero initial capital is allowed for inflow-funded pools."""
        r = _result(initial_capital_eur=0.0)
        assert r.initial_capital_eur == 0.0

    def test_negative_capital_raises(self) -> None:
        with pytest.raises(ValueError, match="must be >= 0"):
            _result(initial_capital_eur=-500.0)

    def test_backtest_result_immutable(self) -> None:
        r = _result()
        with pytest.raises(FrozenInstanceError):
            r.initial_capital_eur = 99_999.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# trades_by_strategy consistency
# ---------------------------------------------------------------------------


class TestTradesByStrategyConsistency:
    def test_trades_by_strategy_consistency(self) -> None:
        """Sum of per-strategy trade counts must equal total closed_trades."""
        t1 = _trade(TAG06, pnl=300.0)
        t2 = _trade(TAG07, pnl=500.0)
        t3 = _trade(TAG07, pnl=-100.0)
        r = _result(
            trades=[t1, t2, t3],
            trades_by_strategy={TAG06: [t1], TAG07: [t2, t3]},
        )
        total = sum(len(v) for v in r.trades_by_strategy.values())
        assert total == len(r.closed_trades) == 3

    def test_trades_by_strategy_mismatch_raises(self) -> None:
        """trades_by_strategy with wrong total raises at construction."""
        t1 = _trade(TAG07)
        t2 = _trade(TAG07)
        with pytest.raises(ValueError, match="trades_by_strategy total"):
            _result(
                trades=[t1, t2],
                trades_by_strategy={TAG06: [], TAG07: [t1]},  # missing t2
            )

    def test_empty_trades_consistent(self) -> None:
        r = _result(
            trades=[],
            trades_by_strategy={TAG06: [], TAG07: []},
        )
        assert len(r.closed_trades) == 0
        assert sum(len(v) for v in r.trades_by_strategy.values()) == 0
