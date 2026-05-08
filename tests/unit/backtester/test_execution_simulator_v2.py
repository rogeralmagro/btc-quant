"""Unit tests for ExecutionSimulatorV2 (regime-aware cost model)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pandas as pd
import pytest

from btc_quant.backtester.execution_simulator import (
    ExecutionSimulator,
    ExecutionSimulatorV1,
    ExecutionSimulatorV2,
)
from btc_quant.backtester.models.enums import (
    MarketRegime,
    OrderSide,
    OrderStatus,
    OrderType,
    StrategyTag,
)
from btc_quant.backtester.models.order import Order
from btc_quant.backtester.models.position import Position
from btc_quant.backtester.regime_detector import RegimeDetector

UTC = timezone.utc
START = datetime(2017, 1, 1, tzinfo=UTC)
TS = datetime(2024, 4, 1, tzinfo=UTC)
TS_NEXT = datetime(2024, 4, 2, tzinfo=UTC)
TAG = StrategyTag.STRAT_07_TACTICAL


# ---------------------------------------------------------------------------
# Data helpers (reuse logic from test_regime_detector)
# ---------------------------------------------------------------------------


def _make_df(n: int, amplitude: float, start: datetime = START) -> pd.DataFrame:
    base = 10_000.0
    timestamps = [start + timedelta(days=i) for i in range(n)]
    close_times = [ts + timedelta(hours=23) for ts in timestamps]
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(timestamps, utc=True),
        "close_time_utc": pd.to_datetime(close_times, utc=True),
        "open":   [base] * n,
        "high":   [base + amplitude] * n,
        "low":    [base - amplitude] * n,
        "close":  [base] * n,
        "volume": [1_000.0] * n,
        "is_synthetic": [False] * n,
    })


def _concat(*segs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(segs, ignore_index=True)


def _normal_hist() -> pd.DataFrame:
    return _make_df(400, 50)


def _volatile_hist() -> pd.DataFrame:
    return _concat(_make_df(400, 50), _make_df(100, 100, START + timedelta(days=400)))


def _stress_hist() -> pd.DataFrame:
    return _concat(_make_df(465, 5), _make_df(35, 5_000, START + timedelta(days=465)))


def _bar(
    open_: float = 45_000.0,
    high: float | None = None,
    low: float | None = None,
) -> pd.Series:
    """Bar helper. Default high/low are ±2% of open to avoid triggering CASCADE."""
    if high is None:
        high = open_ * 1.02
    if low is None:
        low = open_ * 0.98
    return pd.Series({
        "timestamp_utc": pd.Timestamp(TS_NEXT),
        "open": open_,
        "high": high,
        "low": low,
        "close": open_,
        "volume": 100.0,
    })


def _cascade_bar() -> pd.Series:
    """Bar where (high-low)/open > 10% → CASCADE regime."""
    return _bar(open_=10_000.0, high=11_500.0, low=8_400.0)  # range=3100/10000=0.31


def _market_order(side: OrderSide = OrderSide.BUY, qty: float = 1.0) -> Order:
    return Order(
        order_id=uuid4(),
        strategy_tag=TAG,
        timestamp_utc=TS,
        symbol="BTCUSDT",
        side=side,
        order_type=OrderType.MARKET,
        quantity_btc=qty,
    )


def _long_position(entry: float = 45_000.0, qty: float = 1.0) -> Position:
    return Position(
        position_id=uuid4(),
        strategy_tag=TAG,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        opened_at_utc=TS,
        quantity_btc=qty,
        entry_price_avg=entry,
        stop_loss_price=43_000.0,
    )


def _v2(fee: float = 0.001) -> ExecutionSimulatorV2:
    return ExecutionSimulatorV2(RegimeDetector(), fee_rate=fee)


# ---------------------------------------------------------------------------
# V1 backward-compatibility regression
# ---------------------------------------------------------------------------


class TestV1StillWorksUnchanged:
    """Confirm the alias and original V1 tests are unaffected."""

    def test_alias_resolves_to_v1(self) -> None:
        assert ExecutionSimulator is ExecutionSimulatorV1

    def test_v1_import_and_instantiate(self) -> None:
        sim = ExecutionSimulatorV1()
        assert sim._fee_rate == 0.001
        assert sim._slippage_rate == 0.0005

    def test_v1_execute_market_buy(self) -> None:
        sim = ExecutionSimulatorV1(fee_rate=0.001, slippage_rate=0.0005)
        order = _market_order(OrderSide.BUY)
        bar = _bar(open_=45_000.0)
        ex = sim.execute(order, bar)
        expected = 45_000.0 * 1.0005
        assert ex.executed_price == pytest.approx(expected)
        assert ex.status == OrderStatus.FILLED

    def test_v1_execute_stop_or_tp_not_triggered(self) -> None:
        sim = ExecutionSimulatorV1()
        pos = _long_position()
        bar = _bar(open_=45_000.0, high=46_000.0, low=44_000.0)
        assert sim.execute_stop_or_tp(pos, bar, "stop", 43_000.0) is None


# ---------------------------------------------------------------------------
# V2 construction validation
# ---------------------------------------------------------------------------


class TestV2Creation:
    def test_default_fee_rate(self) -> None:
        v2 = _v2()
        assert v2._fee_rate == 0.001

    def test_invalid_fee_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="fee_rate"):
            ExecutionSimulatorV2(RegimeDetector(), fee_rate=0.05)


# ---------------------------------------------------------------------------
# V2 execute() — slippage per regime
# ---------------------------------------------------------------------------


class TestV2ExecuteRegimeSlippage:
    def test_normal_regime_uses_normal_slippage(self) -> None:
        v2 = _v2(fee=0.0)
        ex = v2.execute(_market_order(OrderSide.BUY), _bar(open_=10_000.0), _normal_hist())
        expected = 10_000.0 * (1 + 0.0005)
        assert ex.executed_price == pytest.approx(expected)

    def test_volatile_regime_uses_volatile_slippage(self) -> None:
        v2 = _v2(fee=0.0)
        ex = v2.execute(_market_order(OrderSide.BUY), _bar(open_=10_000.0), _volatile_hist())
        expected = 10_000.0 * (1 + 0.0015)
        assert ex.executed_price == pytest.approx(expected)

    def test_stress_regime_uses_stress_slippage(self) -> None:
        v2 = _v2(fee=0.0)
        ex = v2.execute(_market_order(OrderSide.BUY), _bar(open_=10_000.0), _stress_hist())
        expected = 10_000.0 * (1 + 0.005)
        assert ex.executed_price == pytest.approx(expected)

    def test_cascade_bar_uses_cascade_slippage(self) -> None:
        """Bar with (high-low)/open > 10% → CASCADE → 1.0% slippage."""
        v2 = _v2(fee=0.0)
        bar = _cascade_bar()  # open=10_000, cascade detected
        ex = v2.execute(_market_order(OrderSide.BUY), bar, _normal_hist())
        expected = 10_000.0 * (1 + 0.01)
        assert ex.executed_price == pytest.approx(expected)

    def test_slippage_eur_reflects_regime(self) -> None:
        v2 = _v2(fee=0.0)
        bar = _cascade_bar()
        ex = v2.execute(_market_order(OrderSide.BUY, qty=1.0), bar, _normal_hist())
        # slippage = 10_000 * 0.01 * 1.0 = 100
        assert ex.slippage_eur == pytest.approx(100.0)


class TestV2ExecuteFeesIndependentOfRegime:
    def test_fees_equal_in_normal_vs_stress(self) -> None:
        """fee_rate is fixed regardless of regime; only slippage changes."""
        v2 = ExecutionSimulatorV2(RegimeDetector(), fee_rate=0.001)
        order = _market_order(OrderSide.BUY, qty=1.0)

        ex_normal = v2.execute(order, _bar(open_=10_000.0), _normal_hist())
        ex_stress = v2.execute(order, _bar(open_=10_000.0), _stress_hist())

        # fee = notional × fee_rate; notional differs (prices differ by slippage)
        # but fee_rate is same → fees differ proportionally to exec_price
        # Verify fee_rate consistency: fee / notional == 0.001 for both
        assert ex_normal.fees_eur / ex_normal.executed_price == pytest.approx(0.001, rel=1e-9)
        assert ex_stress.fees_eur / ex_stress.executed_price == pytest.approx(0.001, rel=1e-9)

        # Slippage differs between regimes
        assert ex_normal.slippage_eur != pytest.approx(ex_stress.slippage_eur)


class TestV2ExecuteMetadataIncludesRegime:
    def test_notes_contain_regime(self) -> None:
        v2 = _v2()
        ex = v2.execute(_market_order(OrderSide.BUY), _bar(open_=10_000.0), _normal_hist())
        assert "regime:" in ex.notes

    def test_notes_cascade_regime(self) -> None:
        v2 = _v2()
        ex = v2.execute(_market_order(OrderSide.BUY), _cascade_bar(), _normal_hist())
        assert "cascade" in ex.notes

    def test_notes_fallback_on_insufficient_history(self) -> None:
        """When history < atr_period, falls back to NORMAL and records it."""
        v2 = _v2()
        tiny_hist = _make_df(5, 50)  # < 14 rows
        ex = v2.execute(_market_order(OrderSide.BUY), _bar(open_=10_000.0), tiny_hist)
        assert "fallback" in ex.notes
        assert "normal" in ex.notes


# ---------------------------------------------------------------------------
# V2 execute_stop_or_tp()
# ---------------------------------------------------------------------------


class TestV2ExecuteStopOrTp:
    def test_stop_not_triggered_returns_none(self) -> None:
        v2 = _v2(fee=0.0)
        pos = _long_position(entry=45_000.0)
        bar = _bar(open_=45_000.0, high=46_000.0, low=44_000.0)
        assert v2.execute_stop_or_tp(pos, bar, "stop", 43_000.0, _normal_hist()) is None

    def test_stop_triggered_fills_below_level(self) -> None:
        """In NORMAL regime: exec_price = level × (1 - 0.0005)."""
        v2 = _v2(fee=0.0)
        pos = _long_position(entry=45_000.0)
        bar = _bar(open_=45_000.0, high=46_000.0, low=43_000.0)
        result = v2.execute_stop_or_tp(pos, bar, "stop", 43_000.0, _normal_hist())
        assert result is not None
        assert result.executed_price == pytest.approx(43_000.0 * (1 - 0.0005))

    def test_cascade_stop_uses_cascade_slippage(self) -> None:
        """Stop on a CASCADE bar: slippage = 1%."""
        v2 = _v2(fee=0.0)
        pos = _long_position(entry=45_000.0)
        # Gap: bar opens below stop level AND is cascade
        bar = _bar(open_=9_000.0, high=11_000.0, low=7_500.0)  # (11k-7.5k)/9k ≈ 0.39 → cascade
        result = v2.execute_stop_or_tp(pos, bar, "stop", 10_000.0, _normal_hist())
        assert result is not None
        # Gap fill: base_fill = bar_open = 9_000 (open < level)
        # cascade slippage: 9_000 × (1 - 0.01) = 8_910
        assert result.executed_price == pytest.approx(9_000.0 * (1 - 0.01))

    def test_tp_triggered_notes_contain_label_and_regime(self) -> None:
        v2 = _v2(fee=0.0)
        pos = _long_position(entry=45_000.0)
        bar = _bar(open_=45_000.0, high=48_000.0, low=44_000.0)
        result = v2.execute_stop_or_tp(pos, bar, "tp1", 48_000.0, _normal_hist())
        assert result is not None
        assert "tp1" in result.notes
        assert "regime:" in result.notes
