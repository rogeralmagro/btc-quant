"""Unit tests for BacktestEngine."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import pandas as pd
import pytest

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulator
from btc_quant.backtester.market_context import MarketContext
from btc_quant.backtester.models.enums import OrderSide, StrategyTag
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.backtester.models.signal import Signal
from btc_quant.backtester.strategy_base import StrategyBase
from tests.unit.backtester.conftest import make_ohlcv_df

UTC = timezone.utc
START = datetime(2020, 1, 1, tzinfo=UTC)

TAG_BAH = StrategyTag.BUY_AND_HOLD_BENCHMARK
TAG_06 = StrategyTag.STRAT_06_BASELINE
TAG_07 = StrategyTag.STRAT_07_TACTICAL


# ---------------------------------------------------------------------------
# Minimal concrete strategy helpers
# ---------------------------------------------------------------------------


class _NoSignalStrategy(StrategyBase):
    """Strategy that never signals — used for engine structural tests."""

    def __init__(self, tag: StrategyTag = TAG_07, tf: str = "1d") -> None:
        super().__init__(tag, {})
        self._tf = tf

    def required_timeframes(self) -> list[str]:
        return [self._tf]

    def primary_timeframe(self) -> str:
        return self._tf

    def generate_signals(self, context: MarketContext, portfolio: Portfolio) -> list[Signal]:
        return []


class _SingleBuyStrategy(StrategyBase):
    """Sends one BUY of 1 BTC on the first bar, then nothing."""

    def __init__(self, tag: StrategyTag = TAG_07) -> None:
        super().__init__(tag, {})
        self._done = False

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context: MarketContext, portfolio: Portfolio) -> list[Signal]:
        if self._done:
            return []
        self._done = True
        return [Signal(
            strategy_tag=self.strategy_tag,
            timestamp_utc=context.current_time,
            side=OrderSide.BUY,
            symbol=context.symbol,
            quantity_btc=1.0,
        )]


class _BuyThenSellStrategy(StrategyBase):
    """Sends BUY on bar 0, SELL on bar 2."""

    def __init__(self, tag: StrategyTag = TAG_07) -> None:
        super().__init__(tag, {})
        self._bar = 0

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context: MarketContext, portfolio: Portfolio) -> list[Signal]:
        self._bar += 1
        if self._bar == 1:
            return [Signal(
                strategy_tag=self.strategy_tag,
                timestamp_utc=context.current_time,
                side=OrderSide.BUY,
                symbol=context.symbol,
                quantity_btc=1.0,
            )]
        if self._bar == 3:
            pool = portfolio.get_pool(self.strategy_tag)
            if pool.btc_held > 0:
                return [Signal(
                    strategy_tag=self.strategy_tag,
                    timestamp_utc=context.current_time,
                    side=OrderSide.SELL,
                    symbol=context.symbol,
                    quantity_btc=pool.btc_held,
                    reason_tag="manual_sell",
                )]
        return []


class _DCAStrategy(StrategyBase):
    """Sends two BUYs (bars 0 and 1) for accumulation test."""

    def __init__(self, tag: StrategyTag = TAG_07) -> None:
        super().__init__(tag, {})
        self._bar = 0

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def allows_position_accumulation(self) -> bool:
        return True

    def generate_signals(self, context: MarketContext, portfolio: Portfolio) -> list[Signal]:
        self._bar += 1
        if self._bar <= 2:
            return [Signal(
                strategy_tag=self.strategy_tag,
                timestamp_utc=context.current_time,
                side=OrderSide.BUY,
                symbol=context.symbol,
                quantity_btc=0.5,
            )]
        return []


class _StopStrategy(StrategyBase):
    """Buys on bar 0 with a stop-loss set to be hit on bar 1."""

    def __init__(self, stop_price: float, tag: StrategyTag = TAG_07) -> None:
        super().__init__(tag, {})
        self._stop = stop_price
        self._done = False

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context: MarketContext, portfolio: Portfolio) -> list[Signal]:
        if self._done:
            return []
        self._done = True
        return [Signal(
            strategy_tag=self.strategy_tag,
            timestamp_utc=context.current_time,
            side=OrderSide.BUY,
            symbol=context.symbol,
            quantity_btc=1.0,
            stop_loss_price=self._stop,
        )]


class _TpStrategy(StrategyBase):
    """Buys on bar 0 with a take-profit set to be hit on bar 1."""

    def __init__(self, tp_price: float, tag: StrategyTag = TAG_07) -> None:
        super().__init__(tag, {})
        self._tp = tp_price
        self._done = False

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context: MarketContext, portfolio: Portfolio) -> list[Signal]:
        if self._done:
            return []
        self._done = True
        return [Signal(
            strategy_tag=self.strategy_tag,
            timestamp_utc=context.current_time,
            side=OrderSide.BUY,
            symbol=context.symbol,
            quantity_btc=1.0,
            take_profit_prices=[self._tp],
            take_profit_quantities_pct=[1.0],
        )]


# ---------------------------------------------------------------------------
# Data factories
# ---------------------------------------------------------------------------


def _data(n: int = 10, tf: str = "1d", extra_tfs: dict | None = None) -> dict[str, pd.DataFrame]:
    d = {tf: make_ohlcv_df(n, START, tf)}
    if extra_tfs:
        d.update(extra_tfs)
    return d


def _sim(fee: float = 0.001, slip: float = 0.0) -> ExecutionSimulator:
    return ExecutionSimulator(fee_rate=fee, slippage_rate=slip)


def _engine(
    strategies=None,
    n: int = 10,
    tf: str = "1d",
    capital: float = 10_000.0,
    tag: StrategyTag = TAG_07,
    fee: float = 0.001,
    slip: float = 0.0,
    profit_recycle_pct: float = 0.0,
    extra_tfs: dict | None = None,
) -> BacktestEngine:
    if strategies is None:
        strategies = [_NoSignalStrategy(tag=tag, tf=tf)]
    data = _data(n=n, tf=tf, extra_tfs=extra_tfs)
    capital_map = {s.strategy_tag: capital for s in strategies}
    return BacktestEngine(
        strategies=strategies,
        data=data,
        initial_capital_per_strategy=capital_map,
        execution_simulator=_sim(fee=fee, slip=slip),
        profit_recycle_pct=profit_recycle_pct,
        symbol="BTCUSDT",
    )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestEngineCreationValidation:
    def test_no_strategies_raises(self) -> None:
        with pytest.raises(ValueError, match="strategies must not be empty"):
            BacktestEngine(
                strategies=[],
                data=_data(),
                initial_capital_per_strategy={},
                execution_simulator=_sim(),
            )

    def test_missing_required_tf_raises(self) -> None:
        strat = _NoSignalStrategy(tag=TAG_07, tf="4h")
        with pytest.raises(ValueError, match="4h"):
            BacktestEngine(
                strategies=[strat],
                data=_data(tf="1d"),  # 4h not included
                initial_capital_per_strategy={TAG_07: 1000.0},
                execution_simulator=_sim(),
                profit_recycle_pct=0.0,
            )

    def test_missing_capital_entry_raises(self) -> None:
        strat = _NoSignalStrategy(tag=TAG_07)
        with pytest.raises(ValueError, match="No initial capital"):
            BacktestEngine(
                strategies=[strat],
                data=_data(),
                initial_capital_per_strategy={},  # missing TAG_07
                execution_simulator=_sim(),
                profit_recycle_pct=0.0,
            )

    def test_zero_capital_raises(self) -> None:
        strat = _NoSignalStrategy(tag=TAG_07)
        with pytest.raises(ValueError, match="must be > 0"):
            BacktestEngine(
                strategies=[strat],
                data=_data(),
                initial_capital_per_strategy={TAG_07: 0.0},
                execution_simulator=_sim(),
                profit_recycle_pct=0.0,
            )

    def test_invalid_profit_recycle_pct_raises(self) -> None:
        strat = _NoSignalStrategy(tag=TAG_07)
        with pytest.raises(ValueError, match="profit_recycle_pct"):
            BacktestEngine(
                strategies=[strat],
                data=_data(),
                initial_capital_per_strategy={TAG_07: 1000.0},
                execution_simulator=_sim(),
                profit_recycle_pct=1.5,
            )

    def test_recycle_pct_positive_but_no_target_pool_raises(self) -> None:
        strat = _NoSignalStrategy(tag=TAG_07)
        with pytest.raises(ValueError, match="strat06_baseline"):
            BacktestEngine(
                strategies=[strat],
                data=_data(),
                initial_capital_per_strategy={TAG_07: 1000.0},
                execution_simulator=_sim(),
                profit_recycle_pct=0.5,  # recycling enabled but STRAT_06 not registered
            )


# ---------------------------------------------------------------------------
# Iteration TF determination
# ---------------------------------------------------------------------------


class TestIterationTf:
    def test_single_tf_is_iteration_tf(self) -> None:
        engine = _engine(tf="1d")
        assert engine._iteration_tf == "1d"

    def test_finest_tf_selected(self) -> None:
        strat_4h = _NoSignalStrategy(tag=TAG_07, tf="4h")
        strat_1d = _NoSignalStrategy(tag=TAG_06, tf="1d")
        data = _data(tf="1d")
        data["4h"] = make_ohlcv_df(40, START, "4h")
        engine = BacktestEngine(
            strategies=[strat_4h, strat_1d],
            data=data,
            initial_capital_per_strategy={TAG_07: 1000.0, TAG_06: 1000.0},
            execution_simulator=_sim(),
            profit_recycle_pct=0.0,
        )
        assert engine._iteration_tf == "4h"


# ---------------------------------------------------------------------------
# run() — no signals
# ---------------------------------------------------------------------------


class TestEngineRunNoSignals:
    def test_equity_curve_length_equals_bar_count(self) -> None:
        engine = _engine(n=10)
        result = engine.run()
        assert len(result.equity_curve) == 10

    def test_no_closed_trades(self) -> None:
        result = _engine(n=10).run()
        assert result.closed_trades == []

    def test_no_open_positions(self) -> None:
        result = _engine(n=10).run()
        assert result.final_portfolio.open_positions == {}

    def test_initial_capital_preserved(self) -> None:
        cap = 10_000.0
        result = _engine(n=10, capital=cap).run()
        pool = result.final_portfolio.get_pool(TAG_07)
        assert pool.cash_eur == pytest.approx(cap)

    def test_strategies_used_in_result(self) -> None:
        result = _engine(n=5).run()
        assert TAG_07 in result.strategies_used


# ---------------------------------------------------------------------------
# run() — BUY signal
# ---------------------------------------------------------------------------


class TestEngineRunBuy:
    def test_buy_opens_position(self) -> None:
        engine = _engine(strategies=[_SingleBuyStrategy(TAG_07)], n=5, capital=100_000.0)
        result = engine.run()
        assert len(result.final_portfolio.open_positions) == 1

    def test_buy_deducts_cash(self) -> None:
        engine = _engine(
            strategies=[_SingleBuyStrategy(TAG_07)],
            n=5,
            capital=100_000.0,
            fee=0.001,
            slip=0.0,
        )
        result = engine.run()
        pool = result.final_portfolio.get_pool(TAG_07)
        # After buying 1 BTC: cash should be < initial
        assert pool.cash_eur < 100_000.0

    def test_buy_adds_btc_to_pool(self) -> None:
        engine = _engine(strategies=[_SingleBuyStrategy(TAG_07)], n=5, capital=100_000.0)
        result = engine.run()
        pool = result.final_portfolio.get_pool(TAG_07)
        assert pool.btc_held == pytest.approx(1.0)

    def test_buy_executes_at_bar1_open(self) -> None:
        """Signal from bar[0] close executes at bar[1] open."""
        engine = _engine(
            strategies=[_SingleBuyStrategy(TAG_07)],
            n=5,
            capital=100_000.0,
            fee=0.0,
            slip=0.0,
        )
        result = engine.run()
        pos = next(iter(result.final_portfolio.open_positions.values()))
        # bar[1] open = 10_000 + 1*100 - 50 = 10_050 (from make_ohlcv_df: open = close - 50)
        assert pos.entry_price_avg == pytest.approx(10_050.0)


# ---------------------------------------------------------------------------
# run() — BUY then SELL
# ---------------------------------------------------------------------------


class TestEngineRunBuyThenSell:
    def test_sell_closes_position(self) -> None:
        engine = _engine(strategies=[_BuyThenSellStrategy(TAG_07)], n=10, capital=100_000.0)
        result = engine.run()
        assert result.final_portfolio.open_positions == {}
        assert len(result.closed_trades) == 1

    def test_sell_credits_cash(self) -> None:
        engine = _engine(
            strategies=[_BuyThenSellStrategy(TAG_07)],
            n=10,
            capital=100_000.0,
            fee=0.0,
            slip=0.0,
        )
        result = engine.run()
        pool = result.final_portfolio.get_pool(TAG_07)
        assert pool.cash_eur > 0
        assert pool.btc_held == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run() — accumulation
# ---------------------------------------------------------------------------


class TestEngineRunAccumulation:
    def test_accumulation_raises_when_disabled(self) -> None:
        engine = _engine(strategies=[_SingleBuyStrategy(TAG_07)], n=5, capital=100_000.0)
        # Replace strategy with one that sends two BUYs but has accumulation disabled
        class _TwoBuysNoAccum(_SingleBuyStrategy):
            def allows_position_accumulation(self) -> bool:
                return False

            def generate_signals(self, context, portfolio):
                return [Signal(
                    strategy_tag=self.strategy_tag,
                    timestamp_utc=context.current_time,
                    side=OrderSide.BUY,
                    symbol=context.symbol,
                    quantity_btc=0.5,
                )]

        engine2 = _engine(strategies=[_TwoBuysNoAccum(TAG_07)], n=5, capital=100_000.0)
        with pytest.raises(ValueError, match="allows_position_accumulation"):
            engine2.run()

    def test_accumulation_works_when_enabled(self) -> None:
        engine = _engine(strategies=[_DCAStrategy(TAG_07)], n=5, capital=100_000.0)
        result = engine.run()
        # Two buys of 0.5 BTC each → 1 open position with 1.0 BTC
        assert len(result.final_portfolio.open_positions) == 1
        pos = next(iter(result.final_portfolio.open_positions.values()))
        assert pos.quantity_btc == pytest.approx(1.0)

    def test_accumulation_weighted_avg_correct(self) -> None:
        """Second buy at higher price → avg entry between the two prices."""
        engine = _engine(
            strategies=[_DCAStrategy(TAG_07)],
            n=5,
            capital=100_000.0,
            fee=0.0,
            slip=0.0,
        )
        result = engine.run()
        pos = next(iter(result.final_portfolio.open_positions.values()))
        # bar[1] open = 10_050, bar[2] open = 10_150 → avg = 10_100
        assert pos.entry_price_avg == pytest.approx(10_100.0)


# ---------------------------------------------------------------------------
# run() — stop-loss
# ---------------------------------------------------------------------------


class TestEngineRunStop:
    def test_stop_hit_closes_position(self) -> None:
        """Stop at 9_000 is below the low of every bar in make_ohlcv_df (close-150).

        bar[0] close = 10_000, low = 9_850. Stop at 9_800 is not hit.
        bar[1] close = 10_100, low = 9_950. Stop at 9_800 not hit.
        Let's use stop at 10_050 which is above bar[1] low (9_950) — no wait.

        make_ohlcv_df: close[i] = 10_000 + i*100, low[i] = close[i] - 150.
        bar[0]: close=10_000, low=9_850, open=9_950
        bar[1]: close=10_100, low=9_950, open=10_050
        bar[2]: close=10_200, low=10_050, open=10_150

        The stop fires if bar.low <= stop_price. Bar[1] is the first bar that can
        check stops (position opened there from bar[0] signal).
        But bar[1] check skips newly opened position. So earliest check is bar[2].
        bar[2].low = 10_050 → stop at 10_100 triggers on bar[2].
        """
        stop = 10_100.0  # above bar[2].low=10_050 → triggered
        engine = _engine(
            strategies=[_StopStrategy(stop, TAG_07)],
            n=10,
            capital=100_000.0,
            fee=0.0,
            slip=0.0,
        )
        result = engine.run()
        assert result.final_portfolio.open_positions == {}
        assert len(result.closed_trades) == 1
        assert result.closed_trades[0].exit_reason == "stop_loss_hit"

    def test_stop_not_hit_position_remains_open(self) -> None:
        stop = 1.0  # far below all prices — never triggers
        engine = _engine(
            strategies=[_StopStrategy(stop, TAG_07)],
            n=10,
            capital=100_000.0,
        )
        result = engine.run()
        assert len(result.final_portfolio.open_positions) == 1
        assert result.closed_trades == []


# ---------------------------------------------------------------------------
# run() — take-profit
# ---------------------------------------------------------------------------


class TestEngineRunTp:
    def test_tp_hit_closes_position(self) -> None:
        """TP at 10_250 triggered when bar[2].high = close[2] + 200 = 10_400 >= 10_250."""
        tp = 10_250.0
        engine = _engine(
            strategies=[_TpStrategy(tp, TAG_07)],
            n=10,
            capital=100_000.0,
            fee=0.0,
            slip=0.0,
        )
        result = engine.run()
        assert result.final_portfolio.open_positions == {}
        assert len(result.closed_trades) == 1
        assert "tp1" in result.closed_trades[0].exit_reason

    def test_tp_not_hit_position_remains(self) -> None:
        tp = 999_999.0  # unreachable
        engine = _engine(
            strategies=[_TpStrategy(tp, TAG_07)],
            n=10,
            capital=100_000.0,
        )
        result = engine.run()
        assert len(result.final_portfolio.open_positions) == 1
        assert result.closed_trades == []


# ---------------------------------------------------------------------------
# Profit recycling
# ---------------------------------------------------------------------------


class TestProfitRecycling:
    def _recycle_engine(self, pct: float) -> BacktestEngine:
        strat06 = _NoSignalStrategy(tag=TAG_06)
        strat07 = _BuyThenSellStrategy(tag=TAG_07)
        data = _data(n=10)
        return BacktestEngine(
            strategies=[strat06, strat07],
            data=data,
            initial_capital_per_strategy={TAG_06: 10_000.0, TAG_07: 100_000.0},
            execution_simulator=_sim(fee=0.0, slip=0.0),
            profit_recycle_pct=pct,
        )

    def test_zero_recycle_skips_transfer(self) -> None:
        engine = self._recycle_engine(0.0)
        result = engine.run()
        pool06 = result.final_portfolio.get_pool(TAG_06)
        # No transfer occurred; pool06 still has its initial 10_000 cash
        assert pool06.cash_eur == pytest.approx(10_000.0)

    def test_positive_recycle_transfers_on_winning_trade(self) -> None:
        engine = self._recycle_engine(0.5)
        result = engine.run()
        pool06 = result.final_portfolio.get_pool(TAG_06)
        trade = result.closed_trades[0]
        if trade.net_pnl_eur > 0:
            expected_transfer = trade.net_pnl_eur * 0.5
            assert pool06.cash_eur == pytest.approx(10_000.0 + expected_transfer)
        else:
            assert pool06.cash_eur == pytest.approx(10_000.0)

    def test_engine_raises_when_recycle_target_missing(self) -> None:
        strat07 = _NoSignalStrategy(tag=TAG_07)
        data = _data(n=5)
        with pytest.raises(ValueError, match="strat06_baseline"):
            BacktestEngine(
                strategies=[strat07],
                data=data,
                initial_capital_per_strategy={TAG_07: 1000.0},
                execution_simulator=_sim(),
                profit_recycle_pct=0.3,
            )


# ---------------------------------------------------------------------------
# Strategy alignment
# ---------------------------------------------------------------------------


class TestStrategyAlignment:
    def test_coarser_strategy_only_fires_on_aligned_bars(self) -> None:
        """1w strategy with 1d iteration TF fires only on weekly bars."""
        bar_count = []

        class _CountingWeeklyStrategy(StrategyBase):
            def required_timeframes(self) -> list[str]:
                return ["1w"]

            def primary_timeframe(self) -> str:
                return "1w"

            def generate_signals(self, context, portfolio) -> list[Signal]:
                bar_count.append(1)
                return []

        weekly_df = make_ohlcv_df(4, START, "1w")
        daily_df = make_ohlcv_df(28, START, "1d")
        strat = _CountingWeeklyStrategy(TAG_07, {})
        engine = BacktestEngine(
            strategies=[strat],
            data={"1d": daily_df, "1w": weekly_df},
            initial_capital_per_strategy={TAG_07: 1000.0},
            execution_simulator=_sim(),
            profit_recycle_pct=0.0,
        )
        engine.run()
        # 4 weekly bars: strategy fires at most 4 times
        assert len(bar_count) <= 4


# ---------------------------------------------------------------------------
# PortfolioSnapshot.open_positions_by_strategy
# ---------------------------------------------------------------------------


class TestSnapshotOpenPositionsByStrategy:
    def test_snapshot_tracks_positions_by_strategy(self) -> None:
        """After a BUY, the snapshot must reflect open_positions_by_strategy for that tag."""
        result = _engine(strategies=[_SingleBuyStrategy(TAG_07)], n=5, capital=100_000.0).run()
        # BUY fires on bar 0, executes on bar 1; bars 1-4 snapshot have TAG_07 count=1
        snapshots_with_position = [
            s for s in result.equity_curve
            if s.open_positions_by_strategy.get(TAG_07, 0) > 0
        ]
        assert len(snapshots_with_position) > 0

    def test_snapshot_no_position_before_buy(self) -> None:
        """Bar 0 snapshot: no position opened yet → TAG_07 count == 0."""
        result = _engine(strategies=[_SingleBuyStrategy(TAG_07)], n=5, capital=100_000.0).run()
        first_snap = result.equity_curve[0]
        assert first_snap.open_positions_by_strategy.get(TAG_07, 0) == 0

    def test_snapshot_two_strategies_tracked_independently(self) -> None:
        """Two strategies: one buys, one doesn't → counts are independent."""
        buyer = _SingleBuyStrategy(TAG_07)
        idle = _NoSignalStrategy(TAG_06)
        engine = BacktestEngine(
            strategies=[buyer, idle],
            data=_data(n=5),
            initial_capital_per_strategy={TAG_07: 100_000.0, TAG_06: 5_000.0},
            execution_simulator=_sim(),
            profit_recycle_pct=0.0,
        )
        result = engine.run()
        assert any(s.open_positions_by_strategy.get(TAG_07, 0) > 0 for s in result.equity_curve)
        assert all(s.open_positions_by_strategy.get(TAG_06, 0) == 0 for s in result.equity_curve)

    def test_open_positions_count_equals_sum_of_by_strategy(self) -> None:
        """open_positions_count must equal sum(open_positions_by_strategy.values())."""
        result = _engine(strategies=[_SingleBuyStrategy(TAG_07)], n=5, capital=100_000.0).run()
        for snap in result.equity_curve:
            total = sum(snap.open_positions_by_strategy.values())
            assert snap.open_positions_count == total


# ---------------------------------------------------------------------------
# Full-cash buy: no float64 overdraft
# ---------------------------------------------------------------------------


class _FullCashBuyStrategy(StrategyBase):
    """Sends one BUY with quote_amount_eur = entire pool cash on first bar."""

    def __init__(self, tag: StrategyTag = TAG_BAH) -> None:
        super().__init__(tag, {})
        self._done = False

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context: MarketContext, portfolio: Portfolio) -> list[Signal]:
        if self._done:
            return []
        self._done = True
        pool = portfolio.get_pool(self.strategy_tag)
        return [Signal(
            strategy_tag=self.strategy_tag,
            timestamp_utc=context.current_time,
            side=OrderSide.BUY,
            symbol=context.symbol,
            quote_amount_eur=pool.cash_eur,
        )]


class TestFullCashBuyNoOverdraft:
    def test_signal_with_full_cash_does_not_overdraft(self) -> None:
        """Engine must not raise overdraft when strategy spends entire pool cash.

        The SAFETY_FACTOR=0.999 in _execute_signal prevents float64 rounding
        from causing notional+fees to exceed available cash by a sub-cent amount.
        After the buy, cash must be > 0 (small residual from the safety margin).
        """
        strategy = _FullCashBuyStrategy(TAG_BAH)
        engine = BacktestEngine(
            strategies=[strategy],
            data=_data(n=5),
            initial_capital_per_strategy={TAG_BAH: 10_000.0},
            execution_simulator=ExecutionSimulator(fee_rate=0.001, slippage_rate=0.0005),
            profit_recycle_pct=0.0,
        )
        result = engine.run()  # must not raise

        # After the buy executes (bar 1), total portfolio value must be positive
        snap_after_buy = result.equity_curve[1]
        assert snap_after_buy.total_value_eur > 0
        # The residual cash comes from SAFETY_FACTOR: at most 0.1% of capital
        cash_residual = snap_after_buy.cash_by_pool.get(TAG_BAH, 0.0)
        assert 0.0 < cash_residual < 10_000.0 * 0.001 + 1.0  # < ~11 EUR
