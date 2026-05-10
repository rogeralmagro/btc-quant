"""Unit tests for BacktestEngine inflow processing."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulator
from btc_quant.backtester.inflow_scheduler import CapitalInflow
from btc_quant.backtester.market_context import MarketContext
from btc_quant.backtester.models.enums import OrderSide, StrategyTag
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.backtester.models.signal import Signal
from btc_quant.backtester.strategy_base import StrategyBase
from tests.unit.backtester.conftest import make_ohlcv_df

UTC = timezone.utc
START = datetime(2020, 1, 1, tzinfo=UTC)
TAG = StrategyTag.BUY_AND_HOLD_BENCHMARK
TAG2 = StrategyTag.STRAT_06_BASELINE


class _NoSignalStrategy(StrategyBase):
    def __init__(self, tag: StrategyTag = TAG) -> None:
        super().__init__(tag, {})

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context: MarketContext, portfolio: Portfolio) -> list[Signal]:
        return []


def _data(n: int = 10) -> dict[str, pd.DataFrame]:
    import pandas as pd
    return {"1d": make_ohlcv_df(n, START, "1d")}


def _sim() -> ExecutionSimulator:
    return ExecutionSimulator(fee_rate=0.001, slippage_rate=0.0)


def _engine(
    tag: StrategyTag = TAG,
    capital: float = 0.0,
    inflows: list[CapitalInflow] | None = None,
    n: int = 10,
) -> BacktestEngine:
    import pandas as pd
    data = {"1d": make_ohlcv_df(n, START, "1d")}
    return BacktestEngine(
        strategies=[_NoSignalStrategy(tag=tag)],
        data=data,
        initial_capital_per_strategy={tag: capital},
        execution_simulator=_sim(),
        profit_recycle_pct=0.0,
        inflow_schedule=inflows,
    )


# Need pandas for _data helper
import pandas as pd


# ---------------------------------------------------------------------------
# No inflow schedule: backward compatibility
# ---------------------------------------------------------------------------

class TestEngineWithoutInflowSchedule:
    def test_engine_without_inflow_schedule_works(self) -> None:
        """Engine with no inflow_schedule behaves identically to before."""
        result = _engine(capital=10_000.0).run()
        assert len(result.equity_curve) == 10
        assert result.processed_inflows == []

    def test_engine_none_inflow_schedule_works(self) -> None:
        result = BacktestEngine(
            strategies=[_NoSignalStrategy()],
            data={"1d": make_ohlcv_df(10, START, "1d")},
            initial_capital_per_strategy={TAG: 5_000.0},
            execution_simulator=_sim(),
            profit_recycle_pct=0.0,
            inflow_schedule=None,
        ).run()
        assert result.processed_inflows == []


# ---------------------------------------------------------------------------
# Inflow processing on correct timestamp
# ---------------------------------------------------------------------------

class TestInflowProcessing:
    def test_engine_processes_inflow_on_correct_timestamp(self) -> None:
        """Pool starts at 0; inflow on bar[3] adds €100. Check after bar[3]."""
        inflow_ts = START + timedelta(days=3)  # bar[3].timestamp_utc
        inflows = [CapitalInflow(inflow_ts, 100.0, TAG)]
        result = _engine(capital=0.0, inflows=inflows).run()

        # Inflow timestamp == bar[3].timestamp_utc. The engine advances context
        # to bar[3].close_time_utc, which is > bar[3].timestamp_utc, so the
        # inflow fires at bar[3]. cash_by_pool[TAG] at snapshot[3] should be 100.
        snap = result.equity_curve[3]
        assert snap.cash_by_pool[TAG] == pytest.approx(100.0)

    def test_engine_processes_multiple_inflows_in_order(self) -> None:
        inflows = [
            CapitalInflow(START + timedelta(days=1), 100.0, TAG),
            CapitalInflow(START + timedelta(days=3), 200.0, TAG),
            CapitalInflow(START + timedelta(days=7), 300.0, TAG),
        ]
        result = _engine(capital=0.0, inflows=inflows).run()

        assert result.equity_curve[1].cash_by_pool[TAG] == pytest.approx(100.0)
        assert result.equity_curve[3].cash_by_pool[TAG] == pytest.approx(300.0)
        assert result.equity_curve[7].cash_by_pool[TAG] == pytest.approx(600.0)

    def test_inflow_adds_to_existing_capital(self) -> None:
        """Pool starts with €500; inflow of €200 → total €700."""
        inflows = [CapitalInflow(START + timedelta(days=2), 200.0, TAG)]
        result = _engine(capital=500.0, inflows=inflows).run()
        assert result.equity_curve[2].cash_by_pool[TAG] == pytest.approx(700.0)

    def test_inflows_appear_in_backtest_result(self) -> None:
        inflows = [
            CapitalInflow(START + timedelta(days=1), 100.0, TAG),
            CapitalInflow(START + timedelta(days=5), 200.0, TAG),
        ]
        result = _engine(capital=0.0, inflows=inflows).run()
        assert len(result.processed_inflows) == 2
        assert result.processed_inflows[0].amount_eur == pytest.approx(100.0)
        assert result.processed_inflows[1].amount_eur == pytest.approx(200.0)

    def test_inflow_before_backtest_range_is_processed_on_first_bar(self) -> None:
        """Inflow scheduled before data start is processed at first bar."""
        past_ts = START - timedelta(days=10)
        inflows = [CapitalInflow(past_ts, 50.0, TAG)]
        result = _engine(capital=0.0, inflows=inflows).run()
        assert result.equity_curve[0].cash_by_pool[TAG] == pytest.approx(50.0)

    def test_inflow_after_backtest_range_is_not_processed(self) -> None:
        """Inflow scheduled after the last bar timestamp is never processed."""
        future_ts = START + timedelta(days=100)
        inflows = [CapitalInflow(future_ts, 999.0, TAG)]
        result = _engine(capital=0.0, inflows=inflows, n=10).run()
        assert result.processed_inflows == []
        assert result.equity_curve[-1].cash_by_pool[TAG] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class TestInflowValidation:
    def test_engine_inflow_to_nonexistent_pool_raises(self) -> None:
        """Inflow targeting a pool with no registered strategy raises at construction."""
        bad_inflow = CapitalInflow(START, 100.0, TAG2)  # TAG2 has no pool
        with pytest.raises(ValueError, match="has no registered pool"):
            BacktestEngine(
                strategies=[_NoSignalStrategy(TAG)],  # only TAG pool exists
                data={"1d": make_ohlcv_df(5, START, "1d")},
                initial_capital_per_strategy={TAG: 0.0},
                execution_simulator=_sim(),
                profit_recycle_pct=0.0,
                inflow_schedule=[bad_inflow],
            )


# ---------------------------------------------------------------------------
# Zero-capital pool funded entirely via inflows
# ---------------------------------------------------------------------------

class TestZeroCapitalWithInflow:
    def test_engine_zero_capital_with_inflow_works(self) -> None:
        """Pool starts at 0 EUR; inflow provides all capital."""
        inflows = [CapitalInflow(START + timedelta(days=1), 1_000.0, TAG)]
        result = _engine(capital=0.0, inflows=inflows).run()

        assert result.processed_inflows[0].amount_eur == pytest.approx(1_000.0)
        # After inflow: pool has 1000 EUR
        assert result.equity_curve[1].cash_by_pool[TAG] == pytest.approx(1_000.0)
        # Final equity reflects the inflow
        assert result.equity_curve[-1].total_value_eur == pytest.approx(1_000.0)
