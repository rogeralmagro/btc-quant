"""Unit tests for MetricsCalculator — each static helper validated individually."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import numpy as np
import pytest

from btc_quant.backtester.inflow_scheduler import CapitalInflow
from btc_quant.backtester.metrics.calculator import MetricsCalculator
from btc_quant.backtester.models.enums import OrderSide, StrategyTag, TradeOutcome
from btc_quant.backtester.models.pool import CapitalPool
from btc_quant.backtester.models.portfolio import Portfolio, PortfolioSnapshot
from btc_quant.backtester.models.result import BacktestResult
from btc_quant.backtester.models.trade import Trade

UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)
TAG = StrategyTag.STRAT_07_TACTICAL


# ---------------------------------------------------------------------------
# Test factories
# ---------------------------------------------------------------------------


def _make_trade(
    net_pnl: float,
    fees: float = 10.0,
    r_multiple: float | None = None,
) -> Trade:
    outcome = (
        TradeOutcome.WIN if net_pnl > 0
        else TradeOutcome.LOSS if net_pnl < 0
        else TradeOutcome.BREAKEVEN
    )
    gross = net_pnl + fees
    return Trade(
        trade_id=uuid4(),
        strategy_tag=TAG,
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        opened_at_utc=START,
        entry_price_avg=10_000.0,
        quantity_btc=1.0,
        closed_at_utc=START + timedelta(days=1),
        exit_price_avg=10_000.0 + gross,
        gross_pnl_eur=gross,
        fees_eur=fees,
        net_pnl_eur=net_pnl,
        return_pct=gross / 10_000.0,
        outcome=outcome,
        r_multiple_realized=r_multiple,
    )


def _make_snapshot(
    total_value: float,
    has_positions: bool = False,
    ts: datetime | None = None,
) -> PortfolioSnapshot:
    if ts is None:
        ts = START
    return PortfolioSnapshot(
        timestamp_utc=ts,
        btc_price=10_000.0,
        total_value_eur=total_value,
        total_btc=0.0,
        cash_by_pool={TAG: total_value},
        btc_by_pool={TAG: 0.0},
        open_positions_count=1 if has_positions else 0,
    )


def _make_inflow(amount: float, day_offset: int = 0) -> CapitalInflow:
    return CapitalInflow(
        timestamp_utc=START + timedelta(days=day_offset),
        amount_eur=amount,
        target_strategy=TAG,
    )


def _make_result(
    equity_values: list[float],
    start: datetime = START,
    days: int = 365,
    initial_capital: float = 10_000.0,
    trades: list[Trade] | None = None,
    has_positions: list[bool] | None = None,
    processed_inflows: list[CapitalInflow] | None = None,
) -> BacktestResult:
    end = start + timedelta(days=days)
    if trades is None:
        trades = []
    if has_positions is None:
        has_positions = [False] * len(equity_values)
    snapshots = [
        _make_snapshot(v, hp, start + timedelta(days=i))
        for i, (v, hp) in enumerate(zip(equity_values, has_positions))
    ]
    pool = CapitalPool(
        strategy_tag=TAG,
        cash_eur=equity_values[-1] if equity_values else initial_capital,
        total_invested_eur=initial_capital,
    )
    portfolio = Portfolio(pools={TAG: pool})
    return BacktestResult(
        start_date_utc=start,
        end_date_utc=end,
        initial_capital_eur=initial_capital,
        strategies_used=[TAG],
        final_portfolio=portfolio,
        equity_curve=snapshots,
        closed_trades=trades,
        trades_by_strategy={TAG: list(trades)},
        processed_inflows=processed_inflows or [],
    )


# ---------------------------------------------------------------------------
# calculate_returns
# ---------------------------------------------------------------------------


class TestCalculateReturns:
    def test_returns_basic(self) -> None:
        returns = MetricsCalculator.calculate_returns([100.0, 110.0, 99.0])
        # (110-100)/100 = 0.1, (99-110)/110 = -0.1̄
        assert returns[0] == pytest.approx(0.1)
        assert returns[1] == pytest.approx(-11.0 / 110.0)

    def test_returns_empty_curve(self) -> None:
        returns = MetricsCalculator.calculate_returns([])
        assert len(returns) == 0

    def test_returns_single_value_empty(self) -> None:
        returns = MetricsCalculator.calculate_returns([10_000.0])
        assert len(returns) == 0

    def test_returns_length(self) -> None:
        returns = MetricsCalculator.calculate_returns([1.0, 2.0, 3.0, 4.0])
        assert len(returns) == 3


# ---------------------------------------------------------------------------
# calculate_cagr
# ---------------------------------------------------------------------------


class TestCalculateCagr:
    def test_cagr_basic(self) -> None:
        # 1000 → 2000 in exactly 365 days → CAGR = 100%
        result = MetricsCalculator.calculate_cagr(1_000.0, 2_000.0, 365)
        assert result == pytest.approx(1.0)

    def test_cagr_short_period_returns_none(self) -> None:
        result = MetricsCalculator.calculate_cagr(1_000.0, 2_000.0, 20)
        assert result is None

    def test_cagr_exactly_30_days_not_none(self) -> None:
        result = MetricsCalculator.calculate_cagr(1_000.0, 1_100.0, 30)
        assert result is not None

    def test_cagr_negative(self) -> None:
        # 1000 → 500 in 365 days → CAGR = -50%
        result = MetricsCalculator.calculate_cagr(1_000.0, 500.0, 365)
        assert result == pytest.approx(-0.5)

    def test_cagr_two_years(self) -> None:
        # 1000 → 4000 in 730 days → CAGR = (4)^(1/2) - 1 = 1.0 (100%)
        result = MetricsCalculator.calculate_cagr(1_000.0, 4_000.0, 730)
        assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# calculate_max_drawdown
# ---------------------------------------------------------------------------


class TestCalculateMaxDrawdown:
    def test_max_dd_basic(self) -> None:
        dd, peak_i, trough_i = MetricsCalculator.calculate_max_drawdown([100.0, 120.0, 90.0, 110.0])
        assert dd == pytest.approx(-0.25)   # (90-120)/120
        assert peak_i == 1
        assert trough_i == 2

    def test_max_dd_no_drawdown(self) -> None:
        dd, peak_i, trough_i = MetricsCalculator.calculate_max_drawdown([100.0, 110.0, 120.0])
        assert dd == 0.0

    def test_max_dd_returns_correct_indices(self) -> None:
        # Two drawdowns; second is larger
        curve = [100.0, 110.0, 105.0, 120.0, 80.0, 100.0]
        dd, peak_i, trough_i = MetricsCalculator.calculate_max_drawdown(curve)
        # Max DD: 120 → 80 = -33.3%
        assert dd == pytest.approx((80.0 - 120.0) / 120.0)
        assert peak_i == 3
        assert trough_i == 4

    def test_max_dd_single_value(self) -> None:
        dd, _, _ = MetricsCalculator.calculate_max_drawdown([10_000.0])
        assert dd == 0.0

    def test_max_dd_empty(self) -> None:
        dd, _, _ = MetricsCalculator.calculate_max_drawdown([])
        assert dd == 0.0


# ---------------------------------------------------------------------------
# calculate_drawdown_duration
# ---------------------------------------------------------------------------


def _ts(days_offset: int) -> datetime:
    return START + timedelta(days=days_offset)


class TestCalculateDrawdownDuration:
    def test_duration_with_recovery_calendar_days(self) -> None:
        # Peak at day 1 (120), trough at day 2 (90), recovery at day 3 (130)
        curve = [100.0, 120.0, 90.0, 130.0]
        ts = [_ts(i) for i in range(4)]
        assert MetricsCalculator.calculate_drawdown_duration(curve, ts) == 2  # day3 - day1

    def test_duration_without_recovery_calendar_days(self) -> None:
        # Peak at day 1 (120), never recovers by end (day 4)
        curve = [100.0, 120.0, 90.0, 100.0, 110.0]
        ts = [_ts(i) for i in range(5)]
        assert MetricsCalculator.calculate_drawdown_duration(curve, ts) == 3  # day4 - day1

    def test_duration_no_drawdown_is_zero(self) -> None:
        curve = [100.0, 110.0, 120.0]
        ts = [_ts(i) for i in range(3)]
        assert MetricsCalculator.calculate_drawdown_duration(curve, ts) == 0

    def test_duration_without_timestamps_returns_bars(self) -> None:
        # Backwards compat: no timestamps → bar count
        curve = [100.0, 120.0, 90.0, 130.0]
        assert MetricsCalculator.calculate_drawdown_duration(curve) == 2

    def test_duration_non_daily_timestamps(self) -> None:
        # 4-hour bars: peak at 0h, recovery at 12h → 0 calendar days (same day)
        from datetime import timedelta as td
        base = START
        curve = [100.0, 120.0, 90.0, 130.0]
        ts = [base, base + td(hours=4), base + td(hours=8), base + td(hours=12)]
        # (base+12h) - base = 0 days (timedelta.days = 0 for < 24h)
        assert MetricsCalculator.calculate_drawdown_duration(curve, ts) == 0


# ---------------------------------------------------------------------------
# _calculate_average_recovery_time (private, tested via snapshots)
# ---------------------------------------------------------------------------


class TestAverageRecoveryTime:
    def test_recovery_time_with_recovered_drawdowns(self) -> None:
        """Drawdown from day 1 peak to day 3 recovery → 2 calendar days."""
        snaps = [
            _make_snapshot(100.0, ts=_ts(0)),
            _make_snapshot(120.0, ts=_ts(1)),  # peak
            _make_snapshot(90.0,  ts=_ts(2)),  # trough
            _make_snapshot(130.0, ts=_ts(3)),  # recovery → duration = 3-1 = 2 days
        ]
        result = MetricsCalculator._calculate_average_recovery_time(snaps)
        assert result == pytest.approx(2.0)

    def test_recovery_time_with_only_unrecovered_returns_none(self) -> None:
        """Drawdown never recovers → None (unrecovered drawdowns excluded)."""
        snaps = [
            _make_snapshot(100.0, ts=_ts(0)),
            _make_snapshot(120.0, ts=_ts(1)),  # peak
            _make_snapshot(90.0,  ts=_ts(2)),  # never recovers
            _make_snapshot(100.0, ts=_ts(3)),  # still below 120
        ]
        assert MetricsCalculator._calculate_average_recovery_time(snaps) is None

    def test_recovery_time_average_of_multiple(self) -> None:
        """Two recovered drawdowns: (2 days + 4 days) / 2 = 3 days."""
        snaps = [
            _make_snapshot(100.0, ts=_ts(0)),
            _make_snapshot(120.0, ts=_ts(1)),  # peak 1
            _make_snapshot(90.0,  ts=_ts(2)),  # trough 1
            _make_snapshot(130.0, ts=_ts(3)),  # recovery 1 → 2 days
            _make_snapshot(150.0, ts=_ts(4)),  # new peak
            _make_snapshot(100.0, ts=_ts(5)),  # trough 2
            _make_snapshot(120.0, ts=_ts(6)),  # still in dd
            _make_snapshot(140.0, ts=_ts(7)),  # still in dd
            _make_snapshot(160.0, ts=_ts(8)),  # recovery 2 → 4 days (8-4)
        ]
        result = MetricsCalculator._calculate_average_recovery_time(snaps)
        assert result == pytest.approx(3.0)  # (2 + 4) / 2

    def test_recovery_time_no_drawdown_returns_none(self) -> None:
        snaps = [
            _make_snapshot(100.0, ts=_ts(0)),
            _make_snapshot(110.0, ts=_ts(1)),
            _make_snapshot(120.0, ts=_ts(2)),
        ]
        assert MetricsCalculator._calculate_average_recovery_time(snaps) is None


# ---------------------------------------------------------------------------
# calculate_sharpe
# ---------------------------------------------------------------------------


class TestCalculateSharpe:
    def test_sharpe_known_value(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        expected = (np.mean(returns) / np.std(returns, ddof=1)) * np.sqrt(365)
        actual = MetricsCalculator.calculate_sharpe(returns, 0.0, 365)
        assert actual == pytest.approx(expected)

    def test_sharpe_zero_volatility_returns_none(self) -> None:
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        assert MetricsCalculator.calculate_sharpe(returns, 0.0, 365) is None

    def test_sharpe_empty_returns_none(self) -> None:
        assert MetricsCalculator.calculate_sharpe(np.array([]), 0.0, 365) is None

    def test_sharpe_nonzero_rfr_subtracts_excess(self) -> None:
        returns = np.array([0.01, 0.02, 0.01, 0.02])
        rfr = 0.05  # 5% annual
        daily_rfr = rfr / 365
        excess = returns - daily_rfr
        expected = (np.mean(excess) / np.std(returns, ddof=1)) * np.sqrt(365)
        actual = MetricsCalculator.calculate_sharpe(returns, rfr, 365)
        assert actual == pytest.approx(expected)


# ---------------------------------------------------------------------------
# calculate_sortino
# ---------------------------------------------------------------------------


class TestCalculateSortino:
    def test_sortino_only_positive_returns_none(self) -> None:
        returns = np.array([0.01, 0.02, 0.03])
        assert MetricsCalculator.calculate_sortino(returns, 0.0, 365) is None

    def test_sortino_known_value(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        negative = returns[returns < 0]
        downside_std = float(np.std(negative, ddof=1))
        expected = (np.mean(returns) / downside_std) * np.sqrt(365)
        actual = MetricsCalculator.calculate_sortino(returns, 0.0, 365)
        assert actual == pytest.approx(expected)

    def test_sortino_empty_returns_none(self) -> None:
        assert MetricsCalculator.calculate_sortino(np.array([]), 0.0, 365) is None


# ---------------------------------------------------------------------------
# calculate_calmar
# ---------------------------------------------------------------------------


class TestCalculateCalmar:
    def test_calmar_basic(self) -> None:
        result = MetricsCalculator.calculate_calmar(0.30, -0.20)
        assert result == pytest.approx(1.5)  # 0.30 / 0.20

    def test_calmar_zero_dd_returns_none(self) -> None:
        assert MetricsCalculator.calculate_calmar(0.30, 0.0) is None

    def test_calmar_none_cagr_returns_none(self) -> None:
        assert MetricsCalculator.calculate_calmar(None, -0.20) is None

    def test_calmar_negative_cagr(self) -> None:
        result = MetricsCalculator.calculate_calmar(-0.10, -0.20)
        assert result == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# calculate_var
# ---------------------------------------------------------------------------


class TestCalculateVar:
    def test_var_95_at_5th_percentile(self) -> None:
        returns = np.linspace(-0.1, 0.1, 100)
        expected = float(np.percentile(returns, 5))
        assert MetricsCalculator.calculate_var(returns, 0.95) == pytest.approx(expected)

    def test_var_known_distribution(self) -> None:
        # All returns are -0.05 except one: var must be <= -0.05
        returns = np.array([-0.05] * 10 + [0.10])
        var = MetricsCalculator.calculate_var(returns, 0.95)
        assert var <= -0.05 + 1e-9

    def test_var_empty_returns_zero(self) -> None:
        assert MetricsCalculator.calculate_var(np.array([]), 0.95) == 0.0


# ---------------------------------------------------------------------------
# calculate_expected_shortfall
# ---------------------------------------------------------------------------


class TestCalculateExpectedShortfall:
    def test_es_is_average_of_worst_returns(self) -> None:
        returns = np.linspace(-0.1, 0.1, 100)
        var_val = float(np.percentile(returns, 5))
        expected_es = float(np.mean(returns[returns <= var_val]))
        es = MetricsCalculator.calculate_expected_shortfall(returns, 0.95)
        assert es == pytest.approx(expected_es)

    def test_es_more_extreme_than_var(self) -> None:
        returns = np.array([0.01, -0.02, 0.03, -0.04, -0.01, 0.02, -0.03])
        var = MetricsCalculator.calculate_var(returns, 0.95)
        es = MetricsCalculator.calculate_expected_shortfall(returns, 0.95)
        assert es <= var + 1e-9  # ES is at least as extreme as VaR

    def test_es_empty_returns_zero(self) -> None:
        assert MetricsCalculator.calculate_expected_shortfall(np.array([]), 0.95) == 0.0


# ---------------------------------------------------------------------------
# calculate_largest_losing_streak
# ---------------------------------------------------------------------------


class TestCalculateLargestLosingStreak:
    def test_streak_no_trades_returns_zero_zero(self) -> None:
        n, loss = MetricsCalculator.calculate_largest_losing_streak([])
        assert n == 0
        assert loss == 0.0

    def test_streak_with_alternating_wins_losses(self) -> None:
        trades = [
            _make_trade(100.0),
            _make_trade(-50.0),
            _make_trade(80.0),
            _make_trade(-30.0),
        ]
        n, loss = MetricsCalculator.calculate_largest_losing_streak(trades)
        assert n == 1

    def test_streak_with_consecutive_losses(self) -> None:
        trades = [
            _make_trade(100.0),
            _make_trade(-50.0),
            _make_trade(-30.0),
            _make_trade(80.0),
        ]
        n, loss = MetricsCalculator.calculate_largest_losing_streak(trades)
        assert n == 2
        assert loss == pytest.approx(-80.0)

    def test_streak_all_losses(self) -> None:
        trades = [_make_trade(-10.0), _make_trade(-20.0), _make_trade(-30.0)]
        n, loss = MetricsCalculator.calculate_largest_losing_streak(trades)
        assert n == 3
        assert loss == pytest.approx(-60.0)

    def test_streak_all_wins(self) -> None:
        trades = [_make_trade(100.0), _make_trade(200.0)]
        n, loss = MetricsCalculator.calculate_largest_losing_streak(trades)
        assert n == 0
        assert loss == 0.0


# ---------------------------------------------------------------------------
# calculate_profit_factor
# ---------------------------------------------------------------------------


class TestCalculateProfitFactor:
    def test_profit_factor_basic(self) -> None:
        trades = [_make_trade(100.0), _make_trade(50.0), _make_trade(-30.0)]
        pf = MetricsCalculator.calculate_profit_factor(trades)
        assert pf == pytest.approx(150.0 / 30.0)

    def test_profit_factor_no_losses_returns_none(self) -> None:
        trades = [_make_trade(100.0), _make_trade(50.0)]
        assert MetricsCalculator.calculate_profit_factor(trades) is None

    def test_profit_factor_empty_returns_none(self) -> None:
        assert MetricsCalculator.calculate_profit_factor([]) is None

    def test_profit_factor_above_one_when_wins_dominate(self) -> None:
        trades = [_make_trade(200.0), _make_trade(-100.0)]
        pf = MetricsCalculator.calculate_profit_factor(trades)
        assert pf is not None and pf > 1.0

    def test_profit_factor_below_one_when_losses_dominate(self) -> None:
        trades = [_make_trade(50.0), _make_trade(-200.0)]
        pf = MetricsCalculator.calculate_profit_factor(trades)
        assert pf is not None and pf < 1.0


# ---------------------------------------------------------------------------
# calculate_expectancy
# ---------------------------------------------------------------------------


class TestCalculateExpectancy:
    def test_expectancy_with_known_trades(self) -> None:
        trades = [
            _make_trade(100.0),
            _make_trade(-50.0),
            _make_trade(80.0),
            _make_trade(-30.0),
        ]
        exp = MetricsCalculator.calculate_expectancy(trades)
        assert exp == pytest.approx((100 - 50 + 80 - 30) / 4)  # 25.0

    def test_expectancy_no_trades_returns_none(self) -> None:
        assert MetricsCalculator.calculate_expectancy([]) is None

    def test_expectancy_all_wins_positive(self) -> None:
        trades = [_make_trade(100.0), _make_trade(200.0)]
        exp = MetricsCalculator.calculate_expectancy(trades)
        assert exp is not None and exp > 0

    def test_expectancy_all_losses_negative(self) -> None:
        trades = [_make_trade(-50.0), _make_trade(-30.0)]
        exp = MetricsCalculator.calculate_expectancy(trades)
        assert exp is not None and exp < 0


# ---------------------------------------------------------------------------
# calculate_time_in_market
# ---------------------------------------------------------------------------


class TestCalculateTimeInMarket:
    def test_always_in_market_returns_one(self) -> None:
        result = MetricsCalculator.calculate_time_in_market([True, True, True, True])
        assert result == pytest.approx(1.0)

    def test_never_in_market_returns_zero(self) -> None:
        result = MetricsCalculator.calculate_time_in_market([False, False, False])
        assert result == pytest.approx(0.0)

    def test_partial_returns_correct_pct(self) -> None:
        result = MetricsCalculator.calculate_time_in_market([True, False, True, False])
        assert result == pytest.approx(0.5)

    def test_empty_returns_zero(self) -> None:
        assert MetricsCalculator.calculate_time_in_market([]) == 0.0


# ---------------------------------------------------------------------------
# calculate() — main method
# ---------------------------------------------------------------------------


class TestCalculateMain:
    def test_calculate_handles_empty_trades_list(self) -> None:
        """No closed trades → operational metrics are None."""
        equity = [10_000.0 + i * 10 for i in range(400)]  # steady growth
        result = _make_result(equity, days=400)
        metrics = MetricsCalculator().calculate(result)

        assert metrics.num_trades == 0
        assert metrics.win_rate is None
        assert metrics.profit_factor is None
        assert metrics.avg_win_eur is None
        assert metrics.avg_loss_eur is None
        assert metrics.expectancy_eur is None
        assert metrics.largest_losing_streak_trades is None
        assert metrics.total_fees_paid_eur == 0.0

    def test_calculate_with_buy_and_hold_result(self) -> None:
        """BAH pattern: one position held all bars, no closed trades."""
        n = 400
        equity = [10_000.0 * (1 + 0.001 * i) for i in range(n)]
        has_pos = [True] * n
        result = _make_result(equity, days=n, has_positions=has_pos)
        metrics = MetricsCalculator().calculate(result)

        assert metrics.time_in_market_pct == pytest.approx(1.0)
        assert metrics.num_trades == 0
        assert metrics.max_drawdown_pct == 0.0  # monotonically increasing
        expected_total_return = (equity[-1] - equity[0]) / equity[0]
        assert metrics.total_return_pct == pytest.approx(expected_total_return)

    def test_calculate_total_return_consistency(self) -> None:
        equity = [10_000.0, 12_000.0]
        result = _make_result(equity, days=365)
        metrics = MetricsCalculator().calculate(result)
        assert metrics.total_return_eur == pytest.approx(2_000.0)
        assert metrics.total_return_pct == pytest.approx(0.2)
        assert metrics.final_capital_eur == pytest.approx(12_000.0)

    def test_calculate_with_trades_aggregates_fees(self) -> None:
        trades = [_make_trade(100.0, fees=15.0), _make_trade(-50.0, fees=15.0)]
        equity = [10_000.0, 10_100.0, 10_050.0]
        result = _make_result(equity, days=365, trades=trades)
        metrics = MetricsCalculator().calculate(result)
        assert metrics.total_fees_paid_eur == pytest.approx(30.0)
        assert metrics.num_trades == 2
        assert metrics.win_rate == pytest.approx(0.5)

    def test_calculate_with_benchmarks(self) -> None:
        """excess_return fields are populated when benchmarks are provided."""
        equity = [10_000.0, 12_000.0]  # +20% strategy
        bah = [10_000.0, 11_000.0]     # +10% BAH
        dca = [10_000.0, 10_500.0]     # +5% DCA
        result = _make_result(equity, days=365)
        metrics = MetricsCalculator().calculate(result, bah_equity_curve=bah, dca_equity_curve=dca)
        assert metrics.excess_return_vs_bah_pct == pytest.approx(0.10)
        assert metrics.excess_return_vs_dca_pct == pytest.approx(0.15)

    def test_calculate_no_benchmarks_gives_none(self) -> None:
        result = _make_result([10_000.0, 11_000.0], days=365)
        metrics = MetricsCalculator().calculate(result)
        assert metrics.excess_return_vs_bah_pct is None
        assert metrics.excess_return_vs_dca_pct is None

    def test_calculate_cagr_none_for_short_period(self) -> None:
        result = _make_result([10_000.0, 11_000.0], days=20)
        metrics = MetricsCalculator().calculate(result)
        assert metrics.cagr is None

    def test_calculate_cagr_computed_for_lump_sum_no_inflows(self) -> None:
        """CAGR is computed when initial > 0 and no inflows (lump-sum case)."""
        result = _make_result([10_000.0, 20_000.0], days=365)
        metrics = MetricsCalculator().calculate(result)
        assert metrics.cagr == pytest.approx(1.0)  # 100% CAGR over 1 year

    def test_calculate_cagr_none_when_initial_zero_with_inflows(self) -> None:
        """CAGR is None for DCA strategies (initial=0, periodic inflows)."""
        inflows = [_make_inflow(500.0, day_offset=i * 30) for i in range(12)]
        result = _make_result(
            [0.0] + [500.0 * (i + 1) for i in range(12)],
            days=365,
            initial_capital=0.0,
            processed_inflows=inflows,
        )
        metrics = MetricsCalculator().calculate(result)
        assert metrics.cagr is None

    def test_calculate_max_drawdown_captured(self) -> None:
        equity = [10_000.0, 12_000.0, 9_000.0, 11_000.0]
        result = _make_result(equity, days=365)
        metrics = MetricsCalculator().calculate(result)
        assert metrics.max_drawdown_pct == pytest.approx((9_000.0 - 12_000.0) / 12_000.0)
        assert metrics.max_drawdown_eur == pytest.approx(9_000.0 - 12_000.0)

    def test_calculate_time_in_market_from_snapshots(self) -> None:
        equity = [10_000.0, 11_000.0, 12_000.0, 11_500.0]
        has_pos = [False, True, True, False]
        result = _make_result(equity, days=365, has_positions=has_pos)
        metrics = MetricsCalculator().calculate(result)
        assert metrics.time_in_market_pct == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# total_return_pct with inflows — the DCA / STRAT-06 bug fix
# ---------------------------------------------------------------------------


class TestTotalReturnWithInflows:
    def test_total_return_pct_with_inflows_only(self) -> None:
        """initial=0, inflows=€51k total, final=€100k → pct = (100k-51k)/51k."""
        inflows = [_make_inflow(500.0, day_offset=i * 30) for i in range(102)]
        total_inflow = 102 * 500.0  # 51_000
        final = 100_000.0
        equity = [float(500 * (i + 1)) for i in range(102)]
        equity[-1] = final
        result = _make_result(
            equity,
            days=3060,
            initial_capital=0.0,
            processed_inflows=inflows,
        )
        metrics = MetricsCalculator().calculate(result)

        expected_pct = (final - total_inflow) / total_inflow
        assert metrics.total_return_pct == pytest.approx(expected_pct)
        assert metrics.total_return_eur == pytest.approx(final - total_inflow)
        assert metrics.total_invested_eur == pytest.approx(total_inflow)

    def test_total_return_pct_mixed_initial_and_inflows(self) -> None:
        """initial=€10k, inflows=€5k, final=€20k → pct = (20k-15k)/15k."""
        inflows = [_make_inflow(1_000.0, day_offset=i * 30) for i in range(5)]
        total_inflow = 5 * 1_000.0
        initial = 10_000.0
        total_invested = initial + total_inflow  # 15_000
        final = 20_000.0
        result = _make_result(
            [10_000.0, 20_000.0],
            days=365,
            initial_capital=initial,
            processed_inflows=inflows,
        )
        metrics = MetricsCalculator().calculate(result)

        expected_pct = (final - total_invested) / total_invested
        assert metrics.total_return_pct == pytest.approx(expected_pct)
        assert metrics.total_invested_eur == pytest.approx(total_invested)

    def test_total_return_pct_zero_total_invested_no_raise(self) -> None:
        """initial=0, no inflows → total_invested=0 → return 0.0, no exception."""
        result = _make_result(
            [0.0, 0.0],
            days=365,
            initial_capital=0.0,
            processed_inflows=[],
        )
        metrics = MetricsCalculator().calculate(result)
        assert metrics.total_return_pct == pytest.approx(0.0)
        assert metrics.total_invested_eur == pytest.approx(0.0)

    def test_total_return_pct_initial_only_unchanged(self) -> None:
        """No inflows: behaviour is identical to the pre-fix formula."""
        result = _make_result([10_000.0, 12_000.0], days=365, initial_capital=10_000.0)
        metrics = MetricsCalculator().calculate(result)
        assert metrics.total_return_pct == pytest.approx(0.2)
        assert metrics.total_return_eur == pytest.approx(2_000.0)
        assert metrics.total_invested_eur == pytest.approx(10_000.0)
