"""Unit tests for MetricsReport and StrategyMetrics dataclasses."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from btc_quant.backtester.metrics.report import MetricsReport, StrategyMetrics
from btc_quant.backtester.models.enums import StrategyTag

UTC = timezone.utc
TAG = StrategyTag.STRAT_07_TACTICAL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_strategy_metrics(**overrides) -> StrategyMetrics:
    defaults: dict = {
        "strategy_tag": TAG,
        "num_trades": 0,
        "win_rate": None,
        "profit_factor": None,
        "total_pnl_eur": 0.0,
        "avg_pnl_per_trade_eur": None,
        "avg_r_multiple": None,
        "contribution_to_total_return_pct": 0.0,
        "time_in_market_pct": 0.0,
    }
    defaults.update(overrides)
    return StrategyMetrics(**defaults)


def _minimal_report(**overrides) -> MetricsReport:
    defaults: dict = {
        "start_date_utc": datetime(2024, 1, 1, tzinfo=UTC),
        "end_date_utc": datetime(2024, 12, 31, tzinfo=UTC),
        "duration_days": 365,
        "initial_capital_eur": 10_000.0,
        "total_invested_eur": 10_000.0,
        "final_capital_eur": 12_000.0,
        "total_return_eur": 2_000.0,
        "total_return_pct": 0.2,
        "cagr": 0.2,
        "time_in_market_pct": 0.5,
        "annualized_volatility": 0.3,
        "max_drawdown_pct": -0.1,
        "max_drawdown_eur": -1_000.0,
        "max_drawdown_duration_days": 10,
        "average_recovery_time_days": None,
        "largest_losing_streak_trades": None,
        "largest_losing_streak_eur": None,
        "var_95_daily": -0.02,
        "expected_shortfall_95_daily": -0.03,
        "sharpe_ratio": 1.0,
        "sortino_ratio": 1.5,
        "calmar_ratio": 2.0,
        "num_trades": 0,
        "win_rate": None,
        "profit_factor": None,
        "avg_win_eur": None,
        "avg_loss_eur": None,
        "expectancy_eur": None,
        "avg_r_multiple_realized": None,
        "total_fees_paid_eur": 0.0,
        "metrics_by_strategy": {},
        "avg_cost_basis_eur_per_btc": None,
        "total_btc_accumulated": None,
        "btc_per_eur_invested": None,
        "excess_return_vs_bah_pct": None,
        "excess_return_vs_dca_pct": None,
    }
    defaults.update(overrides)
    return MetricsReport(**defaults)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_metrics_report_immutable(self) -> None:
        report = _minimal_report()
        with pytest.raises(FrozenInstanceError):
            report.total_return_pct = 0.99  # type: ignore[misc]

    def test_strategy_metrics_immutable(self) -> None:
        sm = _minimal_strategy_metrics()
        with pytest.raises(FrozenInstanceError):
            sm.num_trades = 99  # type: ignore[misc]

    def test_metrics_report_fields_are_readable(self) -> None:
        report = _minimal_report()
        assert report.initial_capital_eur == 10_000.0
        assert report.final_capital_eur == 12_000.0
        assert report.total_return_eur == 2_000.0


# ---------------------------------------------------------------------------
# None propagation for no-trade scenarios
# ---------------------------------------------------------------------------


class TestNoTradesNonePropagation:
    def test_metrics_with_no_trades_has_none_operational_metrics(self) -> None:
        report = _minimal_report(num_trades=0)
        assert report.win_rate is None
        assert report.profit_factor is None
        assert report.avg_win_eur is None
        assert report.avg_loss_eur is None
        assert report.expectancy_eur is None
        assert report.avg_r_multiple_realized is None
        assert report.largest_losing_streak_trades is None
        assert report.largest_losing_streak_eur is None

    def test_strategy_metrics_no_trades_optional_fields_none(self) -> None:
        sm = _minimal_strategy_metrics(num_trades=0)
        assert sm.win_rate is None
        assert sm.profit_factor is None
        assert sm.avg_pnl_per_trade_eur is None
        assert sm.avg_r_multiple is None

    def test_cagr_none_when_period_too_short(self) -> None:
        report = _minimal_report(
            start_date_utc=datetime(2024, 1, 1, tzinfo=UTC),
            end_date_utc=datetime(2024, 1, 20, tzinfo=UTC),
            duration_days=19,
            cagr=None,
        )
        assert report.cagr is None

    def test_sharpe_none_when_no_volatility(self) -> None:
        report = _minimal_report(sharpe_ratio=None, sortino_ratio=None)
        assert report.sharpe_ratio is None
        assert report.sortino_ratio is None

    def test_calmar_none_when_no_drawdown(self) -> None:
        report = _minimal_report(
            max_drawdown_pct=0.0,
            max_drawdown_eur=0.0,
            calmar_ratio=None,
        )
        assert report.calmar_ratio is None

    def test_benchmark_fields_none_when_not_provided(self) -> None:
        report = _minimal_report()
        assert report.excess_return_vs_bah_pct is None
        assert report.excess_return_vs_dca_pct is None

    def test_strat06_fields_none_when_not_applicable(self) -> None:
        report = _minimal_report()
        assert report.avg_cost_basis_eur_per_btc is None
        assert report.total_btc_accumulated is None
        assert report.btc_per_eur_invested is None


# ---------------------------------------------------------------------------
# Field values round-trip
# ---------------------------------------------------------------------------


class TestFieldValues:
    def test_duration_days_stored(self) -> None:
        report = _minimal_report(duration_days=730)
        assert report.duration_days == 730

    def test_max_drawdown_pct_negative(self) -> None:
        report = _minimal_report(max_drawdown_pct=-0.25)
        assert report.max_drawdown_pct == -0.25

    def test_time_in_market_pct_full(self) -> None:
        report = _minimal_report(time_in_market_pct=1.0)
        assert report.time_in_market_pct == 1.0

    def test_metrics_by_strategy_stored(self) -> None:
        sm = _minimal_strategy_metrics(num_trades=5, total_pnl_eur=500.0)
        report = _minimal_report(metrics_by_strategy={TAG: sm})
        assert report.metrics_by_strategy[TAG].num_trades == 5
        assert report.metrics_by_strategy[TAG].total_pnl_eur == 500.0

    def test_benchmark_values_stored(self) -> None:
        report = _minimal_report(
            excess_return_vs_bah_pct=0.05,
            excess_return_vs_dca_pct=-0.02,
        )
        assert report.excess_return_vs_bah_pct == pytest.approx(0.05)
        assert report.excess_return_vs_dca_pct == pytest.approx(-0.02)
