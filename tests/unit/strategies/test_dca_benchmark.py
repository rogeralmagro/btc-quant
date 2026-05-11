"""Unit tests for DCABenchmarkStrategy."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.backtester.models.pool import CapitalPool
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.strategies.dca_benchmark import DCABenchmarkStrategy

UTC = timezone.utc

# Monday 12:00 UTC (close_time of a 1d bar = 23:59:59 UTC, but weekday check
# uses whatever hour the bar close lands on; synthetic 1d bars close at 23:59)
MONDAY = datetime(2024, 1, 8, 23, 59, 59, tzinfo=UTC)   # weekday() == 0
TUESDAY = datetime(2024, 1, 9, 23, 59, 59, tzinfo=UTC)  # weekday() == 1
WEDNESDAY = datetime(2024, 1, 10, 23, 59, 59, tzinfo=UTC)

WEEKLY = round(500.0 * 12 / 52, 2)  # 115.38


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_context(dt: datetime) -> MagicMock:
    ctx = MagicMock()
    ctx.current_time = dt
    return ctx


def _make_portfolio(cash: float) -> Portfolio:
    pool = CapitalPool(
        strategy_tag=StrategyTag.DCA_BENCHMARK,
        cash_eur=cash,
        total_invested_eur=cash,
    )
    return Portfolio(pools={StrategyTag.DCA_BENCHMARK: pool})


def _make_strategy(**kwargs) -> DCABenchmarkStrategy:
    defaults = {"weekly_amount_eur": WEEKLY}
    defaults.update(kwargs)
    return DCABenchmarkStrategy(**defaults)


# ── Construction ──────────────────────────────────────────────────────────────


class TestDCABenchmarkConstruction:
    def test_default_attributes(self):
        s = _make_strategy()
        assert s.strategy_tag == StrategyTag.DCA_BENCHMARK
        assert s._weekly_amount_eur == WEEKLY
        assert s._weekday == 0
        assert s._symbol == "BTCUSDT"

    def test_custom_weekday(self):
        s = _make_strategy(weekday=2)
        assert s._weekday == 2

    def test_zero_weekly_amount_raises(self):
        with pytest.raises(ValueError, match="weekly_amount_eur"):
            _make_strategy(weekly_amount_eur=0.0)

    def test_negative_weekly_amount_raises(self):
        with pytest.raises(ValueError, match="weekly_amount_eur"):
            _make_strategy(weekly_amount_eur=-10.0)

    def test_invalid_weekday_raises(self):
        with pytest.raises(ValueError, match="weekday"):
            _make_strategy(weekday=7)
        with pytest.raises(ValueError, match="weekday"):
            _make_strategy(weekday=-1)


# ── Signal generation ─────────────────────────────────────────────────────────


class TestDCABenchmarkSignals:
    def test_buy_on_correct_weekday(self):
        s = _make_strategy()
        ctx = _make_context(MONDAY)
        port = _make_portfolio(cash=500.0)
        signals = s.generate_signals(ctx, port)
        assert len(signals) == 1
        assert signals[0].quote_amount_eur == pytest.approx(WEEKLY)
        assert signals[0].reason_tag == "dca_weekly"

    def test_no_buy_on_other_weekday(self):
        s = _make_strategy()
        for day in (TUESDAY, WEDNESDAY):
            signals = s.generate_signals(_make_context(day), _make_portfolio(500.0))
            assert signals == [], f"expected no signal on {day.strftime('%A')}"

    def test_no_buy_twice_in_same_iso_week(self):
        s = _make_strategy()
        port = _make_portfolio(cash=500.0)
        # First Monday: buys
        s.generate_signals(_make_context(MONDAY), port)
        # Same ISO week (still Monday after forced reset of mock):
        # use a second Monday in the *same* ISO week — impossible in reality,
        # so simulate by checking _last_buy_week guard directly.
        signals2 = s.generate_signals(_make_context(MONDAY), port)
        assert signals2 == []

    def test_no_signal_when_cash_below_threshold(self):
        s = _make_strategy()
        port = _make_portfolio(cash=0.50)  # below _MIN_ORDER_EUR=1.0
        signals = s.generate_signals(_make_context(MONDAY), port)
        assert signals == []

    def test_no_signal_when_cash_zero(self):
        s = _make_strategy()
        port = _make_portfolio(cash=0.0)
        signals = s.generate_signals(_make_context(MONDAY), port)
        assert signals == []

    def test_buys_available_cash_when_below_weekly_target(self):
        """If cash < weekly_amount_eur, buys what's available."""
        s = _make_strategy(weekly_amount_eur=115.38)
        port = _make_portfolio(cash=50.0)
        signals = s.generate_signals(_make_context(MONDAY), port)
        assert len(signals) == 1
        assert signals[0].quote_amount_eur == pytest.approx(50.0)

    def test_buys_on_consecutive_mondays_in_different_iso_weeks(self):
        s = _make_strategy()
        port = _make_portfolio(cash=1000.0)
        # Monday Jan 8 (ISO week 2) and Monday Jan 15 (ISO week 3)
        mon1 = datetime(2024, 1, 8, 23, 59, 59, tzinfo=UTC)
        mon2 = datetime(2024, 1, 15, 23, 59, 59, tzinfo=UTC)
        sig1 = s.generate_signals(_make_context(mon1), port)
        sig2 = s.generate_signals(_make_context(mon2), port)
        assert len(sig1) == 1
        assert len(sig2) == 1

    def test_weekday_config_respected(self):
        """Strategy configured for Wednesday should not fire on Monday."""
        s = _make_strategy(weekday=2)  # Wednesday
        port = _make_portfolio(cash=500.0)
        assert s.generate_signals(_make_context(MONDAY), port) == []
        assert len(s.generate_signals(_make_context(WEDNESDAY), port)) == 1

    def test_signal_strategy_tag_is_dca_benchmark(self):
        s = _make_strategy()
        port = _make_portfolio(cash=500.0)
        signals = s.generate_signals(_make_context(MONDAY), port)
        assert signals[0].strategy_tag == StrategyTag.DCA_BENCHMARK

    def test_required_timeframes(self):
        assert _make_strategy().required_timeframes() == ["1d"]

    def test_primary_timeframe(self):
        assert _make_strategy().primary_timeframe() == "1d"

    def test_allows_accumulation(self):
        assert _make_strategy().allows_position_accumulation() is True
