"""Unit tests for InflowScheduler and CapitalInflow."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from btc_quant.backtester.inflow_scheduler import CapitalInflow, InflowScheduler
from btc_quant.backtester.models.enums import StrategyTag

UTC = timezone.utc
TAG = StrategyTag.STRAT_06_BASELINE


# ---------------------------------------------------------------------------
# CapitalInflow validation
# ---------------------------------------------------------------------------

class TestCapitalInflow:
    def test_valid_inflow(self) -> None:
        cf = CapitalInflow(datetime(2020, 1, 1, tzinfo=UTC), 500.0, TAG)
        assert cf.amount_eur == 500.0

    def test_naive_datetime_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            CapitalInflow(datetime(2020, 1, 1), 500.0, TAG)

    def test_zero_amount_raises(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            CapitalInflow(datetime(2020, 1, 1, tzinfo=UTC), 0.0, TAG)

    def test_negative_amount_raises(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            CapitalInflow(datetime(2020, 1, 1, tzinfo=UTC), -100.0, TAG)


# ---------------------------------------------------------------------------
# InflowScheduler.monthly
# ---------------------------------------------------------------------------

class TestMonthlyScheduler:
    def test_monthly_inflow_creates_correct_number_of_events(self) -> None:
        """start=2020-01-15, end=2020-12-31, day=1 → 11 inflows (Feb–Dec)."""
        inflows = InflowScheduler.monthly(
            amount_eur=500.0,
            target_strategy=TAG,
            start_date=datetime(2020, 1, 15, tzinfo=UTC),
            end_date=datetime(2020, 12, 31, tzinfo=UTC),
            day_of_month=1,
        )
        assert len(inflows) == 11

    def test_monthly_inflow_uses_correct_day(self) -> None:
        inflows = InflowScheduler.monthly(
            amount_eur=100.0,
            target_strategy=TAG,
            start_date=datetime(2020, 1, 1, tzinfo=UTC),
            end_date=datetime(2020, 12, 31, tzinfo=UTC),
            day_of_month=15,
        )
        for inflow in inflows:
            assert inflow.timestamp_utc.day == 15

    def test_monthly_inflow_amount_correct(self) -> None:
        inflows = InflowScheduler.monthly(
            amount_eur=750.0,
            target_strategy=TAG,
            start_date=datetime(2020, 1, 1, tzinfo=UTC),
            end_date=datetime(2020, 6, 30, tzinfo=UTC),
        )
        for inflow in inflows:
            assert inflow.amount_eur == pytest.approx(750.0)

    def test_monthly_inflow_chronological_order(self) -> None:
        inflows = InflowScheduler.monthly(
            amount_eur=100.0,
            target_strategy=TAG,
            start_date=datetime(2020, 1, 1, tzinfo=UTC),
            end_date=datetime(2021, 6, 30, tzinfo=UTC),
        )
        for i in range(len(inflows) - 1):
            assert inflows[i].timestamp_utc < inflows[i + 1].timestamp_utc

    def test_monthly_start_on_day_of_month_includes_it(self) -> None:
        """start_date exactly on day_of_month → first inflow is start_date."""
        inflows = InflowScheduler.monthly(
            amount_eur=100.0,
            target_strategy=TAG,
            start_date=datetime(2020, 3, 1, tzinfo=UTC),
            end_date=datetime(2020, 5, 31, tzinfo=UTC),
            day_of_month=1,
        )
        assert inflows[0].timestamp_utc == datetime(2020, 3, 1, tzinfo=UTC)
        assert len(inflows) == 3  # Mar, Apr, May

    def test_monthly_end_date_exclusive_of_later(self) -> None:
        """Inflow whose date > end_date is not included."""
        inflows = InflowScheduler.monthly(
            amount_eur=100.0,
            target_strategy=TAG,
            start_date=datetime(2020, 1, 1, tzinfo=UTC),
            end_date=datetime(2020, 3, 14, tzinfo=UTC),
            day_of_month=15,
        )
        # Jan 15, Feb 15 are within range; Mar 15 exceeds end_date
        assert len(inflows) == 2

    def test_monthly_all_timestamps_utc(self) -> None:
        inflows = InflowScheduler.monthly(
            amount_eur=100.0,
            target_strategy=TAG,
            start_date=datetime(2020, 1, 1, tzinfo=UTC),
            end_date=datetime(2020, 6, 30, tzinfo=UTC),
        )
        for inflow in inflows:
            assert inflow.timestamp_utc.tzinfo is not None

    def test_monthly_crosses_year_boundary(self) -> None:
        inflows = InflowScheduler.monthly(
            amount_eur=100.0,
            target_strategy=TAG,
            start_date=datetime(2020, 11, 1, tzinfo=UTC),
            end_date=datetime(2021, 2, 28, tzinfo=UTC),
        )
        months = [i.timestamp_utc.month for i in inflows]
        assert months == [11, 12, 1, 2]


# ---------------------------------------------------------------------------
# InflowScheduler validation errors
# ---------------------------------------------------------------------------

class TestSchedulerValidation:
    def test_invalid_amount_raises(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            InflowScheduler.monthly(
                amount_eur=-100.0,
                target_strategy=TAG,
                start_date=datetime(2020, 1, 1, tzinfo=UTC),
                end_date=datetime(2020, 12, 31, tzinfo=UTC),
            )

    def test_invalid_day_of_month_29_raises(self) -> None:
        with pytest.raises(ValueError, match="1 and 28"):
            InflowScheduler.monthly(
                amount_eur=100.0,
                target_strategy=TAG,
                start_date=datetime(2020, 1, 1, tzinfo=UTC),
                end_date=datetime(2020, 12, 31, tzinfo=UTC),
                day_of_month=29,
            )

    def test_invalid_day_of_month_31_raises(self) -> None:
        with pytest.raises(ValueError, match="1 and 28"):
            InflowScheduler.monthly(
                amount_eur=100.0,
                target_strategy=TAG,
                start_date=datetime(2020, 1, 1, tzinfo=UTC),
                end_date=datetime(2020, 12, 31, tzinfo=UTC),
                day_of_month=31,
            )

    def test_end_before_start_raises(self) -> None:
        with pytest.raises(ValueError, match="before end_date"):
            InflowScheduler.monthly(
                amount_eur=100.0,
                target_strategy=TAG,
                start_date=datetime(2020, 6, 1, tzinfo=UTC),
                end_date=datetime(2020, 1, 1, tzinfo=UTC),
            )

    def test_equal_start_end_raises(self) -> None:
        ts = datetime(2020, 1, 1, tzinfo=UTC)
        with pytest.raises(ValueError, match="before end_date"):
            InflowScheduler.monthly(100.0, TAG, ts, ts)

    def test_naive_start_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            InflowScheduler.monthly(
                amount_eur=100.0,
                target_strategy=TAG,
                start_date=datetime(2020, 1, 1),  # naive
                end_date=datetime(2020, 12, 31, tzinfo=UTC),
            )

    def test_naive_end_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            InflowScheduler.monthly(
                amount_eur=100.0,
                target_strategy=TAG,
                start_date=datetime(2020, 1, 1, tzinfo=UTC),
                end_date=datetime(2020, 12, 31),  # naive
            )


# ---------------------------------------------------------------------------
# InflowScheduler.initial_plus_monthly
# ---------------------------------------------------------------------------

class TestInitialPlusMonthly:
    def test_initial_plus_monthly_includes_initial(self) -> None:
        inflows = InflowScheduler.initial_plus_monthly(
            initial_amount_eur=5_000.0,
            monthly_amount_eur=500.0,
            target_strategy=TAG,
            start_date=datetime(2020, 1, 10, tzinfo=UTC),
            end_date=datetime(2020, 6, 30, tzinfo=UTC),
            day_of_month=1,
        )
        assert inflows[0].timestamp_utc == datetime(2020, 1, 10, tzinfo=UTC)
        assert inflows[0].amount_eur == pytest.approx(5_000.0)

    def test_initial_plus_monthly_subsequent_are_monthly(self) -> None:
        inflows = InflowScheduler.initial_plus_monthly(
            initial_amount_eur=5_000.0,
            monthly_amount_eur=500.0,
            target_strategy=TAG,
            start_date=datetime(2020, 1, 10, tzinfo=UTC),
            end_date=datetime(2020, 6, 30, tzinfo=UTC),
            day_of_month=1,
        )
        for inflow in inflows[1:]:
            assert inflow.amount_eur == pytest.approx(500.0)
            assert inflow.timestamp_utc.day == 1

    def test_initial_on_day_of_month_no_double_injection(self) -> None:
        """If start_date is day_of_month, monthly series starts next month."""
        inflows = InflowScheduler.initial_plus_monthly(
            initial_amount_eur=5_000.0,
            monthly_amount_eur=500.0,
            target_strategy=TAG,
            start_date=datetime(2020, 1, 1, tzinfo=UTC),
            end_date=datetime(2020, 4, 30, tzinfo=UTC),
            day_of_month=1,
        )
        # Initial on Jan 1, then monthly: Feb 1, Mar 1, Apr 1 → 4 total
        assert len(inflows) == 4
        assert inflows[0].timestamp_utc == datetime(2020, 1, 1, tzinfo=UTC)
        assert inflows[1].timestamp_utc == datetime(2020, 2, 1, tzinfo=UTC)

    def test_initial_amount_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            InflowScheduler.initial_plus_monthly(
                initial_amount_eur=0.0,
                monthly_amount_eur=500.0,
                target_strategy=TAG,
                start_date=datetime(2020, 1, 1, tzinfo=UTC),
                end_date=datetime(2020, 12, 31, tzinfo=UTC),
            )
