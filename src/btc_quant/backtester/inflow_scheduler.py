"""InflowScheduler: generates periodic capital inflow schedules for the backtester.

A CapitalInflow represents a discrete cash injection into a specific strategy
pool. The BacktestEngine reads a list of these and processes each inflow when
the bar timestamp reaches or passes the inflow's timestamp_utc.

Typical usage for STRAT-06:
    inflows = InflowScheduler.monthly(
        amount_eur=500.0,
        target_strategy=StrategyTag.STRAT_06_BASELINE,
        start_date=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    engine = BacktestEngine(..., inflow_schedule=inflows)
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timezone

from btc_quant.backtester.models.enums import StrategyTag


@dataclass(frozen=True)
class CapitalInflow:
    """A single scheduled capital injection into a strategy pool.

    Attributes:
        timestamp_utc: when the capital becomes available (tz-aware UTC).
        amount_eur: EUR to inject. Must be > 0.
        target_strategy: pool that receives the capital.
    """

    timestamp_utc: datetime
    amount_eur: float
    target_strategy: StrategyTag

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be tz-aware (UTC)")
        if self.timestamp_utc.utcoffset().total_seconds() != 0:
            raise ValueError("timestamp_utc timezone must be UTC (offset must be 0)")
        if self.amount_eur <= 0:
            raise ValueError(f"amount_eur must be > 0, got {self.amount_eur}")


class InflowScheduler:
    """Factory for common CapitalInflow schedule patterns."""

    @staticmethod
    def monthly(
        amount_eur: float,
        target_strategy: StrategyTag,
        start_date: datetime,
        end_date: datetime,
        day_of_month: int = 1,
    ) -> list[CapitalInflow]:
        """Generate monthly inflows on a fixed day of each month.

        The first inflow is the first occurrence of day_of_month on or after
        start_date. Inflows are generated for every month where the target
        date falls within [start_date, end_date].

        Args:
            amount_eur: EUR per inflow. Must be > 0.
            target_strategy: pool receiving the capital.
            start_date: earliest possible inflow date (tz-aware UTC).
            end_date: latest allowed inflow date (tz-aware UTC).
            day_of_month: day on which the inflow occurs (1–28).

        Returns:
            Chronologically ordered list of CapitalInflow events.

        Raises:
            ValueError: if amount_eur <= 0, day_of_month not in [1, 28],
                        start_date >= end_date, or either date is naive.
        """
        InflowScheduler._validate_inputs(amount_eur, start_date, end_date, day_of_month)

        inflows: list[CapitalInflow] = []
        year = start_date.year
        month = start_date.month

        while True:
            ts = datetime(year, month, day_of_month, tzinfo=timezone.utc)
            if ts > end_date:
                break
            if ts >= start_date:
                inflows.append(CapitalInflow(
                    timestamp_utc=ts,
                    amount_eur=amount_eur,
                    target_strategy=target_strategy,
                ))
            # Advance to next month
            if month == 12:
                month = 1
                year += 1
            else:
                month += 1

        return inflows

    @staticmethod
    def initial_plus_monthly(
        initial_amount_eur: float,
        monthly_amount_eur: float,
        target_strategy: StrategyTag,
        start_date: datetime,
        end_date: datetime,
        day_of_month: int = 1,
    ) -> list[CapitalInflow]:
        """One-time initial inflow at start_date, then monthly recurring inflows.

        The initial inflow fires exactly at start_date, regardless of
        day_of_month. The monthly inflows begin from the first occurrence
        of day_of_month strictly after start_date.

        Args:
            initial_amount_eur: one-time EUR injection at start_date.
            monthly_amount_eur: EUR per monthly inflow.
            target_strategy: pool receiving all capital.
            start_date: date of the initial inflow (tz-aware UTC).
            end_date: latest allowed date for monthly inflows (tz-aware UTC).
            day_of_month: day for monthly inflows (1–28).

        Returns:
            Chronologically ordered list starting with the initial inflow.

        Raises:
            ValueError: same conditions as monthly(), plus initial_amount_eur <= 0.
        """
        if initial_amount_eur <= 0:
            raise ValueError(f"initial_amount_eur must be > 0, got {initial_amount_eur}")

        InflowScheduler._validate_inputs(
            monthly_amount_eur, start_date, end_date, day_of_month
        )

        initial = CapitalInflow(
            timestamp_utc=start_date,
            amount_eur=initial_amount_eur,
            target_strategy=target_strategy,
        )

        # Monthly inflows start from the day_of_month >= start_date.
        # If start_date itself is day_of_month, begin the monthly series from
        # the following month to avoid a double-injection on day 1.
        year = start_date.year
        month = start_date.month

        first_monthly = datetime(year, month, day_of_month, tzinfo=timezone.utc)
        if first_monthly <= start_date:
            # Advance one month
            if month == 12:
                month = 1
                year += 1
            else:
                month += 1

        monthly_start = datetime(year, month, day_of_month, tzinfo=timezone.utc)
        monthly_inflows = InflowScheduler.monthly(
            amount_eur=monthly_amount_eur,
            target_strategy=target_strategy,
            start_date=monthly_start,
            end_date=end_date,
            day_of_month=day_of_month,
        )

        return [initial] + monthly_inflows

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _validate_inputs(
        amount_eur: float,
        start_date: datetime,
        end_date: datetime,
        day_of_month: int,
    ) -> None:
        if amount_eur <= 0:
            raise ValueError(f"amount_eur must be > 0, got {amount_eur}")
        if start_date.tzinfo is None:
            raise ValueError("start_date must be tz-aware (UTC)")
        if end_date.tzinfo is None:
            raise ValueError("end_date must be tz-aware (UTC)")
        if start_date >= end_date:
            raise ValueError(
                f"start_date ({start_date}) must be before end_date ({end_date})"
            )
        if not (1 <= day_of_month <= 28):
            raise ValueError(
                f"day_of_month must be between 1 and 28 (got {day_of_month}). "
                "Values 29–31 are excluded to guarantee validity in all months."
            )
