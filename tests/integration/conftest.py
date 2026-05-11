"""Shared fixtures and helpers for integration tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from btc_quant.backtester.inflow_scheduler import InflowScheduler
from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.strategies.strat06.config import DCAModulatedConfig

UTC = timezone.utc

_BASELINE = StrategyTag.STRAT_06_BASELINE
_BUFFER = StrategyTag.STRAT_06_BUFFER
_RESERVE = StrategyTag.STRAT_06_RESERVE


# ── Synthetic data ────────────────────────────────────────────────────────────


def make_synthetic_1d_dataset(
    n_days: int,
    start_date: datetime,
    price_series: list[float] | None = None,
    constant_price: float | None = None,
) -> pd.DataFrame:
    """Generate a synthetic 1D OHLCV DataFrame.

    Args:
        n_days: number of daily candles.
        start_date: UTC timestamp of the first candle's open.
        price_series: per-candle close prices (len must equal n_days).
        constant_price: all candles at this close price (default 50_000).

    close_time_utc[i] = timestamp_utc[i] + 1 day - 1 ms  (= 23:59:59.999 UTC).
    open[i] = close[i-1] for i > 0, else close[0].
    """
    period = timedelta(days=1)
    ms = timedelta(milliseconds=1)

    if price_series is not None:
        if len(price_series) != n_days:
            raise ValueError(
                f"price_series length ({len(price_series)}) must equal n_days ({n_days})"
            )
        closes = list(price_series)
    else:
        closes = [float(constant_price or 50_000.0)] * n_days

    timestamps = [start_date + i * period for i in range(n_days)]
    close_times = [ts + period - ms for ts in timestamps]
    opens = [closes[0]] + closes[:-1]

    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(timestamps, utc=True),
            "close_time_utc": pd.to_datetime(close_times, utc=True),
            "open": opens,
            "high": [c * 1.005 for c in closes],
            "low": [c * 0.995 for c in closes],
            "close": closes,
            "volume": [1_000.0] * n_days,
            "is_synthetic": [False] * n_days,
        }
    )


@pytest.fixture(name="make_synthetic_1d_dataset")
def make_synthetic_1d_dataset_fixture():
    """Pytest fixture wrapping make_synthetic_1d_dataset as a factory."""
    return make_synthetic_1d_dataset


# ── STRAT-06 inflow helpers ───────────────────────────────────────────────────


def make_strat06_inflows(
    config: DCAModulatedConfig,
    start: datetime,
    end: datetime,
    day_of_month: int = 1,
):
    """Generate monthly inflows split across the three STRAT-06 pools.

    Calls InflowScheduler.monthly() once per pool with the proportional amount:
        BASELINE: monthly_inflow_eur * baseline_pct
        BUFFER:   monthly_inflow_eur * buffer_pct
        RESERVE:  monthly_inflow_eur * reserve_pct

    Returns a single sorted list of CapitalInflow events.
    """
    baseline_inflows = InflowScheduler.monthly(
        amount_eur=config.monthly_inflow_eur * config.baseline_pct,
        target_strategy=_BASELINE,
        start_date=start,
        end_date=end,
        day_of_month=day_of_month,
    )
    buffer_inflows = InflowScheduler.monthly(
        amount_eur=config.monthly_inflow_eur * config.buffer_pct,
        target_strategy=_BUFFER,
        start_date=start,
        end_date=end,
        day_of_month=day_of_month,
    )
    reserve_inflows = InflowScheduler.monthly(
        amount_eur=config.monthly_inflow_eur * config.reserve_pct,
        target_strategy=_RESERVE,
        start_date=start,
        end_date=end,
        day_of_month=day_of_month,
    )
    return sorted(
        baseline_inflows + buffer_inflows + reserve_inflows,
        key=lambda x: x.timestamp_utc,
    )
