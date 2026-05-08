"""Shared fixtures for backtester unit tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Callable

import pandas as pd
import pytest

UTC = timezone.utc

_TF_PERIODS: dict[str, timedelta] = {
    "1h": timedelta(hours=1),
    "4h": timedelta(hours=4),
    "1d": timedelta(days=1),
    "1w": timedelta(weeks=1),
}


def make_ohlcv_df(n: int, start: datetime, tf: str) -> pd.DataFrame:
    """Generate n deterministic OHLCV candles.

    close[i] = 10_000 + i * 100  (strictly increasing, unique per candle)
    close_time_utc[i] = timestamp_utc[i] + period - 1 ms
    """
    period = _TF_PERIODS[tf]
    ms = timedelta(milliseconds=1)

    timestamps = [start + i * period for i in range(n)]
    close_times = [ts + period - ms for ts in timestamps]
    closes = [10_000.0 + i * 100.0 for i in range(n)]

    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(timestamps, utc=True),
            "close_time_utc": pd.to_datetime(close_times, utc=True),
            "open": [c - 50.0 for c in closes],
            "high": [c + 200.0 for c in closes],
            "low": [c - 150.0 for c in closes],
            "close": closes,
            "volume": [1_000.0] * n,
            "is_synthetic": [False] * n,
        }
    )
    return df


@pytest.fixture
def daily_df() -> pd.DataFrame:
    """200 daily candles from 2020-01-01."""
    return make_ohlcv_df(200, datetime(2020, 1, 1, tzinfo=UTC), "1d")


@pytest.fixture
def hourly_4h_df() -> pd.DataFrame:
    """200 four-hour candles from 2020-01-01 (≈ 33 days)."""
    return make_ohlcv_df(200, datetime(2020, 1, 1, tzinfo=UTC), "4h")
