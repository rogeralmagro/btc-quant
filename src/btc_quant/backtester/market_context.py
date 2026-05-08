"""MarketContext: zero-look-ahead market data access for strategies."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

# Supported Binance-standard timeframes and their durations.
# Used to determine the finest available TF for get_current_btc_price().
_TF_DURATIONS: dict[str, timedelta] = {
    "1m":  timedelta(minutes=1),
    "3m":  timedelta(minutes=3),
    "5m":  timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h":  timedelta(hours=1),
    "2h":  timedelta(hours=2),
    "4h":  timedelta(hours=4),
    "6h":  timedelta(hours=6),
    "8h":  timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d":  timedelta(days=1),
    "3d":  timedelta(days=3),
    "1w":  timedelta(weeks=1),
}

_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    "timestamp_utc",
    "close_time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "is_synthetic",
})


def _assert_utc(dt: datetime, field_name: str) -> None:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{field_name} must be tz-aware (UTC)")
    if dt.utcoffset().total_seconds() != 0:
        raise ValueError(f"{field_name} timezone must be UTC (offset must be 0)")


class MarketContext:
    """Provides strategy-safe access to market data with zero look-ahead.

    Anti-look-ahead contract
    ========================
    A candle is visible if and only if::

        candle.close_time_utc <= self.current_time

    This means at ``current_time = T``:
    - The bar that just closed at T IS visible (``close_time == T``).
    - Any bar closing after T is NOT visible.

    The engine sets ``current_time`` by calling ``advance_to(bar.close_time_utc)``
    immediately before asking the strategy to evaluate. It is physically
    impossible for a strategy to read data from a bar that has not yet closed.

    Multi-timeframe
    ===============
    Data for multiple TFs is pre-loaded at construction. Strategies call
    ``get_history(tf, lookback)`` to access any registered TF.
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        current_time_utc: datetime,
        symbol: str = "BTCUSDT",
    ) -> None:
        _assert_utc(current_time_utc, "current_time_utc")

        for tf, df in data.items():
            missing = _REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(
                    f"DataFrame for timeframe {tf!r} is missing columns: {missing}"
                )
            if tf not in _TF_DURATIONS:
                raise ValueError(
                    f"Unknown timeframe {tf!r}. Supported: {sorted(_TF_DURATIONS)}"
                )

        self._data = data
        self._current_time_utc = current_time_utc
        self._symbol = symbol

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def current_time(self) -> datetime:
        """The point in time up to which data is visible."""
        return self._current_time_utc

    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def available_timeframes(self) -> list[str]:
        return list(self._data.keys())

    # ── Data access ──────────────────────────────────────────────────────────

    def get_history(self, timeframe: str, lookback: int) -> pd.DataFrame:
        """Return the last ``lookback`` closed candles for the given timeframe.

        Only candles with ``close_time_utc <= current_time`` are returned.
        May return fewer than ``lookback`` rows if insufficient history exists.

        Args:
            timeframe: e.g. "1d", "4h", "1w".
            lookback:  number of candles to return (must be >= 1).

        Returns:
            DataFrame sorted ascending by timestamp_utc, at most ``lookback`` rows.

        Raises:
            ValueError: if timeframe is not available or lookback < 1.
        """
        if timeframe not in self._data:
            raise ValueError(
                f"Timeframe {timeframe!r} not available. "
                f"Available: {self.available_timeframes}"
            )
        if lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {lookback}")

        df = self._data[timeframe]
        visible = df[df["close_time_utc"] <= self._current_time_utc]
        return visible.tail(lookback)

    def get_latest_close(self, timeframe: str) -> float:
        """Close price of the most recent closed candle for ``timeframe``.

        Raises:
            ValueError: if no closed candles exist for timeframe at current_time.
        """
        hist = self.get_history(timeframe, lookback=1)
        if hist.empty:
            raise ValueError(
                f"No closed candles for timeframe {timeframe!r} "
                f"at current_time={self._current_time_utc}"
            )
        return float(hist["close"].iloc[-1])

    def get_current_btc_price(self) -> float:
        """Reference price: close of the latest candle in the finest available TF.

        Used for mark-to-market calculations. Picks the shortest-duration
        timeframe registered in this context.
        """
        finest_tf = min(
            self._data.keys(),
            key=lambda tf: _TF_DURATIONS[tf],
        )
        return self.get_latest_close(finest_tf)

    # ── Time control (engine-only) ───────────────────────────────────────────

    def advance_to(self, new_time_utc: datetime) -> None:
        """Advance (or keep) the context's current time.

        Only the engine should call this, between bar iterations.

        Accepts:
        - ``new_time_utc > current_time``: normal forward advance.
        - ``new_time_utc == current_time``: idempotent; engine may re-evaluate.

        Raises:
            ValueError: if ``new_time_utc < current_time`` (time cannot go back).
        """
        _assert_utc(new_time_utc, "new_time_utc")
        if new_time_utc < self._current_time_utc:
            raise ValueError(
                f"Cannot advance backwards: new_time_utc ({new_time_utc}) < "
                f"current_time ({self._current_time_utc})"
            )
        self._current_time_utc = new_time_utc
