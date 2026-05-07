"""Data quality utilities for strategy-layer consumption.

Strategies use these helpers to decide whether a given lookback window
contains synthetic (forward-filled) candles and, if so, to abstain from
generating signals or placing trades.

The exclusion logic lives here — NOT in the indicator layer. Indicators
compute over all candles (real and synthetic alike) so their output is
always defined. The strategy checks is_period_clean() before acting on
indicator values.
"""

from __future__ import annotations

import pandas as pd


def is_period_clean(
    df: pd.DataFrame,
    end_idx: int,
    lookback: int = 20,
) -> bool:
    """Return True if the lookback window ending at end_idx contains no synthetic candles.

    Strategies call this before generating signals: if any candle in the
    window is synthetic (is_synthetic=True), the window is considered dirty
    and the strategy should abstain.

    The window is [max(0, end_idx - lookback + 1) .. end_idx] inclusive,
    so it always contains at most `lookback` candles.

    Args:
        df: OHLCV DataFrame with an is_synthetic column. Must be sorted
            ascending by timestamp_utc (as guaranteed by ParquetStorage.load).
        end_idx: Positional index of the last candle in the window (inclusive).
            Typically the current bar index in a backtest loop.
        lookback: Number of candles to inspect. Default 20.

    Returns:
        True  — window has no synthetic candles; safe to generate signals.
        False — at least one synthetic candle in the window; strategy should skip.

    Raises:
        KeyError: if df does not have an is_synthetic column.
        IndexError: if end_idx is out of bounds.
    """
    if "is_synthetic" not in df.columns:
        raise KeyError("DataFrame must have an 'is_synthetic' column")

    if end_idx < 0 or end_idx >= len(df):
        raise IndexError(
            f"end_idx={end_idx} is out of bounds for DataFrame of length {len(df)}"
        )

    start_idx = max(0, end_idx - lookback + 1)
    window = df["is_synthetic"].iloc[start_idx : end_idx + 1]
    return not bool(window.any())
