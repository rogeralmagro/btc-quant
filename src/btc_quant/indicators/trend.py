"""Trend indicators: SMA, EMA, MACD.

INDICATOR CONTRACT
==================
All functions in this module follow this contract:

1. Input   : pd.Series (prices) or multiple Series. Must not be empty.
2. Output  : pd.Series with the same index as the primary input.
3. Warmup  : first N positions are explicit NaN. Never filled, never interpolated.
4. Dtype   : output is always float64.
5. Purity  : stateless — same input always produces the same output.
6. Validation: invalid arguments raise ValueError with a descriptive message.
7. Vectorized with pandas/numpy. Iterative only where mathematically unavoidable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.

    Args:
        series: Price series. Must not be empty.
        period: Lookback window. Must be >= 1.

    Returns:
        Float64 Series with the same index as *series*.
        First ``period - 1`` values are NaN (warmup).

    Raises:
        ValueError: if *series* is empty or *period* < 1.
    """
    if series.empty:
        raise ValueError("series must not be empty")
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return series.rolling(window=period).mean().astype("float64")


def ema(series: pd.Series, period: int, adjust: bool = False) -> pd.Series:
    """Exponential Moving Average.

    Uses pandas ``ewm(span=period)``.

    Args:
        series: Price series. Must not be empty.
        period: Span parameter (α = 2 / (period + 1)). Must be >= 1.
        adjust: If False (default), uses the recursive formula
            EMA[t] = α * price[t] + (1 - α) * EMA[t-1], seeded at EMA[0] = price[0].
            This matches TradingView / MetaTrader behaviour.
            If True, uses the weighted-average form (more mathematically precise
            but not the trading-platform standard).

    Returns:
        Float64 Series with the same index as *series*.
        With adjust=False the series has no NaN (seeds from the first value).

    Raises:
        ValueError: if *series* is empty or *period* < 1.
    """
    if series.empty:
        raise ValueError("series must not be empty")
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    return series.ewm(span=period, adjust=adjust).mean().astype("float64")


def macd(
    series: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, pd.Series]:
    """Moving Average Convergence Divergence.

    Both EMAs use adjust=False (Wilder-compatible recursive formula).
    The first ``slow_period - 1`` values of the MACD line are masked as NaN
    because the slow EMA has not yet processed enough bars to be meaningful.
    The signal line additionally requires ``signal_period`` non-NaN MACD values,
    so signal_line and histogram are NaN for the first
    ``(slow_period - 1) + (signal_period - 1)`` bars.

    Args:
        series: Price series. Must not be empty.
        fast_period: Fast EMA span. Must be >= 1 and < slow_period.
        slow_period: Slow EMA span. Must be >= 1 and > fast_period.
        signal_period: Signal EMA span. Must be >= 1.

    Returns:
        Dict with three float64 Series sharing the input index:
        - ``'macd_line'``: EMA(fast) − EMA(slow)
        - ``'signal_line'``: EMA(macd_line, signal_period)
        - ``'histogram'``: macd_line − signal_line

    Raises:
        ValueError: on invalid arguments.
    """
    if series.empty:
        raise ValueError("series must not be empty")
    for name, val in [
        ("fast_period", fast_period),
        ("slow_period", slow_period),
        ("signal_period", signal_period),
    ]:
        if val < 1:
            raise ValueError(f"{name} must be >= 1, got {val}")
    if fast_period >= slow_period:
        raise ValueError(
            f"fast_period ({fast_period}) must be < slow_period ({slow_period})"
        )

    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = (fast_ema - slow_ema).astype("float64")

    # Mask warmup: slow EMA needs slow_period bars before it is meaningful
    macd_line.iloc[: slow_period - 1] = np.nan

    # Signal is an EMA of the MACD line; min_periods enforces full warmup
    signal_line = (
        macd_line.ewm(span=signal_period, adjust=False, min_periods=signal_period)
        .mean()
        .astype("float64")
    )
    histogram = (macd_line - signal_line).astype("float64")

    return {
        "macd_line": macd_line,
        "signal_line": signal_line,
        "histogram": histogram,
    }
