"""Oscillator indicators: RSI.

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


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index with Wilder's smoothing (the trading standard).

    Wilder's smoothing uses α = 1/period (equivalent to EMA span = 2*period − 1).
    This differs from a standard EMA (α = 2/(period+1)) and produces the RSI
    values shown on TradingView, MetaTrader, and most charting platforms.

    Algorithm
    ---------
    1. delta  = series.diff()
    2. gain   = delta.clip(lower=0)   (positive moves only; zero otherwise)
    3. loss   = (−delta).clip(lower=0) (magnitude of negative moves; zero otherwise)
    4. Seed:  avg_gain[period] = mean(gain[1 : period+1])
              avg_loss[period] = mean(loss[1 : period+1])
              (SMA of the first *period* valid differences)
    5. Recursive (Wilder):
              avg_gain[t] = (avg_gain[t−1] * (period−1) + gain[t]) / period
              avg_loss[t] = (avg_loss[t−1] * (period−1) + loss[t]) / period
    6. RS     = avg_gain / avg_loss
    7. RSI    = 100 − 100 / (1 + RS)

    Edge cases
    ----------
    - Constant series (all deltas = 0): avg_gain = avg_loss = 0 → RSI = NaN.
    - Only gains (avg_loss = 0, avg_gain > 0): RSI = 100.
    - Only losses (avg_gain = 0): RSI = 0 (naturally: RS = 0 → RSI = 0).

    Args:
        series: Price series. Must not be empty.
        period: Lookback window. Must be >= 2 (need at least 2 diffs to separate
            gains and losses meaningfully).

    Returns:
        Float64 Series with the same index as *series*.
        First ``period`` values are NaN (warmup). RSI[period] is the first
        defined value, computed from diffs at positions 1..period.

    Raises:
        ValueError: if *series* is empty or *period* < 2.
    """
    if series.empty:
        raise ValueError("series must not be empty")
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")

    n = len(series)
    delta = series.diff()
    gain = delta.clip(lower=0).to_numpy(dtype="float64")
    loss = (-delta).clip(lower=0).to_numpy(dtype="float64")

    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)

    if n <= period:
        return pd.Series(np.nan, index=series.index, dtype="float64")

    # SMA seed from diffs at positions 1..period (delta[0] is NaN from diff)
    avg_gain[period] = gain[1 : period + 1].mean()
    avg_loss[period] = loss[1 : period + 1].mean()

    # Wilder's recursive smoothing
    alpha = 1.0 / period
    for i in range(period + 1, n):
        avg_gain[i] = avg_gain[i - 1] * (1.0 - alpha) + gain[i] * alpha
        avg_loss[i] = avg_loss[i - 1] * (1.0 - alpha) + loss[i] * alpha

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss
        rsi_vals = 100.0 - (100.0 / (1.0 + rs))

    # avg_loss == 0 and avg_gain == 0 → both zero (constant series) → NaN
    # avg_loss == 0 and avg_gain  > 0 → only gains → RSI = 100
    only_gains = (avg_loss == 0.0) & (avg_gain > 0.0)
    both_zero = (avg_loss == 0.0) & (avg_gain == 0.0)
    rsi_vals = np.where(only_gains, 100.0, rsi_vals)
    rsi_vals = np.where(both_zero, np.nan, rsi_vals)

    return pd.Series(rsi_vals, index=series.index, dtype="float64")
