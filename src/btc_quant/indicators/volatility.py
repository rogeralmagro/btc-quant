"""Volatility indicators: ATR, Bollinger Bands.

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


_VALID_ATR_METHODS = frozenset({"wilders", "ema", "sma"})


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    method: str = "wilders",
) -> pd.Series:
    """Average True Range.

    True Range for bar t:
        TR[t] = max(high[t] − low[t],
                    |high[t] − close[t−1]|,
                    |low[t]  − close[t−1]|)

    At t=0 there is no previous close, so pandas naturally returns NaN for
    the gap components. Because ``.max(axis=1, skipna=True)`` ignores NaN,
    TR[0] = high[0] − low[0] (the bar's own range only). No special-casing
    needed.

    Smoothing methods
    -----------------
    'wilders' (default)
        α = 1/period. Identical smoothing to ``rsi()`` in oscillators.py:
        same seed-then-recursive pattern, same α formula. The only structural
        difference is the index of the first defined value — ATR seeds at
        index ``period − 1`` (TR is available from t=0), whereas RSI seeds at
        index ``period`` (the first diff is NaN at t=0).

        Seed:  ATR[period−1]  = mean(TR[0 : period])
        Loop:  ATR[t]         = ATR[t−1] * (1 − α) + TR[t] * α   for t ≥ period
        Warmup: first ``period − 1`` values are NaN.

    'sma'
        Simple rolling mean of TR with window=period.
        Warmup: first ``period − 1`` values are NaN (from rolling).

    'ema'
        pandas ewm(span=period, adjust=False). Seeds at TR[0]; no NaN
        inserted (consistent with ``ema()`` in trend.py).

    Args:
        high:   High prices. Must share index with *low* and *close*.
        low:    Low prices. Must share index with *high* and *close*.
        close:  Close prices. Must share index with *high* and *low*.
        period: Smoothing window. Must be >= 1.
        method: One of ``'wilders'`` (default), ``'ema'``, ``'sma'``.

    Returns:
        Float64 Series with the same index as *close*.

    Raises:
        ValueError: on invalid arguments or if any bar has low > high.
    """
    if not (high.index.equals(low.index) and high.index.equals(close.index)):
        raise ValueError("high, low, close must share the same index")
    if high.empty:
        raise ValueError("series must not be empty")
    if period < 1:
        raise ValueError(f"period must be >= 1, got {period}")
    if method not in _VALID_ATR_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_VALID_ATR_METHODS)}, got {method!r}"
        )
    if (low > high).any():
        first_bad = (low > high).idxmax()
        raise ValueError(f"low > high at index {first_bad!r}")

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    n = len(close)

    if method == "sma":
        return tr.rolling(window=period).mean().astype("float64")

    if method == "ema":
        return tr.ewm(span=period, adjust=False).mean().astype("float64")

    # --- 'wilders': pure-Python loop, same pattern as rsi() ---
    tr_arr = tr.to_numpy(dtype="float64")
    avg = np.full(n, np.nan)

    if n < period:
        return pd.Series(avg, index=close.index, dtype="float64")

    avg[period - 1] = tr_arr[:period].mean()
    alpha = 1.0 / period
    for i in range(period, n):
        avg[i] = avg[i - 1] * (1.0 - alpha) + tr_arr[i] * alpha

    return pd.Series(avg, index=close.index, dtype="float64")


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> dict[str, pd.Series]:
    """Bollinger Bands.

    Uses population standard deviation (ddof=0), matching TradingView's
    implementation. The first ``period − 1`` values of all three bands are NaN.

    Args:
        series:  Price series (typically close). Must not be empty.
        period:  Rolling window. Must be >= 2.
        num_std: Band width in standard deviations. Must be > 0.

    Returns:
        Dict with three float64 Series sharing the input index:
        - ``'middle'``: SMA(series, period)
        - ``'upper'``:  middle + num_std × σ
        - ``'lower'``:  middle − num_std × σ

    Raises:
        ValueError: on invalid arguments.
    """
    if series.empty:
        raise ValueError("series must not be empty")
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")
    if num_std <= 0:
        raise ValueError(f"num_std must be > 0, got {num_std}")

    middle = series.rolling(window=period).mean().astype("float64")
    std = series.rolling(window=period).std(ddof=0).astype("float64")
    band = (num_std * std).astype("float64")

    return {
        "middle": middle,
        "upper": (middle + band).astype("float64"),
        "lower": (middle - band).astype("float64"),
    }
