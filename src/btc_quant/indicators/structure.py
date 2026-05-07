"""Market structure indicators: drawdown from ATH, swing highs/lows.

INDICATOR CONTRACT
==================
All functions in this module follow this contract:

1. Input   : pd.Series (prices) or multiple Series. Must not be empty.
2. Output  : pd.Series with the same index as the primary input.
3. Warmup  : first N positions are explicit NaN. Never filled, never interpolated.
4. Dtype   : output is always float64 (price/ratio series) or bool (flag series).
5. Purity  : stateless — same input always produces the same output.
6. Validation: invalid arguments raise ValueError with a descriptive message.
7. Vectorized with pandas/numpy. Iterative only where mathematically unavoidable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_from_ath(
    close: pd.Series,
    ath_seed: float | None = None,
) -> dict[str, pd.Series]:
    """Drawdown from the running All-Time High.

    Computes two series:
    - ``'ath'``:      running maximum close up to and including each bar.
    - ``'drawdown'``: (close − ath) / ath, always ≤ 0. Zero when close == ath.

    The optional *ath_seed* is useful when loading a data subset: you can
    provide the true historical ATH so the running max starts from the correct
    baseline rather than from the first close in the subset.

    Args:
        close:    Close price series. All values must be > 0.
        ath_seed: If provided, the initial ATH value before the first bar.
                  Must be > 0. If None, the seed defaults to close[0].

    Returns:
        Dict with two float64 Series sharing the input index:
        - ``'ath'``:      running maximum price (non-decreasing).
        - ``'drawdown'``: fractional drawdown from ATH (values in [−1, 0]).

    Raises:
        ValueError: if any close ≤ 0, or if ath_seed ≤ 0 when provided.
    """
    if close.empty:
        raise ValueError("close series must not be empty")
    if (close <= 0).any():
        raise ValueError("all close prices must be > 0")
    if ath_seed is not None and ath_seed <= 0:
        raise ValueError(f"ath_seed must be > 0, got {ath_seed}")

    close_arr = close.to_numpy(dtype="float64")

    if ath_seed is not None:
        # Prepend the seed so accumulate starts from the historical ATH
        extended = np.concatenate([[float(ath_seed)], close_arr])
        ath_arr = np.maximum.accumulate(extended)[1:]
    else:
        ath_arr = np.maximum.accumulate(close_arr)

    ath = pd.Series(ath_arr, index=close.index, dtype="float64")
    drawdown = ((close - ath) / ath).astype("float64")

    return {"ath": ath, "drawdown": drawdown}


def swing_highs_lows(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 5,
) -> dict[str, pd.Series]:
    """Detect swing highs and swing lows using a symmetric lookback window.

    A bar at position t is a swing high when its high is strictly greater than
    the high of every bar within ±lookback positions:

        is_swing_high[t] = True  iff  high[t] > high[t+k]  and
                                       high[t] > high[t-k]  for all k in 1..lookback

    Swing lows are symmetric (strictly less than all lows in the window).

    Boundary positions (first and last *lookback* bars) are always False
    because there are insufficient bars on one side to evaluate the condition.

    WARNING — Look-Ahead Implications
    ==================================
    This function uses future data by design: confirming that bar t is a swing
    requires examining bars t+1 through t+lookback.

    A swing detected at position t can only be CONFIRMED once position
    t+lookback has been observed. Any strategy acting on a swing at bar t
    directly (without waiting lookback bars) introduces look-ahead bias and
    will produce unrealistically optimistic backtest results.

    Correct usage in a backtest loop::

        for t in range(n):
            if is_swing_high[t - lookback]:   # confirmed lookback bars ago
                ...act on the swing...

    Args:
        high:     High price series. Must share index with *low*.
        low:      Low price series. Must share index with *high*.
        lookback: Number of bars to examine on each side. Must be >= 1.

    Returns:
        Dict with two bool Series sharing the input index:
        - ``'is_swing_high'``: True at confirmed swing high bars.
        - ``'is_swing_low'``:  True at confirmed swing low bars.

    Raises:
        ValueError: on invalid arguments or if any bar has low > high.
    """
    if not high.index.equals(low.index):
        raise ValueError("high and low must share the same index")
    if high.empty:
        raise ValueError("series must not be empty")
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")
    if (low > high).any():
        first_bad = (low > high).idxmax()
        raise ValueError(f"low > high at index {first_bad!r}")

    is_swing_high = pd.Series(True, index=high.index)
    is_swing_low = pd.Series(True, index=low.index)

    for k in range(1, lookback + 1):
        is_swing_high &= (high > high.shift(k)).fillna(False)
        is_swing_high &= (high > high.shift(-k)).fillna(False)
        is_swing_low &= (low < low.shift(k)).fillna(False)
        is_swing_low &= (low < low.shift(-k)).fillna(False)

    return {
        "is_swing_high": is_swing_high.astype(bool),
        "is_swing_low": is_swing_low.astype(bool),
    }
