"""Volume indicators: OBV.

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


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume.

    OBV is a cumulative volume indicator that tracks whether volume is flowing
    into or out of an asset:

        OBV[0] = 0
        OBV[t] = OBV[t-1] + volume[t]   if close[t] > close[t-1]
        OBV[t] = OBV[t-1] - volume[t]   if close[t] < close[t-1]
        OBV[t] = OBV[t-1]               if close[t] == close[t-1]

    Design decision — OBV[0] = 0:
        The first bar has no prior close to compare against. By convention
        (Granville's original definition and TradingView) OBV starts at 0.
        NaN would convey no additional information because OBV is always
        interpreted in terms of *changes*, not absolute levels.

    Synthetic candles (is_synthetic=True) have volume=0, so they contribute
    0 to OBV regardless of the flat price direction — correct behaviour since
    there was no real trading during those periods.

    Args:
        close:  Close price series. Must share index with *volume*.
        volume: Volume series. Must be >= 0 for all bars.

    Returns:
        Float64 Series with the same index as *close*. No NaN (warmup = 0).

    Raises:
        ValueError: if indices differ, volume has negatives, or series are empty.
    """
    if not close.index.equals(volume.index):
        raise ValueError("close and volume must share the same index")
    if close.empty:
        raise ValueError("series must not be empty")
    if (volume < 0).any():
        first_bad = (volume < 0).idxmax()
        raise ValueError(f"volume must be >= 0 for all bars; negative at index {first_bad!r}")

    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum().astype("float64")
