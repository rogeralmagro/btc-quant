import math
from typing import List, Tuple

import pandas as pd

from btc_quant.signals.types import SignalResult

_NAN_RESULT = SignalResult(active=False, value=float("nan"))

# Swing is represented as (bar_index, price_value)
Swing = Tuple[int, float]


def _find_swings(
    series: pd.Series,
    lookback: int,
    is_high: bool,
) -> List[Swing]:
    """Detect symmetric pivot highs or lows.

    A bar at index i is a swing high if series[i] is strictly greater than
    every bar in [i-lookback, i) and every bar in (i, i+lookback].
    Symmetric pivots require `lookback` confirmed bars on both sides, so
    the last `lookback` bars of the series can never be swing candidates
    (they lack right-side confirmation).

    Uses strict inequality (>/<) — equal-level plateaus are NOT pivots.

    Args:
        series: Price series (high or low).
        lookback: Number of bars required on each side.
        is_high: True for swing highs, False for swing lows.

    Returns:
        List of (index, value) tuples in chronological order.
    """
    values = series.values
    n = len(values)
    swings: List[Swing] = []

    for i in range(lookback, n - lookback):
        pivot = values[i]
        left = values[i - lookback : i]
        right = values[i + 1 : i + lookback + 1]
        if is_high:
            if all(pivot > v for v in left) and all(pivot > v for v in right):
                swings.append((i, float(pivot)))
        else:
            if all(pivot < v for v in left) and all(pivot < v for v in right):
                swings.append((i, float(pivot)))

    return swings


def is_bullish_market_structure(
    high: pd.Series,
    low: pd.Series,
    swing_lookback: int = 5,
) -> SignalResult:
    """Evaluate bullish market structure via Higher Highs and Higher Lows.

    Detects symmetric pivot highs and lows with `swing_lookback` bars of
    confirmation on each side. Compares the two most recent confirmed swings
    of each type:

        HH = last_swing_high  > prev_swing_high   (strictly greater)
        HL = last_swing_low   > prev_swing_low     (strictly greater)
        active = HH AND HL

    value = (last_sh / prev_sh - 1) + (last_sl / prev_sl - 1)
            Sum of relative changes; positive when both components are
            bullish, negative when both are bearish.

    Swings are detected using strict inequality (>/<): equal-level pivots
    (plateaus) do NOT qualify as swing highs or lows. This means two swings
    at the same price level yield HH=False even though structure is "flat",
    which is correct — flat structure is not bullish per this definition.

    Structural note: the most recent confirmed swing is always at least
    `swing_lookback` bars before the last bar (right-side confirmation lag).
    This lag is structural and expected — it prevents look-ahead.

    Edge cases:
        - len < 2*swing_lookback + 1: active=False, value=NaN.
        - Fewer than 2 swing highs or 2 swing lows detected: active=False,
          value=NaN. This can happen on short, smooth, or noisy series.
        - NaN or inf in high or low: active=False, value=NaN.
        - len(high) != len(low): raises ValueError.

    Args:
        high: Bar highs, chronologically ordered.
        low: Bar lows, chronologically ordered.
        swing_lookback: Bars required on each side of a pivot. Default 5.

    Returns:
        SignalResult(active, relative_change_sum).
    """
    if len(high) != len(low):
        raise ValueError(
            f"high and low must have the same length, "
            f"got {len(high)} and {len(low)}"
        )

    if len(high) < 2 * swing_lookback + 1:
        return _NAN_RESULT

    for series in (high, low):
        if series.isnull().any() or not all(math.isfinite(v) for v in series):
            return _NAN_RESULT

    swing_highs = _find_swings(high, swing_lookback, is_high=True)
    swing_lows = _find_swings(low, swing_lookback, is_high=False)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return _NAN_RESULT

    prev_sh, last_sh = swing_highs[-2][1], swing_highs[-1][1]
    prev_sl, last_sl = swing_lows[-2][1], swing_lows[-1][1]

    hh = last_sh > prev_sh
    hl = last_sl > prev_sl

    delta_high = (last_sh / prev_sh) - 1.0
    delta_low = (last_sl / prev_sl) - 1.0
    value = float(delta_high + delta_low)

    return SignalResult(active=bool(hh and hl), value=value)
