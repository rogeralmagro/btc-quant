import math

import numpy as np
import pandas as pd

from btc_quant.signals.types import SignalResult

_NAN_RESULT = SignalResult(active=False, value=float("nan"))


def is_obv_bullish_trend(
    close: pd.Series,
    volume: pd.Series,
    ma_period: int = 20,
) -> SignalResult:
    """Evaluate the OBV bullish accumulation trend signal.

    Computes On-Balance Volume by accumulating signed volume based on
    whether price closed up, down, or flat vs the previous bar:
        direction = sign(close.diff())   # 1 / -1 / 0 / NaN at index 0
        direction[0] = 0                 # first bar has no prior close
        OBV = cumsum(direction * volume)

    sma_obv = OBV.rolling(ma_period).mean()
    active  = OBV[-1] > sma_obv[-1]
    value   = OBV[-1] - sma_obv[-1]  (positive = OBV in bullish regime)

    Long-only: no "OBV < SMA(OBV)" bearish branch implemented.

    Edge cases:
        - len < ma_period + 1: active=False, value=NaN.
        - len(close) != len(volume): raises ValueError.
        - NaN or inf in either series: active=False, value=NaN.

    Args:
        close: Bar close prices, chronologically ordered.
        volume: Bar volume, chronologically ordered (same length as close).
        ma_period: Rolling window for OBV moving average. Default 20.

    Returns:
        SignalResult(active, obv_minus_sma).
    """
    if len(close) != len(volume):
        raise ValueError(
            f"close and volume must have the same length, "
            f"got {len(close)} and {len(volume)}"
        )

    if len(close) < ma_period + 1:
        return _NAN_RESULT

    for series in (close, volume):
        if series.isnull().any() or not all(math.isfinite(v) for v in series):
            return _NAN_RESULT

    direction = np.sign(close.diff())
    direction.iloc[0] = 0.0
    obv = (direction * volume).cumsum()

    sma_obv = obv.rolling(ma_period).mean()
    diff = float(obv.iloc[-1] - sma_obv.iloc[-1])
    return SignalResult(active=bool(diff > 0), value=diff)
