import math

import pandas as pd

from btc_quant.signals.types import SignalResult

_NAN_RESULT = SignalResult(active=False, value=float("nan"))


def is_high_volume(
    volume: pd.Series,
    rolling_window: int = 252,
    threshold_percentile: float = 75.0,
) -> SignalResult:
    """Evaluate the high-volume regime signal.

    Ranks the current bar's volume against the trailing rolling_window bars
    using the midpoint percentile convention:
        historical = volume[-(rolling_window+1) : -1]  (excludes current bar)
        n_below    = (historical < current).sum()
        n_equal    = (historical == current).sum()
        percentile = (n_below + 0.5 * n_equal) / len(historical) * 100

    active = percentile > threshold_percentile
    value  = percentile (0–100)

    Direction-agnostic by design: high volume signals participation —
    whether buying or selling pressure — without implying direction.
    Directional confirmation comes from Category 1 (regime) and Category
    3 (momentum) signals. In the confluence context, this signal is
    permissive context, not a bullish trigger by itself.

    Midpoint convention (same as classify_atr_regime): ties contribute 0.5
    each. A constant historical window yields percentile=50, NOT 0 or 100.

    Edge cases:
        - len(volume) < rolling_window + 1: active=False, value=NaN.
        - NaN or inf in volume: active=False, value=NaN.

    Args:
        volume: Bar volume, chronologically ordered.
        rolling_window: Trailing window for percentile ranking. Default 252.
        threshold_percentile: Percentile above which signal fires. Default 75.

    Returns:
        SignalResult(active, percentile).
    """
    if len(volume) < rolling_window + 1:
        return _NAN_RESULT

    if volume.isnull().any() or not all(math.isfinite(v) for v in volume):
        return _NAN_RESULT

    current = float(volume.iloc[-1])
    historical = volume.iloc[-(rolling_window + 1) : -1]

    n = len(historical)
    n_below = int((historical < current).sum())
    n_equal = int((historical == current).sum())
    percentile = float((n_below + 0.5 * n_equal) / n * 100)

    return SignalResult(active=bool(percentile > threshold_percentile), value=percentile)
