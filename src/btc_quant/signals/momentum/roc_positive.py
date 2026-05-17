import math

import pandas as pd

from btc_quant.signals.types import SignalResult

_NAN_RESULT = SignalResult(active=False, value=float("nan"))


def is_roc_positive(
    close: pd.Series,
    period: int = 10,
    threshold_pct: float = 0.0,
) -> SignalResult:
    """Evaluate the Rate of Change (ROC) positive momentum signal.

    ROC measures the percentage price change over a fixed look-back:
        ROC = (close[-1] - close[-period-1]) / close[-period-1] * 100

    active = ROC > threshold_pct  (strictly greater, not >=)
    value  = ROC percentage (positive = price higher than N bars ago)

    Default threshold 0% fires when price is strictly above where it was
    period bars ago. threshold_pct is exposed as a parameter for stricter
    momentum filters (e.g., 5% to require meaningful positive momentum).

    Long-only: no symmetric "ROC below negative threshold" branch.

    Edge cases:
        - len(close) < period + 1: active=False, value=NaN.
        - close[-period-1] == 0: degenerate; active=False, value=NaN.
        - NaN or inf in close: active=False, value=NaN.

    Args:
        close: Bar close prices, chronologically ordered.
        period: Look-back period for ROC. Default 10.
        threshold_pct: ROC level above which signal fires. Default 0.0.

    Returns:
        SignalResult(active, roc_pct).
    """
    if len(close) < period + 1:
        return _NAN_RESULT

    if close.isnull().any() or not all(math.isfinite(v) for v in close):
        return _NAN_RESULT

    current = float(close.iloc[-1])
    prior = float(close.iloc[-(period + 1)])

    if prior == 0.0:
        return _NAN_RESULT

    roc = (current - prior) / prior * 100.0
    return SignalResult(active=bool(roc > threshold_pct), value=float(roc))
