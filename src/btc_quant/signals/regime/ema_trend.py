import math

import pandas as pd

from btc_quant.signals.types import SignalResult

_NAN_RESULT = SignalResult(active=False, value=float("nan"))


def is_bullish_ema_trend(
    close: pd.Series,
    fast: int = 50,
    slow: int = 200,
) -> SignalResult:
    """Evaluate the EMA golden-cross regime signal.

    Formula:
        EMA_fast = ewm(span=fast, adjust=False).mean()
        EMA_slow = ewm(span=slow, adjust=False).mean()
        value    = EMA_fast[-1] - EMA_slow[-1]
        active   = value > 0

    Active when the fast EMA is above the slow EMA on the last bar, indicating
    a bullish trending regime. A negative value indicates bearish trend.

    Edge cases:
        - len(close) < slow: returns active=False, value=NaN (insufficient history).
        - Any NaN or inf in close: returns active=False, value=NaN.

    Args:
        close: Daily close prices, chronologically ordered.
        fast: Span for the fast EMA. Default 50.
        slow: Span for the slow EMA. Default 200.

    Returns:
        SignalResult with active=True when EMA(fast) > EMA(slow).
    """
    if len(close) < slow:
        return _NAN_RESULT

    if close.isnull().any() or not all(math.isfinite(v) for v in close):
        return _NAN_RESULT

    ema_fast = close.ewm(span=fast, adjust=False).mean().iloc[-1]
    ema_slow = close.ewm(span=slow, adjust=False).mean().iloc[-1]

    diff = float(ema_fast - ema_slow)
    return SignalResult(active=bool(diff > 0), value=diff)
