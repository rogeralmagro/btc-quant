import math

import pandas as pd

from btc_quant.signals.types import SignalResult

_NAN_RESULT = SignalResult(active=False, value=float("nan"))


def is_macd_bullish(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> SignalResult:
    """Evaluate the MACD bullish momentum signal.

    Computes the classic MACD indicator using EMA with adjust=False:
        macd_line   = EMA(close, fast) - EMA(close, slow)
        signal_line = EMA(macd_line, signal)
        histogram   = macd_line - signal_line

    active = macd_line[-1] > signal_line[-1]  (MACD above signal line)
    value  = histogram at last bar (positive = bullish momentum)

    "Bullish" captures both a freshly crossed MACD (small positive histogram)
    and sustained uptrend (large positive histogram). This signal does NOT
    detect the crossing event itself — only the current state.

    Long-only: no bearish branch (MACD < Signal) implemented.

    Edge cases:
        - len(close) < slow + signal: active=False, value=NaN.
        - NaN or inf in close: active=False, value=NaN.

    Args:
        close: Bar close prices, chronologically ordered.
        fast: Fast EMA span. Default 12.
        slow: Slow EMA span. Default 26.
        signal: Signal line EMA span. Default 9.

    Returns:
        SignalResult(active, histogram).
    """
    if len(close) < slow + signal:
        return _NAN_RESULT

    if close.isnull().any() or not all(math.isfinite(v) for v in close):
        return _NAN_RESULT

    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    histogram = float(macd_line.iloc[-1] - signal_line.iloc[-1])
    return SignalResult(active=bool(histogram > 0), value=histogram)
