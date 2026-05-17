import math

import pandas as pd

from btc_quant.signals.types import SignalResult

_NAN_RESULT = SignalResult(active=False, value=float("nan"))


def is_htf_aligned_bullish(
    close_1d: pd.Series,
    close_1w: pd.Series,
    daily_fast: int = 50,
    daily_slow: int = 200,
    weekly_fast: int = 10,
    weekly_slow: int = 40,
) -> SignalResult:
    """Evaluate higher-timeframe trend alignment.

    Both the daily and weekly timeframes must show a bullish EMA cross for the
    signal to fire. Either bearish frame disables entry.

    Formulas:
        daily_bullish  = EMA(close_1d, daily_fast)[-1]  > EMA(close_1d, daily_slow)[-1]
        weekly_bullish = EMA(close_1w, weekly_fast)[-1] > EMA(close_1w, weekly_slow)[-1]
        active         = daily_bullish AND weekly_bullish
        value          = (EMA_d_fast - EMA_d_slow) + (EMA_w_fast - EMA_w_slow)
                         (sum of spreads; larger = stronger multi-TF alignment)

    Edge cases:
        - len(close_1d) < daily_slow or len(close_1w) < weekly_slow:
          returns active=False, value=NaN.
        - Any NaN or inf in either series: returns active=False, value=NaN.

    Args:
        close_1d: Daily close prices, chronologically ordered.
        close_1w: Weekly close prices, chronologically ordered.
        daily_fast: Fast EMA span on the daily frame. Default 50.
        daily_slow: Slow EMA span on the daily frame. Default 200.
        weekly_fast: Fast EMA span on the weekly frame. Default 10.
        weekly_slow: Slow EMA span on the weekly frame. Default 40.

    Returns:
        SignalResult with active=True only when both timeframes confirm bullish.
    """
    if len(close_1d) < daily_slow or len(close_1w) < weekly_slow:
        return _NAN_RESULT

    for series in (close_1d, close_1w):
        if series.isnull().any() or not all(math.isfinite(v) for v in series):
            return _NAN_RESULT

    ema_d_fast = close_1d.ewm(span=daily_fast, adjust=False).mean().iloc[-1]
    ema_d_slow = close_1d.ewm(span=daily_slow, adjust=False).mean().iloc[-1]
    ema_w_fast = close_1w.ewm(span=weekly_fast, adjust=False).mean().iloc[-1]
    ema_w_slow = close_1w.ewm(span=weekly_slow, adjust=False).mean().iloc[-1]

    daily_bullish = bool(ema_d_fast > ema_d_slow)
    weekly_bullish = bool(ema_w_fast > ema_w_slow)

    value = float((ema_d_fast - ema_d_slow) + (ema_w_fast - ema_w_slow))
    return SignalResult(active=daily_bullish and weekly_bullish, value=value)
