import math

import pandas as pd

from btc_quant.signals.types import SignalResult

_NAN_RESULT = SignalResult(active=False, value=float("nan"))


def is_bullish_trending(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    adx_threshold: float = 25.0,
) -> SignalResult:
    """Evaluate the ADX bullish trending momentum signal.

    Computes the ADX directional movement system using Wilder's smoothing
    (alpha=1/period, adjust=False, min_periods=period):

        up_move   = high.diff()
        down_move = -low.diff()
        +DM = up_move   where up_move > down_move AND up_move > 0, else 0
        -DM = down_move where down_move > up_move AND down_move > 0, else 0

        TR  = max(high - low, |high - prev_close|, |low - prev_close|)
        ATR = Wilder EMA of TR
        +DI = 100 * Wilder EMA(+DM) / ATR
        -DI = 100 * Wilder EMA(-DM) / ATR
        DX  = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = Wilder EMA of DX

    active = ADX[-1] > adx_threshold AND +DI[-1] > -DI[-1]
    value  = ADX at last bar (0–100; does not encode direction)

    ADX measures trend STRENGTH only, not direction. This long-only signal
    requires BOTH sufficient trend strength (ADX > 25 by default) AND
    bullish direction (+DI > -DI). A strong downtrend (ADX > 25, -DI > +DI)
    returns active=False — the signal never opens short positions.

    Edge cases:
        - len < period * 2: active=False, value=NaN (insufficient warm-up).
        - ATR == 0 (degenerate constant prices): active=False, value=NaN.
        - NaN or inf in inputs: active=False, value=NaN.

    Args:
        high: Bar highs, chronologically ordered.
        low: Bar lows, chronologically ordered.
        close: Bar closes, chronologically ordered.
        period: Wilder smoothing period for DM, ATR, and ADX. Default 14.
        adx_threshold: Minimum ADX for a "trending" classification. Default 25.

    Returns:
        SignalResult(active, adx_value).
    """
    if len(close) < period * 2:
        return _NAN_RESULT

    for series in (high, low, close):
        if series.isnull().any() or not all(math.isfinite(v) for v in series):
            return _NAN_RESULT

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smoothed_plus = plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    smoothed_minus = minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    last_atr = float(atr.iloc[-1])
    if not math.isfinite(last_atr) or last_atr == 0.0:
        return _NAN_RESULT

    plus_di = 100.0 * smoothed_plus / atr
    minus_di = 100.0 * smoothed_minus / atr

    last_plus = float(plus_di.iloc[-1])
    last_minus = float(minus_di.iloc[-1])
    di_sum = last_plus + last_minus

    if di_sum == 0.0:
        return _NAN_RESULT

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    last_adx = float(adx.iloc[-1])
    if not math.isfinite(last_adx):
        return _NAN_RESULT

    active = bool(last_adx > adx_threshold and last_plus > last_minus)
    return SignalResult(active=active, value=last_adx)
