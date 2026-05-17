import math

import pandas as pd

from btc_quant.signals.types import SignalResult

_NAN_RESULT = SignalResult(active=False, value=float("nan"))


def is_rsi_oversold(
    close: pd.Series,
    period: int = 14,
    threshold: float = 30.0,
) -> SignalResult:
    """Evaluate the RSI oversold mean-reversion signal.

    Computes Wilder's RSI using exponential smoothing with alpha=1/period:
        delta    = close.diff()
        gain     = delta.clip(lower=0)
        loss     = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
        RS       = avg_gain / avg_loss
        RSI      = 100 - 100 / (1 + RS)

    Active when RSI < threshold on the last bar, indicating an oversold
    condition and a potential mean-reversion long setup.

    Long-only: this signal does NOT detect overbought conditions (RSI > 70).
    The complementary overbought branch is intentionally absent — STRAT-07
    is a long-only strategy.

    value = raw RSI value, 0–100. NaN when insufficient data.
    active = RSI < threshold (default 30 = standard oversold level).

    Special cases:
        - avg_loss == 0 (no losing bars in window): RSI = 100. active=False.
        - len(close) < period + 1: insufficient data. active=False, value=NaN.
        - NaN or inf in close: active=False, value=NaN.

    Args:
        close: Bar close prices, chronologically ordered.
        period: Wilder smoothing period. Default 14.
        threshold: RSI level below which the signal fires. Default 30.0.

    Returns:
        SignalResult(active, rsi_value).
    """
    if len(close) < period + 1:
        return _NAN_RESULT

    if close.isnull().any() or not all(math.isfinite(v) for v in close):
        return _NAN_RESULT

    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()

    last_gain = float(avg_gain.iloc[-1])
    last_loss = float(avg_loss.iloc[-1])

    if math.isnan(last_gain) or math.isnan(last_loss):
        return _NAN_RESULT

    if last_loss == 0.0:
        rsi = 100.0
    else:
        rs = last_gain / last_loss
        rsi = 100.0 - 100.0 / (1.0 + rs)

    return SignalResult(active=bool(rsi < threshold), value=rsi)
