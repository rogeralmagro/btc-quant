import math

import pandas as pd

from btc_quant.signals.types import SignalResult

_NAN_RESULT = SignalResult(active=False, value=float("nan"))


def is_far_below_ma(
    close: pd.Series,
    period: int = 50,
    threshold_pct: float = -10.0,
) -> SignalResult:
    """Evaluate the distance-from-moving-average mean-reversion signal.

    Computes the percentage deviation of the last close from its SMA:
        sma           = close.rolling(period).mean()
        deviation_pct = (close[-1] - sma[-1]) / sma[-1] * 100

    Active when deviation_pct <= threshold_pct, indicating price is
    significantly below its moving average — a mean-reversion long setup.

    value = percentage deviation from SMA (negative = below MA).
    active = deviation_pct <= threshold_pct (default -10%).

    Default threshold -10% is BTC-calibrated: at SMA(50), a -10% deviation
    lies beyond the typical 1-sigma range in normal BTC volatility regimes.
    threshold_pct is exposed as a parameter so walk-forward optimization
    (F10) can test variants without modifying the function.

    Long-only: this signal does NOT detect significantly above-MA conditions.

    SMA (not EMA) used per SYSTEM_FINAL convention for this signal.

    Special cases:
        - sma[-1] == 0: degenerate; returns active=False, value=NaN.
        - len(close) < period: active=False, value=NaN.
        - NaN or inf in close: active=False, value=NaN.

    Args:
        close: Bar close prices, chronologically ordered.
        period: SMA look-back period. Default 50.
        threshold_pct: Deviation level (negative %) below which signal fires.
                       Default -10.0.

    Returns:
        SignalResult(active, deviation_pct).
    """
    if len(close) < period:
        return _NAN_RESULT

    if close.isnull().any() or not all(math.isfinite(v) for v in close):
        return _NAN_RESULT

    sma = close.rolling(period).mean()
    last_close = float(close.iloc[-1])
    last_sma = float(sma.iloc[-1])

    if math.isnan(last_sma) or last_sma == 0.0:
        return _NAN_RESULT

    deviation_pct = (last_close - last_sma) / last_sma * 100.0
    return SignalResult(active=bool(deviation_pct <= threshold_pct), value=float(deviation_pct))
