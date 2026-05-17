import math

import pandas as pd

from btc_quant.signals.types import SignalResult

_NAN_RESULT = SignalResult(active=False, value=float("nan"))


def is_at_bb_lower(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> SignalResult:
    """Evaluate the Bollinger Band lower-touch mean-reversion signal.

    Computes Bollinger Bands and the normalized %B position:
        mid   = close.rolling(period).mean()
        std   = close.rolling(period).std(ddof=0)   # population std
        upper = mid + num_std * std
        lower = mid - num_std * std
        %B    = (close[-1] - lower[-1]) / (upper[-1] - lower[-1])

    %B convention:
        %B < 0   → price below lower band  (signal fires)
        %B = 0   → price exactly at lower band  (signal fires)
        %B = 0.5 → price at midline
        %B = 1.0 → price at upper band
        %B > 1   → price above upper band

    Active when %B <= 0 on the last bar, indicating price at or below the
    lower band — a mean-reversion long setup.

    Long-only: this signal does NOT detect upper-band touches (overbought).

    value = %B (Bollinger %B), the normalized position within the bands.
    active = %B <= 0.0.

    ddof=0 (population std) used for reproducibility with standard TA
    libraries (TradingView, TA-Lib).

    Special cases:
        - std == 0 (constant price): bands collapse, %B is undefined.
          Returns active=False, value=NaN. No division by zero.
        - len(close) < period: active=False, value=NaN.
        - NaN or inf in close: active=False, value=NaN.

    Args:
        close: Bar close prices, chronologically ordered.
        period: Rolling window for mean and std. Default 20.
        num_std: Number of standard deviations for band width. Default 2.0.

    Returns:
        SignalResult(active, pct_b).
    """
    if len(close) < period:
        return _NAN_RESULT

    if close.isnull().any() or not all(math.isfinite(v) for v in close):
        return _NAN_RESULT

    mid = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)

    last_close = float(close.iloc[-1])
    last_mid = float(mid.iloc[-1])
    last_std = float(std.iloc[-1])

    if math.isnan(last_std) or last_std == 0.0:
        return _NAN_RESULT

    lower = last_mid - num_std * last_std
    upper = last_mid + num_std * last_std
    band_width = upper - lower

    pct_b = (last_close - lower) / band_width
    return SignalResult(active=bool(pct_b <= 0.0), value=float(pct_b))
