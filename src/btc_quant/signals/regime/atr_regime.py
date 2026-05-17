import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from btc_quant.signals.types import SignalResult


class AtrRegime(str, Enum):
    CALM = "calm"
    NORMAL = "normal"
    VOLATILE = "volatile"
    EXTREME = "extreme"


@dataclass(frozen=True)
class AtrRegimeResult:
    """Rich result from ATR regime classification.

    Attributes:
        regime: Categorical regime label (CALM / NORMAL / VOLATILE / EXTREME).
        percentile: Current ATR rank vs trailing rolling_window, 0–100.
                    NaN when insufficient data.
        atr_value: Current ATR(atr_period) value. NaN when insufficient data.
        active: True when regime is not EXTREME (position entry is allowed).
                STRAT-07 blocks new entries during extreme volatility (Circuit
                Breaker trigger 4 per SYSTEM_FINAL).
    """

    regime: AtrRegime
    percentile: float
    atr_value: float
    active: bool


_INSUFFICIENT = AtrRegimeResult(
    regime=AtrRegime.NORMAL,
    percentile=float("nan"),
    atr_value=float("nan"),
    active=False,
)


def _regime_from_percentile(p: float) -> AtrRegime:
    if p >= 95:
        return AtrRegime.EXTREME
    if p >= 75:
        return AtrRegime.VOLATILE
    if p >= 25:
        return AtrRegime.NORMAL
    return AtrRegime.CALM


def classify_atr_regime(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    rolling_window: int = 252,
) -> AtrRegimeResult:
    """Classify the current ATR-based volatility regime.

    True Range (TR) = max(high - low, |high - prev_close|, |low - prev_close|).
    ATR = simple moving average of TR over atr_period bars (not Wilder's EMA,
    for reproducibility).

    The current ATR value is ranked as a percentile against the ATR values over
    the preceding rolling_window bars using the midpoint convention: ties
    contribute 0.5 each, so a constant historical window yields percentile=50
    (NORMAL regime), not 0 (CALM). Regime bands:
        percentile < 25       → CALM
        25 <= percentile < 75 → NORMAL
        75 <= percentile < 95 → VOLATILE
        percentile >= 95      → EXTREME

    Active when regime is not EXTREME — STRAT-07 does not open new positions
    during extreme volatility (SYSTEM_FINAL Circuit Breaker trigger 4).

    Edge cases:
        - len < atr_period + rolling_window: returns _INSUFFICIENT
          (regime=NORMAL, percentile=NaN, atr_value=NaN, active=False).
        - NaN rows in TR are skipped when computing the SMA.

    Args:
        high: Bar highs, chronologically ordered.
        low: Bar lows, chronologically ordered.
        close: Bar closes, chronologically ordered.
        atr_period: Look-back for ATR SMA. Default 14.
        rolling_window: Historical window for percentile ranking. Default 252.

    Returns:
        AtrRegimeResult with regime, percentile, atr_value, and active flag.
    """
    min_required = atr_period + rolling_window
    if len(close) < min_required:
        return _INSUFFICIENT

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.rolling(window=atr_period, min_periods=atr_period).mean()

    current_atr = atr.iloc[-1]
    if not math.isfinite(current_atr):
        return _INSUFFICIENT

    window_atr = atr.iloc[-(rolling_window + 1) : -1].dropna()
    if len(window_atr) == 0:
        return _INSUFFICIENT

    n = len(window_atr)
    n_below = int((window_atr < current_atr).sum())
    n_equal = int((window_atr == current_atr).sum())
    # Midpoint convention: equal values contribute 0.5 each, so a value
    # identical to all historical values ranks at the 50th percentile (NORMAL).
    percentile = float((n_below + 0.5 * n_equal) / n * 100)

    regime = _regime_from_percentile(percentile)
    return AtrRegimeResult(
        regime=regime,
        percentile=percentile,
        atr_value=float(current_atr),
        active=bool(regime is not AtrRegime.EXTREME),
    )


def is_atr_regime_tradeable(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 14,
    rolling_window: int = 252,
) -> SignalResult:
    """Wrapper that reduces AtrRegimeResult to SignalResult for the confluence scorer.

    active = True when regime is not EXTREME.
    value  = percentile rank (0–100). NaN when insufficient data.

    Args:
        high: Bar highs.
        low: Bar lows.
        close: Bar closes.
        atr_period: ATR look-back period. Default 14.
        rolling_window: Percentile ranking window. Default 252.

    Returns:
        SignalResult(active, percentile).
    """
    result = classify_atr_regime(high, low, close, atr_period, rolling_window)
    return SignalResult(active=result.active, value=result.percentile)
