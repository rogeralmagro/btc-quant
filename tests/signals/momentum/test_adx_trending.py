import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.momentum.adx_trending import is_bullish_trending


def _ohlc_trending_up(n: int, start: float = 100.0, end: float = 200.0, spread: float = 2.0):
    """Rising close with fixed spread; +DM dominates."""
    close = pd.Series(np.linspace(start, end, n), dtype=float)
    high = close + spread
    low = close - spread
    return high, low, close


def _ohlc_trending_down(n: int, start: float = 200.0, end: float = 100.0, spread: float = 2.0):
    """Falling close with fixed spread; -DM dominates."""
    close = pd.Series(np.linspace(start, end, n), dtype=float)
    high = close + spread
    low = close - spread
    return high, low, close


def _ohlc_sideways(n: int, center: float = 100.0, amplitude: float = 1.5):
    """Oscillating prices: no consistent directional movement → low ADX."""
    t = np.arange(n)
    close = pd.Series(center + amplitude * np.sin(t * 0.3), dtype=float)
    high = close + 0.5
    low = close - 0.5
    return high, low, close


class TestAdxBullishSetup:
    def test_strong_uptrend_fires(self):
        # Steep uptrend over many bars → ADX > 25 and +DI > -DI
        high, low, close = _ohlc_trending_up(n=60, start=100.0, end=250.0, spread=1.0)
        result = is_bullish_trending(high, low, close)
        assert result.active is True
        assert result.value > 25.0


class TestAdxBearishSetup:
    def test_falling_trend_does_not_fire(self):
        # Strong downtrend: ADX > 25 but -DI > +DI → long-only, active=False
        high, low, close = _ohlc_trending_down(n=60, start=250.0, end=100.0, spread=1.0)
        result = is_bullish_trending(high, low, close)
        assert result.active is False
        # ADX may still be high (trend is strong); value is the raw ADX
        assert math.isfinite(result.value)


class TestAdxSideways:
    def test_sideways_low_adx_does_not_fire(self):
        # Oscillating prices: ADX < 25, no trending signal regardless of DI sign
        high, low, close = _ohlc_sideways(n=60)
        result = is_bullish_trending(high, low, close)
        assert result.active is False


class TestAdxInsufficientData:
    def test_too_short(self):
        # Needs period * 2 = 28 bars minimum
        high, low, close = _ohlc_trending_up(n=27)
        result = is_bullish_trending(high, low, close)
        assert result.active is False
        assert math.isnan(result.value)

    def test_empty_series(self):
        empty = pd.Series([], dtype=float)
        result = is_bullish_trending(empty, empty, empty)
        assert result.active is False
        assert math.isnan(result.value)


class TestAdxBadInput:
    def test_nan_in_close(self):
        high, low, close = _ohlc_trending_up(n=60)
        close = close.copy()
        close.iloc[30] = float("nan")
        result = is_bullish_trending(high, low, close)
        assert result.active is False
        assert math.isnan(result.value)

    def test_inf_in_high(self):
        high, low, close = _ohlc_trending_up(n=60)
        high = high.copy()
        high.iloc[5] = float("inf")
        result = is_bullish_trending(high, low, close)
        assert result.active is False
        assert math.isnan(result.value)


class TestAdxEdgeCases:
    def test_trending_bearish_critical_long_only(self):
        # This is the key long-only gate: strong downtrend must NOT fire
        high, low, close = _ohlc_trending_down(n=80, start=300.0, end=100.0, spread=1.5)
        result = is_bullish_trending(high, low, close)
        assert result.active is False

    def test_custom_threshold_stricter(self):
        # Build a noisy uptrend that produces a moderate ADX (not 100).
        # Random walk biased upward with significant noise dampens DI consistency.
        rng = np.random.default_rng(42)
        n = 80
        returns = 0.5 + rng.normal(0, 3.0, n)  # small positive bias, high noise
        close_vals = 100.0 + np.cumsum(returns)
        close = pd.Series(close_vals, dtype=float)
        high = close + 2.0
        low = close - 2.0

        result = is_bullish_trending(high, low, close, adx_threshold=25.0)
        adx_value = result.value
        assert math.isfinite(adx_value)

        # Threshold just above the measured ADX should NOT fire; at or below should
        threshold_above = adx_value + 1.0
        threshold_below = adx_value - 1.0

        r_above = is_bullish_trending(high, low, close, adx_threshold=threshold_above)
        r_below = is_bullish_trending(high, low, close, adx_threshold=threshold_below)

        assert r_above.active is False
        # r_below fires only if also +DI > -DI; value is the same either way
        assert r_above.value == pytest.approx(r_below.value, rel=1e-6)
