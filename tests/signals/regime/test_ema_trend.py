import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.regime.ema_trend import is_bullish_ema_trend


def _series(values) -> pd.Series:
    return pd.Series(values, dtype=float)


class TestEmaTrendBullish:
    def test_bullish_synthetic(self):
        # Linearly rising prices: EMA50 will be above EMA200 at the end.
        prices = _series(np.linspace(100, 300, 300))
        result = is_bullish_ema_trend(prices)
        assert result.active is True
        assert result.value > 0

    def test_value_equals_ema_diff(self):
        prices = _series(np.linspace(100, 300, 300))
        result = is_bullish_ema_trend(prices)
        ema_fast = prices.ewm(span=50, adjust=False).mean().iloc[-1]
        ema_slow = prices.ewm(span=200, adjust=False).mean().iloc[-1]
        assert result.value == pytest.approx(ema_fast - ema_slow, rel=1e-3)


class TestEmaTrendBearish:
    def test_bearish_synthetic(self):
        # Linearly falling prices: EMA50 will be below EMA200 at the end.
        prices = _series(np.linspace(300, 100, 300))
        result = is_bullish_ema_trend(prices)
        assert result.active is False
        assert result.value < 0


class TestEmaTrendSideways:
    def test_sideways_synthetic(self):
        # Oscillating around a flat level: EMAs converge, spread near zero.
        t = np.arange(300)
        prices = _series(100 + 5 * np.sin(t * 0.2))
        result = is_bullish_ema_trend(prices)
        # Both EMAs track the mean closely; spread is small but not guaranteed
        # to be positive. We assert value is finite and behaviour is consistent.
        assert math.isfinite(result.value)
        assert isinstance(result.active, bool)


class TestEmaTrendInsufficientData:
    def test_too_short(self):
        prices = _series(np.linspace(100, 200, 199))  # < slow=200
        result = is_bullish_ema_trend(prices)
        assert result.active is False
        assert math.isnan(result.value)

    def test_exactly_slow_minus_one(self):
        prices = _series(np.ones(199))
        result = is_bullish_ema_trend(prices, fast=50, slow=200)
        assert result.active is False
        assert math.isnan(result.value)

    def test_empty_series(self):
        result = is_bullish_ema_trend(_series([]))
        assert result.active is False
        assert math.isnan(result.value)


class TestEmaTrendBadInput:
    def test_nan_in_close(self):
        values = list(np.linspace(100, 300, 300))
        values[150] = float("nan")
        result = is_bullish_ema_trend(_series(values))
        assert result.active is False
        assert math.isnan(result.value)

    def test_inf_in_close(self):
        values = list(np.linspace(100, 300, 300))
        values[50] = float("inf")
        result = is_bullish_ema_trend(_series(values))
        assert result.active is False
        assert math.isnan(result.value)
