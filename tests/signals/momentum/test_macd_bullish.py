import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.momentum.macd_bullish import is_macd_bullish


def _rising(n: int, start: float = 100.0, end: float = 200.0) -> pd.Series:
    return pd.Series(np.linspace(start, end, n), dtype=float)


def _falling(n: int, start: float = 200.0, end: float = 100.0) -> pd.Series:
    return pd.Series(np.linspace(start, end, n), dtype=float)


def _sideways(n: int, center: float = 100.0, amplitude: float = 2.0) -> pd.Series:
    t = np.arange(n)
    return pd.Series(center + amplitude * np.sin(t * 0.3), dtype=float)


class TestMacdBullishSetup:
    def test_rising_series_fires(self):
        # Sustained uptrend: EMA(12) pulls above EMA(26), MACD > Signal
        prices = _rising(60)
        result = is_macd_bullish(prices)
        assert result.active is True
        assert result.value > 0.0

    def test_recent_bullish_cross(self):
        # Flat for a long time, then sharply rises — MACD just crossed Signal up
        flat = np.full(50, 100.0)
        ramp = np.linspace(100.0, 115.0, 10)
        prices = pd.Series(np.concatenate([flat, ramp]), dtype=float)
        result = is_macd_bullish(prices)
        assert result.active is True
        assert result.value > 0.0

    def test_sustained_bullish_large_histogram(self):
        # Long, steep uptrend → histogram clearly positive
        prices = _rising(80, start=100.0, end=300.0)
        result = is_macd_bullish(prices)
        assert result.active is True
        assert result.value > 0.0


class TestMacdBearishSetup:
    def test_falling_series_does_not_fire(self):
        prices = _falling(60)
        result = is_macd_bullish(prices)
        assert result.active is False
        assert result.value < 0.0


class TestMacdSideways:
    def test_sideways_not_bullish(self):
        prices = _sideways(80)
        result = is_macd_bullish(prices)
        assert isinstance(result.active, bool)
        assert math.isfinite(result.value)


class TestMacdInsufficientData:
    def test_too_short(self):
        # Needs slow + signal = 26 + 9 = 35 bars minimum
        prices = _rising(34)
        result = is_macd_bullish(prices)
        assert result.active is False
        assert math.isnan(result.value)

    def test_empty_series(self):
        result = is_macd_bullish(pd.Series([], dtype=float))
        assert result.active is False
        assert math.isnan(result.value)


class TestMacdBadInput:
    def test_nan_in_close(self):
        values = list(_rising(60))
        values[30] = float("nan")
        result = is_macd_bullish(pd.Series(values))
        assert result.active is False
        assert math.isnan(result.value)

    def test_inf_in_close(self):
        values = list(_rising(60))
        values[10] = float("inf")
        result = is_macd_bullish(pd.Series(values))
        assert result.active is False
        assert math.isnan(result.value)
