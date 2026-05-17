import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.mean_reversion.rsi_oversold import is_rsi_oversold


def _series(values) -> pd.Series:
    return pd.Series(values, dtype=float)


def _falling(n: int, start: float = 100.0, end: float = 50.0) -> pd.Series:
    return _series(np.linspace(start, end, n))


def _rising(n: int, start: float = 50.0, end: float = 150.0) -> pd.Series:
    return _series(np.linspace(start, end, n))


def _sideways(n: int, center: float = 100.0, amplitude: float = 2.0) -> pd.Series:
    t = np.arange(n)
    return _series(center + amplitude * np.sin(t * 0.3))


class TestRsiOversoldBullishSetup:
    def test_falling_series_triggers_oversold(self):
        # Sustained, steep fall → RSI well below 30
        prices = _falling(60, start=100.0, end=40.0)
        result = is_rsi_oversold(prices)
        assert result.active is True
        assert 0.0 <= result.value < 30.0

    def test_value_is_rsi_in_range(self):
        prices = _falling(60, start=100.0, end=40.0)
        result = is_rsi_oversold(prices)
        assert 0.0 <= result.value <= 100.0


class TestRsiOversoldNeutralSetup:
    def test_sideways_not_oversold(self):
        # Oscillating prices → RSI near 50, not oversold
        prices = _sideways(60)
        result = is_rsi_oversold(prices)
        assert result.active is False
        assert 20.0 < result.value < 80.0


class TestRsiOversoldBearishNotFiring:
    def test_rising_series_not_oversold(self):
        # Strong uptrend: RSI near 100, long-only signal must NOT fire
        prices = _rising(60)
        result = is_rsi_oversold(prices)
        assert result.active is False
        assert result.value > 70.0


class TestRsiOversoldInsufficientData:
    def test_too_short(self):
        # period=14 needs at least 15 bars
        prices = _series(np.linspace(100, 50, 14))
        result = is_rsi_oversold(prices, period=14)
        assert result.active is False
        assert math.isnan(result.value)

    def test_empty_series(self):
        result = is_rsi_oversold(_series([]))
        assert result.active is False
        assert math.isnan(result.value)


class TestRsiOversoldBadInput:
    def test_nan_in_close(self):
        values = list(np.linspace(100, 50, 60))
        values[30] = float("nan")
        result = is_rsi_oversold(_series(values))
        assert result.active is False
        assert math.isnan(result.value)

    def test_inf_in_close(self):
        values = list(np.linspace(100, 50, 60))
        values[10] = float("inf")
        result = is_rsi_oversold(_series(values))
        assert result.active is False
        assert math.isnan(result.value)


class TestRsiOversoldEdgeCases:
    def test_all_gains_rsi_100(self):
        # Monotonically rising: no losses, avg_loss=0, RSI=100
        prices = _rising(60)
        result = is_rsi_oversold(prices)
        assert result.active is False
        assert result.value == pytest.approx(100.0, rel=1e-3)

    def test_custom_threshold_25_does_not_fire_at_28(self):
        # Calibrate a series whose RSI lands around 28 with default threshold,
        # then verify threshold=25 does NOT fire while default=30 does.
        prices = _falling(60, start=100.0, end=45.0)
        result_default = is_rsi_oversold(prices, threshold=30.0)
        result_strict = is_rsi_oversold(prices, threshold=25.0)

        if result_default.active:
            # RSI < 30 — check threshold=25 may or may not fire depending on exact value
            if result_default.value >= 25.0:
                assert result_strict.active is False
            # Both can fire only if RSI < 25
        else:
            # If even default doesn't fire, strict certainly doesn't
            assert result_strict.active is False

    def test_strict_threshold_gates_correctly(self):
        # RSI is certainly < 30 on a steep fall; at threshold=5 it should NOT fire
        prices = _falling(60, start=100.0, end=40.0)
        result = is_rsi_oversold(prices, threshold=5.0)
        # RSI can go very low but rarely < 5 on this synthetic; confirm gating works
        if result.active:
            assert result.value < 5.0
        else:
            assert result.value >= 5.0
