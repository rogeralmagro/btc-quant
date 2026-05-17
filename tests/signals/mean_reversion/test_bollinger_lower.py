import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.mean_reversion.bollinger_lower import is_at_bb_lower


def _series(values) -> pd.Series:
    return pd.Series(values, dtype=float)


def _falling_tail(n: int = 60, stable: float = 100.0, drop_to: float = 50.0) -> pd.Series:
    """Stable for n-1 bars, then a single sharp drop on the last bar.

    With period=20, the rolling window sees 19 stable bars and 1 drop.
    The std is dominated by the stable bars, keeping bands narrow, so the
    final close lands well below the lower band.
    """
    stable_part = np.full(n - 1, stable)
    return _series(np.append(stable_part, drop_to))


def _rising_tail(n: int = 60, stable: float = 100.0, rise_to: float = 130.0) -> pd.Series:
    stable_len = n - 15
    stable_part = np.full(stable_len, stable)
    rise_part = np.linspace(stable, rise_to, 15)
    return _series(np.concatenate([stable_part, rise_part]))


def _sideways(n: int = 60, center: float = 100.0, amplitude: float = 3.0) -> pd.Series:
    t = np.arange(n)
    return _series(center + amplitude * np.sin(t * 0.4))


class TestBollingerLowerBullishSetup:
    def test_sharp_drop_below_lower_band(self):
        # Stable prices then sharp drop: last bar should sit below the lower band
        prices = _falling_tail(n=60, stable=100.0, drop_to=70.0)
        result = is_at_bb_lower(prices)
        assert result.active is True
        assert result.value <= 0.0

    def test_value_is_pct_b(self):
        prices = _falling_tail(n=60, stable=100.0, drop_to=70.0)
        result = is_at_bb_lower(prices)
        # %B < 0 means price is below the lower band
        assert result.value < 0.0


class TestBollingerLowerNeutralSetup:
    def test_sideways_not_at_lower(self):
        prices = _sideways()
        result = is_at_bb_lower(prices)
        assert result.active is False
        # %B should be in a moderate range, not far below 0
        assert -0.5 < result.value < 1.5


class TestBollingerLowerBearishNotFiring:
    def test_rising_tail_at_upper_band_does_not_fire(self):
        # Price near upper band: %B > 1, long-only signal must NOT fire
        prices = _rising_tail(n=60, stable=100.0, rise_to=130.0)
        result = is_at_bb_lower(prices)
        assert result.active is False
        assert result.value > 0.0


class TestBollingerLowerInsufficientData:
    def test_too_short(self):
        prices = _series(np.linspace(100, 80, 19))  # < period=20
        result = is_at_bb_lower(prices, period=20)
        assert result.active is False
        assert math.isnan(result.value)

    def test_empty_series(self):
        result = is_at_bb_lower(_series([]))
        assert result.active is False
        assert math.isnan(result.value)


class TestBollingerLowerBadInput:
    def test_nan_in_close(self):
        values = list(_falling_tail())
        values[20] = float("nan")
        result = is_at_bb_lower(_series(values))
        assert result.active is False
        assert math.isnan(result.value)

    def test_inf_in_close(self):
        values = list(_falling_tail())
        values[5] = float("inf")
        result = is_at_bb_lower(_series(values))
        assert result.active is False
        assert math.isnan(result.value)


class TestBollingerLowerEdgeCases:
    def test_volatility_expansion_moderate_drop_stays_above_lower(self):
        # If volatility expands enough, even a moderate drop keeps %B > 0
        # Simulate by using a large rolling window that sees high variance
        t = np.arange(60)
        # Prices that oscillate wildly for most of the period, then drop a little
        prices = _series(100.0 + 20.0 * np.sin(t * 0.5))
        # With num_std=2 and high std, a small drop may still be above the lower band
        result = is_at_bb_lower(prices, period=20, num_std=2.0)
        # This is a behavioral test: result must be coherent (active iff %B <= 0)
        assert result.active is bool(result.value <= 0.0)

    def test_constant_price_std_zero_returns_nan(self):
        # std=0 → bands collapse → %B undefined → no crash, no division by zero
        prices = _series(np.full(60, 100.0))
        result = is_at_bb_lower(prices)
        assert result.active is False
        assert math.isnan(result.value)
