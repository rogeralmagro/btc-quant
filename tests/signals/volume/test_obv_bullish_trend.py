import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.volume.obv_bullish_trend import is_obv_bullish_trend


def _rising_close(n: int, start: float = 100.0, end: float = 150.0) -> pd.Series:
    return pd.Series(np.linspace(start, end, n), dtype=float)


def _falling_close(n: int, start: float = 150.0, end: float = 100.0) -> pd.Series:
    return pd.Series(np.linspace(start, end, n), dtype=float)


def _flat_close(n: int, val: float = 100.0) -> pd.Series:
    return pd.Series(np.full(n, val), dtype=float)


def _uniform_volume(n: int, vol: float = 1000.0) -> pd.Series:
    return pd.Series(np.full(n, vol), dtype=float)


class TestObvBullishSetup:
    def test_rising_price_constant_volume_fires(self):
        # Every bar up → OBV grows monotonically → OBV > SMA(OBV)
        close = _rising_close(40)
        volume = _uniform_volume(40)
        result = is_obv_bullish_trend(close, volume)
        assert result.active is True
        assert result.value > 0.0

    def test_constant_volume_on_rising_prices(self):
        # Explicit: 30 rising bars, constant 500 volume each
        close = pd.Series(np.arange(100.0, 130.0), dtype=float)
        volume = pd.Series(np.full(30, 500.0), dtype=float)
        result = is_obv_bullish_trend(close, volume, ma_period=10)
        assert result.active is True
        assert result.value > 0.0


class TestObvBearishSetup:
    def test_falling_price_obv_below_ma(self):
        # Every bar down → OBV decreases → OBV < SMA(OBV)
        close = _falling_close(40)
        volume = _uniform_volume(40)
        result = is_obv_bullish_trend(close, volume)
        assert result.active is False
        assert result.value < 0.0

    def test_volume_spike_on_down_bar(self):
        # Stable close, then one huge down-bar with massive volume → OBV craters
        close_vals = list(np.full(30, 100.0)) + [90.0]  # last bar is down
        volume_vals = list(np.full(30, 100.0)) + [50_000.0]  # huge volume on down bar
        close = pd.Series(close_vals, dtype=float)
        volume = pd.Series(volume_vals, dtype=float)
        result = is_obv_bullish_trend(close, volume, ma_period=20)
        assert result.active is False


class TestObvSideways:
    def test_flat_close_no_obv_movement(self):
        # No price change → direction=0 on every bar → OBV=0 → OBV == SMA(OBV)
        # active=False since OBV is NOT strictly > SMA(OBV)
        close = _flat_close(30)
        volume = _uniform_volume(30)
        result = is_obv_bullish_trend(close, volume, ma_period=20)
        assert result.active is False
        assert result.value == pytest.approx(0.0, abs=1e-9)


class TestObvInsufficientData:
    def test_too_short(self):
        # ma_period=20 needs at least 21 bars
        close = _rising_close(20)
        volume = _uniform_volume(20)
        result = is_obv_bullish_trend(close, volume, ma_period=20)
        assert result.active is False
        assert math.isnan(result.value)

    def test_empty_series(self):
        result = is_obv_bullish_trend(
            pd.Series([], dtype=float), pd.Series([], dtype=float)
        )
        assert result.active is False
        assert math.isnan(result.value)


class TestObvBadInput:
    def test_nan_in_close(self):
        close = _rising_close(30)
        close = close.copy()
        close.iloc[15] = float("nan")
        result = is_obv_bullish_trend(close, _uniform_volume(30))
        assert result.active is False
        assert math.isnan(result.value)

    def test_inf_in_volume(self):
        volume = _uniform_volume(30)
        volume = volume.copy()
        volume.iloc[10] = float("inf")
        result = is_obv_bullish_trend(_rising_close(30), volume)
        assert result.active is False
        assert math.isnan(result.value)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            is_obv_bullish_trend(_rising_close(30), _uniform_volume(25))
