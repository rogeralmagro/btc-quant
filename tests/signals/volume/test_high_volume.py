import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.volume.high_volume import is_high_volume


def _volume(n: int, base: float = 1000.0) -> pd.Series:
    return pd.Series(np.full(n, base), dtype=float)


def _volume_spike(window: int = 252, base: float = 1000.0, spike: float = 10_000.0) -> pd.Series:
    """window+1 bars: window historical bars at base, last bar is spike."""
    vals = list(np.full(window, base)) + [spike]
    return pd.Series(vals, dtype=float)


def _volume_low(window: int = 252, base: float = 1000.0, low: float = 10.0) -> pd.Series:
    """window+1 bars: window historical bars at base, last bar is very low."""
    vals = list(np.full(window, base)) + [low]
    return pd.Series(vals, dtype=float)


class TestHighVolumeBullishSetup:
    def test_spike_above_threshold_fires(self):
        # Current bar volume >> all historical → percentile ≈ 100 > 75
        volume = _volume_spike()
        result = is_high_volume(volume)
        assert result.active is True
        assert result.value > 75.0


class TestHighVolumeBearishSetup:
    def test_low_volume_below_threshold(self):
        # Current bar volume << historical → percentile ≈ 0
        volume = _volume_low()
        result = is_high_volume(volume)
        assert result.active is False
        assert result.value < 75.0


class TestHighVolumeNeutral:
    def test_median_volume_not_high(self):
        # Current bar exactly matches historical mean: percentile ≈ 50
        volume = _volume(253)  # 252 historical + 1 current all equal
        result = is_high_volume(volume)
        assert result.active is False
        assert 40.0 < result.value < 60.0


class TestHighVolumeInsufficientData:
    def test_too_short(self):
        # rolling_window=252 needs at least 253 bars
        volume = pd.Series(np.full(252, 1000.0), dtype=float)
        result = is_high_volume(volume, rolling_window=252)
        assert result.active is False
        assert math.isnan(result.value)

    def test_empty_series(self):
        result = is_high_volume(pd.Series([], dtype=float))
        assert result.active is False
        assert math.isnan(result.value)


class TestHighVolumeBadInput:
    def test_nan_in_volume(self):
        volume = _volume_spike()
        volume = volume.copy()
        volume.iloc[100] = float("nan")
        result = is_high_volume(volume)
        assert result.active is False
        assert math.isnan(result.value)

    def test_inf_in_volume(self):
        volume = _volume_spike()
        volume = volume.copy()
        volume.iloc[50] = float("inf")
        result = is_high_volume(volume)
        assert result.active is False
        assert math.isnan(result.value)


class TestHighVolumeEdgeCases:
    def test_custom_threshold_gates(self):
        # Build series where current bar ranks at percentile 80:
        # 252 historical: 200 bars at 1000, 52 bars at 2000
        # current = 1500 → n_below = 200, n_equal = 0
        # percentile = 200/252 * 100 ≈ 79.4
        historical = list(np.full(200, 1000.0)) + list(np.full(52, 2000.0))
        current_bar = [1500.0]
        volume = pd.Series(historical + current_bar, dtype=float)

        r_default = is_high_volume(volume, threshold_percentile=75.0)
        r_strict = is_high_volume(volume, threshold_percentile=90.0)

        assert r_default.active is True
        assert r_strict.active is False
        assert r_default.value == pytest.approx(r_strict.value, rel=1e-6)

    def test_constant_historical_midpoint_convention(self):
        # Current == every historical bar → ties contribute 0.5
        # percentile = (0 + 0.5 * 252) / 252 * 100 = 50.0 → active=False (< 75)
        volume = _volume(253)
        result = is_high_volume(volume, rolling_window=252, threshold_percentile=75.0)
        assert result.active is False
        assert result.value == pytest.approx(50.0, rel=1e-3)
