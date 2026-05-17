import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.mean_reversion.distance_from_ma import is_far_below_ma


def _series(values) -> pd.Series:
    return pd.Series(values, dtype=float)


def _below_ma(period: int = 50, deviation_pct: float = -15.0) -> pd.Series:
    """Series whose SMA(period) is ~100 and last bar sits at the specified deviation."""
    stable = np.full(period - 1, 100.0)
    last = 100.0 * (1.0 + deviation_pct / 100.0)
    return _series(np.append(stable, last))


def _above_ma(period: int = 50, deviation_pct: float = 15.0) -> pd.Series:
    stable = np.full(period - 1, 100.0)
    last = 100.0 * (1.0 + deviation_pct / 100.0)
    return _series(np.append(stable, last))


def _at_ma(period: int = 50) -> pd.Series:
    return _series(np.full(period, 100.0))


class TestDistanceBullishSetup:
    def test_price_far_below_ma_fires(self):
        # Last bar is 15% below SMA → exceeds -10% threshold
        prices = _below_ma(period=50, deviation_pct=-15.0)
        result = is_far_below_ma(prices)
        assert result.active is True
        assert result.value < -10.0

    def test_value_is_negative_deviation(self):
        # _below_ma([100]*49 + [85]): SMA(50) = (49*100 + 85)/50 = 99.7
        # deviation = (85 - 99.7) / 99.7 * 100 ≈ -14.74%
        prices = _below_ma(period=50, deviation_pct=-15.0)
        result = is_far_below_ma(prices)
        expected_sma = (49 * 100.0 + 85.0) / 50.0
        expected_dev = (85.0 - expected_sma) / expected_sma * 100.0
        assert result.value == pytest.approx(expected_dev, rel=1e-3)


class TestDistanceNeutralSetup:
    def test_price_at_ma_does_not_fire(self):
        prices = _at_ma()
        result = is_far_below_ma(prices)
        assert result.active is False
        assert result.value == pytest.approx(0.0, abs=1e-6)

    def test_mild_deviation_does_not_fire(self):
        # -5% deviation is below threshold of -10% (closer to MA) → no fire
        prices = _below_ma(period=50, deviation_pct=-5.0)
        result = is_far_below_ma(prices)
        assert result.active is False
        assert -10.0 < result.value < 0.0


class TestDistanceBearishNotFiring:
    def test_price_far_above_ma_does_not_fire(self):
        # Long-only: +15% above MA must NOT trigger
        prices = _above_ma(period=50, deviation_pct=15.0)
        result = is_far_below_ma(prices)
        assert result.active is False
        assert result.value > 0.0


class TestDistanceInsufficientData:
    def test_too_short(self):
        prices = _series(np.linspace(100, 80, 49))  # < period=50
        result = is_far_below_ma(prices, period=50)
        assert result.active is False
        assert math.isnan(result.value)

    def test_empty_series(self):
        result = is_far_below_ma(_series([]))
        assert result.active is False
        assert math.isnan(result.value)


class TestDistanceBadInput:
    def test_nan_in_close(self):
        values = list(_below_ma(deviation_pct=-15.0))
        values[25] = float("nan")
        result = is_far_below_ma(_series(values))
        assert result.active is False
        assert math.isnan(result.value)

    def test_inf_in_close(self):
        values = list(_below_ma(deviation_pct=-15.0))
        values[10] = float("inf")
        result = is_far_below_ma(_series(values))
        assert result.active is False
        assert math.isnan(result.value)


class TestDistanceEdgeCases:
    def test_exactly_at_threshold(self):
        # -10% exactly → active=True (uses <=, not <)
        # Build a series where SMA ≈ 100 and last bar ≈ 90
        stable = np.full(49, 100.0)
        # SMA of 49×100 + 90 = (4900 + 90) / 50 = 99.8; deviation = (90 - 99.8) / 99.8 * 100 ≈ -9.82%
        # Use a longer stable window to make SMA≈100 more precise
        prices = _series(np.append(np.full(99, 100.0), 90.0))
        result = is_far_below_ma(prices, period=50)
        # SMA of last 50 bars: 49×100 + 90 = 4990 / 50 = 99.8
        # deviation = (90 - 99.8) / 99.8 * 100 ≈ -9.82% → does not reach -10%
        # To hit exactly -10%, we need close = sma * 0.9; sma = 100 exactly
        exact = _series(np.append(np.full(49, 100.0), [100.0] * 1))
        # Override last bar to be exactly 10% below a SMA of 100
        vals = list(np.full(49, 100.0)) + [90.0]
        # SMA(50) of [100]*49 + [90] = (49*100 + 90)/50 = 99.8, not 100
        # Use 200 stable bars so SMA(50) = 100 exactly when last=90
        prices2 = _series(np.append(np.full(200, 100.0), 90.0))
        result2 = is_far_below_ma(prices2, period=50)
        # SMA(50) of 200×100 + 90 → last 50 = [100]*49 + [90] = 99.8; still not exact
        # Compute the correct last value for exactly -10% deviation
        # sma of [100]*49 + [x] = (4900 + x) / 50; want x/sma = 0.9 → x = 0.9 * (4900+x)/50
        # 50x = 0.9*(4900+x) → 50x = 4410 + 0.9x → 49.1x = 4410 → x = 89.817...
        x_exact = 4410.0 / 49.1
        prices3 = _series(np.append(np.full(49, 100.0), x_exact))
        result3 = is_far_below_ma(prices3, period=50)
        assert result3.value == pytest.approx(-10.0, rel=1e-3)
        assert result3.active is True  # <= threshold, should fire

    def test_custom_threshold_stricter(self):
        # deviation = -12%: fires at threshold=-10%, does NOT fire at threshold=-15%
        prices = _below_ma(period=50, deviation_pct=-12.0)
        r_default = is_far_below_ma(prices, threshold_pct=-10.0)
        r_strict = is_far_below_ma(prices, threshold_pct=-15.0)
        assert r_default.active is True
        assert r_strict.active is False
        # Both return same value (deviation doesn't depend on threshold)
        assert r_default.value == pytest.approx(r_strict.value, rel=1e-6)
