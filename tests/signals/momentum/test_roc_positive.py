import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.momentum.roc_positive import is_roc_positive


def _rising(n: int, start: float = 100.0, end: float = 150.0) -> pd.Series:
    return pd.Series(np.linspace(start, end, n), dtype=float)


def _falling(n: int, start: float = 150.0, end: float = 100.0) -> pd.Series:
    return pd.Series(np.linspace(start, end, n), dtype=float)


def _flat(n: int, value: float = 100.0) -> pd.Series:
    return pd.Series(np.full(n, value), dtype=float)


class TestRocPositiveBullishSetup:
    def test_rising_series_fires(self):
        # close[-1] > close[-11]: ROC > 0
        prices = _rising(20)
        result = is_roc_positive(prices)
        assert result.active is True
        assert result.value > 0.0

    def test_value_is_percentage_change(self):
        prices = _rising(20, start=100.0, end=120.0)
        result = is_roc_positive(prices, period=10)
        prior = float(prices.iloc[-(10 + 1)])
        current = float(prices.iloc[-1])
        expected = (current - prior) / prior * 100.0
        assert result.value == pytest.approx(expected, rel=1e-3)


class TestRocBearishSetup:
    def test_falling_series_does_not_fire(self):
        # close[-1] < close[-11]: ROC < 0
        prices = _falling(20)
        result = is_roc_positive(prices)
        assert result.active is False
        assert result.value < 0.0


class TestRocSideways:
    def test_flat_roc_zero(self):
        # Constant price: ROC = 0.0; threshold default is 0 (strictly >), so active=False
        prices = _flat(20)
        result = is_roc_positive(prices)
        assert result.active is False
        assert result.value == pytest.approx(0.0, abs=1e-9)


class TestRocInsufficientData:
    def test_too_short(self):
        # period=10 needs at least 11 bars
        prices = _rising(10)
        result = is_roc_positive(prices, period=10)
        assert result.active is False
        assert math.isnan(result.value)

    def test_empty_series(self):
        result = is_roc_positive(pd.Series([], dtype=float))
        assert result.active is False
        assert math.isnan(result.value)


class TestRocBadInput:
    def test_nan_in_close(self):
        values = list(_rising(20))
        values[5] = float("nan")
        result = is_roc_positive(pd.Series(values))
        assert result.active is False
        assert math.isnan(result.value)

    def test_inf_in_close(self):
        values = list(_rising(20))
        values[15] = float("inf")
        result = is_roc_positive(pd.Series(values))
        assert result.active is False
        assert math.isnan(result.value)


class TestRocEdgeCases:
    def test_exactly_at_threshold_default_does_not_fire(self):
        # ROC = 0.0 exactly; default threshold 0.0 uses >, so active=False
        prices = _flat(20)
        result = is_roc_positive(prices, threshold_pct=0.0)
        assert result.active is False
        assert result.value == pytest.approx(0.0, abs=1e-9)

    def test_custom_threshold_5pct_gates(self):
        # ROC ≈ 3%: fires at threshold=0, does NOT fire at threshold=5
        # Build series: prior = 100, current = 103 → ROC = 3%
        prior_val = 100.0
        current_val = 103.0
        # 11 bars: first bar is prior_val, last is current_val
        values = np.linspace(prior_val, current_val, 11)
        prices = pd.Series(values, dtype=float)
        result_default = is_roc_positive(prices, period=10, threshold_pct=0.0)
        result_strict = is_roc_positive(prices, period=10, threshold_pct=5.0)
        assert result_default.active is True
        assert result_strict.active is False
        assert result_default.value == pytest.approx(result_strict.value, rel=1e-6)
