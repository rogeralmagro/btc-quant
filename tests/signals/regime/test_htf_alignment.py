import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.regime.htf_alignment import is_htf_aligned_bullish


def _rising(n: int) -> pd.Series:
    return pd.Series(np.linspace(100, 300, n), dtype=float)


def _falling(n: int) -> pd.Series:
    return pd.Series(np.linspace(300, 100, n), dtype=float)


def _flat(n: int) -> pd.Series:
    t = np.arange(n)
    return pd.Series(100 + 5 * np.sin(t * 0.2), dtype=float)


class TestHtfAlignmentBullish:
    def test_both_bullish(self):
        # Both daily (200 bars) and weekly (40 bars) are rising.
        result = is_htf_aligned_bullish(
            close_1d=_rising(300),
            close_1w=_rising(60),
        )
        assert result.active is True
        assert result.value > 0


class TestHtfAlignmentBearish:
    def test_both_bearish(self):
        result = is_htf_aligned_bullish(
            close_1d=_falling(300),
            close_1w=_falling(60),
        )
        assert result.active is False
        assert result.value < 0


class TestHtfAlignmentSideways:
    def test_sideways_both_frames(self):
        # Oscillating: EMAs converge to the mean; spread near zero.
        # active state is non-deterministic — verify it's a valid bool and value finite.
        result = is_htf_aligned_bullish(
            close_1d=_flat(300),
            close_1w=_flat(60),
        )
        assert isinstance(result.active, bool)
        assert math.isfinite(result.value)


class TestHtfAlignmentInsufficientData:
    def test_daily_too_short(self):
        # daily needs ≥ daily_slow=200 bars
        result = is_htf_aligned_bullish(
            close_1d=_rising(199),
            close_1w=_rising(60),
        )
        assert result.active is False
        assert math.isnan(result.value)

    def test_weekly_too_short(self):
        # weekly needs ≥ weekly_slow=40 bars
        result = is_htf_aligned_bullish(
            close_1d=_rising(300),
            close_1w=_rising(39),
        )
        assert result.active is False
        assert math.isnan(result.value)

    def test_both_too_short(self):
        result = is_htf_aligned_bullish(
            close_1d=_rising(50),
            close_1w=_rising(10),
        )
        assert result.active is False
        assert math.isnan(result.value)


class TestHtfAlignmentBadInput:
    def test_nan_in_daily(self):
        daily = list(np.linspace(100, 300, 300))
        daily[100] = float("nan")
        result = is_htf_aligned_bullish(
            close_1d=pd.Series(daily),
            close_1w=_rising(60),
        )
        assert result.active is False
        assert math.isnan(result.value)

    def test_inf_in_weekly(self):
        weekly = list(np.linspace(100, 300, 60))
        weekly[20] = float("inf")
        result = is_htf_aligned_bullish(
            close_1d=_rising(300),
            close_1w=pd.Series(weekly),
        )
        assert result.active is False
        assert math.isnan(result.value)


class TestHtfAlignmentMixed:
    def test_daily_bullish_weekly_bearish(self):
        # Daily rising, weekly falling → both must confirm → active=False.
        result = is_htf_aligned_bullish(
            close_1d=_rising(300),
            close_1w=_falling(60),
        )
        assert result.active is False

    def test_daily_bearish_weekly_bullish(self):
        # Daily falling, weekly rising → active=False.
        result = is_htf_aligned_bullish(
            close_1d=_falling(300),
            close_1w=_rising(60),
        )
        assert result.active is False
