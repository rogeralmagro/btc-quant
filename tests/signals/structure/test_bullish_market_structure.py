import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.structure.bullish_market_structure import (
    is_bullish_market_structure,
    _find_swings,
)


# ---------------------------------------------------------------------------
# Hand-verified synthetic series (swing_lookback=2, 14 bars)
# Valid pivot range: indices 2..11 (need 2 bars on each side)
#
# BULLISH (HH + HL):
#   high = [98,99, 102, 100,99,98, 101, 100,99,98, 110, 108,107,106]
#                   ^2                   ^6                  ^10
#   swing highs at: (2,102), (6,101), (10,110)  → last two: 101→110 (HH ✓)
#
#   low  = [95,94, 93, 95,96,94, 92, 94,96,95, 93, 95,96,97]
#                   ^2                ^6                ^10
#   swing lows at: (2,93), (6,92), (10,93)  → last two: 92→93 (HL ✓)
#
# BEARISH (LH + LL):
#   Reverse the high/low so each successive swing is lower
#   high = [106,107,110, 108,107,106, 101, 100,99,98, 102, 100,99,98]
#                    ^2                    ^6                  ^10
#   swing highs: (2,110), (6,101), (10,102) → but 102>101 makes this messy
#   → use a cleaner approach: descending swings
#
# MIXED (HH + LL): descending lows despite ascending highs
# MIXED (LH + HL): descending highs despite ascending lows
# ---------------------------------------------------------------------------

LB = 2  # swing_lookback used in all tests below

BULLISH_HIGH = pd.Series(
    [98.0, 99.0, 102.0, 100.0, 99.0, 98.0, 101.0, 100.0, 99.0, 98.0, 110.0, 108.0, 107.0, 106.0],
    dtype=float,
)
BULLISH_LOW = pd.Series(
    [95.0, 94.0, 93.0, 95.0, 96.0, 94.0, 92.0, 94.0, 96.0, 95.0, 93.0, 95.0, 96.0, 97.0],
    dtype=float,
)
# Verified: swing_highs = [(2,102),(6,101),(10,110)], swing_lows = [(2,93),(6,92),(10,93)]
# last_sh=110 > prev_sh=101 ✓ (HH), last_sl=93 > prev_sl=92 ✓ (HL)


class TestBullishStructureBullishSetup:
    def test_hh_and_hl_fires(self):
        result = is_bullish_market_structure(BULLISH_HIGH, BULLISH_LOW, swing_lookback=LB)
        assert result.active is True
        assert result.value > 0.0

    def test_value_is_sum_of_relative_changes(self):
        result = is_bullish_market_structure(BULLISH_HIGH, BULLISH_LOW, swing_lookback=LB)
        # delta_high = 110/101 - 1, delta_low = 93/92 - 1
        expected = (110.0 / 101.0 - 1.0) + (93.0 / 92.0 - 1.0)
        assert result.value == pytest.approx(expected, rel=1e-3)


class TestBullishStructureBearishSetup:
    def test_lh_and_ll_does_not_fire(self):
        # Explicit bearish structure: descending swing highs AND descending swing lows.
        # Verified by hand (swing_lookback=2):
        #   swing_highs: (2, 110), (10, 108)  → 110 → 108 = LH (HH=False)
        #   swing_lows:  (2, 93), (6, 93), (10, 91) → 93 → 91 = LL (HL=False)
        high = pd.Series(
            [98., 99., 110., 108., 107., 106., 105., 104., 103., 102., 108., 106., 105., 104.],
            dtype=float,
        )
        low = pd.Series(
            [95., 94., 93., 95., 96., 95., 93., 95., 96., 95., 91., 93., 94., 95.],
            dtype=float,
        )
        result = is_bullish_market_structure(high, low, swing_lookback=LB)
        assert result.active is False
        assert result.value < 0.0  # both deltas negative


class TestBullishStructureSideways:
    def test_oscillating_no_clear_structure(self):
        # Oscillating prices around a flat level: swings may or may not be detected
        # but active must be a valid bool regardless
        t = np.arange(20)
        high = pd.Series(100.0 + 3.0 * np.sin(t * 0.8), dtype=float)
        low = pd.Series(100.0 - 3.0 * np.sin(t * 0.8), dtype=float)
        result = is_bullish_market_structure(high, low, swing_lookback=LB)
        assert isinstance(result.active, bool)
        if result.active:
            assert math.isfinite(result.value)


class TestBullishStructureInsufficientData:
    def test_too_short(self):
        # Needs 2*2+1 = 5 bars minimum
        high = pd.Series([100.0, 102.0, 101.0, 100.0], dtype=float)
        low = pd.Series([98.0, 99.0, 98.0, 97.0], dtype=float)
        result = is_bullish_market_structure(high, low, swing_lookback=LB)
        assert result.active is False
        assert math.isnan(result.value)

    def test_not_enough_swings(self):
        # Long enough but too smooth to produce 2 confirmed pivots of each type
        high = pd.Series(np.linspace(100.0, 110.0, 20), dtype=float)
        low = pd.Series(np.linspace(98.0, 108.0, 20), dtype=float)
        result = is_bullish_market_structure(high, low, swing_lookback=LB)
        assert result.active is False
        assert math.isnan(result.value)


class TestBullishStructureBadInput:
    def test_nan_in_high(self):
        high = BULLISH_HIGH.copy()
        high.iloc[5] = float("nan")
        result = is_bullish_market_structure(high, BULLISH_LOW, swing_lookback=LB)
        assert result.active is False
        assert math.isnan(result.value)

    def test_inf_in_low(self):
        low = BULLISH_LOW.copy()
        low.iloc[3] = float("inf")
        result = is_bullish_market_structure(BULLISH_HIGH, low, swing_lookback=LB)
        assert result.active is False
        assert math.isnan(result.value)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            is_bullish_market_structure(
                BULLISH_HIGH, BULLISH_LOW.iloc[:-1], swing_lookback=LB
            )


class TestBullishStructureMixedSignals:
    def test_hh_but_ll_does_not_fire(self):
        # Ascending swing highs but descending swing lows → no HL → active=False
        # Build a 14-bar series: highs create HH, lows create LL
        # Highs: same as BULLISH_HIGH → HH pattern
        # Lows: mirror of BULLISH_LOW (reversed relative to center) → LL pattern
        #   We swap last_sl and prev_sl so that last < prev
        #   Achieved by negating the delta: set lows so last swing low < prev swing low
        lows_ll = pd.Series(
            [97.0, 96.0, 95.0, 96.0, 97.0, 95.0, 94.0, 95.0, 97.0, 96.0, 91.0, 93.0, 95.0, 96.0],
            dtype=float,
        )
        # swing lows at i=2 (95), i=6 (94), i=10 (91) → 91 < 94 → LL not HL
        result = is_bullish_market_structure(BULLISH_HIGH, lows_ll, swing_lookback=LB)
        assert result.active is False

    def test_hl_but_lh_does_not_fire(self):
        # Ascending swing lows but descending swing highs → no HH → active=False
        highs_lh = pd.Series(
            [98.0, 99.0, 115.0, 110.0, 109.0, 108.0, 107.0, 106.0, 105.0, 104.0, 110.0, 108.0, 107.0, 106.0],
            dtype=float,
        )
        # swing highs at i=2 (115), i=10 (110) → only 2 swings,
        # but 110 < 115 → LH, not HH
        # Verify at least 2 swing highs are detected, then active=False
        sh = _find_swings(highs_lh, LB, is_high=True)
        if len(sh) >= 2:
            result = is_bullish_market_structure(highs_lh, BULLISH_LOW, swing_lookback=LB)
            assert result.active is False

    def test_equal_swing_highs_plateau_not_hh(self):
        # last_swing_high == prev_swing_high: strict >, so active=False
        highs_equal = pd.Series(
            [98.0, 99.0, 102.0, 100.0, 99.0, 98.0, 102.0, 100.0, 99.0, 98.0, 110.0, 108.0, 107.0, 106.0],
            dtype=float,
        )
        # swing highs: i=2 (102), i=6 (102), i=10 (110)
        # last two: 102→110 → HH. But let's use a case where the last two are equal:
        highs_flat = pd.Series(
            [98.0, 99.0, 102.0, 100.0, 99.0, 98.0, 110.0, 108.0, 107.0, 106.0, 110.0, 108.0, 107.0, 106.0],
            dtype=float,
        )
        # swing high at i=2 (102), i=6 (110), i=10 (110) → last two: 110 == 110 → NOT HH
        sh = _find_swings(highs_flat, LB, is_high=True)
        if len(sh) >= 2 and sh[-1][1] == sh[-2][1]:
            result = is_bullish_market_structure(highs_flat, BULLISH_LOW, swing_lookback=LB)
            assert result.active is False

    def test_only_one_swing_high_detected(self):
        # Monotonically rising highs: no pivot (each bar is higher than next)
        # → 0 swing highs → insufficient → active=False, value=NaN
        high = pd.Series(np.linspace(100.0, 120.0, 20), dtype=float)
        low = BULLISH_LOW.iloc[:20].reset_index(drop=True) if len(BULLISH_LOW) >= 20 else pd.Series(np.linspace(95.0, 115.0, 20), dtype=float)
        result = is_bullish_market_structure(high, low, swing_lookback=LB)
        assert result.active is False
        assert math.isnan(result.value)
