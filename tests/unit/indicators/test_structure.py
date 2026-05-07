"""Unit tests for market structure indicators: drawdown_from_ath, swing_highs_lows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_quant.indicators.structure import drawdown_from_ath, swing_highs_lows


# ---------------------------------------------------------------------------
# drawdown_from_ath
# ---------------------------------------------------------------------------


class TestDrawdownFromATH:
    def test_drawdown_basic(self) -> None:
        """Manual verification of ATH and drawdown for a known series.

        close = [100, 120, 90, 110, 80]

        ATH (running max):  100, 120, 120, 120, 120
        drawdown:             0,   0, (90-120)/120, (110-120)/120, (80-120)/120
                          =   0,   0,  -0.25,        -1/12,         -1/3
        """
        close = pd.Series([100.0, 120.0, 90.0, 110.0, 80.0])
        result = drawdown_from_ath(close)

        ath = result["ath"]
        dd = result["drawdown"]

        np.testing.assert_array_almost_equal(ath.values, [100.0, 120.0, 120.0, 120.0, 120.0])
        assert abs(dd.iloc[0] - 0.0) < 1e-10
        assert abs(dd.iloc[1] - 0.0) < 1e-10
        assert abs(dd.iloc[2] - (-30 / 120)) < 1e-10
        assert abs(dd.iloc[3] - (-10 / 120)) < 1e-10
        assert abs(dd.iloc[4] - (-40 / 120)) < 1e-10

    def test_drawdown_at_ath_is_zero(self) -> None:
        """Whenever close equals the ATH, drawdown is 0."""
        close = pd.Series([100.0, 120.0, 90.0, 130.0, 100.0])
        result = drawdown_from_ath(close)
        dd = result["drawdown"]
        # Bars 0, 1, 3 are all-time highs → drawdown = 0
        assert dd.iloc[0] == 0.0
        assert dd.iloc[1] == 0.0
        assert dd.iloc[3] == 0.0

    def test_drawdown_always_non_positive(self) -> None:
        """Drawdown must always be <= 0."""
        rng = np.random.default_rng(seed=0)
        close = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 500)))
        close = close.clip(lower=0.1)  # ensure all positive
        result = drawdown_from_ath(close)
        assert (result["drawdown"] <= 0).all()

    def test_ath_monotonic_non_decreasing(self) -> None:
        """ATH series must be non-decreasing."""
        rng = np.random.default_rng(seed=1)
        close = pd.Series(100.0 + np.cumsum(rng.normal(0, 2, 300))).clip(lower=0.1)
        ath = drawdown_from_ath(close)["ath"]
        assert (ath.diff().dropna() >= 0).all()

    def test_drawdown_with_ath_seed(self) -> None:
        """ath_seed=100 with close[0]=80 → ATH stays at 100, drawdown[0]=-0.2."""
        close = pd.Series([80.0, 90.0, 110.0, 95.0])
        result = drawdown_from_ath(close, ath_seed=100.0)

        assert result["ath"].iloc[0] == 100.0          # seed dominates
        assert abs(result["drawdown"].iloc[0] - (-0.2)) < 1e-10
        assert result["ath"].iloc[2] == 110.0          # new ATH after seed
        assert result["drawdown"].iloc[2] == 0.0

    def test_drawdown_seed_below_first_close(self) -> None:
        """ath_seed below close[0] → ATH immediately becomes close[0]."""
        close = pd.Series([150.0, 140.0])
        result = drawdown_from_ath(close, ath_seed=100.0)
        assert result["ath"].iloc[0] == 150.0
        assert result["drawdown"].iloc[0] == 0.0

    def test_drawdown_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        close = pd.Series([100.0, 110, 90, 120, 80], index=idx)
        result = drawdown_from_ath(close)
        for key in ("ath", "drawdown"):
            assert list(result[key].index) == list(idx)

    def test_drawdown_output_dtype(self) -> None:
        close = pd.Series([100.0, 90.0, 110.0])
        result = drawdown_from_ath(close)
        assert result["ath"].dtype == np.float64
        assert result["drawdown"].dtype == np.float64

    def test_drawdown_zero_close_raises(self) -> None:
        with pytest.raises(ValueError, match="close"):
            drawdown_from_ath(pd.Series([100.0, 0.0, 90.0]))

    def test_drawdown_negative_close_raises(self) -> None:
        with pytest.raises(ValueError, match="close"):
            drawdown_from_ath(pd.Series([100.0, -10.0, 90.0]))

    def test_drawdown_negative_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="ath_seed"):
            drawdown_from_ath(pd.Series([100.0, 90.0]), ath_seed=-50.0)

    def test_drawdown_zero_seed_raises(self) -> None:
        with pytest.raises(ValueError, match="ath_seed"):
            drawdown_from_ath(pd.Series([100.0, 90.0]), ath_seed=0.0)


# ---------------------------------------------------------------------------
# swing_highs_lows
# ---------------------------------------------------------------------------


class TestSwingHighsLows:
    def test_swing_high_basic(self) -> None:
        """Clear swing high in the middle of a series with lookback=2."""
        # high peaks at index 3 (value 20), surrounded by lower values
        high = pd.Series([5.0, 8, 12, 20, 12, 8, 5, 4, 3])
        low = high * 0.5
        result = swing_highs_lows(high, low, lookback=2)

        # Index 3 should be the only swing high (clearly the highest local peak)
        assert result["is_swing_high"].iloc[3], "Index 3 must be a swing high"
        # Surrounding bars that are lower should not be swing highs
        assert not result["is_swing_high"].iloc[2]
        assert not result["is_swing_high"].iloc[4]

    def test_swing_low_basic(self) -> None:
        """Clear swing low in the middle of a series with lookback=2."""
        high = pd.Series([20.0, 15, 10, 3, 10, 15, 20, 25, 30])
        low = high * 0.8
        result = swing_highs_lows(high, low, lookback=2)

        # low[3] = 2.4 — smallest value surrounded by higher lows
        assert result["is_swing_low"].iloc[3], "Index 3 must be a swing low"
        assert not result["is_swing_low"].iloc[2]
        assert not result["is_swing_low"].iloc[4]

    def test_no_swings_in_monotonic_series(self) -> None:
        """Strictly monotonic series has no interior swing highs or swing lows."""
        high = pd.Series(np.arange(1.0, 21.0))  # strictly increasing
        low = high - 0.5
        result = swing_highs_lows(high, low, lookback=3)
        # No swing highs in monotonic up (each bar is beaten by the next)
        assert not result["is_swing_high"].any()
        # No swing lows in monotonic up (each bar beats the previous as a low)
        assert not result["is_swing_low"].any()

    def test_swing_at_boundary(self) -> None:
        """First and last lookback bars are always False (insufficient context)."""
        lookback = 3
        high = pd.Series([5.0, 8, 12, 20, 12, 8, 5, 4, 3, 2, 1])
        low = high * 0.5
        result = swing_highs_lows(high, low, lookback=lookback)

        # First lookback bars
        assert not result["is_swing_high"].iloc[:lookback].any()
        assert not result["is_swing_low"].iloc[:lookback].any()
        # Last lookback bars
        assert not result["is_swing_high"].iloc[-lookback:].any()
        assert not result["is_swing_low"].iloc[-lookback:].any()

    def test_swing_lookback_window(self) -> None:
        """Swing is detected only if ALL lookback bars on each side are lower/higher."""
        # Peak at index 3 with lookback=2: needs high[3] > high[1,2] and high[4,5]
        # Place a higher bar at distance 3 (index 0 = 25, beyond lookback=2 → irrelevant)
        high = pd.Series([25.0, 8.0, 12.0, 20.0, 12.0, 8.0, 5.0])
        low = high * 0.5
        result = swing_highs_lows(high, low, lookback=2)

        # Index 3 (high=20) vs lookback=2: checks indices 1,2 on left and 4,5 on right
        # high[1]=8 < 20 ✓, high[2]=12 < 20 ✓, high[4]=12 < 20 ✓, high[5]=8 < 20 ✓
        assert result["is_swing_high"].iloc[3]

        # With lookback=3: also checks index 0 (high=25 > 20) → NOT a swing high
        result_lb3 = swing_highs_lows(high, low, lookback=3)
        assert not result_lb3["is_swing_high"].iloc[3]

    def test_swing_high_low_mutually_exclusive(self) -> None:
        """In a standard zigzag series, no bar is simultaneously a swing high and low."""
        # Alternating peaks (even indices) and troughs (odd indices)
        high = pd.Series([5.0, 15.0, 5.0, 20.0, 5.0, 15.0, 5.0, 20.0, 5.0, 15.0, 5.0])
        low = high * 0.5  # lows are always well below the corresponding high
        result = swing_highs_lows(high, low, lookback=1)
        both = result["is_swing_high"] & result["is_swing_low"]
        assert not both.any(), "No bar should be simultaneously a swing high and swing low"

    def test_swing_requires_lookback_to_confirm(self) -> None:
        """Swing at t is confirmed only when bars through t+lookback exist.

        This test documents the anti-look-ahead contract explicitly:

        - swing_highs_lows() uses bars [t+1 .. t+lookback] to confirm a swing at t.
        - With a dataset truncated BEFORE t+lookback, the swing cannot be confirmed
          (shift(-lookback) returns NaN → fillna(False) → unconfirmed).
        - With a dataset extending TO t+lookback, the swing IS confirmed.

        In backtest/live use: treat a swing detected at bar t as actionable only
        at bar t+lookback, when all confirmation bars have been observed.
        """
        lookback = 3
        # Clear peak at index 3 (high=20), all neighbors strictly lower
        highs = pd.Series([10.0, 12.0, 14.0, 20.0, 14.0, 12.0, 10.0, 8.0, 6.0])
        lows = highs - 2.0

        # Truncated at index 5: only 2 bars to the right of t=3 (need 3)
        # → shift(-3) at index 3 maps to index 6 which does NOT exist → NaN → False
        partial_h = highs.iloc[:6]
        partial_l = lows.iloc[:6]
        partial = swing_highs_lows(partial_h, partial_l, lookback=lookback)
        assert not partial["is_swing_high"].iloc[3], (
            "Only 2 right-side bars available (need lookback=3): swing must be unconfirmed"
        )

        # Extended to index 6: exactly lookback=3 bars to the right of t=3 exist
        # → shift(-3) at index 3 maps to index 6 (high=10 < 20) → condition True
        full_h = highs.iloc[:7]
        full_l = lows.iloc[:7]
        full = swing_highs_lows(full_h, full_l, lookback=lookback)
        assert full["is_swing_high"].iloc[3], (
            "Exactly lookback=3 right-side bars available: swing must be confirmed"
        )

    def test_swing_output_dtype(self) -> None:
        high = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0])
        low = high * 0.5
        result = swing_highs_lows(high, low, lookback=1)
        assert result["is_swing_high"].dtype == bool
        assert result["is_swing_low"].dtype == bool

    def test_swing_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=9, freq="D")
        high = pd.Series([5.0, 8, 12, 20, 12, 8, 5, 4, 3], index=idx)
        low = high * 0.5
        result = swing_highs_lows(high, low, lookback=2)
        for key in ("is_swing_high", "is_swing_low"):
            assert list(result[key].index) == list(idx)

    def test_swing_invalid_lookback_raises(self) -> None:
        high = pd.Series([1.0, 2.0, 3.0])
        low = high * 0.5
        with pytest.raises(ValueError, match="lookback"):
            swing_highs_lows(high, low, lookback=0)

    def test_swing_mismatched_indices_raises(self) -> None:
        high = pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(3))
        low = pd.Series([0.5, 1.0, 1.5], index=pd.RangeIndex(1, 4))
        with pytest.raises(ValueError, match="index"):
            swing_highs_lows(high, low, lookback=1)

    def test_swing_low_greater_than_high_raises(self) -> None:
        high = pd.Series([10.0, 8.0, 12.0])
        low = pd.Series([8.0, 11.0, 10.0])  # low[1]=11 > high[1]=8
        with pytest.raises(ValueError, match="low > high"):
            swing_highs_lows(high, low, lookback=1)
