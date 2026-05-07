"""Unit tests for trend indicators: SMA, EMA, MACD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_quant.indicators.trend import ema, macd, sma


# ---------------------------------------------------------------------------
# SMA
# ---------------------------------------------------------------------------


class TestSMA:
    def test_sma_basic(self) -> None:
        """[1..10] period=3 → first 2 NaN, then sliding means."""
        series = pd.Series([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = sma(series, period=3)

        assert result.iloc[0] != result.iloc[0]  # NaN
        assert result.iloc[1] != result.iloc[1]  # NaN
        expected = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
        np.testing.assert_array_almost_equal(result.iloc[2:].values, expected)

    def test_sma_period_1_returns_input(self) -> None:
        """period=1 — every bar is its own mean."""
        series = pd.Series([3.0, 1.0, 4.0, 1.0, 5.0])
        result = sma(series, period=1)
        np.testing.assert_array_equal(result.values, series.values)

    def test_sma_period_equal_length(self) -> None:
        """period=len(series) — only the last value is non-NaN."""
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = sma(series, period=5)
        assert result.iloc[:-1].isna().all()
        assert abs(result.iloc[-1] - 3.0) < 1e-10

    def test_sma_period_greater_than_length(self) -> None:
        """period > len(series) — all NaN."""
        series = pd.Series([1.0, 2.0, 3.0])
        result = sma(series, period=10)
        assert result.isna().all()

    def test_sma_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        series = pd.Series([1.0, 2, 3, 4, 5], index=idx)
        result = sma(series, period=2)
        assert list(result.index) == list(idx)

    def test_sma_output_dtype(self) -> None:
        result = sma(pd.Series([1, 2, 3], dtype="int64"), period=2)
        assert result.dtype == np.float64

    def test_sma_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            sma(pd.Series([1.0, 2.0]), period=0)

    def test_sma_negative_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            sma(pd.Series([1.0, 2.0]), period=-1)

    def test_sma_empty_series_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            sma(pd.Series([], dtype=float), period=3)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


class TestEMA:
    def test_ema_basic(self) -> None:
        """Manual step-by-step EMA verification for period=3, adjust=False.

        α = 2 / (3 + 1) = 0.5

        EMA[0] = 10.0  (seeds with first value)
        EMA[1] = 0.5 * 20 + 0.5 * 10 = 15.0
        EMA[2] = 0.5 * 30 + 0.5 * 15 = 22.5
        EMA[3] = 0.5 * 40 + 0.5 * 22.5 = 31.25
        """
        series = pd.Series([10.0, 20.0, 30.0, 40.0])
        result = ema(series, period=3)
        expected = [10.0, 15.0, 22.5, 31.25]
        np.testing.assert_array_almost_equal(result.values, expected, decimal=6)

    def test_ema_first_value_equals_input(self) -> None:
        """With adjust=False the series seeds at price[0], so EMA[0] == price[0]."""
        series = pd.Series([42.0, 43.0, 44.0, 45.0])
        result = ema(series, period=3, adjust=False)
        assert result.iloc[0] == pytest.approx(42.0)

    def test_ema_smoother_than_sma_after_jump(self) -> None:
        """EMA reacts faster than SMA to an abrupt price jump.

        Series: 10 bars at 10.0, then 10 bars at 100.0, period=5.
        At index 11 (2nd bar after jump, α=1/3):
          SMA[11] = mean(series[7..11]) = mean(10,10,10,100,100) = 46.0
          EMA[11] = α*100 + (1-α)*EMA[10]
                  = 1/3*100 + 2/3*(1/3*100 + 2/3*10) = 60.0
          EMA[11]=60 > SMA[11]=46 — EMA weights recent bars more heavily.
        """
        series = pd.Series([10.0] * 10 + [100.0] * 10)
        period = 5
        s = sma(series, period)
        e = ema(series, period)
        assert abs(s.iloc[11] - 46.0) < 1e-10
        assert abs(e.iloc[11] - 60.0) < 1e-10
        assert e.iloc[11] > s.iloc[11]

    def test_ema_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        series = pd.Series([1.0, 2, 3, 4, 5], index=idx)
        result = ema(series, period=3)
        assert list(result.index) == list(idx)

    def test_ema_output_dtype(self) -> None:
        result = ema(pd.Series([1, 2, 3], dtype="int64"), period=2)
        assert result.dtype == np.float64

    def test_ema_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            ema(pd.Series([1.0, 2.0]), period=0)

    def test_ema_empty_series_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ema(pd.Series([], dtype=float), period=3)


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


class TestMACD:
    def test_macd_returns_three_series(self) -> None:
        series = pd.Series(range(50), dtype=float)
        result = macd(series)
        assert set(result.keys()) == {"macd_line", "signal_line", "histogram"}
        for key, val in result.items():
            assert isinstance(val, pd.Series), f"{key} is not a Series"

    def test_macd_histogram_equals_macd_minus_signal(self) -> None:
        """histogram must equal macd_line - signal_line everywhere."""
        series = pd.Series(range(60), dtype=float)
        result = macd(series, fast_period=3, slow_period=5, signal_period=3)
        diff = (result["macd_line"] - result["signal_line"] - result["histogram"]).abs()
        # Only check non-NaN positions
        mask = diff.notna()
        assert mask.any(), "No non-NaN positions found"
        np.testing.assert_array_almost_equal(diff[mask].values, 0.0, decimal=10)

    def test_macd_warmup_period(self) -> None:
        """signal_line and histogram are NaN for the first
        (slow_period - 1) + (signal_period - 1) bars.

        With fast=3, slow=5, signal=3:
          macd_line NaN: indices 0..3  (slow_period - 1 = 4 bars)
          signal_line needs signal_period=3 non-NaN MACD values,
          first non-NaN MACD is at index 4, so signal defined at index 4+2=6.
          NaN: indices 0..5 → 6 = (5-1) + (3-1)
        """
        n = 50
        fast_p, slow_p, sig_p = 3, 5, 3
        series = pd.Series(range(n), dtype=float)
        result = macd(series, fast_period=fast_p, slow_period=slow_p, signal_period=sig_p)

        warmup = (slow_p - 1) + (sig_p - 1)  # 6
        signal = result["signal_line"]
        hist = result["histogram"]

        assert signal.iloc[:warmup].isna().all(), "signal_line warmup contains non-NaN"
        assert pd.notna(signal.iloc[warmup]), "signal_line should be defined after warmup"
        assert hist.iloc[:warmup].isna().all(), "histogram warmup contains non-NaN"
        assert pd.notna(hist.iloc[warmup]), "histogram should be defined after warmup"

    def test_macd_fast_greater_than_slow_raises(self) -> None:
        with pytest.raises(ValueError, match="fast_period"):
            macd(pd.Series(range(30), dtype=float), fast_period=26, slow_period=12)

    def test_macd_fast_equal_slow_raises(self) -> None:
        with pytest.raises(ValueError, match="fast_period"):
            macd(pd.Series(range(30), dtype=float), fast_period=12, slow_period=12)

    def test_macd_with_known_values(self) -> None:
        """Verify MACD on a linearly rising series.

        For a perfectly linear series price[i] = i, both fast and slow EMAs
        lag behind the price by their respective spans. The MACD line should be
        positive (fast EMA > slow EMA) and roughly constant once warmed up,
        because the lag difference between two EMAs on a linear series is
        proportional to the difference in their lags.
        """
        n = 100
        series = pd.Series(np.arange(n, dtype=float))
        result = macd(series, fast_period=3, slow_period=5, signal_period=3)
        ml = result["macd_line"].dropna()
        # Fast EMA converges faster → fast_ema > slow_ema → macd_line > 0
        assert (ml > 0).all(), "MACD line should be positive on rising series"

    def test_macd_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=50, freq="D")
        series = pd.Series(range(50), dtype=float, index=idx)
        result = macd(series, fast_period=3, slow_period=5, signal_period=3)
        for key, val in result.items():
            assert list(val.index) == list(idx), f"{key} index mismatch"

    def test_macd_empty_series_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            macd(pd.Series([], dtype=float))

    def test_macd_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="fast_period"):
            macd(pd.Series(range(30), dtype=float), fast_period=0)
