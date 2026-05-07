"""Unit tests for volatility indicators: ATR and Bollinger Bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_quant.indicators.volatility import atr, bollinger_bands


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlc(
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> tuple[pd.Series, pd.Series, pd.Series]:
    idx = pd.RangeIndex(len(highs))
    return (
        pd.Series(highs, index=idx, dtype="float64"),
        pd.Series(lows, index=idx, dtype="float64"),
        pd.Series(closes, index=idx, dtype="float64"),
    )


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------


class TestATR:
    def test_atr_basic(self) -> None:
        """Verify TR computation for bars with and without a gap.

        Dataset (4 bars):
            Bar 0: H=10, L=8,  C=9   → TR = H-L = 2     (no prev close)
            Bar 1: H=12, L=10, C=11  → prevC=9
                   max(12-10=2, |12-9|=3, |10-9|=1)  = 3
            Bar 2: H=9,  L=7,  C=8   → prevC=11
                   max(9-7=2, |9-11|=2, |7-11|=4)   = 4
            Bar 3: H=12, L=10, C=11  → prevC=8
                   max(12-10=2, |12-8|=4, |10-8|=2) = 4

        ATR[3] with period=4, SMA:
            mean(2, 3, 4, 4) = 13/4 = 3.25
        """
        h, l, c = _make_ohlc(
            [10, 12, 9, 12],
            [8, 10, 7, 10],
            [9, 11, 8, 11],
        )
        result = atr(h, l, c, period=4, method="sma")
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        assert abs(result.iloc[3] - 3.25) < 1e-10

    def test_atr_known_reference(self) -> None:
        """Synthetic 7-bar reference with Wilders smoothing, period=3.

        Dataset:
            H = [10, 12, 14, 12, 14, 12, 14]
            L = [ 8, 10, 12, 10, 12, 10, 12]
            C = [ 9, 11, 13, 11, 13, 11, 13]

        True Range step by step:
            t=0: TR = 10-8 = 2           (no prev close)
            t=1: prevC=9  → max(12-10=2, |12-9|=3, |10-9|=1)  = 3
            t=2: prevC=11 → max(14-12=2, |14-11|=3, |12-11|=1) = 3
            t=3: prevC=13 → max(12-10=2, |12-13|=1, |10-13|=3) = 3
            t=4: prevC=11 → max(14-12=2, |14-11|=3, |12-11|=1) = 3
            t=5: prevC=13 → max(12-10=2, |12-13|=1, |10-13|=3) = 3
            t=6: prevC=11 → max(14-12=2, |14-11|=3, |12-11|=1) = 3

        Wilders ATR (period=3, α=1/3):
            Seed:  ATR[2] = mean(2,3,3) = 8/3
            ATR[3] = 8/3 * 2/3 + 3 * 1/3 = 16/9 + 9/9   = 25/9
            ATR[4] = 25/9 * 2/3 + 3 * 1/3 = 50/27 + 27/27 = 77/27
            ATR[5] = 77/27 * 2/3 + 3 * 1/3 = 154/81 + 81/81 = 235/81
            ATR[6] = 235/81 * 2/3 + 3 * 1/3 = 470/243 + 243/243 = 713/243
        """
        h, l, c = _make_ohlc(
            [10, 12, 14, 12, 14, 12, 14],
            [8,  10, 12, 10, 12, 10, 12],
            [9,  11, 13, 11, 13, 11, 13],
        )
        result = atr(h, l, c, period=3, method="wilders")

        assert abs(result.iloc[2] - 8 / 3) < 1e-10
        assert abs(result.iloc[3] - 25 / 9) < 1e-10
        assert abs(result.iloc[4] - 77 / 27) < 1e-10
        assert abs(result.iloc[5] - 235 / 81) < 1e-10
        assert abs(result.iloc[6] - 713 / 243) < 1e-10

    def test_atr_first_value_is_nan(self) -> None:
        """Wilders ATR: first period-1 values are NaN, ATR[period-1] is defined."""
        n, period = 30, 14
        h = pd.Series(np.linspace(10, 20, n))
        l = h - 1.0
        c = h - 0.5
        result = atr(h, l, c, period=period, method="wilders")
        assert result.iloc[: period - 1].isna().all()
        assert pd.notna(result.iloc[period - 1])

    def test_atr_method_wilders_vs_ema_differ(self) -> None:
        """Wilders (α=1/period) and EMA (α=2/(period+1)) produce different values."""
        rng = np.random.default_rng(seed=7)
        n = 200
        h = pd.Series(100.0 + np.cumsum(rng.uniform(0, 2, n)))
        l = h - rng.uniform(0.5, 2.0, n)
        c = l + rng.uniform(0, 1, n) * (h - l)

        w = atr(h, l, c, period=14, method="wilders")
        e = atr(h, l, c, period=14, method="ema")

        both_valid = w.notna() & e.notna()
        diff = (w[both_valid] - e[both_valid]).abs()
        assert (diff > 0.01).any(), "Wilders and EMA ATR are unexpectedly identical"

    def test_atr_always_non_negative(self) -> None:
        """ATR must be >= 0 for all non-NaN values."""
        rng = np.random.default_rng(seed=42)
        n = 300
        h = pd.Series(100.0 + np.cumsum(rng.uniform(0, 3, n)))
        l = h - rng.uniform(0.1, 3.0, n)
        c = l + rng.uniform(0, 1, n) * (h - l)
        result = atr(h, l, c, period=14, method="wilders")
        assert (result.dropna() >= 0).all()

    def test_atr_output_dtype(self) -> None:
        h, l, c = _make_ohlc([10, 11, 12, 13, 14], [8, 9, 10, 11, 12], [9, 10, 11, 12, 13])
        assert atr(h, l, c, period=3).dtype == np.float64

    def test_atr_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        h = pd.Series(np.linspace(10, 20, 20), index=idx)
        l = h - 1.0
        c = h - 0.5
        result = atr(h, l, c, period=5)
        assert list(result.index) == list(idx)

    def test_atr_low_greater_than_high_raises(self) -> None:
        h, l, c = _make_ohlc([10, 9, 12], [8, 11, 10], [9, 10, 11])  # l[1] > h[1]
        with pytest.raises(ValueError, match="low > high"):
            atr(h, l, c, period=2)

    def test_atr_invalid_method_raises(self) -> None:
        h, l, c = _make_ohlc([10, 11, 12], [8, 9, 10], [9, 10, 11])
        with pytest.raises(ValueError, match="method"):
            atr(h, l, c, period=2, method="rma")

    def test_atr_invalid_period_raises(self) -> None:
        h, l, c = _make_ohlc([10, 11, 12], [8, 9, 10], [9, 10, 11])
        with pytest.raises(ValueError, match="period"):
            atr(h, l, c, period=0)

    def test_atr_mismatched_indices_raises(self) -> None:
        idx_a = pd.RangeIndex(5)
        idx_b = pd.RangeIndex(1, 6)
        h = pd.Series([10.0] * 5, index=idx_a)
        l = pd.Series([8.0] * 5, index=idx_b)
        c = pd.Series([9.0] * 5, index=idx_a)
        with pytest.raises(ValueError, match="index"):
            atr(h, l, c, period=3)

    def test_atr_sma_method(self) -> None:
        """SMA ATR with period=3 on constant TR=2 series."""
        h, l, c = _make_ohlc([10] * 10, [8] * 10, [9] * 10)
        result = atr(h, l, c, period=3, method="sma")
        # TR[0] = 2, TR[t>=1] = max(2, 0, 0) = 2 → ATR = 2 everywhere after warmup
        assert result.iloc[:2].isna().all()
        np.testing.assert_allclose(result.iloc[2:].values, 2.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Bollinger Bands
# ---------------------------------------------------------------------------


class TestBollingerBands:
    def test_bb_returns_three_series(self) -> None:
        series = pd.Series(range(30), dtype=float)
        result = bollinger_bands(series)
        assert set(result.keys()) == {"middle", "upper", "lower"}
        for key, val in result.items():
            assert isinstance(val, pd.Series), f"{key} is not a Series"

    def test_bb_middle_equals_sma(self) -> None:
        """middle band must equal SMA(series, period) exactly."""
        from btc_quant.indicators.trend import sma

        series = pd.Series(range(30), dtype=float)
        result = bollinger_bands(series, period=10)
        expected_middle = sma(series, period=10)
        pd.testing.assert_series_equal(result["middle"], expected_middle, check_names=False)

    def test_bb_upper_above_middle(self) -> None:
        """upper > middle whenever std > 0 (non-constant window)."""
        series = pd.Series(range(30), dtype=float)
        result = bollinger_bands(series, period=5)
        valid = result["middle"].notna()
        assert (result["upper"][valid] > result["middle"][valid]).all()

    def test_bb_lower_below_middle(self) -> None:
        """lower < middle whenever std > 0 (non-constant window)."""
        series = pd.Series(range(30), dtype=float)
        result = bollinger_bands(series, period=5)
        valid = result["middle"].notna()
        assert (result["lower"][valid] < result["middle"][valid]).all()

    def test_bb_warmup(self) -> None:
        """First period-1 values must be NaN in all three bands."""
        series = pd.Series(range(30), dtype=float)
        period = 10
        result = bollinger_bands(series, period=period)
        for key in ("middle", "upper", "lower"):
            assert result[key].iloc[: period - 1].isna().all(), f"{key} warmup failed"
            assert pd.notna(result[key].iloc[period - 1]), f"{key} should be defined at period-1"

    def test_bb_constant_series(self) -> None:
        """Constant series → std=0 → upper == middle == lower."""
        series = pd.Series([42.0] * 20)
        result = bollinger_bands(series, period=5)
        valid = result["middle"].notna()
        pd.testing.assert_series_equal(
            result["upper"][valid], result["middle"][valid], check_names=False
        )
        pd.testing.assert_series_equal(
            result["lower"][valid], result["middle"][valid], check_names=False
        )

    def test_bb_known_reference(self) -> None:
        """Verify Bollinger Bands against manually computed values.

        Dataset: series = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        period=5, num_std=2.0, ddof=0 (population std)

        At index 4 (first valid bar, window=[10,11,12,13,14]):
            middle = (10+11+12+13+14)/5 = 12.0
            variance = ((10-12)^2 + (11-12)^2 + (12-12)^2
                        + (13-12)^2 + (14-12)^2) / 5
                     = (4+1+0+1+4)/5 = 2.0
            std      = sqrt(2)
            upper    = 12 + 2*sqrt(2)
            lower    = 12 - 2*sqrt(2)

        At index 5 (window=[11,12,13,14,15]):
            middle = 13.0   (same arithmetic structure → same std = sqrt(2))
            upper  = 13 + 2*sqrt(2)
            lower  = 13 - 2*sqrt(2)
        """
        series = pd.Series([float(x) for x in range(10, 21)])  # 10..20
        result = bollinger_bands(series, period=5, num_std=2.0)

        sqrt2 = np.sqrt(2)

        assert abs(result["middle"].iloc[4] - 12.0) < 1e-10
        assert abs(result["upper"].iloc[4] - (12.0 + 2 * sqrt2)) < 1e-10
        assert abs(result["lower"].iloc[4] - (12.0 - 2 * sqrt2)) < 1e-10

        assert abs(result["middle"].iloc[5] - 13.0) < 1e-10
        assert abs(result["upper"].iloc[5] - (13.0 + 2 * sqrt2)) < 1e-10
        assert abs(result["lower"].iloc[5] - (13.0 - 2 * sqrt2)) < 1e-10

    def test_bb_symmetry(self) -> None:
        """upper - middle == middle - lower (bands are symmetric around middle)."""
        series = pd.Series(range(30), dtype=float)
        result = bollinger_bands(series, period=5)
        valid = result["middle"].notna()
        upper_dist = (result["upper"] - result["middle"])[valid]
        lower_dist = (result["middle"] - result["lower"])[valid]
        pd.testing.assert_series_equal(upper_dist, lower_dist, check_names=False)

    def test_bb_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        series = pd.Series(range(30), dtype=float, index=idx)
        result = bollinger_bands(series, period=5)
        for key, val in result.items():
            assert list(val.index) == list(idx), f"{key} index mismatch"

    def test_bb_output_dtype(self) -> None:
        series = pd.Series(range(30), dtype="int64")
        result = bollinger_bands(series, period=5)
        for key, val in result.items():
            assert val.dtype == np.float64, f"{key} dtype is {val.dtype}"

    def test_bb_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            bollinger_bands(pd.Series(range(10), dtype=float), period=1)

    def test_bb_invalid_num_std_raises(self) -> None:
        with pytest.raises(ValueError, match="num_std"):
            bollinger_bands(pd.Series(range(10), dtype=float), num_std=0.0)

    def test_bb_negative_num_std_raises(self) -> None:
        with pytest.raises(ValueError, match="num_std"):
            bollinger_bands(pd.Series(range(10), dtype=float), num_std=-1.0)
