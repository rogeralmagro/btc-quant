"""Unit tests for oscillator indicators: RSI."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_quant.indicators.oscillators import rsi


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


class TestRSI:
    def test_rsi_known_dataset(self) -> None:
        """Verify RSI[14] against a manually computed reference value.

        Dataset (15 prices, period=14):
            prices = [10, 11, 12, 11, 12, 13, 14, 13, 14, 15, 16, 15, 16, 17, 18]

        Step-by-step manual calculation:
            diffs[1..14]: +1,+1,-1,+1,+1,+1,-1,+1,+1,+1,-1,+1,+1,+1
            gains (idx): 1,2,4,5,6,8,9,10,12,13,14  →  11 bars × 1.0 (sum=11)
            losses(idx): 3,7,11                       →   3 bars × 1.0 (sum=3)

            avg_gain = 11/14
            avg_loss =  3/14

            RS  = 11/3
            RSI = 100 - 100 / (1 + 11/3)
                = 100 - 100 / (14/3)
                = 100 - 300/14
                = 550/7
                ≈ 78.571
        """
        prices = pd.Series([10.0, 11, 12, 11, 12, 13, 14, 13, 14, 15, 16, 15, 16, 17, 18])
        result = rsi(prices, period=14)

        assert pd.isna(result.iloc[13]), "RSI[13] should be NaN (still in warmup)"
        assert pd.notna(result.iloc[14]), "RSI[14] should be defined"
        assert abs(result.iloc[14] - 550 / 7) < 0.001

    def test_rsi_constant_series(self) -> None:
        """A constant series has no change → avg_gain = avg_loss = 0 → RSI = NaN.

        Design decision: RSI is undefined for a constant series because there is
        no directional information. We return NaN rather than 50 to avoid
        implying a neutral signal when data may be corrupt or stale.
        """
        prices = pd.Series([42.0] * 20)
        result = rsi(prices, period=14)
        # Non-warmup positions should all be NaN (no movement)
        assert result.iloc[14:].isna().all()

    def test_rsi_only_gains_returns_100(self) -> None:
        """Strictly rising series → avg_loss → 0 → RSI → 100."""
        prices = pd.Series(np.arange(1.0, 25.0))  # 24 bars, strictly increasing
        result = rsi(prices, period=14)
        non_nan = result.dropna()
        assert len(non_nan) > 0
        # All non-NaN RSI values should be 100.0
        assert (non_nan == 100.0).all(), f"Expected all 100, got: {non_nan.values}"

    def test_rsi_only_losses_returns_0(self) -> None:
        """Strictly falling series → avg_gain = 0 → RS = 0 → RSI = 0."""
        prices = pd.Series(np.arange(24.0, 0.0, -1.0))  # strictly decreasing
        result = rsi(prices, period=14)
        non_nan = result.dropna()
        assert len(non_nan) > 0
        assert (non_nan == 0.0).all(), f"Expected all 0, got: {non_nan.values}"

    def test_rsi_warmup_period(self) -> None:
        """First `period` positions are NaN; RSI[period] is the first defined value."""
        prices = pd.Series(range(30), dtype=float)
        period = 14
        result = rsi(prices, period=period)

        assert result.iloc[:period].isna().all(), "Warmup positions must be NaN"
        assert pd.notna(result.iloc[period]), "RSI[period] must be defined"

    def test_rsi_bounded_0_100(self) -> None:
        """RSI is always in [0, 100] for non-NaN values."""
        rng = np.random.default_rng(seed=42)
        prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 500)))
        result = rsi(prices, period=14)
        non_nan = result.dropna()
        assert (non_nan >= 0.0).all(), "RSI below 0 detected"
        assert (non_nan <= 100.0).all(), "RSI above 100 detected"

    def test_rsi_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        prices = pd.Series(range(20), dtype=float, index=idx)
        result = rsi(prices, period=14)
        assert list(result.index) == list(idx)

    def test_rsi_output_dtype(self) -> None:
        prices = pd.Series(range(20), dtype="int64")
        result = rsi(prices, period=14)
        assert result.dtype == np.float64

    def test_rsi_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            rsi(pd.Series([1.0, 2.0, 3.0]), period=1)

    def test_rsi_period_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="period"):
            rsi(pd.Series([1.0, 2.0, 3.0]), period=0)

    def test_rsi_empty_series_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            rsi(pd.Series([], dtype=float), period=14)

    def test_rsi_series_too_short_returns_all_nan(self) -> None:
        """When len(series) <= period the function returns all NaN (no data to seed)."""
        prices = pd.Series([1.0, 2.0, 3.0])
        result = rsi(prices, period=14)
        assert result.isna().all()

    def test_rsi_wilder_vs_ema_differ(self) -> None:
        """Wilder RSI (α=1/14) must differ from a naive EMA RSI (α=2/15).

        This guards against accidentally swapping the smoothing parameter.
        The two α values produce measurably different results on real data.
        """
        rng = np.random.default_rng(seed=0)
        prices = pd.Series(100.0 + np.cumsum(rng.normal(0, 1, 200)))

        # Wilder RSI
        wilder = rsi(prices, period=14)

        # Naive EMA RSI (α = 2/(14+1) = 2/15, i.e. span=14, NOT span=27)
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_g = gain.ewm(span=14, adjust=False).mean()
        avg_l = loss.ewm(span=14, adjust=False).mean()
        naive = (100.0 - 100.0 / (1.0 + avg_g / avg_l)).astype("float64")

        # They must differ — verify at least 10% of non-NaN values differ by > 0.1
        both_valid = wilder.notna() & naive.notna()
        diff = (wilder[both_valid] - naive[both_valid]).abs()
        assert (diff > 0.1).mean() > 0.1, "Wilder and naive EMA RSI are too similar"
