"""Unit tests for RegimeDetector."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from btc_quant.backtester.models.enums import MarketRegime
from btc_quant.backtester.regime_detector import RegimeDetector

UTC = timezone.utc
START = datetime(2017, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_df(n: int, amplitude: float, start: datetime = START) -> pd.DataFrame:
    """n candles with constant TR = 2×amplitude, constant close (no gap).

    TR per bar = max(high-low, |high-prev_close|, |low-prev_close|)
               = max(2a, a, a) = 2a  (since close is constant)

    With Wilder's ATR after warmup: ATR → 2×amplitude (constant).
    """
    base = 10_000.0
    timestamps = [start + timedelta(days=i) for i in range(n)]
    close_times = [ts + timedelta(hours=23) for ts in timestamps]
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(timestamps, utc=True),
        "close_time_utc": pd.to_datetime(close_times, utc=True),
        "open":   [base] * n,
        "high":   [base + amplitude] * n,
        "low":    [base - amplitude] * n,
        "close":  [base] * n,
        "volume": [1_000.0] * n,
        "is_synthetic": [False] * n,
    })


def _concat_segments(*segments: pd.DataFrame) -> pd.DataFrame:
    """Concatenate segments and reset integer index."""
    return pd.concat(segments, ignore_index=True)


def _make_bar(
    open_: float = 10_000.0,
    high: float = 10_000.0,
    low: float = 10_000.0,
    close: float = 10_000.0,
) -> pd.Series:
    return pd.Series({"open": open_, "high": high, "low": low, "close": close})


# ---------------------------------------------------------------------------
# Regime data fixtures
#
# Computed analytically — see inline comments.
# ---------------------------------------------------------------------------


def _normal_data() -> pd.DataFrame:
    """400 bars, amplitude=50 → TR=100, ATR→100, ratio=1.0 → NORMAL."""
    return _make_df(400, 50)


def _volatile_data() -> pd.DataFrame:
    """400 bars amplitude=50 (ATR→100) + 100 bars amplitude=100 (ATR→200).

    After 100 bars of TR=200, ATR[499] ≈ 199.9 (Wilder's near-convergence).
    Annual avg over last 365 ATR values ≈ 123.8
    ratio ≈ 1.615 → VOLATILE (in [1.5, 3.0)).
    """
    low = _make_df(400, 50)
    high = _make_df(100, 100, start=START + timedelta(days=400))
    return _concat_segments(low, high)


def _stress_data() -> pd.DataFrame:
    """465 bars amplitude=5 (ATR→10) + 35 bars amplitude=5_000 (ATR→10_000).

    After 35 bars of TR=10_000, ATR[499] ≈ 9_204.
    Annual avg over last 365 ATR values ≈ 640.
    ratio ≈ 14.4 → STRESS (>= 3.0).
    """
    low = _make_df(465, 5)
    high = _make_df(35, 5_000, start=START + timedelta(days=465))
    return _concat_segments(low, high)


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------


class TestRegimeDetectorCreation:
    def test_default_params_valid(self) -> None:
        d = RegimeDetector()
        assert d._atr_period == 14
        assert d._volatile_mult == 1.5
        assert d._stress_mult == 3.0
        assert d._cascade_threshold == 0.10

    def test_invalid_atr_period_raises(self) -> None:
        with pytest.raises(ValueError, match="atr_period"):
            RegimeDetector(atr_period=0)

    def test_invalid_annual_lookback_raises(self) -> None:
        with pytest.raises(ValueError, match="annual_lookback_days"):
            RegimeDetector(annual_lookback_days=0)

    def test_invalid_volatile_multiplier_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="volatile_multiplier"):
            RegimeDetector(volatile_multiplier=0.0)

    def test_invalid_volatile_multiplier_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="volatile_multiplier"):
            RegimeDetector(volatile_multiplier=-1.0)

    def test_stress_less_than_volatile_raises(self) -> None:
        with pytest.raises(ValueError, match="stress_multiplier"):
            RegimeDetector(volatile_multiplier=2.0, stress_multiplier=1.5)

    def test_stress_equal_to_volatile_raises(self) -> None:
        with pytest.raises(ValueError, match="stress_multiplier"):
            RegimeDetector(volatile_multiplier=1.5, stress_multiplier=1.5)

    def test_invalid_cascade_threshold_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="cascade_threshold"):
            RegimeDetector(cascade_threshold=0.0)


# ---------------------------------------------------------------------------
# Regime classification — NORMAL
# ---------------------------------------------------------------------------


class TestNormalRegime:
    def test_normal_regime_when_atr_below_volatile_threshold(self) -> None:
        """400 constant bars: ratio = 1.0 < 1.5 → NORMAL."""
        assert RegimeDetector().detect(_normal_data()) == MarketRegime.NORMAL

    def test_normal_regime_without_current_bar(self) -> None:
        assert RegimeDetector().detect(_normal_data(), current_bar=None) == MarketRegime.NORMAL


# ---------------------------------------------------------------------------
# Regime classification — VOLATILE
# ---------------------------------------------------------------------------


class TestVolatileRegime:
    def test_volatile_when_atr_between_thresholds(self) -> None:
        """ratio ≈ 1.615 → VOLATILE."""
        assert RegimeDetector().detect(_volatile_data()) == MarketRegime.VOLATILE

    def test_boundary_at_volatile_threshold_is_volatile(self) -> None:
        """ratio == 1.5 exactly → VOLATILE (lower bound inclusive)."""
        det = RegimeDetector()
        assert det._classify(1.5, 1.0) == MarketRegime.VOLATILE

    def test_just_below_volatile_is_normal(self) -> None:
        """ratio = 1.499 → NORMAL."""
        assert RegimeDetector()._classify(1.499, 1.0) == MarketRegime.NORMAL


# ---------------------------------------------------------------------------
# Regime classification — STRESS
# ---------------------------------------------------------------------------


class TestStressRegime:
    def test_stress_when_atr_above_high_threshold(self) -> None:
        """ratio ≈ 14.4 → STRESS."""
        assert RegimeDetector().detect(_stress_data()) == MarketRegime.STRESS

    def test_boundary_at_stress_threshold_is_stress(self) -> None:
        """ratio == 3.0 exactly → STRESS (lower bound inclusive)."""
        assert RegimeDetector()._classify(3.0, 1.0) == MarketRegime.STRESS

    def test_just_below_stress_is_volatile(self) -> None:
        """ratio = 2.999 → VOLATILE."""
        assert RegimeDetector()._classify(2.999, 1.0) == MarketRegime.VOLATILE


# ---------------------------------------------------------------------------
# CASCADE detection
# ---------------------------------------------------------------------------


class TestCascadeDetection:
    def test_cascade_overrides_normal_regime(self) -> None:
        """(high-low)/open = 0.30 > 0.10 → CASCADE despite NORMAL ATR history."""
        bar = _make_bar(open_=10_000, high=11_500, low=8_500)  # range = 3000/10000 = 0.30
        assert RegimeDetector().detect(_normal_data(), bar) == MarketRegime.CASCADE

    def test_cascade_overrides_volatile_regime(self) -> None:
        bar = _make_bar(open_=10_000, high=11_500, low=8_500)
        assert RegimeDetector().detect(_volatile_data(), bar) == MarketRegime.CASCADE

    def test_no_cascade_when_movement_below_threshold(self) -> None:
        """(high-low)/open = 0.08 < 0.10 → regime determined by ATR (NORMAL)."""
        bar = _make_bar(open_=10_000, high=10_400, low=9_600)  # range = 800/10000 = 0.08
        assert RegimeDetector().detect(_normal_data(), bar) == MarketRegime.NORMAL

    def test_cascade_at_exact_threshold_is_not_cascade(self) -> None:
        """(high-low)/open == 0.10 exactly → NOT CASCADE (strict >).

        The CASCADE check uses strict inequality: intrabar_move > threshold.
        At exactly the threshold, the bar falls through to ATR-based classification.
        """
        bar = _make_bar(open_=10_000, high=10_500, low=9_500)  # range = 1000/10000 = 0.10
        result = RegimeDetector().detect(_normal_data(), bar)
        assert result != MarketRegime.CASCADE
        assert result == MarketRegime.NORMAL

    def test_current_bar_none_skips_cascade_check(self) -> None:
        """Passing current_bar=None never returns CASCADE."""
        assert RegimeDetector().detect(_normal_data(), current_bar=None) == MarketRegime.NORMAL


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_insufficient_history_raises(self) -> None:
        """len(df) < atr_period → ValueError."""
        df = _make_df(13, 50)  # default atr_period=14
        with pytest.raises(ValueError, match="atr_period"):
            RegimeDetector().detect(df)

    def test_exactly_atr_period_rows_valid(self) -> None:
        """len(df) == atr_period → 1 valid ATR value, ratio=1.0 → NORMAL."""
        df = _make_df(14, 50)
        assert RegimeDetector().detect(df) == MarketRegime.NORMAL

    def test_insufficient_history_cascade_still_triggers(self) -> None:
        """CASCADE check happens before history length check."""
        df = _make_df(5, 50)  # too short for ATR
        bar = _make_bar(open_=10_000, high=11_500, low=8_500)  # cascade
        # CASCADE fires before the ValueError → returns CASCADE, not raises
        assert RegimeDetector().detect(df, bar) == MarketRegime.CASCADE


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_data_same_regime(self) -> None:
        df = _volatile_data()
        det = RegimeDetector()
        r1 = det.detect(df)
        r2 = det.detect(df)
        assert r1 == r2

    def test_same_data_with_bar_same_regime(self) -> None:
        df = _normal_data()
        bar = _make_bar(open_=10_000, high=10_300, low=9_700)
        det = RegimeDetector()
        r1 = det.detect(df, bar)
        r2 = det.detect(df, bar)
        assert r1 == r2


# ---------------------------------------------------------------------------
# detect_for_executor
# ---------------------------------------------------------------------------


class TestDetectForExecutor:
    def test_returns_same_as_detect(self) -> None:
        df = _normal_data()
        bar = _make_bar(open_=10_000, high=10_300, low=9_700)
        det = RegimeDetector()
        assert det.detect_for_executor(df, bar) == det.detect(df, bar)
