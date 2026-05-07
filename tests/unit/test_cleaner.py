"""Unit tests for GapFiller.

Every test uses fully synthetic DataFrames — no network, no files.
The helpers intentionally create gaps by dropping rows from otherwise
complete time series so the expected timestamps are unambiguous.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from btc_quant.data.cleaner import GapFiller, FillReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTERVAL_MS_4H = 14_400_000
INTERVAL_MS_1D = 86_400_000
START = datetime(2018, 1, 1, tzinfo=timezone.utc)


def _make_df(
    n: int,
    start: datetime = START,
    interval_ms: int = INTERVAL_MS_1D,
    close: float = 50_000.0,
) -> pd.DataFrame:
    """Gap-free OHLCV DataFrame with n candles."""
    timestamps = pd.date_range(
        start=start,
        periods=n,
        freq=pd.Timedelta(milliseconds=interval_ms),
        tz="UTC",
    ).as_unit("ns")

    return pd.DataFrame({
        "timestamp_utc":  timestamps,
        "open":           close,
        "high":           close + 1_000.0,
        "low":            close - 1_000.0,
        "close":          close,
        "volume":         100.0,
        "close_time_utc": timestamps + pd.Timedelta(milliseconds=interval_ms - 1),
        "quote_volume":   5_000_000.0,
        "num_trades":     1000,
        "is_synthetic":   False,
    })


def _drop_rows(df: pd.DataFrame, indices: list[int]) -> pd.DataFrame:
    return df.drop(index=indices).reset_index(drop=True)


# ---------------------------------------------------------------------------
# No gaps — unchanged
# ---------------------------------------------------------------------------

class TestNoGaps:
    def test_no_gaps_returns_unchanged(self) -> None:
        df = _make_df(10)
        result, report = GapFiller().fill_gaps(df, "1d")

        assert len(result) == len(df)
        assert report.total_synthetic_added == 0
        assert report.synthetic_periods == []
        assert report.rejected_gaps == []
        assert report.is_clean is True

    def test_no_gaps_report_is_clean(self) -> None:
        _, report = GapFiller().fill_gaps(_make_df(5), "1d")
        assert report.is_clean is True


# ---------------------------------------------------------------------------
# Single small gap
# ---------------------------------------------------------------------------

class TestSingleSmallGap:
    def test_single_gap_filled(self) -> None:
        """Drop rows 3,4,5 → gap of 3 × 4h = 12h < 48h → filled."""
        df = _make_df(10, interval_ms=INTERVAL_MS_4H)
        df_gap = _drop_rows(df, [3, 4, 5])

        result, report = GapFiller().fill_gaps(df_gap, "4h")

        assert len(result) == 10  # 7 real + 3 synthetic
        assert report.total_synthetic_added == 3
        assert len(report.synthetic_periods) == 1

    def test_synthetic_candles_have_correct_ohlc(self) -> None:
        """O=H=L=C=prev_close for every synthetic candle."""
        df = _make_df(5, close=42_000.0)
        df_with_gap = _drop_rows(df, [2])  # gap of 1 candle

        result, _ = GapFiller().fill_gaps(df_with_gap, "1d")

        synthetic = result[result["is_synthetic"]]
        assert len(synthetic) == 1
        row = synthetic.iloc[0]
        assert row["open"] == 42_000.0
        assert row["high"] == 42_000.0
        assert row["low"] == 42_000.0
        assert row["close"] == 42_000.0

    def test_synthetic_candles_have_zero_volume(self) -> None:
        df = _drop_rows(_make_df(5), [2])
        result, _ = GapFiller().fill_gaps(df, "1d")

        synthetic = result[result["is_synthetic"]]
        assert (synthetic["volume"] == 0.0).all()
        assert (synthetic["quote_volume"] == 0.0).all()
        assert (synthetic["num_trades"] == 0).all()

    def test_synthetic_candles_marked_true(self) -> None:
        """Synthetic rows have is_synthetic=True; real rows have is_synthetic=False."""
        df = _drop_rows(_make_df(5), [2])
        result, _ = GapFiller().fill_gaps(df, "1d")

        assert result["is_synthetic"].dtype == bool
        assert result["is_synthetic"].sum() == 1
        assert (~result["is_synthetic"]).sum() == 4


# ---------------------------------------------------------------------------
# Timestamp correctness
# ---------------------------------------------------------------------------

class TestTimestamps:
    def test_synthetic_timestamps_are_sequential(self) -> None:
        """Three missing 4h candles get timestamps prev+4h, prev+8h, prev+12h."""
        df = _make_df(6, interval_ms=INTERVAL_MS_4H)
        prev_ts = df["timestamp_utc"].iloc[2]
        df_gap = _drop_rows(df, [3, 4, 5])  # drop last 3

        # We need a candle after the gap, so rebuild with 8 candles and drop middle 3
        df8 = _make_df(8, interval_ms=INTERVAL_MS_4H)
        prev_ts = df8["timestamp_utc"].iloc[2]
        df_gap = _drop_rows(df8, [3, 4, 5])

        result, report = GapFiller().fill_gaps(df_gap, "4h")

        synthetic = result[result["is_synthetic"]].reset_index(drop=True)
        assert len(synthetic) == 3

        for k, row in synthetic.iterrows():
            expected_ts = prev_ts + pd.Timedelta(milliseconds=INTERVAL_MS_4H * (k + 1))
            assert row["timestamp_utc"] == expected_ts

    def test_synthetic_close_time_is_timestamp_plus_interval_minus_1ms(self) -> None:
        df = _drop_rows(_make_df(5, interval_ms=INTERVAL_MS_4H), [2])
        result, _ = GapFiller().fill_gaps(df, "4h")

        synthetic = result[result["is_synthetic"]].iloc[0]
        expected_close = synthetic["timestamp_utc"] + pd.Timedelta(milliseconds=INTERVAL_MS_4H - 1)
        assert synthetic["close_time_utc"] == expected_close

    def test_timestamps_are_utc_aware(self) -> None:
        df = _drop_rows(_make_df(5), [2])
        result, _ = GapFiller().fill_gaps(df, "1d")

        assert str(result["timestamp_utc"].dtype) == "datetime64[ns, UTC]"
        assert str(result["close_time_utc"].dtype) == "datetime64[ns, UTC]"

    def test_synthetic_uses_prev_close_not_next_open(self) -> None:
        """No look-ahead: synthetic value equals prev_close, not next_open."""
        df = _make_df(5, close=30_000.0)
        # Manually set a different close for the candle AFTER the gap
        df.loc[3, "open"] = 99_999.0
        df.loc[3, "close"] = 99_999.0
        df_gap = _drop_rows(df, [2])  # remove candle at index 2

        result, _ = GapFiller().fill_gaps(df_gap, "1d")

        synthetic = result[result["is_synthetic"]].iloc[0]
        # Must be 30_000 (from candle at original index 1), never 99_999
        assert synthetic["close"] == 30_000.0


# ---------------------------------------------------------------------------
# Gap above threshold — rejected
# ---------------------------------------------------------------------------

class TestGapAboveThreshold:
    def test_gap_above_48h_rejected(self) -> None:
        """A gap of 13 × 4h = 52h > 48h must NOT be filled."""
        df = _make_df(20, interval_ms=INTERVAL_MS_4H)
        # Drop 13 consecutive candles → 13 × 4h = 52h gap
        df_gap = _drop_rows(df, list(range(5, 18)))

        result, report = GapFiller().fill_gaps(df_gap, "4h", max_gap_hours=48)

        assert report.total_synthetic_added == 0
        assert len(report.rejected_gaps) == 1
        assert report.rejected_gaps[0][2] == 13
        assert report.is_clean is False
        # DataFrame should have only the original real rows, no synthetic
        assert len(result) == len(df_gap)

    def test_gap_exactly_at_threshold_is_filled(self) -> None:
        """A gap of exactly 48h (12 × 4h) must be filled, not rejected."""
        df = _make_df(20, interval_ms=INTERVAL_MS_4H)
        df_gap = _drop_rows(df, list(range(5, 17)))  # 12 candles = 48h

        result, report = GapFiller().fill_gaps(df_gap, "4h", max_gap_hours=48)

        assert report.total_synthetic_added == 12
        assert report.is_clean is True

    def test_custom_threshold_respected(self) -> None:
        """max_gap_hours=4 rejects any gap > 1 candle (4h interval)."""
        df = _drop_rows(_make_df(10, interval_ms=INTERVAL_MS_4H), [3, 4])  # 2-candle gap = 8h

        result, report = GapFiller().fill_gaps(df, "4h", max_gap_hours=4)

        assert len(report.rejected_gaps) == 1
        assert report.is_clean is False


# ---------------------------------------------------------------------------
# Multiple gaps
# ---------------------------------------------------------------------------

class TestMultipleGaps:
    def test_multiple_gaps_all_handled(self) -> None:
        """Two separate gaps are each processed independently."""
        df = _drop_rows(_make_df(15), [3, 10])  # two single-candle gaps

        result, report = GapFiller().fill_gaps(df, "1d")

        assert report.total_synthetic_added == 2
        assert len(report.synthetic_periods) == 2
        assert len(result) == 15

    def test_mixed_filled_and_rejected(self) -> None:
        """Small gap is filled; large gap is rejected in the same DataFrame."""
        df = _make_df(30, interval_ms=INTERVAL_MS_4H)
        # Small gap: 2 candles = 8h < 48h → filled
        # Large gap: 13 candles = 52h > 48h → rejected
        df_gap = _drop_rows(df, [5, 6, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27])

        result, report = GapFiller().fill_gaps(df_gap, "4h", max_gap_hours=48)

        assert report.total_synthetic_added == 2
        assert len(report.synthetic_periods) == 1
        assert len(report.rejected_gaps) == 1
        assert report.is_clean is False


# ---------------------------------------------------------------------------
# Order preservation
# ---------------------------------------------------------------------------

class TestOrder:
    def test_chronological_order_preserved(self) -> None:
        """Result is always sorted ascending by timestamp_utc."""
        df = _drop_rows(_make_df(10), [3, 7])
        df_shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)

        result, _ = GapFiller().fill_gaps(df_shuffled, "1d")

        assert result["timestamp_utc"].is_monotonic_increasing

    def test_result_has_no_duplicate_timestamps(self) -> None:
        df = _drop_rows(_make_df(10), [3])
        result, _ = GapFiller().fill_gaps(df, "1d")

        assert result["timestamp_utc"].duplicated().sum() == 0


# ---------------------------------------------------------------------------
# Fill report __str__
# ---------------------------------------------------------------------------

class TestFillReport:
    def test_str_is_readable(self) -> None:
        df = _drop_rows(_make_df(10), [3])
        _, report = GapFiller().fill_gaps(df, "1d")

        text = str(report)
        assert "synthetic" in text.lower()
        assert "filled" in text.lower()

    def test_unknown_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown interval"):
            GapFiller().fill_gaps(_make_df(5), "2d")
