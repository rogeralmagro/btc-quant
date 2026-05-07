"""Unit tests for DataValidator.

Each test targets exactly one validation rule using minimal synthetic DataFrames,
so failures point directly to the broken check without ambiguity.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from btc_quant.data.validator import DataValidator, ValidationReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INTERVAL = "1d"
INTERVAL_MS = 86_400_000
CLEAN_START = datetime(2017, 8, 17, tzinfo=timezone.utc)


def _make_clean_df(
    n: int = 10,
    start: datetime = CLEAN_START,
    interval_ms: int = INTERVAL_MS,
) -> pd.DataFrame:
    """Build a gap-free, anomaly-free OHLCV DataFrame starting at start."""
    timestamps = pd.date_range(
        start=start,
        periods=n,
        freq=pd.Timedelta(milliseconds=interval_ms),
        tz="UTC",
    ).as_unit("ns")

    return pd.DataFrame({
        "timestamp_utc": timestamps,
        "open":  50_000.0,
        "high":  51_000.0,
        "low":   49_000.0,
        "close": 50_500.0,
        "volume": 100.0,
        "close_time_utc": timestamps + pd.Timedelta(milliseconds=interval_ms - 1),
        "quote_volume": 5_050_000.0,
        "num_trades": 1000,
        "is_synthetic": False,
    })


# ---------------------------------------------------------------------------
# Duplicate tests
# ---------------------------------------------------------------------------

class TestDuplicates:
    def test_duplicate_timestamp_detected(self) -> None:
        df = _make_clean_df(5)
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

        report = DataValidator().validate(df, INTERVAL)

        assert report.duplicates == 1
        assert not report.is_valid

    def test_no_duplicates_in_clean_data(self) -> None:
        report = DataValidator().validate(_make_clean_df(5), INTERVAL)

        assert report.duplicates == 0


# ---------------------------------------------------------------------------
# Gap tests
# ---------------------------------------------------------------------------

class TestGaps:
    def test_single_gap_detected(self) -> None:
        df = _make_clean_df(10).drop(index=3).reset_index(drop=True)

        report = DataValidator().validate(df, INTERVAL)

        assert len(report.gaps) == 1
        assert not report.is_valid

    def test_gap_missing_candle_count_is_correct(self) -> None:
        """Dropping 3 consecutive rows produces a gap of 3 missing candles."""
        df = _make_clean_df(10).drop(index=[3, 4, 5]).reset_index(drop=True)

        report = DataValidator().validate(df, INTERVAL)

        assert len(report.gaps) == 1
        _, _, n_missing = report.gaps[0]
        assert n_missing == 3

    def test_two_separate_gaps_detected(self) -> None:
        df = _make_clean_df(10).drop(index=[2, 7]).reset_index(drop=True)

        report = DataValidator().validate(df, INTERVAL)

        assert len(report.gaps) == 2

    def test_gap_start_is_first_missing_candle(self) -> None:
        """gap_start must equal the timestamp of the first missing candle."""
        df = _make_clean_df(5)
        missing_ts = df["timestamp_utc"].iloc[2].to_pydatetime()
        df = df.drop(index=2).reset_index(drop=True)

        report = DataValidator().validate(df, INTERVAL)

        gap_start, _, _ = report.gaps[0]
        assert gap_start == missing_ts

    def test_no_gaps_in_clean_data(self) -> None:
        report = DataValidator().validate(_make_clean_df(10), INTERVAL)

        assert len(report.gaps) == 0


# ---------------------------------------------------------------------------
# OHLC consistency tests
# ---------------------------------------------------------------------------

class TestOHLCConsistency:
    def test_price_zero_detected(self) -> None:
        df = _make_clean_df(5)
        df.loc[2, "close"] = 0.0

        report = DataValidator().validate(df, INTERVAL)

        assert any(a.check == "price_zero" for a in report.anomalies)
        assert not report.is_valid

    def test_low_greater_than_high(self) -> None:
        df = _make_clean_df(5)
        df.loc[2, "low"] = 52_000.0  # exceeds high (51_000)

        report = DataValidator().validate(df, INTERVAL)

        assert any(a.check == "low_gt_high" for a in report.anomalies)
        assert not report.is_valid

    def test_low_greater_than_open(self) -> None:
        df = _make_clean_df(5)
        df.loc[2, "low"] = 51_500.0  # exceeds open (50_000)

        report = DataValidator().validate(df, INTERVAL)

        assert any(a.check == "low_gt_open" for a in report.anomalies)

    def test_low_greater_than_close(self) -> None:
        df = _make_clean_df(5)
        df.loc[2, "low"] = 51_000.0  # exceeds close (50_500)

        report = DataValidator().validate(df, INTERVAL)

        assert any(a.check == "low_gt_close" for a in report.anomalies)

    def test_high_less_than_open(self) -> None:
        df = _make_clean_df(5)
        df.loc[2, "high"] = 45_000.0  # below open (50_000)

        report = DataValidator().validate(df, INTERVAL)

        assert any(a.check == "high_lt_open" for a in report.anomalies)

    def test_high_less_than_close(self) -> None:
        df = _make_clean_df(5)
        df.loc[2, "high"] = 49_000.0  # below close (50_500)

        report = DataValidator().validate(df, INTERVAL)

        assert any(a.check == "high_lt_close" for a in report.anomalies)

    def test_multiple_ohlc_violations_all_reported(self) -> None:
        """A single bad row may trigger multiple OHLC checks simultaneously."""
        df = _make_clean_df(5)
        df.loc[2, "low"] = 55_000.0   # low > high, low > open, low > close
        df.loc[2, "high"] = 45_000.0  # high < open, high < close (and now < low)

        report = DataValidator().validate(df, INTERVAL)

        checks_hit = {a.check for a in report.anomalies}
        assert "low_gt_high" in checks_hit
        assert "low_gt_open" in checks_hit
        assert "high_lt_open" in checks_hit


# ---------------------------------------------------------------------------
# Volume tests
# ---------------------------------------------------------------------------

class TestVolume:
    def test_negative_volume_detected(self) -> None:
        df = _make_clean_df(5)
        df.loc[2, "volume"] = -1.0

        report = DataValidator().validate(df, INTERVAL)

        assert any(a.check == "volume_negative" for a in report.anomalies)
        assert not report.is_valid

    def test_zero_volume_is_allowed(self) -> None:
        """Zero volume can happen on illiquid candles; it is not an error."""
        df = _make_clean_df(5)
        df.loc[2, "volume"] = 0.0

        report = DataValidator().validate(df, INTERVAL)

        assert not any(a.check == "volume_negative" for a in report.anomalies)


# ---------------------------------------------------------------------------
# Intraday spike tests
# ---------------------------------------------------------------------------

class TestIntradaySpike:
    def test_spike_flagged_as_warning_not_error(self) -> None:
        """A >50% intraday range produces a warning but leaves is_valid=True."""
        df = _make_clean_df(5)
        # open=50_000, high=80_000, low=10_000 → range/open = 140%
        df.loc[2, "low"] = 10_000.0
        df.loc[2, "high"] = 80_000.0

        report = DataValidator().validate(df, INTERVAL)

        spike = [a for a in report.anomalies if a.check == "intraday_spike"]
        assert len(spike) == 1
        assert spike[0].severity == "warning"
        assert report.is_valid

    def test_normal_candle_not_flagged(self) -> None:
        """A regular candle with 2% range must not produce any spike warning."""
        df = _make_clean_df(5)  # high-low = 2% of open by default

        report = DataValidator().validate(df, INTERVAL)

        assert not any(a.check == "intraday_spike" for a in report.anomalies)

    def test_spike_detail_contains_percentage(self) -> None:
        df = _make_clean_df(5)
        df.loc[2, "low"] = 1_000.0
        df.loc[2, "high"] = 100_000.0

        report = DataValidator().validate(df, INTERVAL)

        spike = next(a for a in report.anomalies if a.check == "intraday_spike")
        assert "%" in spike.detail


# ---------------------------------------------------------------------------
# Coverage tests
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_fails_when_data_starts_too_late(self) -> None:
        """Data starting in 2020 fails the 2017-08-17 coverage requirement."""
        df = _make_clean_df(10, start=datetime(2020, 1, 1, tzinfo=timezone.utc))

        report = DataValidator().validate(df, INTERVAL)

        assert not report.is_valid

    def test_passes_for_correct_start(self) -> None:
        """Data starting on 2017-08-17 satisfies coverage for 1d interval."""
        report = DataValidator().validate(_make_clean_df(10, start=CLEAN_START), INTERVAL)

        assert report.is_valid

    def test_weekly_candle_starting_2017_08_14_passes(self) -> None:
        """1W data starting on 2017-08-14 (Monday before genesis) must pass coverage."""
        start = datetime(2017, 8, 14, tzinfo=timezone.utc)
        df = _make_clean_df(10, start=start, interval_ms=604_800_000)

        report = DataValidator().validate(df, "1w")

        assert report.is_valid

    def test_custom_minimum_start(self) -> None:
        """DataValidator respects a custom minimum_start passed at construction."""
        custom_start = datetime(2020, 1, 1, tzinfo=timezone.utc)
        validator = DataValidator(minimum_start=custom_start)
        df = _make_clean_df(10, start=datetime(2020, 1, 1, tzinfo=timezone.utc))

        report = validator.validate(df, INTERVAL)

        assert report.is_valid


# ---------------------------------------------------------------------------
# Clean dataset
# ---------------------------------------------------------------------------

class TestCleanDataset:
    def test_clean_dataset_passes_all_checks(self) -> None:
        report = DataValidator().validate(_make_clean_df(100), INTERVAL)

        assert report.is_valid
        assert report.duplicates == 0
        assert len(report.gaps) == 0
        assert not any(a.severity == "error" for a in report.anomalies)

    def test_report_fields_populated_correctly(self) -> None:
        df = _make_clean_df(10)
        report = DataValidator().validate(df, INTERVAL)

        assert report.total_rows == 10
        assert report.date_range[0].date() == CLEAN_START.date()
        assert isinstance(report.date_range[0], datetime)
        assert isinstance(report.date_range[1], datetime)


# ---------------------------------------------------------------------------
# Summary text
# ---------------------------------------------------------------------------

class TestSummaryText:
    def test_summary_contains_status_and_rows(self) -> None:
        report = DataValidator().validate(_make_clean_df(10), INTERVAL)

        assert "PASSED" in report.summary_text
        assert "rows=10" in report.summary_text

    def test_failed_summary_contains_failed(self) -> None:
        df = _make_clean_df(10).drop(index=3).reset_index(drop=True)
        report = DataValidator().validate(df, INTERVAL)

        assert "FAILED" in report.summary_text

    def test_summary_text_is_multiline(self) -> None:
        report = DataValidator().validate(_make_clean_df(10), INTERVAL)

        assert "\n" in report.summary_text

    def test_summary_shows_gap_details(self) -> None:
        df = _make_clean_df(10).drop(index=3).reset_index(drop=True)
        report = DataValidator().validate(df, INTERVAL)

        assert "gap:" in report.summary_text
        assert "missing" in report.summary_text

    def test_summary_shows_error_details(self) -> None:
        df = _make_clean_df(5)
        df.loc[2, "close"] = 0.0
        report = DataValidator().validate(df, INTERVAL)

        assert "error:" in report.summary_text

    def test_unknown_interval_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown interval"):
            DataValidator().validate(_make_clean_df(5), "2d")


# ---------------------------------------------------------------------------
# Synthetic candle reporting
# ---------------------------------------------------------------------------

class TestSyntheticCandles:
    def test_validator_reports_synthetic_count(self) -> None:
        """synthetic_count equals the number of is_synthetic=True rows."""
        df = _make_clean_df(10)
        # Inject 5 synthetic candles by flipping the flag
        df.loc[[1, 3, 5, 7, 9], "is_synthetic"] = True

        report = DataValidator().validate(df, INTERVAL)

        assert report.synthetic_count == 5
        assert report.has_synthetic_candles is True

    def test_validator_passes_with_synthetic_candles(self) -> None:
        """A DataFrame with synthetic candles but no real gaps must be is_valid=True."""
        from btc_quant.data.cleaner import GapFiller

        df = _make_clean_df(10)
        df_with_gap = df.drop(index=3).reset_index(drop=True)  # 1-day gap
        df_filled, _ = GapFiller().fill_gaps(df_with_gap, INTERVAL)

        report = DataValidator().validate(df_filled, INTERVAL)

        assert report.is_valid is True
        assert report.synthetic_count == 1
        assert report.has_synthetic_candles is True

    def test_no_synthetic_by_default(self) -> None:
        """Clean DataFrame has synthetic_count=0 and has_synthetic_candles=False."""
        report = DataValidator().validate(_make_clean_df(10), INTERVAL)

        assert report.synthetic_count == 0
        assert report.has_synthetic_candles is False

    def test_summary_mentions_synthetic_when_present(self) -> None:
        """summary_text includes synthetic count when has_synthetic_candles is True."""
        from btc_quant.data.cleaner import GapFiller

        df_filled, _ = GapFiller().fill_gaps(
            _make_clean_df(10).drop(index=3).reset_index(drop=True), INTERVAL
        )
        report = DataValidator().validate(df_filled, INTERVAL)

        assert "synthetic" in report.summary_text.lower()
        assert str(report.synthetic_count) in report.summary_text
