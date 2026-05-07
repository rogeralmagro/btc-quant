"""Unit tests for BinanceDownloader.

All tests mock the Binance client to avoid network calls and make the
suite fast and deterministic.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from binance.client import Client
from binance.exceptions import BinanceAPIException

from btc_quant.data.downloader import BinanceDownloader, OHLCV_COLUMNS


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

INTERVAL_MS_1D = 86_400_000
# 2024-01-01 00:00:00 UTC in milliseconds
START_MS = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _make_raw_kline(open_time_ms: int, interval_ms: int = INTERVAL_MS_1D) -> list:
    """Build a synthetic raw Binance kline row (12 fields, strings for prices)."""
    close_time_ms = open_time_ms + interval_ms - 1
    return [
        open_time_ms,    # 0 open_time
        "50000.00",      # 1 open
        "51000.00",      # 2 high
        "49000.00",      # 3 low
        "50500.00",      # 4 close
        "100.0",         # 5 volume
        close_time_ms,   # 6 close_time
        "5050000.00",    # 7 quote_volume
        1000,            # 8 num_trades
        "50.0",          # 9 taker_buy_base
        "2525000.0",     # 10 taker_buy_quote
        "0",             # 11 ignore
    ]


def _make_klines(n: int, start_ms: int, interval_ms: int = INTERVAL_MS_1D) -> list:
    """Build a list of n consecutive raw klines starting at start_ms."""
    return [_make_raw_kline(start_ms + i * interval_ms, interval_ms) for i in range(n)]


def _downloader_with_mock(side_effect: list) -> BinanceDownloader:
    """Return a BinanceDownloader whose Binance client is mocked with side_effect."""
    d = BinanceDownloader(base_retry_delay=0.0)
    d._client = MagicMock()
    d._client.get_klines.side_effect = side_effect
    return d


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestFetchOhlcvSchema:
    def test_fetch_ohlcv_returns_correct_schema(self) -> None:
        """fetch_ohlcv returns a DataFrame with standard columns and correct dtypes."""
        raw = _make_klines(5, START_MS)
        d = _downloader_with_mock([raw, []])

        df = d.fetch_ohlcv(
            "BTCUSDT",
            Client.KLINE_INTERVAL_1DAY,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        assert list(df.columns) == OHLCV_COLUMNS

        assert str(df["timestamp_utc"].dtype) == "datetime64[ns, UTC]"
        assert str(df["close_time_utc"].dtype) == "datetime64[ns, UTC]"

        for col in ("open", "high", "low", "close", "volume", "quote_volume"):
            assert df[col].dtype == "float64", f"Expected float64 for {col}"

        assert df["num_trades"].dtype == "int64"
        assert df["is_synthetic"].dtype == bool
        assert (df["is_synthetic"] == False).all()  # noqa: E712


# ---------------------------------------------------------------------------
# Pagination tests
# ---------------------------------------------------------------------------

class TestFetchOhlcvPagination:
    def test_fetch_ohlcv_handles_pagination(self) -> None:
        """2500 candles spread over 3 batches triggers 4 get_klines calls (last empty)."""
        page1 = _make_klines(1000, START_MS)
        page2 = _make_klines(1000, START_MS + 1000 * INTERVAL_MS_1D)
        page3 = _make_klines(500, START_MS + 2000 * INTERVAL_MS_1D)

        d = _downloader_with_mock([page1, page2, page3, []])

        # end far enough so no candle is filtered by the look-ahead guard
        # (2024-01-01 + 2500 days ≈ 2030-11 — safe with 2040 as end)
        df = d.fetch_ohlcv(
            "BTCUSDT",
            Client.KLINE_INTERVAL_1DAY,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2040, 1, 1, tzinfo=timezone.utc),
        )

        assert d._client.get_klines.call_count == 4
        assert len(df) == 2500

    def test_fetch_ohlcv_second_batch_starts_after_first(self) -> None:
        """The second batch startTime is last_open_time_of_first_batch + 1ms."""
        page1 = _make_klines(2, START_MS)
        page2 = _make_klines(2, START_MS + 2 * INTERVAL_MS_1D)

        d = _downloader_with_mock([page1, page2, []])
        d.fetch_ohlcv(
            "BTCUSDT",
            Client.KLINE_INTERVAL_1DAY,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        second_call_kwargs = d._client.get_klines.call_args_list[1].kwargs
        expected_start = page1[-1][0] + 1
        assert second_call_kwargs["startTime"] == expected_start


# ---------------------------------------------------------------------------
# Rate limit tests
# ---------------------------------------------------------------------------

class TestFetchOhlcvRateLimit:
    @patch("btc_quant.data.downloader.time.sleep")
    def test_fetch_ohlcv_handles_rate_limit(self, mock_sleep: MagicMock) -> None:
        """A single HTTP 429 triggers a sleep-and-retry; the next attempt succeeds."""
        raw = _make_klines(3, START_MS)
        exc_429 = BinanceAPIException(
            MagicMock(), 429, '{"code":-1003,"msg":"Too many requests"}'
        )

        # Call 1: 429; Call 2: 3 candles (same batch, retried); Call 3: [] ends loop
        d = _downloader_with_mock([exc_429, raw, []])

        df = d.fetch_ohlcv(
            "BTCUSDT",
            Client.KLINE_INTERVAL_1DAY,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        assert len(df) == 3
        assert d._client.get_klines.call_count == 3
        mock_sleep.assert_called_once()

    @patch("btc_quant.data.downloader.time.sleep")
    def test_fetch_ohlcv_raises_after_max_retries(self, _mock_sleep: MagicMock) -> None:
        """RuntimeError is raised when all retries are exhausted."""
        exc_429 = BinanceAPIException(
            MagicMock(), 429, '{"code":-1003,"msg":"Too many requests"}'
        )

        d = BinanceDownloader(max_retries=3, base_retry_delay=0.0)
        d._client = MagicMock()
        d._client.get_klines.side_effect = exc_429

        with pytest.raises(RuntimeError, match="retries"):
            d.fetch_ohlcv(
                "BTCUSDT",
                Client.KLINE_INTERVAL_1DAY,
                start=datetime(2024, 1, 1, tzinfo=timezone.utc),
                end=datetime(2024, 1, 10, tzinfo=timezone.utc),
            )


# ---------------------------------------------------------------------------
# Incremental update tests
# ---------------------------------------------------------------------------

class TestFetchIncrementalUpdate:
    def test_fetch_incremental_returns_only_new(self) -> None:
        """Incremental update fetches starting one interval after the last stored candle."""
        existing_raw = _make_klines(3, START_MS)
        existing_df = BinanceDownloader()._parse_klines(existing_raw)

        new_raw = _make_klines(2, START_MS + 3 * INTERVAL_MS_1D)
        d = _downloader_with_mock([new_raw, []])

        result = d.fetch_incremental_update(
            "BTCUSDT", Client.KLINE_INTERVAL_1DAY, existing_df
        )

        assert len(result) == 2

        last_existing_ts = existing_df["timestamp_utc"].max()
        assert result["timestamp_utc"].min() > last_existing_ts

    def test_fetch_incremental_starttime_is_last_plus_one_interval(self) -> None:
        """The startTime passed to get_klines is exactly max(existing) + 1 interval."""
        existing_raw = _make_klines(3, START_MS)
        existing_df = BinanceDownloader()._parse_klines(existing_raw)

        new_raw = _make_klines(1, START_MS + 3 * INTERVAL_MS_1D)
        d = _downloader_with_mock([new_raw, []])

        d.fetch_incremental_update("BTCUSDT", Client.KLINE_INTERVAL_1DAY, existing_df)

        first_call_kwargs = d._client.get_klines.call_args_list[0].kwargs
        expected_start_ms = START_MS + 3 * INTERVAL_MS_1D
        assert first_call_kwargs["startTime"] == expected_start_ms

    def test_fetch_incremental_empty_existing_falls_back_to_full_history(self) -> None:
        """When existing_df is empty, falls back to fetch_full_history."""
        raw = _make_klines(5, START_MS)
        d = _downloader_with_mock([raw, []])

        result = d.fetch_incremental_update(
            "BTCUSDT", Client.KLINE_INTERVAL_1DAY, pd.DataFrame(columns=OHLCV_COLUMNS)
        )

        assert len(result) == 5

    def test_fetch_incremental_returns_empty_when_up_to_date(self) -> None:
        """Returns empty DataFrame when the last stored candle opened just now.

        open_time = now → next_start = now + 1 day → start >= end → early return.
        """
        now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        existing_df = BinanceDownloader()._parse_klines([_make_raw_kline(now_ms)])

        d = BinanceDownloader()
        d._client = MagicMock()

        result = d.fetch_incremental_update(
            "BTCUSDT", Client.KLINE_INTERVAL_1DAY, existing_df
        )

        assert result.empty
        d._client.get_klines.assert_not_called()


# ---------------------------------------------------------------------------
# Timestamp UTC tests
# ---------------------------------------------------------------------------

class TestTimestampsAreUtc:
    def test_parse_klines_timestamps_are_utc(self) -> None:
        """_parse_klines produces UTC-aware timestamp columns."""
        raw = _make_klines(3, START_MS)
        df = BinanceDownloader()._parse_klines(raw)

        assert hasattr(df["timestamp_utc"].dtype, "tz"), "timestamp_utc must be tz-aware"
        assert str(df["timestamp_utc"].dtype.tz) == "UTC"
        assert hasattr(df["close_time_utc"].dtype, "tz"), "close_time_utc must be tz-aware"
        assert str(df["close_time_utc"].dtype.tz) == "UTC"

    def test_fetch_ohlcv_output_timestamps_are_utc(self) -> None:
        """fetch_ohlcv output has UTC-aware timestamp columns end-to-end."""
        raw = _make_klines(3, START_MS)
        d = _downloader_with_mock([raw, []])

        df = d.fetch_ohlcv(
            "BTCUSDT",
            Client.KLINE_INTERVAL_1DAY,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        assert str(df["timestamp_utc"].dtype.tz) == "UTC"
        assert str(df["close_time_utc"].dtype.tz) == "UTC"

    def test_fetch_ohlcv_timestamp_values_match_start(self) -> None:
        """First candle's timestamp_utc matches the requested start date."""
        raw = _make_klines(3, START_MS)
        d = _downloader_with_mock([raw, []])

        df = d.fetch_ohlcv(
            "BTCUSDT",
            Client.KLINE_INTERVAL_1DAY,
            start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end=datetime(2024, 1, 10, tzinfo=timezone.utc),
        )

        expected_first = pd.Timestamp("2024-01-01", tz="UTC")
        assert df["timestamp_utc"].iloc[0] == expected_first


# ---------------------------------------------------------------------------
# Interval helper tests
# ---------------------------------------------------------------------------

class TestIntervalToMs:
    def test_known_intervals(self) -> None:
        d = BinanceDownloader()
        assert d._interval_to_ms(Client.KLINE_INTERVAL_1DAY) == 86_400_000
        assert d._interval_to_ms(Client.KLINE_INTERVAL_4HOUR) == 14_400_000
        assert d._interval_to_ms(Client.KLINE_INTERVAL_1WEEK) == 604_800_000

    def test_unknown_interval_raises(self) -> None:
        d = BinanceDownloader()
        with pytest.raises(ValueError, match="Unknown interval"):
            d._interval_to_ms("2d")
