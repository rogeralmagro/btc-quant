"""Unit tests for MarketContext — emphasis on anti-look-ahead guarantees."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from btc_quant.backtester.market_context import MarketContext
from tests.unit.backtester.conftest import make_ohlcv_df

UTC = timezone.utc
START = datetime(2020, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(
    current_time: datetime,
    n: int = 200,
    tf: str = "1d",
    extra_tfs: dict | None = None,
) -> MarketContext:
    data: dict[str, pd.DataFrame] = {tf: make_ohlcv_df(n, START, tf)}
    if extra_tfs:
        data.update(extra_tfs)
    return MarketContext(data, current_time)


def _close_time_of(bar_index: int, tf: str = "1d") -> datetime:
    """Return the close_time_utc of candle bar_index in a series starting at START."""
    from tests.unit.backtester.conftest import _TF_PERIODS
    period = _TF_PERIODS[tf]
    open_ts = START + bar_index * period
    return open_ts + period - timedelta(milliseconds=1)


def _expected_close(bar_index: int) -> float:
    """close[i] = 10_000 + i * 100."""
    return 10_000.0 + bar_index * 100.0


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestMarketContextCreation:
    def test_basic_creation(self) -> None:
        ctx = _ctx(START + timedelta(days=10))
        assert ctx.symbol == "BTCUSDT"
        assert "1d" in ctx.available_timeframes

    def test_custom_symbol(self) -> None:
        df = make_ohlcv_df(10, START, "1d")
        ctx = MarketContext({"1d": df}, START + timedelta(days=1), symbol="ETHUSDT")
        assert ctx.symbol == "ETHUSDT"

    def test_naive_datetime_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            _ctx(datetime(2020, 1, 10))

    def test_non_utc_offset_raises(self) -> None:
        ts = datetime(2020, 1, 10, tzinfo=timezone(timedelta(hours=2)))
        with pytest.raises(ValueError, match="UTC"):
            _ctx(ts)

    def test_missing_column_raises(self) -> None:
        df = make_ohlcv_df(10, START, "1d").drop(columns=["close_time_utc"])
        with pytest.raises(ValueError, match="missing columns"):
            MarketContext({"1d": df}, START + timedelta(days=5))

    def test_unknown_timeframe_raises(self) -> None:
        df = make_ohlcv_df(10, START, "1d")
        with pytest.raises(ValueError, match="Unknown timeframe"):
            MarketContext({"999x": df}, START + timedelta(days=5))

    def test_available_timeframes(self) -> None:
        df4h = make_ohlcv_df(100, START, "4h")
        df1d = make_ohlcv_df(100, START, "1d")
        ctx = MarketContext(
            {"4h": df4h, "1d": df1d},
            START + timedelta(days=10),
        )
        assert set(ctx.available_timeframes) == {"4h", "1d"}


# ---------------------------------------------------------------------------
# Anti-look-ahead tests (CRITICAL)
# ---------------------------------------------------------------------------


class TestAntiLookAhead:
    def test_get_history_excludes_future_candles(self) -> None:
        """Requesting history at bar 60's close must not include bar 61+."""
        close_60 = _close_time_of(60)
        ctx = _ctx(close_60)
        hist = ctx.get_history("1d", lookback=200)
        # All returned candle indices must be <= 60
        assert (hist["close"] <= _expected_close(60)).all()
        assert _expected_close(61) not in hist["close"].values

    def test_get_history_includes_only_closed_candles(self) -> None:
        """Bar whose close_time_utc > current_time must never appear."""
        close_10 = _close_time_of(10)
        ctx = _ctx(close_10)
        hist = ctx.get_history("1d", lookback=50)
        assert len(hist) == 11  # bars 0..10 inclusive
        assert hist["close"].iloc[-1] == pytest.approx(_expected_close(10))

    def test_edge_case_exactly_at_close_time_includes_candle(self) -> None:
        """current_time == bar.close_time_utc → that bar IS visible (<=)."""
        close_42 = _close_time_of(42)
        ctx = _ctx(close_42)
        latest = ctx.get_latest_close("1d")
        assert latest == pytest.approx(_expected_close(42))

    def test_edge_case_microsecond_before_close_excludes_candle(self) -> None:
        """current_time = close_time - 1µs → that bar is NOT yet closed."""
        close_42 = _close_time_of(42)
        one_us_before = close_42 - timedelta(microseconds=1)
        ctx = _ctx(one_us_before)
        latest = ctx.get_latest_close("1d")
        # Should see bar 41, not 42
        assert latest == pytest.approx(_expected_close(41))

    def test_get_latest_close_returns_correct_value(self) -> None:
        """Latest close is deterministically verifiable against known formula."""
        close_99 = _close_time_of(99)
        ctx = _ctx(close_99)
        assert ctx.get_latest_close("1d") == pytest.approx(_expected_close(99))

    def test_lookback_returns_at_most_n_candles(self) -> None:
        """Requesting more than available returns all available, not an error."""
        close_5 = _close_time_of(5)
        ctx = _ctx(close_5)
        hist = ctx.get_history("1d", lookback=1000)
        assert len(hist) == 6  # bars 0..5

    def test_lookback_truncates_correctly(self) -> None:
        """Requesting lookback=3 when 10 bars visible → last 3 only."""
        close_9 = _close_time_of(9)
        ctx = _ctx(close_9)
        hist = ctx.get_history("1d", lookback=3)
        assert len(hist) == 3
        assert hist["close"].iloc[-1] == pytest.approx(_expected_close(9))
        assert hist["close"].iloc[0] == pytest.approx(_expected_close(7))

    def test_synthetic_candles_included_in_history(self) -> None:
        """is_synthetic=True candles are NOT filtered out by MarketContext."""
        df = make_ohlcv_df(10, START, "1d")
        df.loc[5, "is_synthetic"] = True
        ctx = MarketContext({"1d": df}, _close_time_of(9))
        hist = ctx.get_history("1d", lookback=10)
        assert hist["is_synthetic"].any()


# ---------------------------------------------------------------------------
# get_current_btc_price
# ---------------------------------------------------------------------------


class TestGetCurrentBtcPrice:
    def test_get_current_btc_price_uses_finest_timeframe(self) -> None:
        """With both 1d and 4h loaded, price comes from 4h (finer)."""
        df_1d = make_ohlcv_df(50, START, "1d")
        df_4h = make_ohlcv_df(200, START, "4h")

        # At close of 4h bar 5: close_time = START + 5*4h + 3h59m59.999s
        from tests.unit.backtester.conftest import _TF_PERIODS
        period_4h = _TF_PERIODS["4h"]
        close_4h_5 = START + 5 * period_4h + period_4h - timedelta(milliseconds=1)

        ctx = MarketContext({"1d": df_1d, "4h": df_4h}, close_4h_5)

        # 4h bar 5 close = 10000 + 5*100 = 10500
        assert ctx.get_current_btc_price() == pytest.approx(10_500.0)

    def test_get_current_btc_price_single_tf(self) -> None:
        close_10 = _close_time_of(10)
        ctx = _ctx(close_10)
        assert ctx.get_current_btc_price() == pytest.approx(_expected_close(10))


# ---------------------------------------------------------------------------
# advance_to
# ---------------------------------------------------------------------------


class TestAdvanceTo:
    def test_advance_to_future_works(self) -> None:
        ctx = _ctx(_close_time_of(10))
        new_time = _close_time_of(20)
        ctx.advance_to(new_time)
        assert ctx.current_time == new_time
        assert ctx.get_latest_close("1d") == pytest.approx(_expected_close(20))

    def test_advance_to_same_time_is_idempotent(self) -> None:
        t = _close_time_of(10)
        ctx = _ctx(t)
        ctx.advance_to(t)  # must not raise
        assert ctx.current_time == t

    def test_advance_to_past_raises(self) -> None:
        ctx = _ctx(_close_time_of(20))
        with pytest.raises(ValueError, match="backwards"):
            ctx.advance_to(_close_time_of(19))

    def test_advance_naive_raises(self) -> None:
        ctx = _ctx(_close_time_of(10))
        with pytest.raises(ValueError, match="tz-aware"):
            ctx.advance_to(datetime(2020, 1, 20))

    def test_advance_reveals_more_history(self) -> None:
        """After advancing, more candles become visible."""
        ctx = _ctx(_close_time_of(5))
        assert len(ctx.get_history("1d", 100)) == 6
        ctx.advance_to(_close_time_of(15))
        assert len(ctx.get_history("1d", 100)) == 16


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_invalid_timeframe_raises(self) -> None:
        ctx = _ctx(_close_time_of(10))
        with pytest.raises(ValueError, match="not available"):
            ctx.get_history("4h", lookback=10)

    def test_invalid_lookback_raises(self) -> None:
        ctx = _ctx(_close_time_of(10))
        with pytest.raises(ValueError, match="lookback must be >= 1"):
            ctx.get_history("1d", lookback=0)

    def test_get_latest_close_no_data_raises(self) -> None:
        """Before any candle closes, get_latest_close raises."""
        # current_time before the very first candle's close_time
        before_first = START + timedelta(hours=1)
        ctx = _ctx(before_first)
        with pytest.raises(ValueError, match="No closed candles"):
            ctx.get_latest_close("1d")
