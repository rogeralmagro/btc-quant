"""BinanceDownloader: fetches OHLCV data from Binance public REST API.

Architectural decisions:
- No API key required: all endpoints used are public (klines).
- Pagination: Binance caps responses at 1000 candles per request; we advance
  start_ms to batch[-1][0] + 1 after each page so pages never overlap.
- Look-ahead bias: fetch_ohlcv filters out any candle whose close_time_utc >= end,
  ensuring only fully-closed candles are returned.
- Rate limits: exponential backoff with jitter on HTTP 429 / 418 responses.
- Timestamps: all datetimes are UTC-aware (datetime64[ns, UTC]).
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone

import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from loguru import logger


OHLCV_COLUMNS = [
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_utc",
    "quote_volume",
    "num_trades",
    "is_synthetic",  # False for all real candles; True for gap-filled synthetic ones
]

# Milliseconds per interval — used for pagination step and incremental updates
_INTERVAL_MS: dict[str, int] = {
    Client.KLINE_INTERVAL_1MINUTE: 60_000,
    Client.KLINE_INTERVAL_3MINUTE: 180_000,
    Client.KLINE_INTERVAL_5MINUTE: 300_000,
    Client.KLINE_INTERVAL_15MINUTE: 900_000,
    Client.KLINE_INTERVAL_30MINUTE: 1_800_000,
    Client.KLINE_INTERVAL_1HOUR: 3_600_000,
    Client.KLINE_INTERVAL_2HOUR: 7_200_000,
    Client.KLINE_INTERVAL_4HOUR: 14_400_000,
    Client.KLINE_INTERVAL_6HOUR: 21_600_000,
    Client.KLINE_INTERVAL_8HOUR: 28_800_000,
    Client.KLINE_INTERVAL_12HOUR: 43_200_000,
    Client.KLINE_INTERVAL_1DAY: 86_400_000,
    Client.KLINE_INTERVAL_3DAY: 259_200_000,
    Client.KLINE_INTERVAL_1WEEK: 604_800_000,
    Client.KLINE_INTERVAL_1MONTH: 2_592_000_000,
}

DEFAULT_START = datetime(2017, 8, 17, tzinfo=timezone.utc)
MAX_CANDLES_PER_REQUEST = 1000
_LOG_PROGRESS_EVERY = 10  # log at INFO level every N batches; DEBUG otherwise


class BinanceDownloader:
    """Downloads OHLCV candles from Binance public klines endpoint.

    Handles pagination, rate-limit retry with exponential backoff, and UTC
    timestamp normalization. No API credentials required.
    """

    def __init__(self, max_retries: int = 5, base_retry_delay: float = 1.0) -> None:
        self._client = Client()
        self._max_retries = max_retries
        self._base_retry_delay = base_retry_delay

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles for symbol/interval in the half-open window [start, end).

        Only fully-closed candles are returned (close_time_utc < end), which
        guarantees zero look-ahead bias: at instant t you only see candles
        whose close_time is strictly before t.

        Args:
            symbol: Binance trading pair, e.g. "BTCUSDT".
            interval: Binance interval string, e.g. Client.KLINE_INTERVAL_1DAY.
            start: Inclusive start, must be UTC-aware.
            end: Exclusive end, must be UTC-aware. Candles open at this boundary are dropped.

        Returns:
            DataFrame sorted ascending by timestamp_utc with columns:
            timestamp_utc, open, high, low, close, volume,
            close_time_utc, quote_volume, num_trades.
        """
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        all_raw: list[list] = []
        batch_count = 0

        logger.info(
            "Starting fetch: symbol={} interval={} start={} end={}",
            symbol, interval, start.isoformat(), end.isoformat(),
        )

        while start_ms < end_ms:
            raw = self._fetch_batch(symbol, interval, start_ms, end_ms)
            if not raw:
                break

            all_raw.extend(raw)
            batch_count += 1

            if batch_count % _LOG_PROGRESS_EVERY == 0:
                up_to = datetime.fromtimestamp(raw[-1][0] / 1000, tz=timezone.utc)
                logger.info(
                    "Progress: {} batches fetched ({} candles, up to {})",
                    batch_count, len(all_raw), up_to.isoformat(),
                )
            else:
                logger.debug(
                    "Batch {}: {} candles starting {}",
                    batch_count, len(raw),
                    datetime.fromtimestamp(raw[0][0] / 1000, tz=timezone.utc).isoformat(),
                )

            start_ms = raw[-1][0] + 1  # advance past the last returned open_time

        if not all_raw:
            logger.warning(
                "No data returned for {} {} [{} — {}]",
                symbol, interval, start.isoformat(), end.isoformat(),
            )
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        df = self._parse_klines(all_raw)

        # Drop open (unclosed) candles — core look-ahead bias guard
        df = df[df["close_time_utc"] < pd.Timestamp(end)]
        df = df.sort_values("timestamp_utc").reset_index(drop=True)

        logger.info(
            "Fetch complete: {} candles for {} {} [{} — {}]",
            len(df), symbol, interval,
            df["timestamp_utc"].iloc[0].isoformat(),
            df["timestamp_utc"].iloc[-1].isoformat(),
        )
        return df

    def fetch_full_history(
        self,
        symbol: str,
        interval: str,
        start: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch complete history from start to the last closed candle.

        Args:
            symbol: Binance trading pair, e.g. "BTCUSDT".
            interval: Binance interval string.
            start: Inclusive start (UTC-aware). Defaults to 2017-08-17,
                   the date BTC/USDT became available on Binance.

        Returns:
            DataFrame with all closed candles from start to now.
        """
        if start is None:
            start = DEFAULT_START
        end = datetime.now(tz=timezone.utc)
        logger.info("fetch_full_history: {} {} from {}", symbol, interval, start.isoformat())
        return self.fetch_ohlcv(symbol, interval, start, end)

    def fetch_incremental_update(
        self,
        symbol: str,
        interval: str,
        existing_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fetch only candles newer than the latest timestamp in existing_df.

        Idempotent: start is derived from max(existing timestamp) so running
        twice yields the same result (second run returns an empty DataFrame if
        data is already current).

        Args:
            symbol: Binance trading pair.
            interval: Binance interval string.
            existing_df: Previously stored DataFrame; must have a timestamp_utc column.

        Returns:
            DataFrame with ONLY new candles not present in existing_df.
            May be empty if data is already up to date.
        """
        if existing_df.empty:
            logger.info("existing_df is empty — falling back to fetch_full_history")
            return self.fetch_full_history(symbol, interval)

        interval_ms = self._interval_to_ms(interval)
        last_ts: pd.Timestamp = existing_df["timestamp_utc"].max()
        start = (last_ts + pd.Timedelta(milliseconds=interval_ms)).to_pydatetime()
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)

        end = datetime.now(tz=timezone.utc)

        logger.info(
            "fetch_incremental_update: {} {} from {} (last stored: {})",
            symbol, interval, start.isoformat(), last_ts,
        )

        if start >= end:
            logger.info("Data already up to date for {} {}", symbol, interval)
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        return self.fetch_ohlcv(symbol, interval, start, end)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_batch(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list:
        """Fetch one page of up to 1000 klines with exponential-backoff retry.

        HTTP 429 (rate limit) and 418 (IP banned temporarily) both trigger
        backoff. Other API errors are re-raised immediately.
        """
        for attempt in range(self._max_retries):
            try:
                return self._client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    startTime=start_ms,
                    endTime=end_ms - 1,  # Binance endTime is inclusive; make it exclusive
                    limit=MAX_CANDLES_PER_REQUEST,
                )
            except BinanceAPIException as exc:
                if exc.status_code not in (418, 429):
                    raise
                delay = min(
                    self._base_retry_delay * (2 ** attempt) + random.uniform(0, 1),
                    60.0,
                )
                logger.warning(
                    "Rate limit (HTTP {}) on attempt {}/{} — sleeping {:.1f}s",
                    exc.status_code, attempt + 1, self._max_retries, delay,
                )
                time.sleep(delay)
            except BinanceRequestException as exc:
                logger.error(
                    "Network error on attempt {}/{}: {}",
                    attempt + 1, self._max_retries, exc,
                )
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(self._base_retry_delay * (2 ** attempt))

        raise RuntimeError(
            f"Failed to fetch {symbol} {interval} klines after {self._max_retries} retries"
        )

    def _parse_klines(self, raw: list) -> pd.DataFrame:
        """Convert raw Binance klines list to a standardized DataFrame.

        Binance kline field mapping (index → meaning):
        0 open_time_ms, 1 open, 2 high, 3 low, 4 close, 5 volume (base asset),
        6 close_time_ms, 7 quote_asset_volume, 8 number_of_trades,
        9 taker_buy_base_volume, 10 taker_buy_quote_volume, 11 ignore
        """
        df = pd.DataFrame(raw, columns=[
            "open_time_ms", "open", "high", "low", "close", "volume",
            "close_time_ms", "quote_volume", "num_trades",
            "_taker_buy_base", "_taker_buy_quote", "_ignore",
        ])

        # .dt.as_unit("ns") forces datetime64[ns, UTC] regardless of pandas version
        # (pandas 2.x defaults to datetime64[ms, UTC] for unit="ms")
        df["timestamp_utc"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True).dt.as_unit("ns")
        df["close_time_utc"] = pd.to_datetime(df["close_time_ms"], unit="ms", utc=True).dt.as_unit("ns")

        for col in ("open", "high", "low", "close", "volume", "quote_volume"):
            df[col] = df[col].astype("float64")
        df["num_trades"] = df["num_trades"].astype("int64")
        df["is_synthetic"] = False

        return df[OHLCV_COLUMNS].copy()

    def _interval_to_ms(self, interval: str) -> int:
        """Return the number of milliseconds in one interval period.

        Raises:
            ValueError: if interval is not a recognized Binance interval string.
        """
        if interval not in _INTERVAL_MS:
            raise ValueError(
                f"Unknown interval '{interval}'. Valid: {sorted(_INTERVAL_MS.keys())}"
            )
        return _INTERVAL_MS[interval]
