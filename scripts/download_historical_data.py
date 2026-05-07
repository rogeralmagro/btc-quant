"""Download and persist historical OHLCV data from Binance public API.

Usage:
    python scripts/download_historical_data.py
    python scripts/download_historical_data.py --update-only
    python scripts/download_historical_data.py --symbol BTCUSDT --intervals 1d,4h,1w
    python scripts/download_historical_data.py --start 2020-01-01 --log-level DEBUG

Exit codes:
    0 — all intervals downloaded and validated successfully
    1 — one or more intervals failed validation or raised an error
"""

import argparse
import sys
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from btc_quant.data.cleaner import GapFiller
from btc_quant.data.downloader import BinanceDownloader
from btc_quant.data.storage import ParquetStorage
from btc_quant.data.validator import DataValidator

BASE_PATH = "data/processed"
DEFAULT_SYMBOL = "BTCUSDT"
DEFAULT_INTERVALS = "1d,4h,1w"
DEFAULT_START = "2017-08-17"


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download historical OHLCV data from Binance"
    )
    p.add_argument(
        "--symbol", default=DEFAULT_SYMBOL,
        help=f"Trading pair (default: {DEFAULT_SYMBOL})",
    )
    p.add_argument(
        "--intervals", default=DEFAULT_INTERVALS,
        help=f"Comma-separated Binance intervals (default: {DEFAULT_INTERVALS})",
    )
    p.add_argument(
        "--start", default=DEFAULT_START,
        help=f"Start date YYYY-MM-DD (default: {DEFAULT_START})",
    )
    p.add_argument(
        "--update-only", action="store_true",
        help="Fetch only candles newer than the last stored one (requires existing file)",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return p.parse_args()


def _process_interval(
    symbol: str,
    interval: str,
    start: datetime,
    update_only: bool,
    downloader: BinanceDownloader,
    gap_filler: GapFiller,
    validator: DataValidator,
    storage: ParquetStorage,
) -> dict:
    """Download, gap-fill, validate, and persist one symbol/interval combination.

    Returns a summary record dict consumed by _print_summary.
    Does NOT raise — all errors are caught, logged, and reflected in the record.
    """
    record: dict = {
        "symbol": symbol,
        "interval": interval,
        "rows": None,
        "synthetic": None,
        "start": None,
        "end": None,
        "status": "FAILED",
    }

    try:
        if update_only and storage.exists(symbol, interval, BASE_PATH):
            logger.info("[{}/{}] Mode: incremental update", symbol, interval)
            existing = storage.load(symbol, interval, BASE_PATH)
            new_data = downloader.fetch_incremental_update(symbol, interval, existing)

            if new_data.empty:
                logger.info("[{}/{}] Already up to date — nothing to fetch", symbol, interval)
                df = existing
            else:
                logger.info("[{}/{}] Merging {} new candles with {} existing",
                            symbol, interval, len(new_data), len(existing))
                df = (
                    pd.concat([existing, new_data], ignore_index=True)
                    .sort_values("timestamp_utc")
                    .reset_index(drop=True)
                )
        else:
            if update_only:
                logger.info(
                    "[{}/{}] --update-only set but no existing file found — running full download",
                    symbol, interval,
                )
            else:
                logger.info("[{}/{}] Mode: full history download", symbol, interval)
            df = downloader.fetch_full_history(symbol, interval, start)

        logger.info("[{}/{}] Applying gap filler...", symbol, interval)
        df, fill_report = gap_filler.fill_gaps(df, interval)
        if fill_report.total_synthetic_added > 0:
            logger.info("[{}/{}] {}", symbol, interval, fill_report)
        if not fill_report.is_clean:
            logger.error(
                "[{}/{}] {} gap(s) exceeded 48h threshold and were NOT filled",
                symbol, interval, len(fill_report.rejected_gaps),
            )

        logger.info("[{}/{}] Validating {} candles...", symbol, interval, len(df))
        report = validator.validate(df, interval)

        if not report.is_valid:
            logger.error(
                "[{}/{}] Validation FAILED — data NOT saved:\n{}",
                symbol, interval, report.summary_text,
            )
            record["status"] = "VALIDATION_FAILED"
            return record

        warnings = [a for a in report.anomalies if a.severity == "warning"]
        if warnings:
            logger.warning(
                "[{}/{}] {} candle(s) flagged with warnings (check ValidationReport)",
                symbol, interval, len(warnings),
            )

        storage.save(df, symbol, interval, BASE_PATH)

        status = "OK*" if report.has_synthetic_candles else "OK"
        record.update({
            "rows": report.total_rows,
            "synthetic": report.synthetic_count,
            "start": report.date_range[0].date(),
            "end": report.date_range[1].date(),
            "status": status,
        })
        logger.info(
            "[{}/{}] Done — {} rows saved ({} synthetic)",
            symbol, interval, report.total_rows, report.synthetic_count,
        )

    except Exception:
        logger.exception("[{}/{}] Unexpected error", symbol, interval)
        record["status"] = "ERROR"

    return record


def _print_summary(records: list[dict]) -> None:
    """Print an ASCII summary table to stdout."""
    headers = ["Symbol", "Interval", "Rows", "Synthetic", "Start", "End", "Status"]

    rows = [
        [
            r["symbol"],
            r["interval"],
            str(r["rows"])      if r["rows"]      is not None else "—",
            str(r["synthetic"]) if r["synthetic"]  is not None else "—",
            str(r["start"])     if r["start"]      is not None else "—",
            str(r["end"])       if r["end"]        is not None else "—",
            r["status"],
        ]
        for r in records
    ]

    col_widths = [
        max(len(h), *(len(row[i]) for row in rows))
        for i, h in enumerate(headers)
    ]

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    def _fmt(cols: list[str]) -> str:
        return "|" + "|".join(f" {c.ljust(w)} " for c, w in zip(cols, col_widths)) + "|"

    print()
    print(sep)
    print(_fmt(headers))
    print(sep)
    for row in rows:
        print(_fmt(row))
    print(sep)
    print()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.log_level)

    try:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        logger.error("Invalid --start date '{}'. Expected format: YYYY-MM-DD", args.start)
        sys.exit(1)

    intervals = [i.strip() for i in args.intervals.split(",") if i.strip()]

    logger.info(
        "Starting | symbol={} intervals={} start={} update_only={}",
        args.symbol, intervals, start.date(), args.update_only,
    )

    downloader = BinanceDownloader()
    gap_filler = GapFiller()
    validator = DataValidator()
    storage = ParquetStorage()

    records = []
    for interval in intervals:
        records.append(
            _process_interval(
                args.symbol, interval, start, args.update_only,
                downloader, gap_filler, validator, storage,
            )
        )

    _print_summary(records)

    failed = [r for r in records if r["status"] not in ("OK", "OK*")]
    if failed:
        logger.error(
            "{} interval(s) failed: {}",
            len(failed), [r["interval"] for r in failed],
        )
        sys.exit(1)

    logger.info("All {} interval(s) completed successfully.", len(records))


if __name__ == "__main__":
    main()
