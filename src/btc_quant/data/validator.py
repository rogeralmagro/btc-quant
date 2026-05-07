"""DataValidator: validates OHLCV DataFrames for quality and consistency.

Architectural decisions:
- Returns ValidationReport instead of raising exceptions so callers can
  log/display failures and decide whether to abort or continue.
- Separates hard errors (data is unusable: gaps, duplicates, OHLC violations,
  negative volume) from soft warnings (suspicious but tradeable: intraday spikes).
  Only hard errors set is_valid=False.
- Anomaly.severity = "error" | "warning" allows downstream filtering.
- Coverage check: first candle must be within one interval of minimum_start
  to accommodate weekly candles that start on 2017-08-14 (Monday before 2017-08-17).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from loguru import logger


MINIMUM_START = datetime(2017, 8, 17, tzinfo=timezone.utc)
INTRADAY_SPIKE_THRESHOLD = 0.50  # flag candles where (high-low)/open > 50%

# Milliseconds per Binance interval string — kept local to avoid coupling with downloader
_INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,
}


@dataclass
class Anomaly:
    timestamp_utc: datetime
    check: str
    detail: str
    severity: str = "error"  # "error" → is_valid=False; "warning" → flagged only


@dataclass
class ValidationReport:
    is_valid: bool
    total_rows: int
    date_range: tuple[datetime, datetime]
    gaps: list[tuple[datetime, datetime, int]]  # (gap_start, gap_end, n_missing_candles)
    anomalies: list[Anomaly]
    duplicates: int
    summary_text: str


class DataValidator:
    """Validates OHLCV DataFrames produced by BinanceDownloader.

    Instantiate once and call validate() for each DataFrame to check.
    """

    def __init__(self, minimum_start: datetime = MINIMUM_START) -> None:
        self._minimum_start = minimum_start

    def validate(self, df: pd.DataFrame, interval: str) -> ValidationReport:
        """Run all quality checks and return a ValidationReport.

        Checks performed (hard errors unless marked WARNING):
        1. No duplicate timestamps
        2. No temporal gaps — consecutive timestamps must differ by exactly interval
        3. OHLC consistency: no zero prices; low ≤ high, open, close; high ≥ open, close
        4. Volume ≥ 0
        5. [WARNING] Intraday range > 50% of open price — suspicious, not necessarily wrong
        6. Coverage: first candle within one interval of minimum_start

        Args:
            df: DataFrame with columns: timestamp_utc, open, high, low, close, volume.
            interval: Binance interval string, e.g. "1d", "4h", "1w".

        Returns:
            ValidationReport. is_valid=True only when all hard checks pass.
        """
        if interval not in _INTERVAL_MS:
            raise ValueError(
                f"Unknown interval '{interval}'. Valid: {sorted(_INTERVAL_MS.keys())}"
            )

        interval_ms = _INTERVAL_MS[interval]
        logger.info("Validating {} rows, interval={}", len(df), interval)

        duplicates = self._check_duplicates(df)
        gaps = self._check_gaps(df, interval_ms)
        anomalies = (
            self._check_ohlc(df)
            + self._check_volume(df)
            + self._check_spikes(df)
        )
        coverage_ok = self._check_coverage(df, interval_ms)

        hard_errors = [a for a in anomalies if a.severity == "error"]
        is_valid = (
            duplicates == 0
            and len(gaps) == 0
            and len(hard_errors) == 0
            and coverage_ok
        )

        if df.empty:
            date_range: tuple[datetime, datetime] = (datetime.min, datetime.max)
        else:
            date_range = (
                df["timestamp_utc"].min().to_pydatetime(),
                df["timestamp_utc"].max().to_pydatetime(),
            )

        report = ValidationReport(
            is_valid=is_valid,
            total_rows=len(df),
            date_range=date_range,
            gaps=gaps,
            anomalies=anomalies,
            duplicates=duplicates,
            summary_text="",
        )
        report.summary_text = self._build_summary(report, interval)

        if is_valid:
            logger.info("Validation PASSED: {}", report.summary_text)
        else:
            logger.warning("Validation FAILED: {}", report.summary_text)

        return report

    # ------------------------------------------------------------------
    # Private checks
    # ------------------------------------------------------------------

    def _check_duplicates(self, df: pd.DataFrame) -> int:
        return int(df["timestamp_utc"].duplicated().sum())

    def _check_gaps(
        self, df: pd.DataFrame, interval_ms: int
    ) -> list[tuple[datetime, datetime, int]]:
        """Detect gaps where consecutive timestamps differ by more than interval_ms."""
        if len(df) < 2:
            return []

        sorted_ts = df["timestamp_utc"].sort_values().reset_index(drop=True)
        diffs_ms = sorted_ts.diff().dt.total_seconds().mul(1000)

        gaps: list[tuple[datetime, datetime, int]] = []
        for i in range(1, len(sorted_ts)):
            diff = diffs_ms.iloc[i]
            if diff > interval_ms + 500:  # 500ms tolerance for float precision
                n_missing = int(round(diff / interval_ms)) - 1
                gap_start = (
                    sorted_ts.iloc[i - 1] + pd.Timedelta(milliseconds=interval_ms)
                ).to_pydatetime()
                gap_end = (
                    sorted_ts.iloc[i] - pd.Timedelta(milliseconds=interval_ms)
                ).to_pydatetime()
                gaps.append((gap_start, gap_end, n_missing))

        return gaps

    def _check_ohlc(self, df: pd.DataFrame) -> list[Anomaly]:
        """Flag OHLC consistency violations as hard errors."""
        anomalies: list[Anomaly] = []

        checks: list[tuple[str, pd.Series, str]] = [
            (
                "price_zero",
                (df[["open", "high", "low", "close"]] == 0).any(axis=1),
                "a price field is zero",
            ),
            ("low_gt_high", df["low"] > df["high"], "low > high"),
            ("low_gt_open", df["low"] > df["open"], "low > open"),
            ("low_gt_close", df["low"] > df["close"], "low > close"),
            ("high_lt_open", df["high"] < df["open"], "high < open"),
            ("high_lt_close", df["high"] < df["close"], "high < close"),
        ]

        for check_name, mask, detail in checks:
            for ts in df.loc[mask, "timestamp_utc"]:
                anomalies.append(
                    Anomaly(
                        timestamp_utc=ts.to_pydatetime(),
                        check=check_name,
                        detail=detail,
                        severity="error",
                    )
                )

        return anomalies

    def _check_volume(self, df: pd.DataFrame) -> list[Anomaly]:
        """Flag rows with negative volume as hard errors."""
        mask = df["volume"] < 0
        return [
            Anomaly(
                timestamp_utc=ts.to_pydatetime(),
                check="volume_negative",
                detail="volume < 0",
                severity="error",
            )
            for ts in df.loc[mask, "timestamp_utc"]
        ]

    def _check_spikes(self, df: pd.DataFrame) -> list[Anomaly]:
        """Flag candles with intraday range > threshold as warnings (not errors)."""
        safe_open = df["open"].where(df["open"] != 0, other=1.0)
        intraday_pct = (df["high"] - df["low"]) / safe_open
        mask = intraday_pct > INTRADAY_SPIKE_THRESHOLD

        return [
            Anomaly(
                timestamp_utc=ts.to_pydatetime(),
                check="intraday_spike",
                detail=f"intraday range {pct:.1%} > {INTRADAY_SPIKE_THRESHOLD:.0%} of open",
                severity="warning",
            )
            for ts, pct in zip(df.loc[mask, "timestamp_utc"], intraday_pct[mask])
        ]

    def _check_coverage(self, df: pd.DataFrame, interval_ms: int) -> bool:
        """Return True if first candle is within one interval of minimum_start.

        One-interval tolerance lets weekly candles starting 2017-08-14 pass
        when minimum_start is 2017-08-17.
        """
        if df.empty:
            return False
        first_ts = df["timestamp_utc"].min()
        cutoff = pd.Timestamp(self._minimum_start) + pd.Timedelta(milliseconds=interval_ms)
        return bool(first_ts <= cutoff)

    def _build_summary(self, report: ValidationReport, interval: str) -> str:
        status = "PASSED" if report.is_valid else "FAILED"
        lines = [
            f"[{status}] interval={interval} rows={report.total_rows}",
            f"  date_range : {report.date_range[0].date()} → {report.date_range[1].date()}",
            f"  duplicates : {report.duplicates}",
            f"  gaps       : {len(report.gaps)}",
        ]

        errors = [a for a in report.anomalies if a.severity == "error"]
        warnings = [a for a in report.anomalies if a.severity == "warning"]
        lines.append(f"  anomalies  : {len(errors)} error(s), {len(warnings)} warning(s)")

        for gs, ge, n in report.gaps[:3]:
            lines.append(f"    gap: {gs.date()} → {ge.date()} ({n} missing)")
        if len(report.gaps) > 3:
            lines.append(f"    ... and {len(report.gaps) - 3} more gap(s)")

        for a in errors[:3]:
            lines.append(f"    error: [{a.check}] at {a.timestamp_utc.date()} — {a.detail}")
        if len(errors) > 3:
            lines.append(f"    ... and {len(errors) - 3} more error(s)")

        return "\n".join(lines)
