"""GapFiller: detects and forward-fills temporal gaps in OHLCV DataFrames.

Architectural decisions:
- Forward-fill only: each synthetic candle copies prev_close to O/H/L/C and
  sets volume=0. This uses only past data → zero look-ahead bias.
- Hard ceiling: gaps longer than max_gap_hours are rejected and left unfilled.
  The caller (download script) treats rejected gaps as validation errors.
- is_synthetic=True marks every filled candle so strategies can exclude them
  from signal generation without touching indicator computation.
- Timestamps are computed as prev_ts + k * interval_ms (integer arithmetic via
  pd.Timedelta). This is exact for all Binance intervals including 1w (always
  Monday 00:00 UTC + 7 days = next Monday 00:00 UTC).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
from loguru import logger

# Local interval map — mirrors validator._INTERVAL_MS to avoid cross-module coupling
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

_GAP_TOLERANCE_MS = 500  # float-precision tolerance when comparing timestamps


@dataclass
class FillReport:
    """Summary of what GapFiller did to a DataFrame."""

    total_synthetic_added: int
    synthetic_periods: list[tuple[datetime, datetime, int]]  # (start, end, n_added)
    rejected_gaps: list[tuple[datetime, datetime, int]]  # (start, end, n_needed) — > threshold
    is_clean: bool  # True when all gaps were filled (no rejections)

    def __str__(self) -> str:
        lines = [
            f"FillReport: {self.total_synthetic_added} synthetic candle(s) added, "
            f"{len(self.rejected_gaps)} gap(s) rejected",
        ]
        for gs, ge, n in self.synthetic_periods:
            lines.append(f"  filled : {gs.date()} → {ge.date()} ({n} candle(s))")
        for gs, ge, n in self.rejected_gaps:
            lines.append(f"  REJECTED: {gs.date()} → {ge.date()} ({n} candle(s) needed, above threshold)")
        return "\n".join(lines)


class GapFiller:
    """Fills temporal gaps in OHLCV DataFrames with synthetic forward-fill candles.

    Usage:
        filler = GapFiller()
        df_filled, report = filler.fill_gaps(df, interval="4h")
    """

    def fill_gaps(
        self,
        df: pd.DataFrame,
        interval: str,
        max_gap_hours: int = 48,
    ) -> tuple[pd.DataFrame, FillReport]:
        """Detect and forward-fill gaps smaller than max_gap_hours.

        For each detected gap:
        - If gap duration > max_gap_hours: record as rejected, leave as-is.
        - If gap duration <= max_gap_hours: insert synthetic candles using
          the closing price of the last real candle before the gap.

        Synthetic candle values (zero look-ahead — only prev_close used):
            open = high = low = close = prev_close
            volume = quote_volume = 0.0
            num_trades = 0
            is_synthetic = True
            timestamp_utc = prev_ts + k * interval_ms  (k = 1 .. n_missing)
            close_time_utc = timestamp_utc + interval_ms - 1ms

        Args:
            df: OHLCV DataFrame with timestamp_utc column (UTC-aware).
            interval: Binance interval string, e.g. "4h".
            max_gap_hours: Maximum gap duration to fill. Gaps exceeding this
                           are left unfilled and recorded in FillReport.rejected_gaps.

        Returns:
            (filled_df, FillReport) — filled_df sorted ascending by timestamp_utc.

        Raises:
            ValueError: if interval is not recognised.
        """
        if interval not in _INTERVAL_MS:
            raise ValueError(
                f"Unknown interval '{interval}'. Valid: {sorted(_INTERVAL_MS.keys())}"
            )

        interval_ms = _INTERVAL_MS[interval]
        max_gap_ms = max_gap_hours * 3_600_000

        df_work = df.sort_values("timestamp_utc").reset_index(drop=True)

        # Ensure is_synthetic column exists (backwards compat with pre-schema files)
        if "is_synthetic" not in df_work.columns:
            df_work["is_synthetic"] = False

        synthetic_blocks: list[pd.DataFrame] = []
        synthetic_periods: list[tuple[datetime, datetime, int]] = []
        rejected_gaps: list[tuple[datetime, datetime, int]] = []

        for i in range(1, len(df_work)):
            prev_ts: pd.Timestamp = df_work["timestamp_utc"].iloc[i - 1]
            curr_ts: pd.Timestamp = df_work["timestamp_utc"].iloc[i]
            diff_ms = (curr_ts - prev_ts).total_seconds() * 1000

            if diff_ms <= interval_ms + _GAP_TOLERANCE_MS:
                continue  # no gap

            n_missing = int(round(diff_ms / interval_ms)) - 1
            gap_duration_ms = n_missing * interval_ms

            gap_start = (prev_ts + pd.Timedelta(milliseconds=interval_ms)).to_pydatetime()
            gap_end = (curr_ts - pd.Timedelta(milliseconds=interval_ms)).to_pydatetime()

            if gap_duration_ms > max_gap_ms:
                logger.warning(
                    "Gap of {} candle(s) ({:.1f}h) exceeds {}h threshold — NOT filled: {} → {}",
                    n_missing,
                    gap_duration_ms / 3_600_000,
                    max_gap_hours,
                    gap_start.date(),
                    gap_end.date(),
                )
                rejected_gaps.append((gap_start, gap_end, n_missing))
                continue

            logger.info(
                "Filling gap of {} candle(s) ({:.1f}h) with forward-fill: {} → {}",
                n_missing,
                gap_duration_ms / 3_600_000,
                gap_start.date(),
                gap_end.date(),
            )

            synthetic = self._make_synthetic_candles(
                prev_candle=df_work.iloc[i - 1],
                n_missing=n_missing,
                interval_ms=interval_ms,
            )
            synthetic_blocks.append(synthetic)
            synthetic_periods.append((gap_start, gap_end, n_missing))

        total_added = sum(len(b) for b in synthetic_blocks)

        if synthetic_blocks:
            result = (
                pd.concat([df_work] + synthetic_blocks, ignore_index=True)
                .sort_values("timestamp_utc")
                .reset_index(drop=True)
            )
        else:
            result = df_work

        report = FillReport(
            total_synthetic_added=total_added,
            synthetic_periods=synthetic_periods,
            rejected_gaps=rejected_gaps,
            is_clean=(len(rejected_gaps) == 0),
        )

        if total_added > 0:
            logger.info("GapFiller: {} synthetic candle(s) inserted", total_added)

        return result, report

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _make_synthetic_candles(
        self,
        prev_candle: pd.Series,
        n_missing: int,
        interval_ms: int,
    ) -> pd.DataFrame:
        """Generate n_missing synthetic candles following prev_candle.

        Uses only prev_candle["close"] — never reads forward in the series.
        """
        prev_ts: pd.Timestamp = prev_candle["timestamp_utc"]
        prev_close: float = float(prev_candle["close"])

        rows = []
        for k in range(1, n_missing + 1):
            ts = prev_ts + pd.Timedelta(milliseconds=interval_ms * k)
            rows.append({
                "timestamp_utc":  ts,
                "open":           prev_close,
                "high":           prev_close,
                "low":            prev_close,
                "close":          prev_close,
                "volume":         0.0,
                "close_time_utc": ts + pd.Timedelta(milliseconds=interval_ms - 1),
                "quote_volume":   0.0,
                "num_trades":     0,
                "is_synthetic":   True,
            })

        synthetic = pd.DataFrame(rows)

        # Ensure datetime columns match the caller's dtype (datetime64[ns, UTC])
        for col in ("timestamp_utc", "close_time_utc"):
            synthetic[col] = synthetic[col].dt.as_unit("ns")

        return synthetic
