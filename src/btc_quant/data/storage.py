"""ParquetStorage: persists OHLCV DataFrames to and from Parquet files.

Architectural decisions:
- One file per (symbol, interval) pair; no partitioning. At our volumes
  (< 100k rows per file) this is faster and simpler than partitioned datasets,
  and makes incremental updates trivial: load → append → save.
- snappy compression: best balance of read speed and file size for OHLCV workloads.
- pyarrow engine: preserves datetime64[ns, UTC] dtype faithfully through roundtrip.
- Files are always sorted by timestamp_utc ascending on write so readers can
  assume sorted order without an extra sort pass.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from loguru import logger


_COMPRESSION = "snappy"
_ENGINE = "pyarrow"


class ParquetStorage:
    """Reads and writes OHLCV DataFrames as Parquet files.

    File naming convention: {symbol_lower}_{interval}.parquet
    Example: data/processed/btcusdt_1d.parquet
    """

    def get_path(
        self,
        symbol: str,
        interval: str,
        base_path: str | Path = "data/processed",
    ) -> Path:
        """Return the canonical Parquet path for a symbol/interval pair."""
        return Path(base_path) / f"{symbol.lower()}_{interval}.parquet"

    def exists(
        self,
        symbol: str,
        interval: str,
        base_path: str | Path = "data/processed",
    ) -> bool:
        """Return True if a Parquet file already exists for this symbol/interval."""
        return self.get_path(symbol, interval, base_path).exists()

    def save(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: str,
        base_path: str | Path = "data/processed",
    ) -> Path:
        """Write df to Parquet sorted by timestamp_utc ascending.

        Overwrites any existing file for this symbol/interval (idempotent writes).
        Parent directory is created automatically if it does not exist.

        Args:
            df: DataFrame to persist. Must contain a timestamp_utc column.
            symbol: Trading pair, e.g. "BTCUSDT".
            interval: Binance interval string, e.g. "1d".
            base_path: Directory to write into. Created if absent.

        Returns:
            Path of the written file.
        """
        path = self.get_path(symbol, interval, base_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df_out = df.sort_values("timestamp_utc").reset_index(drop=True)

        # Ensure ns precision before writing so roundtrips are lossless
        for col in ("timestamp_utc", "close_time_utc"):
            if col in df_out.columns:
                df_out[col] = df_out[col].dt.as_unit("ns")

        df_out.to_parquet(path, engine=_ENGINE, compression=_COMPRESSION, index=False)
        logger.info("Saved {} rows → {} ({})", len(df_out), path, _COMPRESSION)
        return path

    def load(
        self,
        symbol: str,
        interval: str,
        base_path: str | Path = "data/processed",
    ) -> pd.DataFrame:
        """Load a Parquet file and return a DataFrame with UTC-aware timestamps.

        Args:
            symbol: Trading pair, e.g. "BTCUSDT".
            interval: Binance interval string, e.g. "1d".
            base_path: Directory to read from.

        Returns:
            DataFrame sorted ascending by timestamp_utc with UTC-aware timestamp columns.

        Raises:
            FileNotFoundError: if no file exists for this symbol/interval.
        """
        path = self.get_path(symbol, interval, base_path)
        if not path.exists():
            raise FileNotFoundError(
                f"No data file found for {symbol} {interval} at {path}"
            )

        df = pd.read_parquet(path, engine=_ENGINE)

        for col in ("timestamp_utc", "close_time_utc"):
            if col not in df.columns:
                continue
            s = df[col]
            if s.dt.tz is None:
                df[col] = s.dt.tz_localize("UTC").dt.as_unit("ns")
            elif str(s.dt.tz) != "UTC":
                df[col] = s.dt.tz_convert("UTC").dt.as_unit("ns")
            else:
                df[col] = s.dt.as_unit("ns")

        logger.info("Loaded {} rows ← {}", len(df), path)
        return df
