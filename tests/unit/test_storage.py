"""Unit tests for ParquetStorage.

Uses pytest's tmp_path fixture so no test writes to the real data/ directory.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from btc_quant.data.storage import ParquetStorage


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

INTERVAL_MS = 86_400_000
START = datetime(2017, 8, 17, tzinfo=timezone.utc)


def _make_df(n: int = 5, start: datetime = START) -> pd.DataFrame:
    timestamps = pd.date_range(
        start=start,
        periods=n,
        freq=pd.Timedelta(milliseconds=INTERVAL_MS),
        tz="UTC",
    ).as_unit("ns")

    return pd.DataFrame({
        "timestamp_utc":  timestamps,
        "open":           50_000.0,
        "high":           51_000.0,
        "low":            49_000.0,
        "close":          50_500.0,
        "volume":         100.0,
        "close_time_utc": timestamps + pd.Timedelta(milliseconds=INTERVAL_MS - 1),
        "quote_volume":   5_050_000.0,
        "num_trades":     1000,
        "is_synthetic":   False,
    })


# ---------------------------------------------------------------------------
# get_path
# ---------------------------------------------------------------------------

class TestGetPath:
    def test_naming_convention(self) -> None:
        path = ParquetStorage().get_path("BTCUSDT", "1d", base_path="/data/processed")
        assert path.name == "btcusdt_1d.parquet"

    def test_symbol_is_lowercased(self) -> None:
        path = ParquetStorage().get_path("BTCUSDT", "4h", base_path="/data/processed")
        assert "BTCUSDT" not in path.name
        assert "btcusdt" in path.name

    def test_returns_path_object(self) -> None:
        assert isinstance(ParquetStorage().get_path("BTCUSDT", "1d"), Path)

    def test_base_path_is_parent(self, tmp_path: Path) -> None:
        path = ParquetStorage().get_path("BTCUSDT", "1d", base_path=tmp_path)
        assert path.parent == tmp_path


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------

class TestExists:
    def test_returns_false_when_no_file(self, tmp_path: Path) -> None:
        assert ParquetStorage().exists("BTCUSDT", "1d", base_path=tmp_path) is False

    def test_returns_true_after_save(self, tmp_path: Path) -> None:
        storage = ParquetStorage()
        storage.save(_make_df(), "BTCUSDT", "1d", base_path=tmp_path)

        assert storage.exists("BTCUSDT", "1d", base_path=tmp_path) is True

    def test_different_intervals_are_independent(self, tmp_path: Path) -> None:
        storage = ParquetStorage()
        storage.save(_make_df(), "BTCUSDT", "1d", base_path=tmp_path)

        assert storage.exists("BTCUSDT", "4h", base_path=tmp_path) is False


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

class TestSave:
    def test_save_returns_path(self, tmp_path: Path) -> None:
        result = ParquetStorage().save(_make_df(), "BTCUSDT", "1d", base_path=tmp_path)
        assert isinstance(result, Path)

    def test_save_creates_file(self, tmp_path: Path) -> None:
        path = ParquetStorage().save(_make_df(), "BTCUSDT", "1d", base_path=tmp_path)
        assert path.exists()

    def test_save_creates_parent_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        ParquetStorage().save(_make_df(), "BTCUSDT", "1d", base_path=nested)
        assert nested.exists()

    def test_save_sorts_unsorted_input(self, tmp_path: Path) -> None:
        df = _make_df(5)
        df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

        storage = ParquetStorage()
        storage.save(df_shuffled, "BTCUSDT", "1d", base_path=tmp_path)
        loaded = storage.load("BTCUSDT", "1d", base_path=tmp_path)

        assert loaded["timestamp_utc"].is_monotonic_increasing

    def test_save_overwrites_existing(self, tmp_path: Path) -> None:
        storage = ParquetStorage()
        df_v1 = _make_df(3)
        df_v2 = _make_df(7)

        storage.save(df_v1, "BTCUSDT", "1d", base_path=tmp_path)
        storage.save(df_v2, "BTCUSDT", "1d", base_path=tmp_path)

        loaded = storage.load("BTCUSDT", "1d", base_path=tmp_path)
        assert len(loaded) == 7


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

class TestLoad:
    def test_load_raises_when_file_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="BTCUSDT"):
            ParquetStorage().load("BTCUSDT", "1d", base_path=tmp_path)

    def test_load_timestamp_utc_is_utc_aware(self, tmp_path: Path) -> None:
        storage = ParquetStorage()
        storage.save(_make_df(), "BTCUSDT", "1d", base_path=tmp_path)
        df = storage.load("BTCUSDT", "1d", base_path=tmp_path)

        assert str(df["timestamp_utc"].dt.tz) == "UTC"

    def test_load_close_time_utc_is_utc_aware(self, tmp_path: Path) -> None:
        storage = ParquetStorage()
        storage.save(_make_df(), "BTCUSDT", "1d", base_path=tmp_path)
        df = storage.load("BTCUSDT", "1d", base_path=tmp_path)

        assert str(df["close_time_utc"].dt.tz) == "UTC"

    def test_load_result_is_sorted(self, tmp_path: Path) -> None:
        storage = ParquetStorage()
        df_shuffled = _make_df(10).sample(frac=1, random_state=0).reset_index(drop=True)
        storage.save(df_shuffled, "BTCUSDT", "1d", base_path=tmp_path)

        loaded = storage.load("BTCUSDT", "1d", base_path=tmp_path)

        assert loaded["timestamp_utc"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# Roundtrip
# ---------------------------------------------------------------------------

class TestRoundtrip:
    def test_roundtrip_preserves_values(self, tmp_path: Path) -> None:
        df = _make_df(10)
        storage = ParquetStorage()
        storage.save(df, "BTCUSDT", "1d", base_path=tmp_path)
        loaded = storage.load("BTCUSDT", "1d", base_path=tmp_path)

        pd.testing.assert_frame_equal(df, loaded, check_like=False)

    def test_roundtrip_preserves_float_dtypes(self, tmp_path: Path) -> None:
        df = _make_df(5)
        storage = ParquetStorage()
        storage.save(df, "BTCUSDT", "1d", base_path=tmp_path)
        loaded = storage.load("BTCUSDT", "1d", base_path=tmp_path)

        for col in ("open", "high", "low", "close", "volume", "quote_volume"):
            assert loaded[col].dtype == "float64", f"{col} must remain float64"

    def test_roundtrip_preserves_int_dtypes(self, tmp_path: Path) -> None:
        df = _make_df(5)
        storage = ParquetStorage()
        storage.save(df, "BTCUSDT", "1d", base_path=tmp_path)
        loaded = storage.load("BTCUSDT", "1d", base_path=tmp_path)

        assert loaded["num_trades"].dtype == "int64"

    def test_roundtrip_preserves_bool_is_synthetic(self, tmp_path: Path) -> None:
        df = _make_df(5)
        storage = ParquetStorage()
        storage.save(df, "BTCUSDT", "1d", base_path=tmp_path)
        loaded = storage.load("BTCUSDT", "1d", base_path=tmp_path)

        assert loaded["is_synthetic"].dtype == bool
        assert (loaded["is_synthetic"] == False).all()  # noqa: E712

    def test_roundtrip_preserves_timestamp_precision(self, tmp_path: Path) -> None:
        df = _make_df(5)
        storage = ParquetStorage()
        storage.save(df, "BTCUSDT", "1d", base_path=tmp_path)
        loaded = storage.load("BTCUSDT", "1d", base_path=tmp_path)

        assert str(loaded["timestamp_utc"].dtype) == "datetime64[ns, UTC]"

    def test_roundtrip_different_intervals_do_not_collide(self, tmp_path: Path) -> None:
        df_1d = _make_df(5)
        df_4h = _make_df(3)
        storage = ParquetStorage()

        storage.save(df_1d, "BTCUSDT", "1d", base_path=tmp_path)
        storage.save(df_4h, "BTCUSDT", "4h", base_path=tmp_path)

        assert len(storage.load("BTCUSDT", "1d", base_path=tmp_path)) == 5
        assert len(storage.load("BTCUSDT", "4h", base_path=tmp_path)) == 3
