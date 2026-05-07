"""Unit tests for data quality utilities."""

from __future__ import annotations

import pandas as pd
import pytest

from btc_quant.data.quality import is_period_clean


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df(n: int, synthetic_indices: list[int] | None = None) -> pd.DataFrame:
    """Simple DataFrame with is_synthetic column for testing."""
    flags = [False] * n
    for i in (synthetic_indices or []):
        flags[i] = True
    return pd.DataFrame({"is_synthetic": flags})


# ---------------------------------------------------------------------------
# Clean period
# ---------------------------------------------------------------------------

class TestCleanPeriod:
    def test_clean_period_returns_true(self) -> None:
        """Window with no synthetic candles → True."""
        df = _make_df(20)
        assert is_period_clean(df, end_idx=19, lookback=20) is True

    def test_all_real_candles(self) -> None:
        df = _make_df(10)
        for i in range(10):
            assert is_period_clean(df, end_idx=i, lookback=5) is True


# ---------------------------------------------------------------------------
# Dirty period
# ---------------------------------------------------------------------------

class TestDirtyPeriod:
    def test_dirty_period_returns_false(self) -> None:
        """Window containing a synthetic candle → False."""
        df = _make_df(20, synthetic_indices=[10])
        assert is_period_clean(df, end_idx=19, lookback=20) is False

    def test_synthetic_at_end_idx_returns_false(self) -> None:
        """Synthetic candle at end_idx itself → False."""
        df = _make_df(10, synthetic_indices=[9])
        assert is_period_clean(df, end_idx=9, lookback=5) is False

    def test_synthetic_at_start_of_window_returns_false(self) -> None:
        """Synthetic candle at the first position of the window → False."""
        df = _make_df(20, synthetic_indices=[5])
        # lookback=10, end_idx=14 → window [5..14] → synthetic at 5
        assert is_period_clean(df, end_idx=14, lookback=10) is False


# ---------------------------------------------------------------------------
# Lookback boundary
# ---------------------------------------------------------------------------

class TestLookbackBoundary:
    def test_synthetic_outside_window_does_not_affect_result(self) -> None:
        """Synthetic candle outside the lookback window is ignored."""
        df = _make_df(20, synthetic_indices=[0])  # synthetic at index 0
        # lookback=10, end_idx=19 → window [10..19] → synthetic at 0 is outside
        assert is_period_clean(df, end_idx=19, lookback=10) is True

    def test_lookback_larger_than_df_uses_all_rows(self) -> None:
        """When lookback > len(df), window starts at 0 without error."""
        df = _make_df(5)
        assert is_period_clean(df, end_idx=4, lookback=100) is True

    def test_lookback_of_one_checks_only_end_idx(self) -> None:
        """lookback=1 checks only the candle at end_idx."""
        df = _make_df(10, synthetic_indices=[5])
        assert is_period_clean(df, end_idx=4, lookback=1) is True   # real
        assert is_period_clean(df, end_idx=5, lookback=1) is False  # synthetic

    def test_exact_boundary_included(self) -> None:
        """Window is inclusive on both ends."""
        df = _make_df(10, synthetic_indices=[3])
        # lookback=5, end_idx=7 → window [3..7] → synthetic at 3 is included
        assert is_period_clean(df, end_idx=7, lookback=5) is False
        # lookback=4, end_idx=7 → window [4..7] → synthetic at 3 is excluded
        assert is_period_clean(df, end_idx=7, lookback=4) is True


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrors:
    def test_missing_column_raises_key_error(self) -> None:
        df = pd.DataFrame({"close": [1.0, 2.0]})
        with pytest.raises(KeyError, match="is_synthetic"):
            is_period_clean(df, end_idx=1)

    def test_out_of_bounds_end_idx_raises(self) -> None:
        df = _make_df(5)
        with pytest.raises(IndexError, match="out of bounds"):
            is_period_clean(df, end_idx=10)

    def test_negative_end_idx_raises(self) -> None:
        df = _make_df(5)
        with pytest.raises(IndexError, match="out of bounds"):
            is_period_clean(df, end_idx=-1)
