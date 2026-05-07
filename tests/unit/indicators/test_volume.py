"""Unit tests for volume indicators: OBV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from btc_quant.indicators.volume import obv


class TestOBV:
    def test_obv_basic(self) -> None:
        """Manual step-by-step OBV verification.

        Dataset:
            close  = [10, 12, 11, 13, 13, 9]
            volume = [ 1,  2,  3,  4,  5, 6]

        Step by step:
            OBV[0] = 0             (seed, no prior close)
            OBV[1] = 0 + 2 =  2   (close 12 > 10, add)
            OBV[2] = 2 - 3 = -1   (close 11 < 12, subtract)
            OBV[3] = -1 + 4 = 3   (close 13 > 11, add)
            OBV[4] = 3             (close 13 == 13, unchanged)
            OBV[5] = 3 - 6 = -3   (close 9 < 13, subtract)
        """
        c = pd.Series([10.0, 12, 11, 13, 13, 9])
        v = pd.Series([1.0, 2, 3, 4, 5, 6])
        result = obv(c, v)
        expected = [0.0, 2.0, -1.0, 3.0, 3.0, -3.0]
        np.testing.assert_array_almost_equal(result.values, expected)

    def test_obv_first_value_is_zero(self) -> None:
        """OBV[0] is always 0 regardless of volume."""
        c = pd.Series([100.0, 101.0])
        v = pd.Series([999.0, 1.0])
        assert obv(c, v).iloc[0] == 0.0

    def test_obv_constant_close_unchanged(self) -> None:
        """Flat price → OBV stays at 0 throughout."""
        c = pd.Series([50.0] * 10)
        v = pd.Series(range(1, 11), dtype=float)
        result = obv(c, v)
        assert (result == 0.0).all()

    def test_obv_only_up_increases(self) -> None:
        """Strictly rising price → OBV is strictly monotonically increasing."""
        c = pd.Series(np.arange(1.0, 11.0))
        v = pd.Series([1.0] * 10)
        result = obv(c, v)
        # OBV[0]=0, OBV[1..9]=1,2,...,9
        assert (result.diff().dropna() > 0).all()

    def test_obv_only_down_decreases(self) -> None:
        """Strictly falling price → OBV is strictly monotonically decreasing."""
        c = pd.Series(np.arange(10.0, 0.0, -1.0))
        v = pd.Series([1.0] * 10)
        result = obv(c, v)
        assert (result.diff().dropna() < 0).all()

    def test_obv_preserves_index(self) -> None:
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        c = pd.Series([1.0, 2, 3, 4, 5], index=idx)
        v = pd.Series([1.0] * 5, index=idx)
        assert list(obv(c, v).index) == list(idx)

    def test_obv_output_dtype(self) -> None:
        c = pd.Series([1.0, 2.0, 3.0])
        v = pd.Series([1.0, 1.0, 1.0])
        assert obv(c, v).dtype == np.float64

    def test_obv_negative_volume_raises(self) -> None:
        c = pd.Series([10.0, 11.0, 12.0])
        v = pd.Series([1.0, -1.0, 1.0])
        with pytest.raises(ValueError, match="volume"):
            obv(c, v)

    def test_obv_mismatched_indices_raises(self) -> None:
        c = pd.Series([1.0, 2.0], index=pd.RangeIndex(2))
        v = pd.Series([1.0, 1.0], index=pd.RangeIndex(1, 3))
        with pytest.raises(ValueError, match="index"):
            obv(c, v)

    def test_obv_empty_series_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            obv(pd.Series([], dtype=float), pd.Series([], dtype=float))
