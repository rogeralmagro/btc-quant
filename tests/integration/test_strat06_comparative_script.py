"""Smoke test: run_strat06_comparative_backtest runs without crash on synthetic data."""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.integration.conftest import make_synthetic_1d_dataset

UTC = timezone.utc

# Load the script as a module without executing __main__
_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "run_strat06_comparative_backtest.py"
_spec = importlib.util.spec_from_file_location("comparative_script", _SCRIPT_PATH)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]


@pytest.fixture()
def synthetic_parquet(tmp_path):
    """Write 120 bars of synthetic 1D data to a parquet file."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pytest.skip("pyarrow not installed")

    start = datetime(2024, 1, 1, tzinfo=UTC)
    df = make_synthetic_1d_dataset(120, start, constant_price=50_000.0)
    path = tmp_path / "btcusdt_1d.parquet"
    df.to_parquet(path, index=False)
    return path


class TestComparativeScriptSmoke:
    def test_script_runs_without_crash(self, synthetic_parquet, tmp_path):
        """Main function completes and produces all expected output files."""
        output_dir = tmp_path / "report"
        _mod.main(synthetic_parquet, output_dir)

        assert output_dir.exists()
        expected_files = [
            "equity_curve_bah.csv",
            "equity_curve_dca.csv",
            "equity_curve_strat06.csv",
            "trades_bah.csv",
            "trades_dca.csv",
            "trades_strat06.csv",
            "comparative_summary.json",
        ]
        for fname in expected_files:
            assert (output_dir / fname).exists(), f"Missing output: {fname}"

    def test_summary_json_structure(self, synthetic_parquet, tmp_path):
        """comparative_summary.json has the expected top-level keys."""
        import json

        output_dir = tmp_path / "report2"
        _mod.main(synthetic_parquet, output_dir)

        with open(output_dir / "comparative_summary.json") as f:
            summary = json.load(f)

        assert "backtest_period" in summary
        assert "strategies" in summary
        assert len(summary["strategies"]) == 3
        labels = {s["strategy"] for s in summary["strategies"]}
        assert labels == {"buy_and_hold", "dca_benchmark", "strat06"}

    def test_no_negative_btc_or_nan_equity(self, synthetic_parquet, tmp_path):
        """Sanity: no NaN equity values, no negative BTC in any strategy."""
        import csv
        import math

        output_dir = tmp_path / "report3"
        _mod.main(synthetic_parquet, output_dir)

        for fname in ("equity_curve_bah.csv", "equity_curve_dca.csv", "equity_curve_strat06.csv"):
            with open(output_dir / fname) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    val = float(row["total_value_eur"])
                    assert not math.isnan(val), f"NaN in {fname}"
                    assert val >= 0, f"Negative equity in {fname}: {val}"
