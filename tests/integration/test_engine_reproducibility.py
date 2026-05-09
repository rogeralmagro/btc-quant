"""Integration test: validate engine determinism and reproducibility.

Running the engine twice on identical inputs must produce bit-exact results.
Any non-determinism indicates a bug: hidden global state, unordered dict
iteration, or PRNG contamination.

If either test fails: STOP and debug. The engine is not deterministic.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pandas as pd
import pytest

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulator
from btc_quant.backtester.metrics.calculator import MetricsCalculator
from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.strategies.buy_and_hold import BuyAndHoldStrategy

DATA_DIR = Path(__file__).parents[2] / "data" / "processed"
DAILY_PARQUET = DATA_DIR / "btcusdt_1d.parquet"

FEE = 0.001
SLIP = 0.0005
INITIAL_CAPITAL = 10_000.0
TAG = StrategyTag.BUY_AND_HOLD_BENCHMARK


@pytest.fixture(scope="module")
def df_500() -> pd.DataFrame:
    """500 daily bars starting from bar index 1."""
    df = pd.read_parquet(DAILY_PARQUET)
    return df.iloc[1:501].reset_index(drop=True)


def _bah_engine(df: pd.DataFrame) -> BacktestEngine:
    return BacktestEngine(
        strategies=[BuyAndHoldStrategy()],
        data={"1d": df},
        initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
        execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
        profit_recycle_pct=0.0,
    )


class TestReproducibility:
    def test_same_inputs_produce_bit_exact_outputs(self, df_500: pd.DataFrame) -> None:
        """Two independent engine runs on the same data must produce identical equity curves.

        Checks every snapshot's total_value_eur for bit-exact equality (no
        floating-point relaxation). Any divergence means non-determinism.
        """
        result_a = _bah_engine(df_500).run()
        result_b = _bah_engine(df_500).run()

        assert len(result_a.equity_curve) == len(result_b.equity_curve), (
            "Equity curves have different lengths"
        )
        for i, (snap_a, snap_b) in enumerate(
            zip(result_a.equity_curve, result_b.equity_curve)
        ):
            assert snap_a.total_value_eur == snap_b.total_value_eur, (
                f"bar {i}: equity_curve diverged between runs "
                f"({snap_a.total_value_eur} vs {snap_b.total_value_eur})"
            )

    def test_metrics_reproducibility(self, df_500: pd.DataFrame) -> None:
        """Two MetricsCalculator.calculate() calls on identical results must agree.

        Checks every numeric field of MetricsReport for bit-exact equality.
        Dict fields (metrics_by_strategy) are checked by their per-strategy
        total_pnl_eur and num_trades.
        """
        result_a = _bah_engine(df_500).run()
        result_b = _bah_engine(df_500).run()

        calc = MetricsCalculator()
        report_a = calc.calculate(result_a)
        report_b = calc.calculate(result_b)

        # Check every scalar field — both numeric and None — for equality.
        for f in fields(report_a):
            val_a = getattr(report_a, f.name)
            val_b = getattr(report_b, f.name)
            if isinstance(val_a, dict):
                continue  # checked separately below
            assert val_a == val_b, (
                f"MetricsReport.{f.name} differs between runs: {val_a!r} vs {val_b!r}"
            )

        # Check per-strategy breakdown
        assert set(report_a.metrics_by_strategy) == set(report_b.metrics_by_strategy)
        for tag in report_a.metrics_by_strategy:
            sm_a = report_a.metrics_by_strategy[tag]
            sm_b = report_b.metrics_by_strategy[tag]
            assert sm_a.num_trades == sm_b.num_trades
            assert sm_a.total_pnl_eur == sm_b.total_pnl_eur
