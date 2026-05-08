"""Integration test: BacktestEngine × BuyAndHoldStrategy × real BTC data.

Cross-validation: BAH final BTC value must match the manually computed
price-ratio outcome within 0.5%. Any deviation indicates an engine bug
(look-ahead, wrong fee/slippage math, CapitalPool accounting error, etc.).

If this test fails: STOP and debug. Do not widen the tolerance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulator
from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.strategies.buy_and_hold import BuyAndHoldStrategy

DATA_DIR = Path(__file__).parents[2] / "data" / "processed"
DAILY_PARQUET = DATA_DIR / "btcusdt_1d.parquet"

FEE = 0.001
SLIP = 0.0005
INITIAL_CAPITAL = 10_000.0
TAG = StrategyTag.BUY_AND_HOLD_BENCHMARK


@pytest.fixture(scope="module")
def df_1d() -> pd.DataFrame:
    """Real daily BTC data, first 500 bars (2017-08 to ~2019-01)."""
    df = pd.read_parquet(DAILY_PARQUET)
    # Slice a deterministic window; avoid the very first bar (tiny volume) by
    # using bars 1..500 so the engine has enough history.
    return df.iloc[1:501].reset_index(drop=True)


class TestBuyAndHoldValidation:
    def test_bah_no_closed_trades(self, df_1d: pd.DataFrame) -> None:
        """BAH never sells — closed_trades must be empty."""
        engine = BacktestEngine(
            strategies=[BuyAndHoldStrategy()],
            data={"1d": df_1d},
            initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
            execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
            profit_recycle_pct=0.0,
        )
        result = engine.run()
        assert result.closed_trades == []

    def test_bah_one_open_position(self, df_1d: pd.DataFrame) -> None:
        """BAH must hold exactly one open position at the end."""
        engine = BacktestEngine(
            strategies=[BuyAndHoldStrategy()],
            data={"1d": df_1d},
            initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
            execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
            profit_recycle_pct=0.0,
        )
        result = engine.run()
        assert len(result.final_portfolio.open_positions) == 1

    def test_bah_cash_nearly_exhausted(self, df_1d: pd.DataFrame) -> None:
        """After buying, cash must be < 1% of initial capital."""
        engine = BacktestEngine(
            strategies=[BuyAndHoldStrategy()],
            data={"1d": df_1d},
            initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
            execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
            profit_recycle_pct=0.0,
        )
        result = engine.run()
        pool = result.final_portfolio.get_pool(TAG)
        assert pool.cash_eur < INITIAL_CAPITAL * 0.01

    def test_bah_btc_held_is_positive(self, df_1d: pd.DataFrame) -> None:
        engine = BacktestEngine(
            strategies=[BuyAndHoldStrategy()],
            data={"1d": df_1d},
            initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
            execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
            profit_recycle_pct=0.0,
        )
        result = engine.run()
        pool = result.final_portfolio.get_pool(TAG)
        assert pool.btc_held > 0.0

    def test_bah_final_value_matches_price_ratio(self, df_1d: pd.DataFrame) -> None:
        """Core cross-validation.

        The engine's final BTC value must match the manually computed
        expected value (same formula as the engine uses internally) within 0.5%.

        Manual calculation:
          exec_price = bar[1].open * (1 + SLIP)
          qty_btc    = INITIAL_CAPITAL / (exec_price * (1 + FEE))
          final_val  = qty_btc * last_bar.close

        The 0.5% tolerance absorbs floating-point rounding only.
        Any larger deviation indicates an engine accounting bug.
        """
        engine = BacktestEngine(
            strategies=[BuyAndHoldStrategy()],
            data={"1d": df_1d},
            initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
            execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
            profit_recycle_pct=0.0,
        )
        result = engine.run()
        pool = result.final_portfolio.get_pool(TAG)

        # Signal fires at bar[0].close → executes at bar[1].open
        exec_price = df_1d.iloc[1]["open"] * (1.0 + SLIP)
        expected_qty = INITIAL_CAPITAL / (exec_price * (1.0 + FEE))
        last_close = float(df_1d.iloc[-1]["close"])
        expected_btc_value = expected_qty * last_close

        actual_btc_value = pool.btc_held * last_close
        diff_pct = abs(actual_btc_value - expected_btc_value) / expected_btc_value
        print(
    f"\nBAH final value diff: {diff_pct:.8%} | "
    f"expected={expected_btc_value:.6f} | "
    f"actual={actual_btc_value:.6f} | "
    f"qty_expected={expected_qty:.10f} | "
    f"qty_actual={pool.btc_held:.10f}"
)
        assert diff_pct < 0.005, (
            f"Engine BTC value deviates {diff_pct:.4%} from expected. "
            f"Expected: {expected_btc_value:.2f} EUR, "
            f"Actual: {actual_btc_value:.2f} EUR. "
            "Check engine accounting for buy/fee/slippage bugs."
        )

    def test_equity_curve_length_matches_data(self, df_1d: pd.DataFrame) -> None:
        """Equity curve must have one snapshot per bar."""
        engine = BacktestEngine(
            strategies=[BuyAndHoldStrategy()],
            data={"1d": df_1d},
            initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
            execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
            profit_recycle_pct=0.0,
        )
        result = engine.run()
        assert len(result.equity_curve) == len(df_1d)
