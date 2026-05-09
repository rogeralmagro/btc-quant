"""Integration test: verify the BacktestEngine has no look-ahead bias.

If either test fails: STOP and debug. The engine is reading future data.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulator
from btc_quant.backtester.market_context import MarketContext
from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.backtester.models.signal import Signal
from btc_quant.backtester.strategy_base import StrategyBase
from btc_quant.strategies.buy_and_hold import BuyAndHoldStrategy

DATA_DIR = Path(__file__).parents[2] / "data" / "processed"
DAILY_PARQUET = DATA_DIR / "btcusdt_1d.parquet"

FEE = 0.001
SLIP = 0.0005
TAG = StrategyTag.BUY_AND_HOLD_BENCHMARK


class RecordingStrategy(StrategyBase):
    """Records the max timestamp_utc visible at each bar; never generates signals."""

    def __init__(self) -> None:
        super().__init__(strategy_tag=TAG, config={})
        self.max_ts_per_bar: list = []

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context: MarketContext, portfolio: Portfolio) -> list[Signal]:
        hist = context.get_history("1d", lookback=10_000)
        if hist.empty:
            self.max_ts_per_bar.append(None)
        else:
            self.max_ts_per_bar.append(hist["timestamp_utc"].max())
        return []


@pytest.fixture(scope="module")
def df_200() -> pd.DataFrame:
    """200 daily bars starting from bar index 1."""
    df = pd.read_parquet(DAILY_PARQUET)
    return df.iloc[1:201].reset_index(drop=True)


class TestNoLookAhead:
    def test_strategy_only_sees_past_data(self, df_200: pd.DataFrame) -> None:
        """Max timestamp_utc visible to the strategy at bar i must be < bar i+1's open.

        The engine advances the context to bar_i.close_time_utc before calling
        generate_signals. Therefore the latest bar visible has open time ==
        bar_i.timestamp_utc, which must be strictly before bar_{i+1}.timestamp_utc.
        """
        strategy = RecordingStrategy()
        engine = BacktestEngine(
            strategies=[strategy],
            data={"1d": df_200},
            initial_capital_per_strategy={TAG: 10_000.0},
            execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
            profit_recycle_pct=0.0,
        )
        engine.run()

        recorded = strategy.max_ts_per_bar
        assert len(recorded) == len(df_200)

        for i in range(len(df_200) - 1):
            max_ts = recorded[i]
            assert max_ts is not None, f"bar {i}: no history visible (should have >= 1 bar)"
            next_open = df_200.iloc[i + 1]["timestamp_utc"]
            assert max_ts < next_open, (
                f"bar {i}: strategy saw future data — "
                f"max visible timestamp {max_ts} >= next bar open {next_open}"
            )

    def test_modifying_future_data_does_not_change_past_results(
        self, df_200: pd.DataFrame
    ) -> None:
        """Replacing prices after cut_idx with 10× values must not affect equity_curve[:cut+1].

        A look-ahead engine (e.g. one that reads ahead or caches global stats)
        would produce different past equity values when future prices change.
        """
        cut_idx = 100

        df_original = df_200.copy()
        df_modified = df_200.copy()
        ohlcv = ["open", "high", "low", "close", "volume"]
        df_modified.loc[cut_idx + 1:, ohlcv] = df_modified.loc[cut_idx + 1:, ohlcv] * 10

        def _bah_engine(df: pd.DataFrame) -> BacktestEngine:
            return BacktestEngine(
                strategies=[BuyAndHoldStrategy()],
                data={"1d": df},
                initial_capital_per_strategy={TAG: 10_000.0},
                execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
                profit_recycle_pct=0.0,
            )

        result_orig = _bah_engine(df_original).run()
        result_mod = _bah_engine(df_modified).run()

        for i in range(cut_idx + 1):
            orig_val = result_orig.equity_curve[i].total_value_eur
            mod_val = result_mod.equity_curve[i].total_value_eur
            assert orig_val == mod_val, (
                f"bar {i}: equity diverged after future-data modification "
                f"({orig_val} vs {mod_val}) — engine may have look-ahead bias"
            )
