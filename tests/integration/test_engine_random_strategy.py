"""Integration test: random strategy should produce zero-Sharpe baseline.

A random strategy without any predictive edge has zero expected Sharpe.
Across a large ensemble of seeds the mean must sit near zero, and fees
must drag returns well below the buy-and-hold benchmark.

If either test fails: STOP and debug. The engine or metrics path is wrong.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulator
from btc_quant.backtester.market_context import MarketContext
from btc_quant.backtester.metrics.calculator import MetricsCalculator
from btc_quant.backtester.models.enums import OrderSide, StrategyTag
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.backtester.models.signal import Signal
from btc_quant.backtester.strategy_base import StrategyBase
from btc_quant.strategies.buy_and_hold import BuyAndHoldStrategy

DATA_DIR = Path(__file__).parents[2] / "data" / "processed"
DAILY_PARQUET = DATA_DIR / "btcusdt_1d.parquet"

FEE = 0.001
SLIP = 0.0005
INITIAL_CAPITAL = 10_000.0
TAG = StrategyTag.BUY_AND_HOLD_BENCHMARK

SEEDS = [42, 137, 256, 1024, 2048, 9999, 31415, 12345, 98765, 55555]


class RandomBuySellStrategy(StrategyBase):
    """Buys or sells randomly (p=0.5 each direction) using stdlib random.Random.

    State is seeded at construction; the global random state is never touched,
    ensuring test isolation between different seeds.
    """

    def __init__(self, seed: int) -> None:
        super().__init__(strategy_tag=TAG, config={})
        self._rng = random.Random(seed)

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context: MarketContext, portfolio: Portfolio) -> list[Signal]:
        pool = portfolio.get_pool(self.strategy_tag)

        # Find open position for this strategy (if any)
        existing_pos = None
        for pos in portfolio.open_positions.values():
            if pos.strategy_tag == self.strategy_tag:
                existing_pos = pos
                break

        if existing_pos is None and pool.cash_eur > 0:
            if self._rng.random() < 0.5:
                # Reserve a tiny buffer so FP rounding in qty→notional→fees
                # never causes pool.remove_cash to see amount > cash_eur.
                return [Signal(
                    strategy_tag=self.strategy_tag,
                    timestamp_utc=context.current_time,
                    side=OrderSide.BUY,
                    symbol=context.symbol,
                    quote_amount_eur=pool.cash_eur * 0.9999,
                )]
        elif existing_pos is not None:
            if self._rng.random() < 0.5:
                return [Signal(
                    strategy_tag=self.strategy_tag,
                    timestamp_utc=context.current_time,
                    side=OrderSide.SELL,
                    symbol=context.symbol,
                    quantity_btc=existing_pos.quantity_btc,
                )]

        return []


@pytest.fixture(scope="module")
def df_500() -> pd.DataFrame:
    """500 daily bars starting from bar index 1 (~2017-08 to ~2018-12)."""
    df = pd.read_parquet(DAILY_PARQUET)
    return df.iloc[1:501].reset_index(drop=True)


@pytest.fixture(scope="module")
def df_1000() -> pd.DataFrame:
    """1000 daily bars starting from bar index 1 (~2017-08 to ~2020-05).

    BTC rose from ~$4300 to ~$8800 over this window, giving BAH a clearly
    positive return against which fee drag can be measured.
    """
    df = pd.read_parquet(DAILY_PARQUET)
    return df.iloc[1:1001].reset_index(drop=True)


def _run_random(df: pd.DataFrame, seed: int):
    """Run one random strategy and return its MetricsReport."""
    engine = BacktestEngine(
        strategies=[RandomBuySellStrategy(seed)],
        data={"1d": df},
        initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
        execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
        profit_recycle_pct=0.0,
    )
    result = engine.run()
    return MetricsCalculator().calculate(result)


def _run_bah(df: pd.DataFrame):
    """Run buy-and-hold and return its MetricsReport."""
    engine = BacktestEngine(
        strategies=[BuyAndHoldStrategy()],
        data={"1d": df},
        initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
        execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
        profit_recycle_pct=0.0,
    )
    result = engine.run()
    return MetricsCalculator().calculate(result)


class TestRandomStrategy:
    def test_random_strategy_sharpe_near_zero(self, df_500: pd.DataFrame) -> None:
        """Mean Sharpe across 10 seeds must be in [-0.7, 0.7]; each individual < 2.5.

        A random strategy has zero alpha so its expected Sharpe is 0. Fees push
        it slightly negative. A mean outside [-0.7, 0.7] would indicate the
        engine is systematically rewarding (or penalising) random decisions.
        """
        sharpes = []
        for seed in SEEDS:
            report = _run_random(df_500, seed)
            if report.sharpe_ratio is not None:
                sharpes.append(report.sharpe_ratio)

        assert len(sharpes) >= 8, "Too many seeds produced undefined Sharpe"

        mean_sharpe = float(np.mean(sharpes))
        assert -0.7 <= mean_sharpe <= 0.7, (
            f"Mean Sharpe {mean_sharpe:.3f} is outside [-0.7, 0.7] — "
            "random strategy has unexpected systematic edge"
        )
        for s in sharpes:
            assert s < 2.5, (
                f"Individual Sharpe {s:.3f} >= 2.5 — suspiciously high for a random strategy"
            )

    def test_random_underperforms_bah_due_to_fees(self, df_1000: pd.DataFrame) -> None:
        """Random strategy must underperform BAH by >= 15 ppt on average across seeds.

        Each round-trip costs ~0.2% in fees + slippage. With many trades the
        drag compounds well below the zero-cost BAH return.
        """
        bah_report = _run_bah(df_1000)
        bah_return = bah_report.total_return_pct

        random_returns = []
        random_trade_counts = []
        for seed in SEEDS:
            report = _run_random(df_1000, seed)
            random_returns.append(report.total_return_pct)
            random_trade_counts.append(report.num_trades)

        mean_random_return = float(np.mean(random_returns))
        mean_num_trades = float(np.mean(random_trade_counts))

        underperformance = bah_return - mean_random_return
        assert underperformance >= 0.15, (
            f"Random strategy underperforms BAH by only {underperformance:.3f} "
            f"(need >= 0.15). BAH={bah_return:.3f}, random mean={mean_random_return:.3f}"
        )
        assert mean_num_trades >= 50, (
            f"Mean num_trades={mean_num_trades:.1f} — not enough trades to validate fees drag"
        )
