"""Integration test: MetricsCalculator × BacktestEngine × BuyAndHoldStrategy.

Uses ~8 years of real BTC daily data (2017-08 to 2026-05).
Known-range assertions are grounded in publicly documented BTC history:
  - Long-term CAGR > 30% (very conservative lower bound for 8+ years of BTC)
  - Max drawdown between -70% and -90% (BTC experienced -83% in 2018 alone)
  - Sharpe between 0.3 and 2.5 (conservative range for daily equity-hold)
  - BAH never closes a position → num_trades == 0
  - BAH buys on bar[1] and holds → time_in_market > 99%
"""

from __future__ import annotations

from pathlib import Path

import pytest

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulator
from btc_quant.backtester.metrics.calculator import MetricsCalculator
from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.strategies.buy_and_hold import BuyAndHoldStrategy

DATA_DIR = Path(__file__).parents[2] / "data" / "processed"
DAILY_PARQUET = DATA_DIR / "btcusdt_1d.parquet"

TAG = StrategyTag.BUY_AND_HOLD_BENCHMARK
INITIAL_CAPITAL = 10_000.0


@pytest.fixture(scope="module")
def bah_metrics():
    """Run BAH on full BTC history and compute MetricsReport."""
    import pandas as pd

    df = pd.read_parquet(DAILY_PARQUET)
    # Drop the very first bar (tiny volume artefact) but keep all subsequent bars
    df = df.iloc[1:].reset_index(drop=True)

    engine = BacktestEngine(
        strategies=[BuyAndHoldStrategy()],
        data={"1d": df},
        initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
        execution_simulator=ExecutionSimulator(fee_rate=0.001, slippage_rate=0.0005),
        profit_recycle_pct=0.0,
    )
    result = engine.run()
    return MetricsCalculator().calculate(result)


class TestMetricsWithBuyAndHold:
    """Sanity-check MetricsReport values against known BTC history."""

    def test_bah_num_trades_is_zero(self, bah_metrics) -> None:
        """BAH never sells — no closed trades."""
        assert bah_metrics.num_trades == 0

    def test_bah_time_in_market_above_99pct(self, bah_metrics) -> None:
        """BAH buys on bar[1] and holds forever → almost always in market."""
        assert bah_metrics.time_in_market_pct > 0.99

    def test_bah_cagr_is_not_none(self, bah_metrics) -> None:
        assert bah_metrics.cagr is not None

    def test_bah_cagr_above_30pct(self, bah_metrics) -> None:
        """BTC delivered > 30% CAGR over 8+ years (conservative lower bound)."""
        assert bah_metrics.cagr > 0.30

    def test_bah_max_drawdown_in_known_range(self, bah_metrics) -> None:
        """BTC experienced -83% in 2018, -77% in 2022. Range: [-90%, -70%]."""
        assert -0.90 <= bah_metrics.max_drawdown_pct <= -0.70

    def test_bah_sharpe_in_reasonable_range(self, bah_metrics) -> None:
        """Annualised Sharpe on full BTC history expected between 0.3 and 2.5."""
        assert bah_metrics.sharpe_ratio is not None
        assert 0.3 <= bah_metrics.sharpe_ratio <= 2.5

    def test_bah_operational_metrics_are_none(self, bah_metrics) -> None:
        """All per-trade metrics must be None when there are no closed trades."""
        assert bah_metrics.win_rate is None
        assert bah_metrics.profit_factor is None
        assert bah_metrics.avg_win_eur is None
        assert bah_metrics.avg_loss_eur is None
        assert bah_metrics.expectancy_eur is None
        assert bah_metrics.largest_losing_streak_trades is None

    def test_bah_total_return_positive(self, bah_metrics) -> None:
        """BTC price rose ~19× from 2017 to 2026 → strongly positive return."""
        assert bah_metrics.total_return_pct > 5.0  # > 500%

    def test_bah_max_drawdown_duration_nonzero(self, bah_metrics) -> None:
        """BTC had multi-month drawdowns; duration must be well above 0."""
        assert bah_metrics.max_drawdown_duration_days > 30

    def test_bah_annualized_volatility_btc_range(self, bah_metrics) -> None:
        """BTC annualised vol is historically 60–100%."""
        assert 0.50 <= bah_metrics.annualized_volatility <= 1.20

    def test_bah_var_95_is_negative(self, bah_metrics) -> None:
        """5th-percentile daily return must be negative for a volatile asset."""
        assert bah_metrics.var_95_daily < 0.0

    def test_bah_es_more_extreme_than_var(self, bah_metrics) -> None:
        assert bah_metrics.expected_shortfall_95_daily <= bah_metrics.var_95_daily

    def test_bah_metrics_by_strategy_populated(self, bah_metrics) -> None:
        assert TAG in bah_metrics.metrics_by_strategy
        sm = bah_metrics.metrics_by_strategy[TAG]
        assert sm.num_trades == 0
        assert sm.win_rate is None

    def test_bah_duration_days_consistent(self, bah_metrics) -> None:
        """Duration must match end - start."""
        import datetime
        delta = bah_metrics.end_date_utc - bah_metrics.start_date_utc
        assert bah_metrics.duration_days == delta.days
