"""Integration tests: DCAModulatedStrategy × BacktestEngine on synthetic data."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulator
from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.strategies.strat06.config import DCAModulatedConfig
from btc_quant.strategies.strat06.strategy import DCAModulatedStrategy
from tests.integration.conftest import make_strat06_inflows, make_synthetic_1d_dataset

try:
    import pyarrow  # noqa: F401
    _HAS_PARQUET = True
except ImportError:
    _HAS_PARQUET = False

UTC = timezone.utc
START = datetime(2024, 1, 1, tzinfo=UTC)   # Monday

_BASELINE = StrategyTag.STRAT_06_BASELINE
_BUFFER = StrategyTag.STRAT_06_BUFFER
_RESERVE = StrategyTag.STRAT_06_RESERVE

DATA_DIR = Path(__file__).parents[2] / "data" / "processed"
DAILY_PARQUET = DATA_DIR / "btcusdt_1d.parquet"


# ── Engine factory ────────────────────────────────────────────────────────────


def _make_engine(
    config: DCAModulatedConfig,
    df,
    initial_capital: dict | None = None,
    inflows=None,
) -> BacktestEngine:
    strat = DCAModulatedStrategy(config)
    ic = initial_capital or {_BASELINE: 0.0, _BUFFER: 0.0, _RESERVE: 0.0}
    return BacktestEngine(
        strategies=[strat],
        data={"1d": df},
        initial_capital_per_strategy=ic,
        execution_simulator=ExecutionSimulator(fee_rate=0.001, slippage_rate=0.0),
        profit_recycle_pct=0.0,
        inflow_schedule=inflows,
    )


# ── Helper ────────────────────────────────────────────────────────────────────


def _reserve_curve(equity_curve, tag=_RESERVE):
    return [s.cash_by_pool.get(tag, 0.0) for s in equity_curve]


class TestStrat06IntegrationWithEngine:
    """Validates STRAT-06 inside BacktestEngine on controlled synthetic scenarios."""

    # ── Test 1 ────────────────────────────────────────────────────────────────

    def test_one_year_flat_price_basic_flow(self, make_synthetic_1d_dataset):
        """365 daily bars at $50k with monthly €500 inflows.

        Expected:
        - 36 processed inflows (3 pools × 12 months)
        - 0 closed trades (STRAT-06 never sells)
        - ≥ 48 open positions (at least 4 baseline buys per month)
        - Some BTC accumulated in BASELINE pool
        - RESERVE reaches cap (€1500) by end of year
        - BUFFER > 0 (accumulated its own inflows)
        - 0 reserve deployments (no drawdown)
        """
        n = 365
        df = make_synthetic_1d_dataset(n, START, constant_price=50_000.0)
        cfg = DCAModulatedConfig(monthly_inflow_eur=500.0)
        end = START + timedelta(days=n)
        inflows = make_strat06_inflows(cfg, START, end)

        result = _make_engine(cfg, df, inflows=inflows).run()

        assert len(result.processed_inflows) == 36  # 3 pools × 12 months
        assert len(result.closed_trades) == 0
        # All DCA buys accumulate into one position; check fill count ≥ 48
        assert len(result.final_portfolio.open_positions) == 1
        pos = list(result.final_portfolio.open_positions.values())[0]
        assert len(pos.entry_orders) >= 48

        baseline_pool = result.final_portfolio.get_pool(_BASELINE)
        reserve_pool = result.final_portfolio.get_pool(_RESERVE)
        buffer_pool = result.final_portfolio.get_pool(_BUFFER)

        assert baseline_pool.btc_held > 0.060
        assert reserve_pool.cash_eur == pytest.approx(1500.0, abs=1.0)
        assert buffer_pool.cash_eur > 0.0

        # No reserve deployment: RESERVE never decreased below its monthly peak
        reserve_vals = _reserve_curve(result.equity_curve)
        assert reserve_vals[-1] == pytest.approx(1500.0, abs=1.0)

    # ── Test 2 ────────────────────────────────────────────────────────────────

    def test_simulated_bear_triggers_modulation(self, make_synthetic_1d_dataset):
        """200 bars: 50 flat at $50k, then 100-bar linear drop to $25k, then 50 flat.

        Expected:
        - BTC accumulation grows as multiplier activates during bear
        - Reserve tranche deploys after 7 days of sustained -50% drawdown
        - 0 closed trades
        """
        n = 200
        prices = [50_000.0] * 50
        for k in range(100):
            prices.append(50_000.0 - k * 25_000.0 / 99.0)
        prices += [25_000.0] * 50

        df = make_synthetic_1d_dataset(n, START, price_series=prices)
        cfg = DCAModulatedConfig(monthly_inflow_eur=500.0)
        end = START + timedelta(days=n)
        inflows = make_strat06_inflows(cfg, START, end)

        result = _make_engine(cfg, df, inflows=inflows).run()

        assert len(result.closed_trades) == 0
        assert len(result.final_portfolio.open_positions) > 0

        # Reserve should have been deployed (RESERVE.cash drops after sustained -50%)
        reserve_vals = _reserve_curve(result.equity_curve)
        max_reserve = max(reserve_vals)
        # At least one bar where reserve dropped from its peak (= deployment fired)
        assert any(v < max_reserve * 0.90 for v in reserve_vals), (
            "Expected reserve deployment after 7 days of -50% drawdown"
        )

        # BTC accumulated should exceed what flat-price 1x-multiplier would produce
        # Rough lower bound: ~6 months of baseline-only buys at 68.75/week ≈ 24 buys
        assert result.final_portfolio.get_pool(_BASELINE).btc_held > 0.015

    # ── Test 3 ────────────────────────────────────────────────────────────────

    def test_recovery_resets_reserve_cycle(self, make_synthetic_1d_dataset):
        """340 bars: bull → bear (−50%, 10-bar flat) → recovery (ATH $60k) → bear again.

        Each bear cycle drops to -50% then holds flat for ≥7 bars so the
        persistence check fires and the reserve deploys. Expected:
        - Reserve deploys in the first bear cycle
        - New ATH triggers cycle reset
        - Reserve deploys again in the second bear cycle
        - Two distinct drops in the RESERVE equity curve
        """
        # Phase 1 (bars 0-49):   $50k flat
        # Phase 2 (bars 50-89):  $50k → $25k  (40-bar drop, reaches -50%)
        # Phase 2b (bars 90-99): $25k flat     (10 bars to satisfy 7-day persistence)
        # Phase 3 (bars 100-199): $25k → $60k  (new ATH)
        # Phase 4 (bars 200-239): $60k → $30k  (40-bar drop, reaches -50% of ATH)
        # Phase 4b (bars 240-339): $30k flat   (100 bars to satisfy persistence + buffer)
        prices = [50_000.0] * 50
        for k in range(40):
            prices.append(50_000.0 - k * 25_000.0 / 39.0)
        prices += [25_000.0] * 10
        for k in range(100):
            prices.append(25_000.0 + k * 35_000.0 / 99.0)
        for k in range(40):
            prices.append(60_000.0 - k * 30_000.0 / 39.0)
        prices += [30_000.0] * 100
        n = len(prices)  # 340

        df = make_synthetic_1d_dataset(n, START, price_series=prices)
        cfg = DCAModulatedConfig(monthly_inflow_eur=500.0)
        end = START + timedelta(days=n)
        inflows = make_strat06_inflows(cfg, START, end)

        result = _make_engine(cfg, df, inflows=inflows).run()

        assert len(result.closed_trades) == 0
        assert len(result.final_portfolio.open_positions) > 0

        reserve_vals = _reserve_curve(result.equity_curve)
        max_reserve = max(reserve_vals)

        # Identify bars where RESERVE dropped ≥10% from the running maximum
        running_max = 0.0
        drops: list[int] = []
        for i, v in enumerate(reserve_vals):
            if v > running_max:
                running_max = v
            elif running_max > 0 and v < running_max * 0.90:
                drops.append(i)

        assert len(drops) >= 2, (
            f"Expected at least 2 reserve deployment events; "
            f"found {len(drops)} drop(s) in RESERVE equity curve"
        )

    # ── Test 4 ────────────────────────────────────────────────────────────────

    def test_concentration_limit_blocks_buys(self, make_synthetic_1d_dataset):
        """Concentration check blocks baseline buys when BTC dominates the portfolio.

        Setup: BASELINE starts with €5000 and accumulates BTC at $50k.
        After 100 bars, price jumps to $1M. Concentration spikes to ~83% > 70%.
        No further buys should occur during the high-price phase.
        """
        n = 200
        prices = [50_000.0] * 100 + [1_000_000.0] * 100

        df = make_synthetic_1d_dataset(n, START, price_series=prices)
        cfg = DCAModulatedConfig(monthly_inflow_eur=500.0, max_concentration_pct=0.70)
        # No inflows — use fixed initial capital only
        initial = {_BASELINE: 5_000.0, _BUFFER: 0.0, _RESERVE: 0.0}

        result = _make_engine(cfg, df, initial_capital=initial).run()

        assert len(result.closed_trades) == 0

        # Phase 1 (bars 0-99 at 50k): Mondays Jan 1 through ~Apr 8 = 15 Mondays
        # Every Monday fires (BASELINE has plenty of cash, concentration low)
        # Phase 2 (bars 100-199 at 1M): all Mondays blocked by concentration
        # All buys accumulate into one position; check fill count
        assert len(result.final_portfolio.open_positions) == 1
        total_fills = len(list(result.final_portfolio.open_positions.values())[0].entry_orders)
        assert 12 <= total_fills <= 16  # all from phase 1

        baseline_pool = result.final_portfolio.get_pool(_BASELINE)
        # BTC was accumulated in phase 1
        assert baseline_pool.btc_held > 0.015

        # In phase 2, BASELINE cash should be unchanged from end of phase 1
        # (no buys → cash was not deducted). At minimum, remaining cash > 0.
        assert baseline_pool.cash_eur > 0

    # ── Test 5 ────────────────────────────────────────────────────────────────

    def test_overflow_redirects_to_buffer(self, make_synthetic_1d_dataset):
        """RESERVE fills to cap (€1500) after 12 months; subsequent inflows overflow to BUFFER.

        With 18 months of inflows: months 1-12 fill RESERVE to cap, months 13-18
        redirect €125/month overflow to BUFFER instead.
        """
        n = 550  # ≈18 months
        df = make_synthetic_1d_dataset(n, START, constant_price=50_000.0)
        cfg = DCAModulatedConfig(monthly_inflow_eur=500.0)
        end = START + timedelta(days=n)
        inflows = make_strat06_inflows(cfg, START, end)

        result = _make_engine(cfg, df, inflows=inflows).run()

        assert len(result.closed_trades) == 0

        reserve_pool = result.final_portfolio.get_pool(_RESERVE)
        buffer_pool = result.final_portfolio.get_pool(_BUFFER)

        # Reserve must never exceed cap (1 EUR tolerance for float precision)
        reserve_vals = _reserve_curve(result.equity_curve)
        assert max(reserve_vals) <= 1501.0

        # Reserve final value = cap (fills up and stays there)
        assert reserve_pool.cash_eur == pytest.approx(1500.0, abs=1.0)

        # Buffer received its own inflows (18 * 100 = 1800) plus reserve overflow
        # (6 months * 125 = 750), minus supplementary baseline buys from buffer.
        # Lower bound: well above zero.
        assert buffer_pool.cash_eur > 500.0

    # ── Test 6 ────────────────────────────────────────────────────────────────

    def test_no_sell_signals_strat06_only_accumulates(self, make_synthetic_1d_dataset):
        """STRAT-06 never emits a SELL signal: all positions remain open."""
        n = 200
        # Volatile pattern: alternating up/down phases
        prices = []
        for i in range(n):
            if (i // 20) % 2 == 0:
                prices.append(50_000.0 + i * 100)
            else:
                prices.append(50_000.0 - i * 50)
        # Clip to positive prices
        prices = [max(p, 1_000.0) for p in prices]

        df = make_synthetic_1d_dataset(n, START, price_series=prices)
        cfg = DCAModulatedConfig(monthly_inflow_eur=500.0)
        end = START + timedelta(days=n)
        inflows = make_strat06_inflows(cfg, START, end)

        result = _make_engine(cfg, df, inflows=inflows).run()

        assert len(result.closed_trades) == 0
        assert len(result.final_portfolio.open_positions) >= 1
        assert result.final_portfolio.get_pool(_BASELINE).btc_held > 0

    # ── Test 7 ────────────────────────────────────────────────────────────────

    def test_inflow_split_across_three_pools(self, make_synthetic_1d_dataset):
        """Inflows arrive correctly proportioned across the three STRAT-06 pools.

        With monthly_inflow=€500 (55/20/25 split):
          BASELINE receives €275, BUFFER €100, RESERVE €125 per month.

        The snapshot at bar 0 captures pool state AFTER the inflow fires and
        AFTER generate_signals runs, but BEFORE the next bar's buy execution
        (signal execution happens at bar 1's open). Since BASELINE ≥ target
        (€275 ≥ €68.75), no BUFFER→BASELINE transfer occurs during generate_signals,
        so the snapshot reflects the raw inflow amounts.
        """
        n = 5
        df = make_synthetic_1d_dataset(n, START, constant_price=50_000.0)
        cfg = DCAModulatedConfig(monthly_inflow_eur=500.0)
        # Single inflow set on Jan 1 only (end=Jan 2 so only one monthly entry)
        end_exclusive = START + timedelta(days=1, seconds=1)
        inflows = make_strat06_inflows(cfg, START, end_exclusive)

        result = _make_engine(cfg, df, inflows=inflows).run()

        # Bar 0 snapshot: inflow fired, generate_signals ran, buy NOT yet executed
        snap0 = result.equity_curve[0]
        assert snap0.cash_by_pool[_BASELINE] == pytest.approx(275.0, rel=1e-6)
        assert snap0.cash_by_pool[_BUFFER] == pytest.approx(100.0, rel=1e-6)
        assert snap0.cash_by_pool[_RESERVE] == pytest.approx(125.0, rel=1e-6)

    # ── Smoke test ────────────────────────────────────────────────────────────

    @pytest.mark.skipif(not _HAS_PARQUET, reason="pyarrow not installed")
    def test_full_8_year_simulation_does_not_crash(self):
        """Smoke test: STRAT-06 on ~8 years of real BTC data (2017-2026).

        Only verifies: no exception, non-empty equity curve, some trades executed,
        no NaN/Inf in equity values. No quantitative assertions (those are Sub-session 4.4).
        """
        import pandas as pd

        df = pd.read_parquet(DAILY_PARQUET)

        start = datetime(2017, 9, 1, tzinfo=UTC)
        end = datetime(2026, 5, 1, tzinfo=UTC)

        cfg = DCAModulatedConfig(monthly_inflow_eur=500.0)
        inflows = make_strat06_inflows(cfg, start, end)

        result = _make_engine(
            cfg,
            df,
            inflows=inflows,
        ).run()

        assert len(result.equity_curve) > 0
        assert len(result.final_portfolio.open_positions) > 0

        for snap in result.equity_curve:
            assert snap.total_value_eur == snap.total_value_eur  # not NaN
            assert snap.total_value_eur < float("inf")
