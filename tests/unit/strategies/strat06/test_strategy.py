"""Unit tests for DCAModulatedStrategy."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.strategies.strat06.config import BaselineExecutionMode, DCAModulatedConfig
from btc_quant.strategies.strat06.strategy import DCAModulatedStrategy

UTC = timezone.utc

# ── Helpers ───────────────────────────────────────────────────────────────────

MONDAY = datetime(2024, 1, 8, 12, 0, tzinfo=UTC)    # week 2
NEXT_MONDAY = datetime(2024, 1, 15, 12, 0, tzinfo=UTC)  # week 3
TUESDAY = datetime(2024, 1, 9, 12, 0, tzinfo=UTC)
WEDNESDAY = datetime(2024, 1, 10, 12, 0, tzinfo=UTC)

ATH_PRICE = 100_000.0
BELOW_10 = 90_000.0   # -10% drawdown exactly
BELOW_50 = 50_000.0   # -50% drawdown exactly
BELOW_55 = 45_000.0   # -55% drawdown
BELOW_80 = 20_000.0   # -80% drawdown


def _strat(config: DCAModulatedConfig | None = None) -> DCAModulatedStrategy:
    return DCAModulatedStrategy(config or DCAModulatedConfig(monthly_inflow_eur=500.0))


def _day(base: datetime, offset: int) -> datetime:
    return base + timedelta(days=offset)


def _set_ath(strat: DCAModulatedStrategy, price: float, ctx_factory, port_factory) -> None:
    """Call generate_signals on a non-baseline day to establish ATH without triggering a buy."""
    ctx = ctx_factory(TUESDAY, price)
    port = port_factory(baseline_cash=1000.0)
    strat.generate_signals(ctx, port)


def _call_n_days(
    strat: DCAModulatedStrategy,
    price: float,
    n: int,
    ctx_factory,
    port_factory,
    start_day: int = 0,
    baseline_cash: float = 0.0,
    buffer_cash: float = 0.0,
    reserve_cash: float = 0.0,
):
    """Call generate_signals n times on consecutive non-baseline days."""
    port = port_factory(
        baseline_cash=baseline_cash,
        buffer_cash=buffer_cash,
        reserve_cash=reserve_cash,
    )
    last_signals = []
    for i in range(n):
        ts = WEDNESDAY + timedelta(days=start_day + i)
        ctx = ctx_factory(ts, price)
        last_signals = strat.generate_signals(ctx, port)
    return last_signals, port


# ── Baseline execution ────────────────────────────────────────────────────────


class TestBaselineExecution:
    def test_baseline_buy_on_monday_at_target_hour(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        ctx = mock_context_factory(MONDAY, 50_000.0)
        port = mock_portfolio_factory(baseline_cash=200.0)
        signals = strat.generate_signals(ctx, port)
        assert len(signals) == 1
        assert signals[0].quote_amount_eur == pytest.approx(68.75)
        assert signals[0].reason_tag == "baseline_weekly"
        assert signals[0].strategy_tag == StrategyTag.STRAT_06_BASELINE

    def test_no_baseline_buy_outside_target_hour(
        self, mock_context_factory, mock_portfolio_factory
    ) -> None:
        cfg = DCAModulatedConfig(
            monthly_inflow_eur=500.0,
            baseline_execution_mode=BaselineExecutionMode.WEEKLY_AT_HOUR,
        )
        strat = _strat(cfg)
        for hour in (11, 13):
            ts = datetime(2024, 1, 8, hour, 0, tzinfo=UTC)
            ctx = mock_context_factory(ts, 50_000.0)
            port = mock_portfolio_factory(baseline_cash=200.0)
            signals = strat.generate_signals(ctx, port)
            assert signals == [], f"Expected no signal at hour={hour}"

    def test_no_baseline_buy_outside_target_weekday(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        ctx = mock_context_factory(TUESDAY, 50_000.0)
        port = mock_portfolio_factory(baseline_cash=200.0)
        signals = strat.generate_signals(ctx, port)
        assert signals == []

    def test_no_baseline_buy_twice_in_same_iso_week(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        port = mock_portfolio_factory(baseline_cash=500.0)

        # First call — fires
        ctx1 = mock_context_factory(MONDAY, 50_000.0)
        s1 = strat.generate_signals(ctx1, port)
        assert len(s1) == 1

        # Second call same day — no signal
        ctx2 = mock_context_factory(MONDAY, 50_000.0)
        s2 = strat.generate_signals(ctx2, port)
        assert s2 == []

        # Next Monday — fires again
        ctx3 = mock_context_factory(NEXT_MONDAY, 50_000.0)
        s3 = strat.generate_signals(ctx3, port)
        assert len(s3) == 1


# ── Modulation ────────────────────────────────────────────────────────────────


class TestModulation:
    def test_multiplier_1x_at_zero_drawdown(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        ctx = mock_context_factory(MONDAY, 50_000.0)
        port = mock_portfolio_factory(baseline_cash=200.0)
        signals = strat.generate_signals(ctx, port)
        assert len(signals) == 1
        assert signals[0].metadata["multiplier"] == pytest.approx(1.0)
        assert signals[0].quote_amount_eur == pytest.approx(68.75)

    def test_multiplier_125x_at_minus_10_drawdown(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        # Establish ATH on Tuesday
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)
        # Monday at -10%
        ctx = mock_context_factory(NEXT_MONDAY, BELOW_10)
        port = mock_portfolio_factory(baseline_cash=200.0)
        signals = strat.generate_signals(ctx, port)
        assert len(signals) == 1
        assert signals[0].metadata["multiplier"] == pytest.approx(1.25)
        assert signals[0].quote_amount_eur == pytest.approx(68.75 * 1.25)

    def test_multiplier_uses_buffer_when_baseline_insufficient(
        self, mock_context_factory, mock_portfolio_factory
    ) -> None:
        # -35% → multiplier 2.0, target = 68.75 * 2.0 = 137.5
        # baseline=50, buffer=200 → deficit=87.5, transfer 87.5
        cfg = DCAModulatedConfig(monthly_inflow_eur=500.0)
        strat = _strat(cfg)
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)

        port = mock_portfolio_factory(baseline_cash=50.0, buffer_cash=200.0)
        ctx = mock_context_factory(NEXT_MONDAY, ATH_PRICE * 0.65)  # -35% drawdown
        signals = strat.generate_signals(ctx, port)

        assert len(signals) == 1
        assert signals[0].quote_amount_eur == pytest.approx(137.5)
        # Buffer was debited, baseline used
        buffer_pool = port.get_pool(StrategyTag.STRAT_06_BUFFER)
        assert buffer_pool.cash_eur == pytest.approx(200.0 - 87.5)

    def test_multiplier_caps_at_total_cash_when_insufficient(
        self, mock_context_factory, mock_portfolio_factory
    ) -> None:
        # -35% → target=137.5, baseline=20, buffer=30 → total=50, capped at 50
        cfg = DCAModulatedConfig(monthly_inflow_eur=500.0)
        strat = _strat(cfg)
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)

        port = mock_portfolio_factory(baseline_cash=20.0, buffer_cash=30.0)
        ctx = mock_context_factory(NEXT_MONDAY, ATH_PRICE * 0.65)
        signals = strat.generate_signals(ctx, port)

        assert len(signals) == 1
        assert signals[0].quote_amount_eur == pytest.approx(50.0)

    def test_no_signal_when_cash_below_min_order(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        ctx = mock_context_factory(MONDAY, 50_000.0)
        port = mock_portfolio_factory(baseline_cash=5.0, buffer_cash=0.0)
        signals = strat.generate_signals(ctx, port)
        assert signals == []


# ── Reserve deployment ────────────────────────────────────────────────────────


class TestReserveDeployment:
    def test_no_reserve_deployment_above_minus_50(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)
        # 8 days at -45% — not deep enough to trigger tranche 0
        signals, _ = _call_n_days(
            strat, ATH_PRICE * 0.55, 8,
            mock_context_factory, mock_portfolio_factory,
            reserve_cash=1500.0,
        )
        assert all("reserve_tranche" not in s.reason_tag for s in signals)

    def test_reserve_tranche_0_at_minus_50_persistent(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        """1 ATH-setting call + 7 consecutive days at -50% → fires on 8th call."""
        strat = _strat(default_config)
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)
        signals, _ = _call_n_days(
            strat, BELOW_50, 7,
            mock_context_factory, mock_portfolio_factory,
            reserve_cash=1500.0,
        )
        assert len(signals) == 1
        assert signals[0].reason_tag == "reserve_tranche_0"

    def test_no_reserve_deployment_without_persistence(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)
        signals, _ = _call_n_days(
            strat, BELOW_50, 3,
            mock_context_factory, mock_portfolio_factory,
            reserve_cash=1500.0,
        )
        assert all("reserve_tranche" not in s.reason_tag for s in signals)

    def test_reserve_signal_includes_transfer(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)
        signals, port = _call_n_days(
            strat, BELOW_50, 7,
            mock_context_factory, mock_portfolio_factory,
            reserve_cash=1500.0,
        )
        assert len(signals) == 1
        reserve_pool = port.get_pool(StrategyTag.STRAT_06_RESERVE)
        baseline_pool = port.get_pool(StrategyTag.STRAT_06_BASELINE)
        deployed = signals[0].quote_amount_eur
        assert reserve_pool.cash_eur == pytest.approx(1500.0 - deployed)
        assert baseline_pool.cash_eur == pytest.approx(deployed)

    def test_reserve_below_min_order_no_signal(
        self, mock_context_factory, mock_portfolio_factory
    ) -> None:
        # reserve=20, tranche 0 deploys 20% → 4 EUR < min_order=10
        cfg = DCAModulatedConfig(monthly_inflow_eur=500.0)
        strat = _strat(cfg)
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)
        signals, _ = _call_n_days(
            strat, BELOW_50, 7,
            mock_context_factory, mock_portfolio_factory,
            reserve_cash=20.0,
        )
        assert all("reserve_tranche" not in s.reason_tag for s in signals)

    def test_reserve_resets_on_new_ath(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        port = mock_portfolio_factory(reserve_cash=1500.0)

        # Establish ATH at 100k, then 7 days at -50% → tranche 0 fires
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)
        _call_n_days(strat, BELOW_50, 7, mock_context_factory, mock_portfolio_factory,
                     reserve_cash=1500.0)

        # New ATH → reset_cycle
        ctx_ath = mock_context_factory(WEDNESDAY + timedelta(days=10), ATH_PRICE * 1.10)
        strat.generate_signals(ctx_ath, mock_portfolio_factory(reserve_cash=1500.0))

        # Immediately at -55% — persistence not established yet, no fire
        ctx_dd = mock_context_factory(WEDNESDAY + timedelta(days=11), BELOW_55)
        port2 = mock_portfolio_factory(reserve_cash=1500.0)
        signals = strat.generate_signals(ctx_dd, port2)
        assert all("reserve_tranche" not in s.reason_tag for s in signals)


# ── Overflow redirection ──────────────────────────────────────────────────────


class TestOverflowRedirection:
    def test_reserve_overflow_moved_to_buffer(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        # reserve_cap_eur = 12 * 500 * 0.25 = 1500. reserve=2000 → overflow=500
        strat = _strat(default_config)
        ctx = mock_context_factory(TUESDAY, 50_000.0)
        port = mock_portfolio_factory(reserve_cash=2000.0)
        strat.generate_signals(ctx, port)
        reserve_pool = port.get_pool(StrategyTag.STRAT_06_RESERVE)
        buffer_pool = port.get_pool(StrategyTag.STRAT_06_BUFFER)
        assert reserve_pool.cash_eur == pytest.approx(1500.0)
        assert buffer_pool.cash_eur == pytest.approx(500.0)

    def test_no_overflow_when_below_cap(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        ctx = mock_context_factory(TUESDAY, 50_000.0)
        port = mock_portfolio_factory(reserve_cash=1000.0)
        strat.generate_signals(ctx, port)
        reserve_pool = port.get_pool(StrategyTag.STRAT_06_RESERVE)
        buffer_pool = port.get_pool(StrategyTag.STRAT_06_BUFFER)
        assert reserve_pool.cash_eur == pytest.approx(1000.0)
        assert buffer_pool.cash_eur == pytest.approx(0.0)


# ── ATH tracking ─────────────────────────────────────────────────────────────


class TestATHTracking:
    def test_ath_initialized_from_first_price(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        ctx = mock_context_factory(TUESDAY, 50_000.0)
        port = mock_portfolio_factory()
        strat.generate_signals(ctx, port)
        assert strat._ath_tracker.ath == pytest.approx(50_000.0)

    def test_ath_updates_with_higher_price(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        for price in (50_000.0, 80_000.0, 70_000.0):
            ctx = mock_context_factory(TUESDAY, price)
            port = mock_portfolio_factory()
            strat.generate_signals(ctx, port)
        assert strat._ath_tracker.ath == pytest.approx(80_000.0)

    def test_reset_cycle_happens_before_record_drawdown(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        """After a new ATH the deployed tranche set resets, and the 0.0
        drawdown recorded at the ATH break the -55% streak, so the tranche
        cannot fire until 7 more consecutive qualifying days pass."""
        strat = _strat(default_config)

        # Phase 1: establish ATH + deploy tranche 0
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)
        _call_n_days(strat, BELOW_50, 7, mock_context_factory, mock_portfolio_factory,
                     reserve_cash=1500.0)

        # Phase 2: new ATH resets cycle
        ctx_new_ath = mock_context_factory(WEDNESDAY + timedelta(days=10), ATH_PRICE * 1.10)
        strat.generate_signals(ctx_new_ath, mock_portfolio_factory(reserve_cash=1500.0))

        # Phase 3: 6 days at -55% — streak length = 6, not enough
        port = mock_portfolio_factory(reserve_cash=1500.0)
        for i in range(6):
            ts = WEDNESDAY + timedelta(days=11 + i)
            ctx = mock_context_factory(ts, BELOW_55)
            signals = strat.generate_signals(ctx, port)
            assert all("reserve_tranche" not in s.reason_tag for s in signals), \
                f"Unexpected reserve signal on day {i+1}"

        # Phase 4: 7th day at -55% → tranche 0 fires (new cycle)
        ctx_fire = mock_context_factory(WEDNESDAY + timedelta(days=17), BELOW_55)
        signals = strat.generate_signals(ctx_fire, port)
        reserve_signals = [s for s in signals if "reserve_tranche" in s.reason_tag]
        assert len(reserve_signals) == 1


# ── Concentration limit ───────────────────────────────────────────────────────


class TestConcentrationLimit:
    def test_no_buy_when_concentration_above_70(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        # cash=100, btc_held=0.005 at 50k → btc_value=250, total=350, conc=71.4%
        strat = _strat(default_config)
        from btc_quant.backtester.models.pool import CapitalPool
        from btc_quant.backtester.models.portfolio import Portfolio
        port = Portfolio(
            pools={
                StrategyTag.STRAT_06_BASELINE: CapitalPool(
                    strategy_tag=StrategyTag.STRAT_06_BASELINE,
                    cash_eur=100.0,
                    btc_held=0.005,
                ),
                StrategyTag.STRAT_06_BUFFER: CapitalPool(
                    strategy_tag=StrategyTag.STRAT_06_BUFFER,
                    cash_eur=0.0,
                ),
                StrategyTag.STRAT_06_RESERVE: CapitalPool(
                    strategy_tag=StrategyTag.STRAT_06_RESERVE,
                    cash_eur=0.0,
                ),
            }
        )
        ctx = mock_context_factory(MONDAY, 50_000.0)
        signals = strat.generate_signals(ctx, port)
        assert signals == []

    def test_buy_resumes_after_concentration_drops(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        # Low concentration: cash=1000, btc=0 → 0%
        port = mock_portfolio_factory(baseline_cash=1000.0)
        ctx = mock_context_factory(MONDAY, 50_000.0)
        signals = strat.generate_signals(ctx, port)
        assert len(signals) == 1


# ── Signal order ──────────────────────────────────────────────────────────────


class TestSignalOrder:
    def _setup_persistent_drawdown(
        self, strat, price, mock_context_factory, mock_portfolio_factory,
        reserve_cash=1500.0, n=7, start_day=0
    ):
        """Record n days of drawdown at price (non-baseline days)."""
        port = mock_portfolio_factory(
            baseline_cash=500.0,
            buffer_cash=200.0,
            reserve_cash=reserve_cash,
        )
        for i in range(n):
            ts = WEDNESDAY + timedelta(days=start_day + i)
            ctx = mock_context_factory(ts, price)
            strat.generate_signals(ctx, port)
        return port

    def test_both_baseline_and_reserve_in_same_call(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        """Reserve signal comes before baseline signal.

        Setup: ATH on Tue Jan 9, then 6 days Tue-Sun (Jan 9-14) at -50%
        → 6 consecutive qualifying entries. Final call on Mon Jan 15 adds
        the 7th → reserve fires; baseline also fires because it's Monday 12:00.
        (Avoiding WEDNESDAY as base prevents the loop from landing on Jan 15
        prematurely and consuming the baseline-buy quota for that ISO week.)
        """
        strat = _strat(default_config)
        port = mock_portfolio_factory(
            baseline_cash=500.0,
            buffer_cash=200.0,
            reserve_cash=1500.0,
        )

        # ATH + 6 days at -50% using Tue Jan 9 through Sun Jan 14 (none are Monday)
        base = datetime(2024, 1, 9, 12, 0, tzinfo=UTC)  # Tuesday Jan 9
        # First call establishes ATH=100k, records drawdown=0.0
        strat.generate_signals(mock_context_factory(base, ATH_PRICE), port)
        # Days 1-6: Tue Jan 9 through Sun Jan 14 at -50%
        for i in range(6):
            ts = base + timedelta(days=i)
            strat.generate_signals(mock_context_factory(ts, BELOW_50), port)

        # 7th day at -50% on Monday Jan 15 12:00 — reserve + baseline both fire
        ctx = mock_context_factory(NEXT_MONDAY, BELOW_50)
        signals = strat.generate_signals(ctx, port)
        assert len(signals) == 2
        assert "reserve_tranche" in signals[0].reason_tag
        assert "baseline" in signals[1].reason_tag

    def test_only_reserve_when_not_baseline_time(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)
        port = self._setup_persistent_drawdown(
            strat, BELOW_50, mock_context_factory, mock_portfolio_factory, n=6
        )
        # Wednesday (not baseline day) with persistent -50%
        ctx = mock_context_factory(WEDNESDAY + timedelta(days=10), BELOW_50)
        signals = strat.generate_signals(ctx, port)
        assert len(signals) == 1
        assert "reserve_tranche" in signals[0].reason_tag

    def test_only_baseline_when_no_reserve_trigger(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        ctx = mock_context_factory(MONDAY, 50_000.0)
        port = mock_portfolio_factory(baseline_cash=200.0, reserve_cash=1500.0)
        signals = strat.generate_signals(ctx, port)
        assert len(signals) == 1
        assert signals[0].reason_tag == "baseline_weekly"


# ── Edge cases ────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_no_signal_when_all_pools_empty(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        ctx = mock_context_factory(MONDAY, 50_000.0)
        port = mock_portfolio_factory()  # all pools at 0
        signals = strat.generate_signals(ctx, port)
        assert signals == []

    def test_zero_drawdown_normal_baseline_quote_amount(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        strat = _strat(default_config)
        ctx = mock_context_factory(MONDAY, 50_000.0)
        port = mock_portfolio_factory(baseline_cash=500.0)
        signals = strat.generate_signals(ctx, port)
        assert len(signals) == 1
        assert signals[0].quote_amount_eur == pytest.approx(68.75)

    def test_extreme_drawdown_minus_80_uses_max_multiplier(
        self, mock_context_factory, mock_portfolio_factory, default_config
    ) -> None:
        # -80% → multiplier 2.5 (max tier), target = 68.75 * 2.5 = 171.875
        strat = _strat(default_config)
        _set_ath(strat, ATH_PRICE, mock_context_factory, mock_portfolio_factory)
        ctx = mock_context_factory(NEXT_MONDAY, BELOW_80)
        port = mock_portfolio_factory(baseline_cash=500.0)
        signals = strat.generate_signals(ctx, port)
        assert len(signals) == 1
        assert signals[0].metadata["multiplier"] == pytest.approx(2.5)
        assert signals[0].quote_amount_eur == pytest.approx(68.75 * 2.5)


# ── Baseline execution mode ───────────────────────────────────────────────────


class TestBaselineExecutionMode:
    def test_default_mode_is_weekly(self, default_config) -> None:
        assert default_config.baseline_execution_mode == BaselineExecutionMode.WEEKLY

    def test_weekly_mode_triggers_on_weekday_any_hour(
        self, mock_context_factory, mock_portfolio_factory
    ) -> None:
        cfg = DCAModulatedConfig(
            monthly_inflow_eur=500.0,
            baseline_execution_mode=BaselineExecutionMode.WEEKLY,
            baseline_weekday=0,
        )
        strat = _strat(cfg)

        # Monday at 23:59:59 (daily bar close) — should fire
        ts_close = datetime(2024, 1, 8, 23, 59, 59, tzinfo=UTC)
        port = mock_portfolio_factory(baseline_cash=200.0)
        signals = strat.generate_signals(mock_context_factory(ts_close, 50_000.0), port)
        assert len(signals) == 1, "WEEKLY mode should fire at Monday bar close (23:59)"

        # Same ISO week, Monday again at 12:00 — must NOT fire twice
        ts_noon = datetime(2024, 1, 8, 12, 0, tzinfo=UTC)
        port2 = mock_portfolio_factory(baseline_cash=200.0)
        signals2 = strat.generate_signals(mock_context_factory(ts_noon, 50_000.0), port2)
        assert signals2 == [], "WEEKLY mode must not fire twice in same ISO week"

        # Next Monday at 23:59:59 — fires again
        ts_next = datetime(2024, 1, 15, 23, 59, 59, tzinfo=UTC)
        port3 = mock_portfolio_factory(baseline_cash=200.0)
        signals3 = strat.generate_signals(mock_context_factory(ts_next, 50_000.0), port3)
        assert len(signals3) == 1, "WEEKLY mode should fire again next ISO week"

    def test_weekly_at_hour_mode_requires_exact_hour(
        self, mock_context_factory, mock_portfolio_factory
    ) -> None:
        cfg = DCAModulatedConfig(
            monthly_inflow_eur=500.0,
            baseline_execution_mode=BaselineExecutionMode.WEEKLY_AT_HOUR,
            baseline_weekday=0,
            baseline_hour_utc=12,
        )
        strat = _strat(cfg)
        port = mock_portfolio_factory(baseline_cash=200.0)

        # Monday at 11:00 — no fire
        ts_early = datetime(2024, 1, 8, 11, 0, tzinfo=UTC)
        assert strat.generate_signals(mock_context_factory(ts_early, 50_000.0), port) == []

        # Monday at 12:00 — fires
        ts_exact = datetime(2024, 1, 8, 12, 0, tzinfo=UTC)
        signals = strat.generate_signals(mock_context_factory(ts_exact, 50_000.0), port)
        assert len(signals) == 1

        # Monday at 13:00 — no fire (same ISO week already consumed)
        ts_late = datetime(2024, 1, 8, 13, 0, tzinfo=UTC)
        assert strat.generate_signals(mock_context_factory(ts_late, 50_000.0), port) == []
