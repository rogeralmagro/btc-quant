"""Unit tests for ReserveManager, ReserveTranche, and ReserveDeployment."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from btc_quant.strategies.strat06.reserve_manager import (
    ReserveDeployment,
    ReserveManager,
    ReserveTranche,
)

UTC = timezone.utc
BASE_TS = datetime(2020, 1, 1, tzinfo=UTC)
PRICE = 10_000.0


def _ts(day: int) -> datetime:
    return BASE_TS + timedelta(days=day)


def _manager(cap: float = 1_500.0, persistence: int = 7) -> ReserveManager:
    return ReserveManager(reserve_cap_eur=cap, persistence_days=persistence)


def _record_days(rm: ReserveManager, drawdown: float, n_days: int, start_day: int = 0) -> None:
    """Record `n_days` consecutive days of the same drawdown."""
    for i in range(n_days):
        rm.record_drawdown(_ts(start_day + i), drawdown)


# ---------------------------------------------------------------------------
# ReserveTranche validation
# ---------------------------------------------------------------------------

class TestReserveTranche:
    def test_valid_tranche(self) -> None:
        t = ReserveTranche(drawdown_trigger=-0.50, deploy_pct=0.20)
        assert t.drawdown_trigger == -0.50
        assert t.deploy_pct == 0.20

    def test_positive_trigger_raises(self) -> None:
        with pytest.raises(ValueError, match="<= 0"):
            ReserveTranche(drawdown_trigger=0.10, deploy_pct=0.20)

    def test_zero_deploy_pct_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            ReserveTranche(drawdown_trigger=-0.50, deploy_pct=0.0)

    def test_deploy_pct_above_1_raises(self) -> None:
        with pytest.raises(ValueError, match=r"\(0, 1\]"):
            ReserveTranche(drawdown_trigger=-0.50, deploy_pct=1.1)


# ---------------------------------------------------------------------------
# ReserveManager construction validation
# ---------------------------------------------------------------------------

class TestReserveManagerValidation:
    def test_zero_reserve_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            ReserveManager(reserve_cap_eur=0.0)

    def test_negative_reserve_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="> 0"):
            ReserveManager(reserve_cap_eur=-100.0)

    def test_persistence_zero_days_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            ReserveManager(reserve_cap_eur=1000.0, persistence_days=0)

    def test_unordered_tranches_raises(self) -> None:
        tranches = [
            ReserveTranche(-0.70, 0.25),
            ReserveTranche(-0.50, 0.20),  # wrong order
        ]
        with pytest.raises(ValueError, match="ordered"):
            ReserveManager(reserve_cap_eur=1000.0, tranches=tranches)

    def test_tranches_total_exceeds_100pct_raises(self) -> None:
        tranches = [
            ReserveTranche(-0.50, 0.60),
            ReserveTranche(-0.70, 0.60),  # 0.60 + 0.60 = 1.20 > 1.0
        ]
        with pytest.raises(ValueError, match="<= 1.0"):
            ReserveManager(reserve_cap_eur=1000.0, tranches=tranches)

    def test_tranches_exactly_100pct_is_valid(self) -> None:
        tranches = [
            ReserveTranche(-0.50, 0.50),
            ReserveTranche(-0.70, 0.50),
        ]
        rm = ReserveManager(reserve_cap_eur=1000.0, tranches=tranches)
        assert rm.reserve_cap_eur == 1000.0


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------

class TestReserveManagerAccumulation:
    def test_initial_balance_is_zero(self) -> None:
        rm = _manager()
        assert rm.current_balance_eur == 0.0

    def test_inflow_below_cap_no_overflow(self) -> None:
        rm = _manager(cap=1_500.0)
        overflow = rm.add_inflow(500.0)
        assert overflow == pytest.approx(0.0)
        assert rm.current_balance_eur == pytest.approx(500.0)

    def test_inflow_exceeds_cap_returns_overflow(self) -> None:
        rm = _manager(cap=1_000.0)
        overflow = rm.add_inflow(1_200.0)
        assert overflow == pytest.approx(200.0)
        assert rm.current_balance_eur == pytest.approx(1_000.0)

    def test_multiple_inflows_accumulate(self) -> None:
        rm = _manager(cap=1_500.0)
        rm.add_inflow(500.0)
        rm.add_inflow(500.0)
        assert rm.current_balance_eur == pytest.approx(1_000.0)

    def test_full_reserve_overflow_equals_full_amount(self) -> None:
        rm = _manager(cap=1_000.0)
        rm.add_inflow(1_000.0)  # fill to cap
        overflow = rm.add_inflow(300.0)
        assert overflow == pytest.approx(300.0)
        assert rm.current_balance_eur == pytest.approx(1_000.0)

    def test_inflow_zero_raises(self) -> None:
        rm = _manager()
        with pytest.raises(ValueError):
            rm.add_inflow(0.0)

    def test_inflow_negative_raises(self) -> None:
        rm = _manager()
        with pytest.raises(ValueError):
            rm.add_inflow(-50.0)


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------

class TestReserveManagerDeployment:
    def test_no_deployment_when_drawdown_above_trigger(self) -> None:
        rm = _manager()
        rm.add_inflow(1_000.0)
        _record_days(rm, drawdown=-0.30, n_days=10)
        result = rm.evaluate_deployment(-0.30, PRICE, _ts(10))
        assert result is None

    def test_no_deployment_without_persistence(self) -> None:
        rm = _manager(persistence=7)
        rm.add_inflow(1_000.0)
        _record_days(rm, drawdown=-0.55, n_days=3)
        result = rm.evaluate_deployment(-0.55, PRICE, _ts(3))
        assert result is None

    def test_deployment_at_trigger_with_persistence(self) -> None:
        rm = _manager(persistence=7)
        rm.add_inflow(1_000.0)
        _record_days(rm, drawdown=-0.55, n_days=8)
        result = rm.evaluate_deployment(-0.55, PRICE, _ts(8))
        assert result is not None
        assert result.tranche_id == 0  # first tranche (-0.50)
        assert result.amount_deployed_eur == pytest.approx(200.0)  # 20% of 1000
        assert rm.current_balance_eur == pytest.approx(800.0)

    def test_persistence_requires_continuous_below_threshold(self) -> None:
        rm = _manager(persistence=7)
        rm.add_inflow(1_000.0)
        # 4 qualifying days, then 1 day above threshold, then 4 more
        _record_days(rm, drawdown=-0.55, n_days=4, start_day=0)
        rm.record_drawdown(_ts(4), -0.40)  # breaks the streak
        _record_days(rm, drawdown=-0.55, n_days=4, start_day=5)
        result = rm.evaluate_deployment(-0.55, PRICE, _ts(9))
        assert result is None  # streak was broken — only 4 consecutive, not 7

    def test_deeper_tranche_takes_priority(self) -> None:
        rm = _manager(persistence=7)
        rm.add_inflow(1_000.0)
        # -0.80 qualifies ALL tranches; deepest (tranche 3, -0.75) should fire
        _record_days(rm, drawdown=-0.80, n_days=8)
        result = rm.evaluate_deployment(-0.80, PRICE, _ts(8))
        assert result is not None
        assert result.tranche_id == 3  # deepest tranche (-0.75, 30%)

    def test_tranche_not_deployed_twice_in_same_cycle(self) -> None:
        rm = _manager(persistence=7)
        rm.add_inflow(1_000.0)
        _record_days(rm, drawdown=-0.55, n_days=8)
        first = rm.evaluate_deployment(-0.55, PRICE, _ts(8))
        assert first is not None
        second = rm.evaluate_deployment(-0.55, PRICE, _ts(9))
        assert second is None

    def test_cycle_reset_re_enables_tranches(self) -> None:
        rm = _manager(persistence=7)
        rm.add_inflow(1_000.0)
        _record_days(rm, drawdown=-0.55, n_days=8)
        first = rm.evaluate_deployment(-0.55, PRICE, _ts(8))
        assert first is not None
        rm.reset_cycle()
        # Re-record persistence after reset
        _record_days(rm, drawdown=-0.55, n_days=8, start_day=9)
        second = rm.evaluate_deployment(-0.55, PRICE, _ts(17))
        assert second is not None
        assert second.tranche_id == 0

    def test_no_deployment_with_empty_reserve(self) -> None:
        rm = _manager(persistence=1)
        # No inflow — balance stays at 0
        rm.record_drawdown(_ts(0), -0.80)
        result = rm.evaluate_deployment(-0.80, PRICE, _ts(1))
        assert result is None

    def test_deployment_record_fields(self) -> None:
        rm = _manager(persistence=1)
        rm.add_inflow(1_000.0)
        rm.record_drawdown(_ts(0), -0.55)
        result = rm.evaluate_deployment(-0.55, 9_500.0, _ts(1))
        assert result is not None
        assert result.btc_price_at_deploy == pytest.approx(9_500.0)
        assert result.drawdown_at_deploy == pytest.approx(-0.55)
        assert result.timestamp_utc == _ts(1)


# ---------------------------------------------------------------------------
# record_drawdown validation
# ---------------------------------------------------------------------------

class TestReserveManagerRecordDrawdown:
    def test_naive_datetime_raises(self) -> None:
        rm = _manager()
        with pytest.raises(ValueError, match="tz-aware"):
            rm.record_drawdown(datetime(2020, 1, 1), -0.10)

    def test_positive_drawdown_raises(self) -> None:
        rm = _manager()
        with pytest.raises(ValueError, match="<= 0"):
            rm.record_drawdown(_ts(0), 0.10)

    def test_history_bounded_by_maxlen(self) -> None:
        rm = ReserveManager(reserve_cap_eur=1000.0, persistence_days=3)
        for i in range(20):
            rm.record_drawdown(_ts(i), -0.10)
        assert len(rm._drawdown_history) == 6  # maxlen = persistence_days * 2


# ---------------------------------------------------------------------------
# Integration: full cycle simulation
# ---------------------------------------------------------------------------

class TestReserveManagerFullCycle:
    def test_full_cycle_simulation(self) -> None:
        """Inflow fills reserve; sequential tranche deployments in a bear cycle."""
        rm = ReserveManager(reserve_cap_eur=1_500.0, persistence_days=3)

        # Fill reserve: 1500 EUR, overflow = 0
        overflow = rm.add_inflow(1_500.0)
        assert overflow == pytest.approx(0.0)
        assert rm.current_balance_eur == pytest.approx(1_500.0)

        # Simulate gradual drawdown reaching each threshold with 4-day persistence
        # Tranche 0 fires at -50%
        _record_days(rm, -0.55, n_days=4)
        result0 = rm.evaluate_deployment(-0.55, 5_000.0, _ts(4))
        assert result0 is not None
        assert result0.tranche_id == 0
        assert rm.current_balance_eur == pytest.approx(1_200.0)  # 1500 - 20%

        # Tranche 1 fires at -60%
        _record_days(rm, -0.65, n_days=4, start_day=5)
        result1 = rm.evaluate_deployment(-0.65, 4_000.0, _ts(9))
        assert result1 is not None
        assert result1.tranche_id == 1
        assert rm.current_balance_eur == pytest.approx(900.0)  # 1200 - 25%

        # After a new ATH, reset the cycle
        rm.reset_cycle()
        assert rm.tranches_deployed == set()
        assert rm.current_balance_eur == pytest.approx(900.0)  # balance unchanged
