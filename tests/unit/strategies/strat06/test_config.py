"""Unit tests for DCAModulatedConfig."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from btc_quant.strategies.strat06.config import BaselineExecutionMode, DCAModulatedConfig


def _cfg(**overrides) -> DCAModulatedConfig:
    defaults = dict(monthly_inflow_eur=500.0)
    defaults.update(overrides)
    return DCAModulatedConfig(**defaults)


class TestDCAModulatedConfigValid:
    def test_default_config_valid(self) -> None:
        cfg = _cfg()
        assert cfg.monthly_inflow_eur == 500.0
        assert cfg.baseline_pct == 0.55
        assert cfg.buffer_pct == 0.20
        assert cfg.reserve_pct == 0.25
        assert cfg.baseline_execution_mode == BaselineExecutionMode.WEEKLY
        assert cfg.baseline_weekday == 0
        assert cfg.baseline_hour_utc == 12
        assert cfg.reserve_cap_months == 12
        assert cfg.persistence_days_for_tranche == 7
        assert cfg.symbol == "BTCUSDT"
        assert cfg.min_order_eur == 10.0
        assert cfg.max_concentration_pct == 0.70

    def test_reserve_cap_eur_calculation(self) -> None:
        # monthly=500, reserve_pct=0.25, cap_months=12 → 500 * 0.25 * 12 = 1500
        cfg = _cfg(monthly_inflow_eur=500.0, reserve_pct=0.25, reserve_cap_months=12)
        assert cfg.reserve_cap_eur == pytest.approx(1500.0)

    def test_baseline_per_week_calculation(self) -> None:
        # monthly=500, baseline_pct=0.55 → 500 * 0.55 / 4 = 68.75
        cfg = _cfg(monthly_inflow_eur=500.0, baseline_pct=0.55)
        assert cfg.baseline_per_week_eur == pytest.approx(68.75)

    def test_immutability(self) -> None:
        cfg = _cfg()
        with pytest.raises(FrozenInstanceError):
            cfg.monthly_inflow_eur = 999.0  # type: ignore[misc]


class TestDCAModulatedConfigValidation:
    def test_negative_inflow_raises(self) -> None:
        with pytest.raises(ValueError, match="monthly_inflow_eur"):
            _cfg(monthly_inflow_eur=-100.0)

    def test_zero_inflow_raises(self) -> None:
        with pytest.raises(ValueError, match="monthly_inflow_eur"):
            _cfg(monthly_inflow_eur=0.0)

    def test_pct_greater_than_one_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_pct"):
            DCAModulatedConfig(
                monthly_inflow_eur=500.0,
                baseline_pct=1.1,
                buffer_pct=0.20,
                reserve_pct=0.25,
            )

    def test_pcts_not_summing_to_one_raises(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            DCAModulatedConfig(
                monthly_inflow_eur=500.0,
                baseline_pct=0.50,
                buffer_pct=0.20,
                reserve_pct=0.25,
            )

    def test_pcts_must_sum_to_one(self) -> None:
        # Exact sum within 1e-9 tolerance should not raise.
        cfg = DCAModulatedConfig(
            monthly_inflow_eur=500.0,
            baseline_pct=0.55,
            buffer_pct=0.20,
            reserve_pct=0.25,
        )
        assert abs(cfg.baseline_pct + cfg.buffer_pct + cfg.reserve_pct - 1.0) < 1e-9

    def test_invalid_weekday_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_weekday"):
            _cfg(baseline_weekday=-1)
        with pytest.raises(ValueError, match="baseline_weekday"):
            _cfg(baseline_weekday=7)

    def test_invalid_hour_raises(self) -> None:
        with pytest.raises(ValueError, match="baseline_hour_utc"):
            _cfg(baseline_hour_utc=-1)
        with pytest.raises(ValueError, match="baseline_hour_utc"):
            _cfg(baseline_hour_utc=24)

    def test_min_order_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="min_order_eur"):
            _cfg(min_order_eur=-1.0)

    def test_min_order_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="min_order_eur"):
            _cfg(min_order_eur=0.0)

    def test_concentration_pct_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_concentration_pct"):
            _cfg(max_concentration_pct=0.0)

    def test_concentration_pct_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="max_concentration_pct"):
            _cfg(max_concentration_pct=1.1)
