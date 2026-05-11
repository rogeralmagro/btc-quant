"""DCAModulatedConfig: configuration dataclass for STRAT-06."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DCAModulatedConfig:
    """Immutable configuration for STRAT-06 Drawdown-Modulated DCA.

    Defaults aligned with SYSTEM_FINAL §3.5 and STRATEGY_06 §3.

    Allocation of the monthly inflow:
        baseline_pct  → weekly DCA buys (modulated by drawdown depth)
        buffer_pct    → tactical overlay buffer pool
        reserve_pct   → deep value reserve (deployed only at extreme drawdowns)

    The three allocation percentages must sum to 1.0 (tolerance 1e-9).
    """

    # Monthly inflow:
    monthly_inflow_eur: float

    # Allocation split (must sum to 1.0):
    baseline_pct: float = 0.55
    buffer_pct: float = 0.20
    reserve_pct: float = 0.25

    # Baseline DCA execution schedule:
    baseline_weekday: int = 0       # 0 = Monday (Python weekday())
    baseline_hour_utc: int = 12     # 12:00 UTC

    # Reserve parameters:
    reserve_cap_months: int = 12
    persistence_days_for_tranche: int = 7

    # Operational limits:
    symbol: str = "BTCUSDT"
    min_order_eur: float = 10.0
    max_concentration_pct: float = 0.70

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def reserve_cap_eur(self) -> float:
        """Maximum EUR the reserve pool can hold.

        = reserve_cap_months × monthly_inflow_eur × reserve_pct
        """
        return self.reserve_cap_months * self.monthly_inflow_eur * self.reserve_pct

    @property
    def baseline_per_week_eur(self) -> float:
        """Baseline (unmodulated) DCA amount per weekly execution.

        = monthly_inflow_eur × baseline_pct / 4
        """
        return self.monthly_inflow_eur * self.baseline_pct / 4

    # ── Validation ────────────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if self.monthly_inflow_eur <= 0:
            raise ValueError(
                f"monthly_inflow_eur must be > 0, got {self.monthly_inflow_eur}"
            )

        for name, value in (
            ("baseline_pct", self.baseline_pct),
            ("buffer_pct", self.buffer_pct),
            ("reserve_pct", self.reserve_pct),
        ):
            if not (0 < value <= 1.0):
                raise ValueError(f"{name} must be in (0, 1], got {value}")

        total = self.baseline_pct + self.buffer_pct + self.reserve_pct
        if abs(total - 1.0) >= 1e-9:
            raise ValueError(
                f"baseline_pct + buffer_pct + reserve_pct must sum to 1.0, "
                f"got {total:.10f}"
            )

        if not (0 <= self.baseline_weekday <= 6):
            raise ValueError(
                f"baseline_weekday must be in [0, 6], got {self.baseline_weekday}"
            )

        if not (0 <= self.baseline_hour_utc <= 23):
            raise ValueError(
                f"baseline_hour_utc must be in [0, 23], got {self.baseline_hour_utc}"
            )

        if self.reserve_cap_months < 1:
            raise ValueError(
                f"reserve_cap_months must be >= 1, got {self.reserve_cap_months}"
            )

        if self.persistence_days_for_tranche < 1:
            raise ValueError(
                f"persistence_days_for_tranche must be >= 1, "
                f"got {self.persistence_days_for_tranche}"
            )

        if self.min_order_eur <= 0:
            raise ValueError(
                f"min_order_eur must be > 0, got {self.min_order_eur}"
            )

        if not (0 < self.max_concentration_pct <= 1.0):
            raise ValueError(
                f"max_concentration_pct must be in (0, 1], "
                f"got {self.max_concentration_pct}"
            )
