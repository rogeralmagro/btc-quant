"""ReserveManager: Deep Value Reserve mechanics for STRAT-06.

Implements STRAT-06 §3 Componente 3 (Deep Value Reserve):

ACCUMULATION
    Each inflow adds to the reserve balance until the cap is reached.
    Capital that exceeds the cap (overflow) must be redirected to the
    BUFFER pool by the caller.

DEPLOYMENT
    Four tranches deploy in extreme drawdowns, each requiring the drawdown
    to persist for >= persistence_days consecutive days (anti-flash-crash):
        Tranche 1: -50% sustained ≥7d → 20% of reserve
        Tranche 2: -60% sustained ≥7d → 25% of reserve
        Tranche 3: -70% sustained ≥7d → 25% of reserve
        Tranche 4: -75% sustained ≥7d → 30% of reserve

    Only the deepest qualifying tranche fires per evaluate_deployment() call.
    Once a tranche fires in a cycle it cannot fire again until reset_cycle().

CYCLE RESET
    A new ATH resets the deployed-tranche set, readying the reserve for the
    next bear cycle.

MEMORY EFFICIENCY
    _drawdown_history is capped at persistence_days * 2 entries (a deque with
    maxlen). For the default 7-day persistence this means ≤14 entries in memory
    regardless of backtest length (3100+ days).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class ReserveTranche:
    """A single deployment tranche of the Deep Value Reserve.

    Attributes:
        drawdown_trigger: drawdown threshold that activates this tranche,
                          expressed as a non-positive fraction (e.g. -0.50).
        deploy_pct: fraction of the current reserve balance to deploy (0 < x <= 1).
    """

    drawdown_trigger: float
    deploy_pct: float

    def __post_init__(self) -> None:
        if self.drawdown_trigger > 0:
            raise ValueError(
                f"drawdown_trigger must be <= 0, got {self.drawdown_trigger}"
            )
        if not (0 < self.deploy_pct <= 1.0):
            raise ValueError(
                f"deploy_pct must be in (0, 1], got {self.deploy_pct}"
            )


@dataclass(frozen=True)
class ReserveDeployment:
    """Immutable record of a single tranche deployment event.

    Attributes:
        tranche_id: 0-based index of the tranche in the tranches list.
        timestamp_utc: when the deployment was triggered.
        drawdown_at_deploy: drawdown that triggered deployment.
        amount_deployed_eur: EUR amount removed from the reserve.
        btc_price_at_deploy: BTC price at time of deployment (for audit).
    """

    tranche_id: int
    timestamp_utc: datetime
    drawdown_at_deploy: float
    amount_deployed_eur: float
    btc_price_at_deploy: float


class ReserveManager:
    """Manages the Deep Value Reserve for STRAT-06.

    See module docstring for full mechanics.
    """

    DEFAULT_TRANCHES: list[ReserveTranche] = [
        ReserveTranche(drawdown_trigger=-0.50, deploy_pct=0.20),
        ReserveTranche(drawdown_trigger=-0.60, deploy_pct=0.25),
        ReserveTranche(drawdown_trigger=-0.70, deploy_pct=0.25),
        ReserveTranche(drawdown_trigger=-0.75, deploy_pct=0.30),
    ]

    def __init__(
        self,
        reserve_cap_eur: float,
        tranches: list[ReserveTranche] | None = None,
        persistence_days: int = 7,
    ) -> None:
        """
        Args:
            reserve_cap_eur: maximum EUR balance the reserve can hold.
            tranches: deployment tranches ordered from least to most negative
                drawdown_trigger. Defaults to DEFAULT_TRANCHES.
            persistence_days: consecutive days the drawdown must stay at or
                below a trigger before that tranche can deploy. Must be >= 1.

        Raises:
            ValueError: if reserve_cap_eur <= 0, persistence_days < 1,
                        tranches are not strictly ordered, or their total
                        deploy_pct exceeds 1.0.
        """
        if reserve_cap_eur <= 0:
            raise ValueError(
                f"reserve_cap_eur must be > 0, got {reserve_cap_eur}"
            )
        if persistence_days < 1:
            raise ValueError(
                f"persistence_days must be >= 1, got {persistence_days}"
            )

        resolved = tranches if tranches is not None else self.DEFAULT_TRANCHES

        for i in range(1, len(resolved)):
            if resolved[i].drawdown_trigger >= resolved[i - 1].drawdown_trigger:
                raise ValueError(
                    "Tranches must be ordered from least negative to most negative "
                    f"drawdown_trigger (got {resolved[i-1].drawdown_trigger} then "
                    f"{resolved[i].drawdown_trigger})"
                )

        total_pct = sum(t.deploy_pct for t in resolved)
        if total_pct > 1.0:
            raise ValueError(
                f"Sum of deploy_pct across all tranches must be <= 1.0, got {total_pct:.4f}"
            )

        self._reserve_cap_eur = reserve_cap_eur
        self._tranches = resolved
        self._persistence_days = persistence_days

        self._current_balance_eur: float = 0.0
        self._tranches_deployed_this_cycle: set[int] = set()
        # Bounded history: only keeps the last persistence_days * 2 entries.
        self._drawdown_history: deque[tuple[datetime, float]] = deque(
            maxlen=persistence_days * 2
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def current_balance_eur(self) -> float:
        """Current EUR held in the reserve."""
        return self._current_balance_eur

    @property
    def reserve_cap_eur(self) -> float:
        """Maximum reserve capacity in EUR."""
        return self._reserve_cap_eur

    @property
    def tranches_deployed(self) -> set[int]:
        """Indices of tranches already deployed in the current cycle (copy)."""
        return self._tranches_deployed_this_cycle.copy()

    # ── Accumulation ──────────────────────────────────────────────────────────

    def add_inflow(self, amount_eur: float) -> float:
        """Add capital to the reserve, up to the cap.

        Args:
            amount_eur: EUR to deposit. Must be > 0.

        Returns:
            Overflow EUR that did not fit (0.0 if reserve was not full).
            Caller must redirect overflow to the BUFFER pool.

        Raises:
            ValueError: if amount_eur <= 0.
        """
        if amount_eur <= 0:
            raise ValueError(f"amount_eur must be > 0, got {amount_eur}")

        capacity_remaining = self._reserve_cap_eur - self._current_balance_eur
        deposited = min(amount_eur, capacity_remaining)
        overflow = amount_eur - deposited
        self._current_balance_eur += deposited
        return overflow

    # ── Drawdown tracking ─────────────────────────────────────────────────────

    def record_drawdown(self, timestamp_utc: datetime, drawdown: float) -> None:
        """Record the current drawdown for persistence tracking.

        The history is automatically trimmed by the deque's maxlen so memory
        usage stays bounded regardless of backtest length.

        Args:
            timestamp_utc: UTC timestamp of the observation.
            drawdown: current drawdown as a non-positive fraction.

        Raises:
            ValueError: if drawdown > 0 or timestamp is naive.
        """
        if timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be tz-aware")
        if drawdown > 0:
            raise ValueError(f"drawdown must be <= 0, got {drawdown}")
        self._drawdown_history.append((timestamp_utc, drawdown))

    # ── Cycle management ──────────────────────────────────────────────────────

    def reset_cycle(self) -> None:
        """Reset deployed-tranche tracking for a new bear cycle.

        Call this when a new ATH is reached. Does NOT clear the reserve
        balance — accumulated capital persists across cycles.
        """
        self._tranches_deployed_this_cycle.clear()

    # ── Deployment evaluation ─────────────────────────────────────────────────

    def evaluate_deployment(
        self,
        current_drawdown: float,
        current_btc_price: float,
        current_timestamp: datetime,
    ) -> ReserveDeployment | None:
        """Evaluate whether a tranche should deploy now.

        Checks tranches from most negative to least negative (deepest first)
        so that the deepest qualifying tranche takes priority if multiple
        qualify simultaneously.

        Persistence check: scans _drawdown_history for a continuous run of
        >= persistence_days entries where drawdown <= tranche.drawdown_trigger
        ending at the current observation.

        Args:
            current_drawdown: current drawdown (non-positive fraction).
            current_btc_price: BTC price now (used only for the audit record).
            current_timestamp: UTC timestamp of this evaluation.

        Returns:
            ReserveDeployment if a tranche fires, None otherwise.

        Raises:
            ValueError: if current_drawdown > 0 or current_btc_price <= 0.
        """
        if current_drawdown > 0:
            raise ValueError(
                f"current_drawdown must be <= 0, got {current_drawdown}"
            )
        if current_btc_price <= 0:
            raise ValueError(
                f"current_btc_price must be > 0, got {current_btc_price}"
            )
        if self._current_balance_eur <= 0:
            return None

        # Check from deepest to least negative so the most extreme fires first.
        for i in reversed(range(len(self._tranches))):
            tranche = self._tranches[i]

            if i in self._tranches_deployed_this_cycle:
                continue

            if current_drawdown > tranche.drawdown_trigger:
                continue  # drawdown hasn't reached this tranche

            if not self._has_persistence(tranche.drawdown_trigger):
                continue

            # This tranche qualifies — deploy it.
            amount = self._current_balance_eur * tranche.deploy_pct
            self._current_balance_eur -= amount
            self._tranches_deployed_this_cycle.add(i)

            return ReserveDeployment(
                tranche_id=i,
                timestamp_utc=current_timestamp,
                drawdown_at_deploy=current_drawdown,
                amount_deployed_eur=amount,
                btc_price_at_deploy=current_btc_price,
            )

        return None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _has_persistence(self, trigger: float) -> bool:
        """Return True if the drawdown has been <= trigger for persistence_days days.

        Counts consecutive qualifying entries from the END of the history
        (most recent first). Stops as soon as an entry fails the condition.
        """
        consecutive = 0
        for _ts, dd in reversed(self._drawdown_history):
            if dd <= trigger:
                consecutive += 1
            else:
                break
            if consecutive >= self._persistence_days:
                return True
        return False
