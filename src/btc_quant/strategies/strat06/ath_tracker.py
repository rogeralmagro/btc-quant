"""ATHTracker: stateful all-time-high tracker for STRAT-06.

Maintains the running maximum price seen across an entire backtest.
The ATH is monotonically non-decreasing: it never falls once set.

Used by STRAT-06 to compute the current drawdown from peak, which in turn
drives the DrawdownMultiplier and ReserveManager deployment logic.
"""

from __future__ import annotations


class ATHTracker:
    """Tracks the all-time-high price of an asset across a backtest.

    State: a single float (_ath). Updated via update(); queried via ath
    property and get_drawdown().

    Usage:
        tracker = ATHTracker()
        tracker.update(price=100)          # ath = 100
        tracker.update(price=80)           # ath = 100 (never falls)
        tracker.update(price=120)          # ath = 120
        tracker.get_drawdown(current_price=90)  # returns -0.25
    """

    def __init__(self, initial_ath: float | None = None) -> None:
        """
        Args:
            initial_ath: known ATH before the backtest window starts.
                         If None, the ATH is initialised on the first update().

        Raises:
            ValueError: if initial_ath is provided but <= 0.
        """
        if initial_ath is not None and initial_ath <= 0:
            raise ValueError(f"initial_ath must be positive, got {initial_ath}")
        self._ath: float | None = initial_ath

    @property
    def ath(self) -> float | None:
        """Current ATH, or None if update() has never been called."""
        return self._ath

    def update(self, price: float) -> None:
        """Update the ATH if price exceeds the current maximum.

        Args:
            price: latest observed price.

        Raises:
            ValueError: if price <= 0.
        """
        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")
        if self._ath is None or price > self._ath:
            self._ath = price

    def get_drawdown(self, current_price: float) -> float:
        """Return the drawdown from ATH as a non-positive fraction.

        Examples:
            ath=100, current=100  → 0.0
            ath=100, current=80   → -0.20
            ath=100, current=50   → -0.50

        Args:
            current_price: price to measure drawdown against.

        Returns:
            Drawdown in [-1, 0]. Returns 0.0 when at ATH.

        Raises:
            ValueError: if current_price <= 0.
            RuntimeError: if update() has never been called (ath is None).
        """
        if current_price <= 0:
            raise ValueError(f"current_price must be positive, got {current_price}")
        if self._ath is None:
            raise RuntimeError(
                "ATHTracker has no ATH yet — call update() at least once before get_drawdown()"
            )
        return (current_price - self._ath) / self._ath
