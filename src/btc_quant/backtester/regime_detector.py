"""RegimeDetector: ATR-based market regime classification for cost modelling."""

from __future__ import annotations

import pandas as pd

from btc_quant.backtester.models.enums import MarketRegime
from btc_quant.indicators.volatility import atr


class RegimeDetector:
    """Classifies market regime based on realised ATR volatility.

    Regime definitions (aligned with SYSTEM_FINAL §2.2)
    ====================================================
    NORMAL:   ATR(period) <  volatile_multiplier × annual_avg_ATR
    VOLATILE: volatile_multiplier × avg ≤ ATR < stress_multiplier × avg
    STRESS:   ATR(period) >= stress_multiplier × annual_avg_ATR
    CASCADE:  intra-bar move > cascade_threshold of bar open
              (overrides NORMAL/VOLATILE/STRESS — checked first)

    NORMAL/VOLATILE/STRESS are computed on closed candles only.
    CASCADE is computed against ``current_bar`` (which may be forming) so a
    strategy can block new entries while the bar is still in progress.

    Edge case — minimal history
    ===========================
    With exactly ``atr_period`` closed candles: Wilder's ATR produces one
    valid value (the seed average). The annual average equals that single
    value, ratio == 1.0 → NORMAL. This is the expected behaviour; callers
    must ensure at least ``atr_period`` rows before calling ``detect()``.

    Determinism
    ===========
    Stateless: same input always produces the same regime.
    """

    def __init__(
        self,
        atr_period: int = 14,
        annual_lookback_days: int = 365,
        volatile_multiplier: float = 1.5,
        stress_multiplier: float = 3.0,
        cascade_threshold: float = 0.10,
    ) -> None:
        if atr_period < 1:
            raise ValueError(f"atr_period must be >= 1, got {atr_period}")
        if annual_lookback_days < 1:
            raise ValueError(
                f"annual_lookback_days must be >= 1, got {annual_lookback_days}"
            )
        if volatile_multiplier <= 0:
            raise ValueError(
                f"volatile_multiplier must be > 0, got {volatile_multiplier}"
            )
        if stress_multiplier <= volatile_multiplier:
            raise ValueError(
                f"stress_multiplier ({stress_multiplier}) must be strictly greater "
                f"than volatile_multiplier ({volatile_multiplier})"
            )
        if cascade_threshold <= 0:
            raise ValueError(
                f"cascade_threshold must be > 0, got {cascade_threshold}"
            )

        self._atr_period = atr_period
        self._annual_lookback_days = annual_lookback_days
        self._volatile_mult = volatile_multiplier
        self._stress_mult = stress_multiplier
        self._cascade_threshold = cascade_threshold

    # ── Public API ───────────────────────────────────────────────────────────

    def detect(
        self,
        historical_data_1d: pd.DataFrame,
        current_bar: pd.Series | None = None,
    ) -> MarketRegime:
        """Detect the current market regime.

        Args:
            historical_data_1d: DataFrame of closed 1-day candles up to now.
                Must contain ``high``, ``low``, ``close`` columns and at least
                ``atr_period`` rows. More rows → more accurate annual average.
            current_bar: Optional Series with ``open``, ``high``, ``low`` of the
                bar being evaluated (may still be forming). If present and the
                intra-bar range exceeds ``cascade_threshold``, returns CASCADE
                immediately without inspecting historical data.

        Returns:
            MarketRegime enum member.

        Raises:
            ValueError: if ``len(historical_data_1d) < atr_period``.
        """
        # 1. CASCADE fast-path — does not require ATR history
        if current_bar is not None:
            bar_open = float(current_bar["open"])
            if bar_open > 0:
                intrabar_move = (
                    float(current_bar["high"]) - float(current_bar["low"])
                ) / bar_open
                if intrabar_move > self._cascade_threshold:
                    return MarketRegime.CASCADE

        # 2. Validate history length
        if len(historical_data_1d) < self._atr_period:
            raise ValueError(
                f"historical_data_1d has {len(historical_data_1d)} rows, "
                f"but atr_period={self._atr_period} requires at least that many. "
                "Provide more historical data or reduce atr_period."
            )

        # 3. Compute ATR (Wilder's smoothing)
        atr_series = atr(
            high=historical_data_1d["high"],
            low=historical_data_1d["low"],
            close=historical_data_1d["close"],
            period=self._atr_period,
        )
        valid_atr = atr_series.dropna()

        # 4. Current ATR and annual average
        current_atr = float(valid_atr.iloc[-1])
        annual_avg = float(valid_atr.tail(self._annual_lookback_days).mean())

        # 5. Classify
        return self._classify(current_atr, annual_avg)

    def detect_for_executor(
        self,
        historical_data_1d: pd.DataFrame,
        current_bar: pd.Series,
    ) -> MarketRegime:
        """Regime detection for ExecutionSimulatorV2.

        Equivalent to ``detect(historical_data_1d, current_bar)``. The separate
        name documents the intended call-site so it is obvious that the executor
        always supplies both historical data and the bar being executed.
        """
        return self.detect(historical_data_1d, current_bar)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _classify(self, current_atr: float, annual_avg: float) -> MarketRegime:
        """Map a current/annual ATR pair to a regime.

        Boundaries:
        - VOLATILE lower bound (volatile_multiplier): inclusive.
        - STRESS lower bound (stress_multiplier): inclusive.
        - Below volatile_multiplier: NORMAL.
        """
        ratio = current_atr / annual_avg
        if ratio >= self._stress_mult:
            return MarketRegime.STRESS
        if ratio >= self._volatile_mult:
            return MarketRegime.VOLATILE
        return MarketRegime.NORMAL
