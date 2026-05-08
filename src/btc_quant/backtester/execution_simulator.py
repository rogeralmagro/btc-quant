"""ExecutionSimulator: order execution for backtest.

Two implementations are provided:

* ``ExecutionSimulatorV1`` (aliased as ``ExecutionSimulator``):
  Fixed fee + slippage rates. Simple, fast, and always correct regardless
  of how much historical data is available. Use this for unit tests, quick
  strategy sketches, and any context where regime data is unavailable.

* ``ExecutionSimulatorV2``:
  Regime-aware cost model. Slippage varies by ``MarketRegime`` (NORMAL →
  0.05 %, VOLATILE → 0.15 %, STRESS → 0.50 %, CASCADE → 1.00 %). Fee
  remains fixed at 0.1 % per side. Requires a ``RegimeDetector`` instance
  and at least ``atr_period`` rows of 1-day history on each call. When
  history is insufficient the executor falls back to NORMAL-regime slippage
  and records this in ``ExecutedOrder.notes``.

The ``BacktestEngine`` detects which version is in use and passes the
appropriate arguments automatically.
"""

from __future__ import annotations

from datetime import timezone
from typing import Any

import pandas as pd

from btc_quant.backtester.models.enums import MarketRegime, OrderSide, OrderStatus, OrderType, StrategyTag
from btc_quant.backtester.models.order import ExecutedOrder, Order, make_order
from btc_quant.backtester.models.position import Position

UTC = timezone.utc


def _bar_ts(bar: pd.Series) -> Any:
    """Extract a tz-aware UTC datetime from a bar Series."""
    ts = bar["timestamp_utc"]
    if hasattr(ts, "to_pydatetime"):
        dt = ts.to_pydatetime()
    else:
        dt = ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class ExecutionSimulatorV1:
    """Simulates order execution in backtest with a simple cost model.

    Execution model
    ===============
    An Order generated at bar close t is executed at the OPEN of bar t+1.
    This models real-world latency: the decision is made at bar close, and
    the trade fires when the next market session opens.

    Cost model V1 (simple, fixed rates)
    ====================================
    - Fees:     ``fee_rate``     × notional  (default 0.1%, Binance Spot)
    - Slippage: ``slippage_rate`` × notional (default 0.05%, conservative)

    A regime-aware cost model (tighter in low-vol, wider in high-vol) is
    planned for Sub-session 3.3 and will replace this simulator.

    Only MARKET orders are supported in V1. LIMIT/STOP raise NotImplementedError.
    """

    def __init__(
        self,
        fee_rate: float = 0.001,       # 0.1 %
        slippage_rate: float = 0.0005, # 0.05 %
    ) -> None:
        if not (0 <= fee_rate <= 0.01):
            raise ValueError(
                f"fee_rate must be in [0, 0.01], got {fee_rate}"
            )
        if not (0 <= slippage_rate <= 0.01):
            raise ValueError(
                f"slippage_rate must be in [0, 0.01], got {slippage_rate}"
            )
        self._fee_rate = fee_rate
        self._slippage_rate = slippage_rate

    # ── Market-order execution ────────────────────────────────────────────────

    def execute(self, order: Order, next_bar: pd.Series) -> ExecutedOrder:
        """Execute a MARKET order at the open of next_bar.

        Args:
            order:    The Order to fill. Must be MARKET type.
            next_bar: Row from an OHLCV DataFrame representing the bar after
                      the order was submitted. Must have columns:
                      ``timestamp_utc``, ``open``.

        Returns:
            ExecutedOrder with slippage-adjusted price, fees, and status FILLED.

        Price model:
        - BUY:  ``exec_price = open × (1 + slippage_rate)``
        - SELL: ``exec_price = open × (1 − slippage_rate)``

        slippage_eur is signed: positive = paid more than open (BUY),
        negative = received less than open (SELL).

        Raises:
            NotImplementedError: for LIMIT, STOP, STOP_LIMIT orders.
        """
        if order.order_type != OrderType.MARKET:
            raise NotImplementedError(
                f"V1 ExecutionSimulator only handles MARKET orders; "
                f"got {order.order_type!r}. LIMIT/STOP support is planned."
            )

        base_price = float(next_bar["open"])

        if order.side == OrderSide.BUY:
            exec_price = base_price * (1.0 + self._slippage_rate)
        else:
            exec_price = base_price * (1.0 - self._slippage_rate)

        notional = exec_price * order.quantity_btc
        fees_eur = notional * self._fee_rate
        slippage_eur = (exec_price - base_price) * order.quantity_btc

        return ExecutedOrder(
            order=order,
            executed_at_utc=_bar_ts(next_bar),
            executed_price=exec_price,
            executed_quantity_btc=order.quantity_btc,
            fees_eur=fees_eur,
            slippage_eur=slippage_eur,
            status=OrderStatus.FILLED,
        )

    # ── Stop / TP detection ──────────────────────────────────────────────────

    def execute_stop_or_tp(
        self,
        position: Position,
        bar: pd.Series,
        tp_or_stop: str,
        level_price: float,
    ) -> ExecutedOrder | None:
        """Detect and fill a stop-loss or take-profit level against a bar.

        For a LONG position:

        **Stop** (``tp_or_stop == "stop"``):
          - Triggered if ``bar['low'] <= level_price``
          - Gap open (``bar['open'] <= level_price``): fills at ``bar['open']``
          - Intra-bar touch: fills at ``level_price``

        **TP** (any other value, e.g. ``"tp1"``, ``"tp2"``):
          - Triggered if ``bar['high'] >= level_price``
          - Gap open (``bar['open'] >= level_price``): fills at ``bar['open']``
          - Intra-bar touch: fills at ``level_price``

        ``slippage_eur`` reflects the gap: ``(exec_price − level_price) × qty``.
        Zero when filled exactly at the level.

        Args:
            position:   Open LONG position being monitored.
            bar:        Current bar (OHLCV). Must have ``open``, ``high``,
                        ``low``, ``timestamp_utc``.
            tp_or_stop: ``"stop"`` for a stop-loss check, anything else for TP.
            level_price: The stop or TP price to check against.

        Returns:
            ExecutedOrder if the level triggered, None otherwise.
        """
        bar_open = float(bar["open"])
        bar_low = float(bar["low"])
        bar_high = float(bar["high"])
        bar_time = _bar_ts(bar)
        qty = position.quantity_btc

        is_stop = tp_or_stop == "stop"

        if is_stop:
            if bar_low > level_price:
                return None
            exec_price = bar_open if bar_open <= level_price else level_price
        else:
            if bar_high < level_price:
                return None
            exec_price = bar_open if bar_open >= level_price else level_price

        notional = exec_price * qty
        fees_eur = notional * self._fee_rate
        slippage_eur = (exec_price - level_price) * qty

        exit_order = make_order(
            strategy_tag=position.strategy_tag,
            timestamp_utc=bar_time,
            symbol=position.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity_btc=qty,
        )

        return ExecutedOrder(
            order=exit_order,
            executed_at_utc=bar_time,
            executed_price=exec_price,
            executed_quantity_btc=qty,
            fees_eur=fees_eur,
            slippage_eur=slippage_eur,
            status=OrderStatus.FILLED,
            notes=f"{tp_or_stop}_triggered",
        )


# Backward-compatibility alias — all existing imports of ``ExecutionSimulator``
# continue to resolve to the V1 class without any code changes.
ExecutionSimulator = ExecutionSimulatorV1


# ---------------------------------------------------------------------------
# ExecutionSimulatorV2 — regime-aware cost model
# ---------------------------------------------------------------------------


class ExecutionSimulatorV2:
    """Regime-aware execution simulator.

    Slippage depends on the market regime detected by a ``RegimeDetector``
    at execution time. Fees are fixed at ``fee_rate`` per side (same as V1).

    Slippage rates per regime
    -------------------------
    NORMAL:   0.05 % (same as V1 default)
    VOLATILE: 0.15 %
    STRESS:   0.50 %
    CASCADE:  1.00 %

    Fallback
    --------
    If the regime detector raises ``ValueError`` (insufficient history for
    ATR computation), V2 silently falls back to NORMAL-regime slippage and
    records ``"regime:normal(fallback)"`` in ``ExecutedOrder.notes``. This
    keeps the engine functional during the warmup period at the start of a
    backtest.

    Decision responsibility
    -----------------------
    The executor models the *cost* of execution. Whether to block entry
    during CASCADE is the *strategy's* responsibility (the strategy queries
    the RegimeDetector independently). The executor never refuses to fill.
    """

    SLIPPAGE_BY_REGIME: dict[MarketRegime, float] = {
        MarketRegime.NORMAL:   0.0005,
        MarketRegime.VOLATILE: 0.0015,
        MarketRegime.STRESS:   0.005,
        MarketRegime.CASCADE:  0.01,
    }

    def __init__(
        self,
        regime_detector: "RegimeDetector",  # type: ignore[name-defined]  # noqa: F821
        fee_rate: float = 0.001,
    ) -> None:
        if not (0 <= fee_rate <= 0.01):
            raise ValueError(f"fee_rate must be in [0, 0.01], got {fee_rate}")
        self._detector = regime_detector
        self._fee_rate = fee_rate

    # ── Market-order execution ────────────────────────────────────────────────

    def execute(
        self,
        order: Order,
        next_bar: pd.Series,
        historical_data_1d: pd.DataFrame,
    ) -> ExecutedOrder:
        """Execute a MARKET order at the open of next_bar with regime-aware slippage.

        Args:
            order:             MARKET order to fill.
            next_bar:          Bar at whose open the order fires (needs ``open``,
                               ``high``, ``low``, ``timestamp_utc``).
            historical_data_1d: Closed 1-day candles visible at execution time.
                               Used by the RegimeDetector. If insufficient
                               (< atr_period), falls back to NORMAL slippage.

        Raises:
            NotImplementedError: for non-MARKET orders.
        """
        if order.order_type != OrderType.MARKET:
            raise NotImplementedError(
                f"V2 ExecutionSimulator only handles MARKET orders; "
                f"got {order.order_type!r}."
            )

        regime, fallback = self._detect_regime(historical_data_1d, next_bar)
        slippage_rate = self.SLIPPAGE_BY_REGIME[regime]
        base_price = float(next_bar["open"])

        if order.side == OrderSide.BUY:
            exec_price = base_price * (1.0 + slippage_rate)
        else:
            exec_price = base_price * (1.0 - slippage_rate)

        notional = exec_price * order.quantity_btc
        fees_eur = notional * self._fee_rate
        slippage_eur = (exec_price - base_price) * order.quantity_btc

        suffix = "(fallback)" if fallback else ""
        return ExecutedOrder(
            order=order,
            executed_at_utc=_bar_ts(next_bar),
            executed_price=exec_price,
            executed_quantity_btc=order.quantity_btc,
            fees_eur=fees_eur,
            slippage_eur=slippage_eur,
            status=OrderStatus.FILLED,
            notes=f"regime:{regime.value}{suffix}",
        )

    # ── Stop / TP detection ──────────────────────────────────────────────────

    def execute_stop_or_tp(
        self,
        position: Position,
        bar: pd.Series,
        tp_or_stop: str,
        level_price: float,
        historical_data_1d: pd.DataFrame,
    ) -> ExecutedOrder | None:
        """Detect and fill a stop-loss or take-profit with regime-aware slippage.

        Trigger logic is identical to V1. After determining the base fill price
        (gap or level), an additional regime-based slippage is applied to model
        market-impact costs (e.g. wider spreads during CASCADE).

        For LONG-position exits (SELL): slippage worsens the fill downward.
        ``slippage_eur = (exec_price − level_price) × qty`` captures both the
        gap (if any) and the regime impact combined.
        """
        bar_open = float(bar["open"])
        bar_low = float(bar["low"])
        bar_high = float(bar["high"])
        bar_time = _bar_ts(bar)
        qty = position.quantity_btc
        is_stop = tp_or_stop == "stop"

        # Trigger detection (same as V1)
        if is_stop:
            if bar_low > level_price:
                return None
            base_fill = bar_open if bar_open <= level_price else level_price
        else:
            if bar_high < level_price:
                return None
            base_fill = bar_open if bar_open >= level_price else level_price

        # Apply regime slippage (SELL → fills lower by slippage_rate)
        regime, fallback = self._detect_regime(historical_data_1d, bar)
        slippage_rate = self.SLIPPAGE_BY_REGIME[regime]
        exec_price = base_fill * (1.0 - slippage_rate)

        notional = exec_price * qty
        fees_eur = notional * self._fee_rate
        slippage_eur = (exec_price - level_price) * qty

        exit_order = make_order(
            strategy_tag=position.strategy_tag,
            timestamp_utc=bar_time,
            symbol=position.symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity_btc=qty,
        )
        suffix = "(fallback)" if fallback else ""
        return ExecutedOrder(
            order=exit_order,
            executed_at_utc=bar_time,
            executed_price=exec_price,
            executed_quantity_btc=qty,
            fees_eur=fees_eur,
            slippage_eur=slippage_eur,
            status=OrderStatus.FILLED,
            notes=f"{tp_or_stop}_triggered regime:{regime.value}{suffix}",
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _detect_regime(
        self,
        historical_data_1d: pd.DataFrame,
        bar: pd.Series,
    ) -> tuple[MarketRegime, bool]:
        """Return (regime, is_fallback). Fallback=True when history is insufficient."""
        try:
            regime = self._detector.detect_for_executor(historical_data_1d, bar)
            return regime, False
        except ValueError:
            return MarketRegime.NORMAL, True
