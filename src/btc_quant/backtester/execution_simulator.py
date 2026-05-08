"""ExecutionSimulator: V1 simple-cost order execution for backtest."""

from __future__ import annotations

from datetime import timezone
from typing import Any

import pandas as pd

from btc_quant.backtester.models.enums import OrderSide, OrderStatus, OrderType, StrategyTag
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


class ExecutionSimulator:
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
