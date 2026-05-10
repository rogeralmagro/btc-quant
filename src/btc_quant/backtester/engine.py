"""BacktestEngine: event-driven multi-strategy backtest loop."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from btc_quant.backtester.execution_simulator import ExecutionSimulator, ExecutionSimulatorV2
from btc_quant.backtester.inflow_scheduler import CapitalInflow
from btc_quant.backtester.market_context import MarketContext
from btc_quant.backtester.models.enums import OrderSide, OrderType, StrategyTag
from btc_quant.backtester.models.order import ExecutedOrder, make_order
from btc_quant.backtester.models.pool import CapitalPool
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.backtester.models.position import Position, TakeProfitLevel
from btc_quant.backtester.models.result import BacktestResult
from btc_quant.backtester.models.signal import Signal
from btc_quant.backtester.models.trade import Trade
from btc_quant.backtester.strategy_base import StrategyBase

UTC = timezone.utc

# Supported timeframe durations (must match market_context._TF_DURATIONS).
_TF_DURATIONS: dict[str, timedelta] = {
    "1m":  timedelta(minutes=1),
    "3m":  timedelta(minutes=3),
    "5m":  timedelta(minutes=5),
    "15m": timedelta(minutes=15),
    "30m": timedelta(minutes=30),
    "1h":  timedelta(hours=1),
    "2h":  timedelta(hours=2),
    "4h":  timedelta(hours=4),
    "6h":  timedelta(hours=6),
    "8h":  timedelta(hours=8),
    "12h": timedelta(hours=12),
    "1d":  timedelta(days=1),
    "3d":  timedelta(days=3),
    "1w":  timedelta(weeks=1),
}


def _to_utc_datetime(ts: Any) -> datetime:
    """Coerce a pandas Timestamp or datetime to a tz-aware UTC datetime."""
    if hasattr(ts, "to_pydatetime"):
        dt = ts.to_pydatetime()
    else:
        dt = ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


class BacktestEngine:
    """Event-driven backtest engine for multiple simultaneous strategies.

    Execution model (no look-ahead)
    ================================
    Per bar t (iteration TF):

    1. ``context.advance_to(bar_t.close_time_utc)`` — data up to bar t is now visible.
    2. Execute pending orders from bar_{t-1} at ``bar_t.open``.
    3. Check stop-loss / take-profit for all open positions
       (positions opened in step 2 are skipped for this bar).
    4. Strategies aligned to bar t's close time call ``generate_signals``.
    5. Snapshot portfolio at ``bar_t.close_time_utc``.

    Pending orders from the final bar are discarded (no next bar to execute them).

    Capital mutations
    =================
    BUY:  ``pool.remove_cash(notional + fees)``, ``pool.add_btc(qty, notional)``
    SELL: ``pool.remove_btc(qty, notional)``, ``pool.cash_eur += notional - fees``

    SELL uses a direct credit (bypassing ``add_cash``) so that profit recycling
    is not double-counted in ``total_invested_eur``.

    Profit recycling
    ================
    When a trade closes with ``net_pnl_eur > 0`` and ``profit_recycle_pct > 0``,
    the engine transfers that fraction of net P&L from the winning strategy pool
    to ``StrategyTag.STRAT_06_BASELINE``.

    Raises ValueError at construction if ``profit_recycle_pct > 0`` but no
    STRAT_06_BASELINE pool is registered (fail-loud). Pass
    ``profit_recycle_pct=0.0`` to explicitly disable recycling.
    """

    def __init__(
        self,
        strategies: list[StrategyBase],
        data: dict[str, pd.DataFrame],
        initial_capital_per_strategy: dict[StrategyTag, float],
        execution_simulator: ExecutionSimulator,
        profit_recycle_pct: float = 0.5,
        symbol: str = "BTCUSDT",
        inflow_schedule: list[CapitalInflow] | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("strategies must not be empty")
        if not (0.0 <= profit_recycle_pct <= 1.0):
            raise ValueError(
                f"profit_recycle_pct must be in [0, 1], got {profit_recycle_pct}"
            )

        # Validate all required TFs are present
        for strategy in strategies:
            for tf in strategy.required_timeframes():
                if tf not in data:
                    raise ValueError(
                        f"Strategy {strategy!r} requires timeframe {tf!r} "
                        f"but it is not in data. Available: {sorted(data)}"
                    )

        # Validate capital
        strategy_tags = [s.strategy_tag for s in strategies]
        for tag in strategy_tags:
            if tag not in initial_capital_per_strategy:
                raise ValueError(
                    f"No initial capital specified for strategy {tag!r}"
                )
            if initial_capital_per_strategy[tag] < 0:
                raise ValueError(
                    f"initial_capital for {tag!r} must be >= 0, "
                    f"got {initial_capital_per_strategy[tag]}"
                )

        # Validate profit recycling target exists when recycling is enabled
        recycle_target = StrategyTag.STRAT_06_BASELINE
        if profit_recycle_pct > 0 and recycle_target not in strategy_tags:
            raise ValueError(
                f"Profit recycling configured (profit_recycle_pct={profit_recycle_pct}) "
                f"but target pool {recycle_target.value!r} is not registered. "
                "Pass profit_recycle_pct=0.0 to disable recycling."
            )

        # V2 simulator requires 1d data for regime detection
        if isinstance(execution_simulator, ExecutionSimulatorV2) and "1d" not in data:
            raise ValueError(
                "ExecutionSimulatorV2 requires '1d' timeframe data for regime detection. "
                "Add '1d' to the data dict or switch to ExecutionSimulatorV1."
            )

        self._strategies = strategies
        self._strategy_by_tag: dict[StrategyTag, StrategyBase] = {
            s.strategy_tag: s for s in strategies
        }
        self._data = data
        self._simulator = execution_simulator
        self._profit_recycle_pct = profit_recycle_pct
        self._symbol = symbol

        # Build portfolio
        pools = {
            tag: CapitalPool(
                strategy_tag=tag,
                cash_eur=initial_capital_per_strategy[tag],
                total_invested_eur=initial_capital_per_strategy[tag],
            )
            for tag in strategy_tags
        }
        self._portfolio = Portfolio(pools=pools)
        self._initial_capital = sum(initial_capital_per_strategy[t] for t in strategy_tags)

        # Inflow schedule: sorted ascending so we can pop from the front cheaply.
        # Validate all targets have registered pools.
        if inflow_schedule:
            for inflow in inflow_schedule:
                if inflow.target_strategy not in self._portfolio.pools:
                    raise ValueError(
                        f"Inflow targets strategy {inflow.target_strategy!r} "
                        "which has no registered pool. Add it to strategies and "
                        "initial_capital_per_strategy."
                    )
        self._pending_inflows: list[CapitalInflow] = sorted(
            inflow_schedule or [], key=lambda x: x.timestamp_utc
        )
        self._processed_inflows: list[CapitalInflow] = []

        # Precompute iteration TF and alignment sets
        self._iteration_tf = self._determine_iteration_tf()
        self._aligned_close_times: dict[str, set[datetime]] = {
            tf: {_to_utc_datetime(ts) for ts in df["close_time_utc"]}
            for tf, df in data.items()
        }

        # Build initial MarketContext (time = first bar's open, before any close)
        iteration_df = self._data[self._iteration_tf]
        first_open = _to_utc_datetime(iteration_df.iloc[0]["timestamp_utc"])
        self._market_context = MarketContext(data, first_open, symbol=symbol)

        self._equity_curve = []

    # ── Internals ────────────────────────────────────────────────────────────

    def _determine_iteration_tf(self) -> str:
        """Return the finest TF required across all strategies."""
        required: set[str] = set()
        for strategy in self._strategies:
            required.update(strategy.required_timeframes())
        return min(required, key=lambda tf: _TF_DURATIONS[tf])

    def _is_strategy_aligned(self, strategy: StrategyBase, bar_close_time: datetime) -> bool:
        tf_set = self._aligned_close_times.get(strategy.primary_timeframe(), set())
        return bar_close_time in tf_set

    # ── V1 / V2 dispatch helpers ─────────────────────────────────────────────

    def _is_v2(self) -> bool:
        return isinstance(self._simulator, ExecutionSimulatorV2)

    def _get_visible_1d_history(self) -> pd.DataFrame:
        """Return all 1d bars visible to MarketContext right now (for V2 regime detection)."""
        n = len(self._data["1d"])
        return self._market_context.get_history("1d", lookback=n)

    def _slippage_rate_for_sizing(self, bar: pd.Series, side: OrderSide) -> float:
        """Return the slippage rate to use when converting quote_amount_eur → qty.

        For V2 the regime is detected from current visible history so the sizing
        estimate uses the same rate as the actual execution — no overdraft risk.
        """
        if self._is_v2():
            from btc_quant.backtester.models.enums import MarketRegime
            try:
                history = self._get_visible_1d_history()
                regime = self._simulator._detector.detect_for_executor(history, bar)  # type: ignore[union-attr]
            except ValueError:
                regime = MarketRegime.NORMAL
            return ExecutionSimulatorV2.SLIPPAGE_BY_REGIME[regime]
        return self._simulator._slippage_rate  # type: ignore[union-attr]

    def _sim_execute(self, order: "Order", bar: pd.Series) -> "ExecutedOrder":
        if self._is_v2():
            return self._simulator.execute(order, bar, self._get_visible_1d_history())  # type: ignore[union-attr]
        return self._simulator.execute(order, bar)

    def _sim_execute_stop_tp(
        self,
        pos: Position,
        bar: pd.Series,
        label: str,
        price: float,
    ) -> "ExecutedOrder | None":
        if self._is_v2():
            return self._simulator.execute_stop_or_tp(  # type: ignore[union-attr]
                pos, bar, label, price, self._get_visible_1d_history()
            )
        return self._simulator.execute_stop_or_tp(pos, bar, label, price)

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self) -> BacktestResult:
        """Execute the backtest and return the full result."""
        iteration_df = self._data[self._iteration_tf]
        pending_signals: list[Signal] = []

        start_time: datetime | None = None
        end_time: datetime | None = None

        for _, bar in iteration_df.iterrows():
            bar_close_time = _to_utc_datetime(bar["close_time_utc"])

            if start_time is None:
                start_time = _to_utc_datetime(bar["timestamp_utc"])

            # 1. Advance market context
            self._market_context.advance_to(bar_close_time)

            # 1b. Process inflows whose timestamp has arrived (≤ bar close time).
            #     Capital is added here so strategies can see it when generating
            #     signals in step 4, and trades execute at the next bar's open.
            while self._pending_inflows and self._pending_inflows[0].timestamp_utc <= bar_close_time:
                inflow = self._pending_inflows.pop(0)
                pool = self._portfolio.get_pool(inflow.target_strategy)
                pool.add_cash(inflow.amount_eur, source="inflow")
                self._processed_inflows.append(inflow)

            # 2. Execute pending signals from previous bar at this bar's open
            newly_opened: set[UUID] = set()
            for signal in pending_signals:
                pos_id = self._execute_signal(signal, bar)
                if pos_id is not None:
                    newly_opened.add(pos_id)
            pending_signals = []

            # 3. Check stops / TPs (skip positions opened in this step)
            self._check_stops_and_tps(bar, skip_positions=newly_opened)

            # 4. Generate signals from aligned strategies
            for strategy in self._strategies:
                if self._is_strategy_aligned(strategy, bar_close_time):
                    signals = strategy.generate_signals(
                        self._market_context, self._portfolio
                    )
                    pending_signals.extend(signals)

            # 5. Snapshot
            btc_price = float(bar["close"])
            self._equity_curve.append(
                self._portfolio.snapshot(bar_close_time, btc_price)
            )
            end_time = bar_close_time

        # pending_signals from the final bar are discarded (no next bar)

        if start_time is None or end_time is None:
            raise ValueError("data is empty — cannot run backtest")

        return self._build_result(start_time, end_time)

    # ── Signal execution ─────────────────────────────────────────────────────

    def _execute_signal(self, signal: Signal, bar: pd.Series) -> UUID | None:
        """Execute a pending signal at bar.open. Returns position_id for BUYs."""
        pool = self._portfolio.get_pool(signal.strategy_tag)

        # Convert quote_amount_eur → quantity_btc at execution time
        if signal.quantity_btc is not None:
            qty = signal.quantity_btc
        else:
            open_price = float(bar["open"])
            slip_rate = self._slippage_rate_for_sizing(bar, signal.side)
            fee_rate = self._simulator._fee_rate  # type: ignore[union-attr]
            if signal.side == OrderSide.BUY:
                est_exec = open_price * (1.0 + slip_rate)
            else:
                est_exec = open_price * (1.0 - slip_rate)
            if est_exec <= 0:
                return None
            # SAFETY_FACTOR guards against float64 rounding causing notional+fees
            # to exceed available cash by a sub-cent amount (e.g. 10000.0000000001).
            _SAFETY_FACTOR = 0.999
            safe_quote = min(
                signal.quote_amount_eur,  # type: ignore[operator]
                pool.cash_eur * _SAFETY_FACTOR,
            )
            qty = safe_quote / (est_exec * (1.0 + fee_rate))

        if qty <= 0:
            return None

        order = make_order(
            strategy_tag=signal.strategy_tag,
            timestamp_utc=signal.timestamp_utc,
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.MARKET,
            quantity_btc=qty,
            parent_signal_id=signal.signal_id,
        )
        executed = self._sim_execute(order, bar)

        if signal.side == OrderSide.BUY:
            return self._apply_buy(signal, executed, pool)
        else:
            self._apply_sell(signal, executed, pool)
            return None

    def _apply_buy(
        self,
        signal: Signal,
        executed: ExecutedOrder,
        pool: CapitalPool,
    ) -> UUID:
        exec_price = executed.executed_price
        qty = executed.executed_quantity_btc
        fees_eur = executed.fees_eur
        notional = exec_price * qty

        # Find existing open position for this strategy (if any)
        existing_pos: Position | None = None
        for pos in self._portfolio.open_positions.values():
            if pos.strategy_tag == signal.strategy_tag:
                existing_pos = pos
                break

        if existing_pos is not None:
            strategy = self._strategy_by_tag[signal.strategy_tag]
            if not strategy.allows_position_accumulation():
                raise ValueError(
                    f"Strategy {signal.strategy_tag.value!r} generated a BUY while "
                    "a position is already open. Override "
                    "allows_position_accumulation() to return True to enable DCA, "
                    "or ensure the position is closed before buying again."
                )
            pool.remove_cash(notional + fees_eur, reason="buy_accumulate")
            pool.add_btc(qty=qty, cost_eur=notional)
            existing_pos.add_to_position(executed)
            return existing_pos.position_id

        # New position
        pool.remove_cash(notional + fees_eur, reason="buy")
        pool.add_btc(qty=qty, cost_eur=notional)

        tp_levels: list[TakeProfitLevel] = []
        if signal.take_profit_prices and signal.take_profit_quantities_pct:
            for price, pct in zip(signal.take_profit_prices, signal.take_profit_quantities_pct):
                tp_levels.append(TakeProfitLevel(price=price, quantity_pct=pct))

        pos = Position(
            position_id=uuid4(),
            strategy_tag=signal.strategy_tag,
            symbol=signal.symbol,
            side=OrderSide.BUY,
            opened_at_utc=executed.executed_at_utc,
            quantity_btc=qty,
            entry_price_avg=exec_price,
            fees_paid_eur=fees_eur,
            stop_loss_price=signal.stop_loss_price,
            take_profit_levels=tp_levels,
            entry_orders=[executed],
        )
        self._portfolio.open_position(pos)
        return pos.position_id

    def _apply_sell(
        self,
        signal: Signal,
        executed: ExecutedOrder,
        pool: CapitalPool,
    ) -> None:
        exec_price = executed.executed_price
        qty = executed.executed_quantity_btc
        fees_eur = executed.fees_eur
        notional = exec_price * qty

        existing_pos: Position | None = None
        for pos in self._portfolio.open_positions.values():
            if pos.strategy_tag == signal.strategy_tag:
                existing_pos = pos
                break

        if existing_pos is None:
            return  # no open position to close; ignore signal

        pool.remove_btc(qty=qty, revenue_eur=notional)
        pool.cash_eur += notional - fees_eur  # direct credit; bypasses add_cash

        exit_reason = signal.reason_tag or "signal_sell"
        trade = self._portfolio.close_position(
            existing_pos.position_id, [executed], exit_reason
        )
        self._apply_profit_recycling(trade)

    # ── Stop / TP management ─────────────────────────────────────────────────

    def _check_stops_and_tps(
        self, bar: pd.Series, skip_positions: set[UUID]
    ) -> None:
        for pos_id, pos in list(self._portfolio.open_positions.items()):
            if pos_id in skip_positions:
                continue

            pool = self._portfolio.get_pool(pos.strategy_tag)

            # Stop-loss check (closes entire remaining position)
            if pos.stop_loss_price is not None:
                result = self._sim_execute_stop_tp(pos, bar, "stop", pos.stop_loss_price)
                if result is not None:
                    self._close_position_from_exit(pos, result, pool, "stop_loss_hit")
                    continue  # position is gone; skip TP check

            # Take-profit checks (each TP closes the entire remaining position)
            for i, tp in enumerate(pos.take_profit_levels):
                if tp.executed:
                    continue
                label = f"tp{i + 1}"
                result = self._sim_execute_stop_tp(pos, bar, label, tp.price)
                if result is not None:
                    self._close_position_from_exit(pos, result, pool, f"{label}_hit")
                    break  # position closed; stop iterating TPs

    def _close_position_from_exit(
        self,
        pos: Position,
        executed: ExecutedOrder,
        pool: CapitalPool,
        exit_reason: str,
    ) -> None:
        exec_price = executed.executed_price
        qty = executed.executed_quantity_btc
        fees_eur = executed.fees_eur
        notional = exec_price * qty

        pool.remove_btc(qty=qty, revenue_eur=notional)
        pool.cash_eur += notional - fees_eur  # direct credit

        trade = self._portfolio.close_position(pos.position_id, [executed], exit_reason)
        self._apply_profit_recycling(trade)

    # ── Profit recycling ─────────────────────────────────────────────────────

    def _apply_profit_recycling(self, trade: Trade) -> None:
        if self._profit_recycle_pct == 0 or trade.net_pnl_eur <= 0:
            return
        amount = trade.net_pnl_eur * self._profit_recycle_pct
        self._portfolio.transfer_cash(
            from_tag=trade.strategy_tag,
            to_tag=StrategyTag.STRAT_06_BASELINE,
            amount=amount,
            reason="profit_recycling",
        )

    # ── Result construction ──────────────────────────────────────────────────

    def _build_result(self, start_time: datetime, end_time: datetime) -> BacktestResult:
        closed_trades = list(self._portfolio.closed_trades)
        trades_by_strategy: dict[StrategyTag, list[Trade]] = {
            s.strategy_tag: [] for s in self._strategies
        }
        for trade in closed_trades:
            trades_by_strategy.setdefault(trade.strategy_tag, []).append(trade)

        return BacktestResult(
            start_date_utc=start_time,
            end_date_utc=end_time,
            initial_capital_eur=self._initial_capital,
            strategies_used=[s.strategy_tag for s in self._strategies],
            final_portfolio=self._portfolio,
            equity_curve=list(self._equity_curve),
            closed_trades=closed_trades,
            trades_by_strategy=trades_by_strategy,
            processed_inflows=list(self._processed_inflows),
            backtest_metadata={
                "iteration_tf": self._iteration_tf,
                "symbol": self._symbol,
                "profit_recycle_pct": self._profit_recycle_pct,
            },
        )
