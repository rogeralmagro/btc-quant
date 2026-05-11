"""DCAModulatedStrategy: STRAT-06 Drawdown-Modulated DCA + Deep Value Reserve."""

from __future__ import annotations

from datetime import datetime

from btc_quant.backtester.market_context import MarketContext
from btc_quant.backtester.models.enums import OrderSide, StrategyTag
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.backtester.models.signal import Signal
from btc_quant.backtester.strategy_base import StrategyBase
from btc_quant.strategies.strat06.ath_tracker import ATHTracker
from btc_quant.strategies.strat06.config import BaselineExecutionMode, DCAModulatedConfig
from btc_quant.strategies.strat06.drawdown_multiplier import DrawdownMultiplier
from btc_quant.strategies.strat06.reserve_manager import ReserveManager

_BASELINE = StrategyTag.STRAT_06_BASELINE
_BUFFER = StrategyTag.STRAT_06_BUFFER
_RESERVE = StrategyTag.STRAT_06_RESERVE


class DCAModulatedStrategy(StrategyBase):
    """STRAT-06: Drawdown-Modulated DCA + Deep Value Reserve.

    Components:
    1. Weekly baseline DCA (Monday 12:00 UTC, fixed weekly buy)
    2. Drawdown modulation (multiplies baseline using BUFFER cash)
    3. Deep Value Reserve (tranches deployed at extreme sustained drawdowns)

    No exits. Accumulation only.

    SIGNAL TAGGING CONTRACT:
    All Signals use strategy_tag=STRAT_06_BASELINE regardless of which pool
    the capital comes from. The engine uses this tag to route execution back
    to this class (strategy_by_tag lookup), and only this class is registered
    under STRAT_06_BASELINE.

    ATOMIC TRANSFERS:
    Pool transfers (BUFFER→BASELINE, RESERVE→BASELINE) happen atomically
    within generate_signals(), before emitting the corresponding Signal.
    This assumes the engine executes emitted Signals without rejection (at
    the next bar's open). If a future RiskManager can reject Signals, this
    contract must be revisited — a rejected Signal would leave cash
    mis-positioned after an already-committed transfer.

    RESERVE BALANCE SOURCE OF TRUTH:
    ReserveManager is used only for logic: drawdown persistence tracking,
    deployed-tranche state, and new-ATH cycle resets. The actual reserve
    cash lives in portfolio.get_pool(STRAT_06_RESERVE).cash_eur, which is
    the single source of truth. This balance is passed to
    evaluate_deployment() on every call so ReserveManager never needs to
    track cash independently.
    """

    def __init__(self, config: DCAModulatedConfig) -> None:
        super().__init__(
            strategy_tag=_BASELINE,
            config=config.__dict__,
        )
        self._config = config
        self._ath_tracker = ATHTracker()
        self._drawdown_multiplier = DrawdownMultiplier()
        self._reserve_manager = ReserveManager(
            reserve_cap_eur=config.reserve_cap_eur,
            persistence_days=config.persistence_days_for_tranche,
        )
        self._last_baseline_buy_week: tuple[int, int] | None = None
        self._last_known_ath: float | None = None

    # ── StrategyBase interface ────────────────────────────────────────────────

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def additional_pool_tags(self) -> list[StrategyTag]:
        return [_BUFFER, _RESERVE]

    def allows_position_accumulation(self) -> bool:
        return True

    # ── Core logic ────────────────────────────────────────────────────────────

    def generate_signals(
        self, context: MarketContext, portfolio: Portfolio
    ) -> list[Signal]:
        """Generate buy signals for this bar.

        Strict operation order (required by spec):
        1. Update ATH tracker with current price
        2. Detect new ATH → reset_cycle (BEFORE record_drawdown)
        3. Record drawdown for persistence tracking
        4. Redirect reserve overflow to buffer if cap exceeded
        5. Evaluate reserve tranche deployment
        6. Evaluate baseline + modulation buy
        """
        current_price = context.get_current_btc_price()
        current_time = context.current_time

        # 1. Update ATH tracker
        self._ath_tracker.update(current_price)
        drawdown = self._ath_tracker.get_drawdown(current_price)

        # 2. Detect new ATH → reset cycle (before recording current drawdown)
        if self._last_known_ath is None or current_price > self._last_known_ath:
            if self._last_known_ath is not None:
                self._reserve_manager.reset_cycle()
            self._last_known_ath = current_price

        # 3. Record drawdown for persistence tracking
        self._reserve_manager.record_drawdown(current_time, drawdown)

        # 4. Redirect reserve overflow to buffer
        self._maybe_redirect_reserve_overflow(portfolio)

        signals: list[Signal] = []

        # 5. Evaluate reserve tranche deployment
        reserve_signal = self._check_reserve_deployment(
            current_price, drawdown, current_time, portfolio
        )
        if reserve_signal is not None:
            signals.append(reserve_signal)

        # 6. Evaluate baseline + modulation
        baseline_signal = self._check_baseline_buy(
            current_price, drawdown, current_time, portfolio
        )
        if baseline_signal is not None:
            signals.append(baseline_signal)

        return signals

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _maybe_redirect_reserve_overflow(self, portfolio: Portfolio) -> None:
        reserve_pool = portfolio.get_pool(_RESERVE)
        cap = self._config.reserve_cap_eur
        if reserve_pool.cash_eur > cap:
            overflow = reserve_pool.cash_eur - cap
            portfolio.transfer_cash(
                from_tag=_RESERVE,
                to_tag=_BUFFER,
                amount=overflow,
                reason="reserve_overflow",
            )

    def _check_reserve_deployment(
        self,
        current_price: float,
        drawdown: float,
        current_time: datetime,
        portfolio: Portfolio,
    ) -> Signal | None:
        reserve_pool = portfolio.get_pool(_RESERVE)
        deployment = self._reserve_manager.evaluate_deployment(
            current_drawdown=drawdown,
            current_btc_price=current_price,
            current_timestamp=current_time,
            current_balance_eur=reserve_pool.cash_eur,
        )
        if deployment is None:
            return None

        amount = deployment.amount_deployed_eur
        if amount < self._config.min_order_eur:
            return None

        portfolio.transfer_cash(
            from_tag=_RESERVE,
            to_tag=_BASELINE,
            amount=amount,
            reason=f"reserve_tranche_{deployment.tranche_id}",
        )

        return Signal(
            strategy_tag=_BASELINE,
            timestamp_utc=current_time,
            side=OrderSide.BUY,
            symbol=self._config.symbol,
            quote_amount_eur=amount,
            reason_tag=f"reserve_tranche_{deployment.tranche_id}",
            metadata={
                "drawdown_at_deploy": deployment.drawdown_at_deploy,
                "btc_price_at_deploy": deployment.btc_price_at_deploy,
            },
        )

    def _check_baseline_buy(
        self,
        current_price: float,
        drawdown: float,
        current_time: datetime,
        portfolio: Portfolio,
    ) -> Signal | None:
        if not self._is_baseline_time(current_time):
            return None

        if (
            self._calculate_concentration(portfolio, current_price)
            > self._config.max_concentration_pct
        ):
            return None

        multiplier = self._drawdown_multiplier.get_multiplier(drawdown)
        target_amount = self._config.baseline_per_week_eur * multiplier

        baseline_pool = portfolio.get_pool(_BASELINE)
        buffer_pool = portfolio.get_pool(_BUFFER)

        deficit = max(0.0, target_amount - baseline_pool.cash_eur)
        if deficit > 0 and buffer_pool.cash_eur > 0:
            transfer_amount = min(deficit, buffer_pool.cash_eur)
            portfolio.transfer_cash(
                from_tag=_BUFFER,
                to_tag=_BASELINE,
                amount=transfer_amount,
                reason="modulation_transfer",
            )

        actual_amount = min(target_amount, baseline_pool.cash_eur)
        if actual_amount < self._config.min_order_eur:
            return None

        iso_year, iso_week, _ = current_time.isocalendar()
        self._last_baseline_buy_week = (iso_year, iso_week)

        reason_tag = (
            f"baseline_modulated_x{multiplier}" if multiplier > 1.0 else "baseline_weekly"
        )

        return Signal(
            strategy_tag=_BASELINE,
            timestamp_utc=current_time,
            side=OrderSide.BUY,
            symbol=self._config.symbol,
            quote_amount_eur=actual_amount,
            reason_tag=reason_tag,
            metadata={
                "drawdown": drawdown,
                "multiplier": multiplier,
                "target_amount": target_amount,
            },
        )

    def _is_baseline_time(self, current_time: datetime) -> bool:
        if current_time.weekday() != self._config.baseline_weekday:
            return False
        if self._config.baseline_execution_mode == BaselineExecutionMode.WEEKLY_AT_HOUR:
            if current_time.hour != self._config.baseline_hour_utc:
                return False
        iso_year, iso_week, _ = current_time.isocalendar()
        if self._last_baseline_buy_week == (iso_year, iso_week):
            return False
        return True

    def _calculate_concentration(
        self, portfolio: Portfolio, current_price: float
    ) -> float:
        total_cash = 0.0
        total_btc_value = 0.0
        for tag in (_BASELINE, _BUFFER, _RESERVE):
            pool = portfolio.get_pool(tag)
            total_cash += pool.cash_eur
            total_btc_value += pool.btc_held * current_price
        total_value = total_cash + total_btc_value
        if total_value <= 0:
            return 0.0
        return total_btc_value / total_value
