"""DCABenchmarkStrategy: pure fixed-amount weekly DCA, no modulation."""

from __future__ import annotations

from btc_quant.backtester.market_context import MarketContext
from btc_quant.backtester.models.enums import OrderSide, StrategyTag
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.backtester.models.signal import Signal
from btc_quant.backtester.strategy_base import StrategyBase

_MIN_ORDER_EUR = 1.0


class DCABenchmarkStrategy(StrategyBase):
    """Pure fixed-amount weekly DCA — no drawdown modulation, no reserve.

    Purpose: benchmark against DCAModulatedStrategy (STRAT-06) to isolate
    the value added by drawdown modulation and the deep-value reserve.

    Inflow contract:
        Receives the same €500/month inflow schedule as STRAT-06, credited to
        the DCA_BENCHMARK pool. Buys €115.38 every Monday (= 500 * 12 / 52,
        the annualised weekly equivalent), accumulating leftover cash from
        month-boundary misalignment and spending it the following Monday.

        If the available cash is less than weekly_amount_eur (e.g. inflow has
        not yet arrived for the month), buys the available cash instead,
        staying in rhythm without skipping weeks entirely.

    Fairness note:
        Total capital deployed equals STRAT-06's total inflow (same schedule,
        same monthly amount). The only variable is *when* within each month
        the capital enters the market and whether a drawdown multiplier is
        applied (STRAT-06) or not (this benchmark).
    """

    def __init__(
        self,
        weekly_amount_eur: float,
        weekday: int = 0,
        symbol: str = "BTCUSDT",
    ) -> None:
        super().__init__(
            strategy_tag=StrategyTag.DCA_BENCHMARK,
            config={
                "weekly_amount_eur": weekly_amount_eur,
                "weekday": weekday,
                "symbol": symbol,
            },
        )
        if weekly_amount_eur <= 0:
            raise ValueError(
                f"weekly_amount_eur must be > 0, got {weekly_amount_eur}"
            )
        if not (0 <= weekday <= 6):
            raise ValueError(f"weekday must be 0–6, got {weekday}")

        self._weekly_amount_eur = weekly_amount_eur
        self._weekday = weekday
        self._symbol = symbol
        self._last_buy_week: tuple[int, int] | None = None

    # ── StrategyBase interface ────────────────────────────────────────────────

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def allows_position_accumulation(self) -> bool:
        return True

    # ── Core logic ────────────────────────────────────────────────────────────

    def generate_signals(
        self, context: MarketContext, portfolio: Portfolio
    ) -> list[Signal]:
        current_time = context.current_time

        if current_time.weekday() != self._weekday:
            return []

        iso_year, iso_week, _ = current_time.isocalendar()
        if self._last_buy_week == (iso_year, iso_week):
            return []

        pool = portfolio.get_pool(StrategyTag.DCA_BENCHMARK)
        available = pool.cash_eur
        actual = min(self._weekly_amount_eur, available)

        if actual < _MIN_ORDER_EUR:
            return []

        self._last_buy_week = (iso_year, iso_week)

        return [
            Signal(
                strategy_tag=StrategyTag.DCA_BENCHMARK,
                timestamp_utc=current_time,
                side=OrderSide.BUY,
                symbol=self._symbol,
                quote_amount_eur=actual,
                reason_tag="dca_weekly",
            )
        ]
