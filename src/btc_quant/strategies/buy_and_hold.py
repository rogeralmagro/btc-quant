"""BuyAndHoldStrategy: benchmark that buys once and never sells."""

from __future__ import annotations

from btc_quant.backtester.market_context import MarketContext
from btc_quant.backtester.models.enums import OrderSide, StrategyTag
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.backtester.models.signal import Signal
from btc_quant.backtester.strategy_base import StrategyBase


class BuyAndHoldStrategy(StrategyBase):
    """Benchmark strategy: deploy all capital on the first available bar, then hold.

    Used to cross-validate engine P&L against the raw price change. If the
    engine is correct, a Buy-and-Hold run over N bars should produce a final
    BTC value consistent with price_final / price_entry (minus fees/slippage).

    No stops, no take-profits, no reinvestment logic.
    """

    def __init__(self) -> None:
        super().__init__(
            strategy_tag=StrategyTag.BUY_AND_HOLD_BENCHMARK,
            config={},
        )
        self._first_buy_done = False

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(
        self,
        context: MarketContext,
        portfolio: Portfolio,
    ) -> list[Signal]:
        if self._first_buy_done:
            return []

        pool = portfolio.get_pool(self.strategy_tag)
        if pool.cash_eur <= 0:
            return []

        self._first_buy_done = True
        return [
            Signal(
                strategy_tag=self.strategy_tag,
                timestamp_utc=context.current_time,
                side=OrderSide.BUY,
                symbol=context.symbol,
                quote_amount_eur=pool.cash_eur,
                reason_tag="initial_buy",
            )
        ]
