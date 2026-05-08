"""Abstract base class for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.backtester.models.signal import Signal

if TYPE_CHECKING:
    from btc_quant.backtester.market_context import MarketContext
    from btc_quant.backtester.models.portfolio import Portfolio


class StrategyBase(ABC):
    """Abstract base for all system strategies.

    Contract:
    - The engine calls generate_signals() on every bar of the iteration TF.
    - The strategy receives a MarketContext with data up to and including the
      current bar's close (zero look-ahead guaranteed by MarketContext).
    - The strategy returns a list of Signals (may be empty).
    - The engine simulates signal execution as Orders at the OPEN of the next
      bar. The strategy must never assume immediate fill.

    Subclass responsibilities:
    - Implement generate_signals(context, portfolio) -> list[Signal]
    - Declare required_timeframes() -> list[str]
    - Declare primary_timeframe() -> str
    """

    def __init__(self, strategy_tag: StrategyTag, config: dict[str, Any]) -> None:
        self.strategy_tag = strategy_tag
        self.config = config
        self._validate_config()

    @abstractmethod
    def required_timeframes(self) -> list[str]:
        """Timeframes this strategy queries (e.g. ["1d", "4h", "1w"]).

        The engine guarantees these TFs are available in MarketContext.
        """
        ...

    @abstractmethod
    def primary_timeframe(self) -> str:
        """Timeframe in which this strategy makes decisions.

        The engine iterates over the finest TF required by all active strategies.
        Strategies with a coarser primary_timeframe are only evaluated on bars
        aligned to that TF (e.g. STRAT-06 with "1w" fires once per week even
        when the engine iterates daily).
        """
        ...

    @abstractmethod
    def generate_signals(
        self,
        context: MarketContext,
        portfolio: Portfolio,
    ) -> list[Signal]:
        """Evaluate current market context and return trading signals.

        Args:
            context: Market data up to (and including) the current bar's close.
                     Read-only; never mutate the context.
            portfolio: Current portfolio state. Read-only for the strategy;
                       mutations are performed by the engine, not here.

        Returns:
            List of Signals. May be empty. The engine converts each Signal
            into an Order executed at the next bar's open.
        """
        ...

    def allows_position_accumulation(self) -> bool:
        """True if strategy may send multiple BUYs into the same position (DCA).

        False (default): engine rejects a second BUY while a position is open.
        True: engine calls Position.add_to_position() to average down/up.
        Override in strategies that implement DCA or pyramid-entry logic.
        """
        return False

    def _validate_config(self) -> None:
        """Hook for config validation. Called at the end of __init__.

        Default: no-op. Override to raise ValueError on invalid config.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tag={self.strategy_tag.value!r})"
