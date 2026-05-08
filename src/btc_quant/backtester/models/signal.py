"""Signal model: strategy output indicating trading intent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from btc_quant.backtester.models.enums import OrderSide, StrategyTag


@dataclass(frozen=True)
class Signal:
    """Output of a strategy indicating trading intent.

    A Signal is a RECOMMENDATION, not an order. The RiskManager (future)
    may reject, modify, or transform it into an Order.

    The signal_id allows tracing Order → Signal → Strategy in logs and
    reporting.

    Immutable: once generated, it is never modified.
    """

    strategy_tag: StrategyTag
    timestamp_utc: datetime  # when the signal was generated (bar close)
    side: OrderSide
    symbol: str

    # Sizing — exactly one must be set:
    quantity_btc: float | None = None      # explicit BTC quantity
    quote_amount_eur: float | None = None  # EUR amount; engine calculates BTC at execution

    # Optional risk levels:
    stop_loss_price: float | None = None
    take_profit_prices: list[float] | None = None           # multi-level TPs
    take_profit_quantities_pct: list[float] | None = None   # fraction of position per TP level

    # Reporting / debug:
    score: int | None = None  # e.g. confluence score in STRAT-07
    reason_tag: str = ""      # e.g. "tp1_hit", "regime_flip", "weekly_dca"
    metadata: dict[str, Any] = field(default_factory=dict)

    # Auto-generated identity — enables Order.parent_signal_id traceability:
    signal_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None or self.timestamp_utc.utcoffset() is None:
            raise ValueError("timestamp_utc must be tz-aware (UTC)")
        if self.timestamp_utc.utcoffset().total_seconds() != 0:
            raise ValueError("timestamp_utc timezone must be UTC (offset must be 0)")

        has_qty = self.quantity_btc is not None
        has_quote = self.quote_amount_eur is not None
        if has_qty and has_quote:
            raise ValueError(
                "Specify either quantity_btc or quote_amount_eur, not both"
            )
        if not has_qty and not has_quote:
            raise ValueError("One of quantity_btc or quote_amount_eur must be set")

        if has_qty and self.quantity_btc <= 0:  # type: ignore[operator]
            raise ValueError(f"quantity_btc must be > 0, got {self.quantity_btc}")
        if has_quote and self.quote_amount_eur <= 0:  # type: ignore[operator]
            raise ValueError(f"quote_amount_eur must be > 0, got {self.quote_amount_eur}")

        if self.stop_loss_price is not None and self.stop_loss_price <= 0:
            raise ValueError(f"stop_loss_price must be > 0, got {self.stop_loss_price}")

        has_tp_prices = self.take_profit_prices is not None
        has_tp_qtys = self.take_profit_quantities_pct is not None
        if has_tp_prices != has_tp_qtys:
            raise ValueError(
                "take_profit_prices and take_profit_quantities_pct must both be "
                "provided or both be None"
            )
        if has_tp_prices:
            if len(self.take_profit_prices) != len(self.take_profit_quantities_pct):  # type: ignore[arg-type]
                raise ValueError(
                    f"take_profit_prices (len={len(self.take_profit_prices)}) and "  # type: ignore[arg-type]
                    f"take_profit_quantities_pct (len={len(self.take_profit_quantities_pct)}) "  # type: ignore[arg-type]
                    "must have the same length"
                )
            for i, pct in enumerate(self.take_profit_quantities_pct):  # type: ignore[union-attr]
                if not (0 < pct <= 1):
                    raise ValueError(
                        f"take_profit_quantities_pct[{i}]={pct} must be in (0, 1]"
                    )
            total = sum(self.take_profit_quantities_pct)  # type: ignore[union-attr]
            if total > 1.0 + 1e-9:
                raise ValueError(
                    f"Sum of take_profit_quantities_pct ({total:.6f}) must be <= 1.0"
                )
