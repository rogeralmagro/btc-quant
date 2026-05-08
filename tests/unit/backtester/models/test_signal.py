"""Unit tests for Signal model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from btc_quant.backtester.models.enums import OrderSide, StrategyTag
from btc_quant.backtester.models.signal import Signal

UTC = timezone.utc
TS = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)


def _base(**overrides) -> Signal:
    """Minimal valid Signal with quantity_btc."""
    defaults = dict(
        strategy_tag=StrategyTag.STRAT_07_TACTICAL,
        timestamp_utc=TS,
        side=OrderSide.BUY,
        symbol="BTCUSDT",
        quantity_btc=0.1,
    )
    defaults.update(overrides)
    return Signal(**defaults)


class TestSignalCreation:
    def test_signal_creation_with_quantity(self) -> None:
        s = _base()
        assert s.strategy_tag == StrategyTag.STRAT_07_TACTICAL
        assert s.timestamp_utc == TS
        assert s.side == OrderSide.BUY
        assert s.symbol == "BTCUSDT"
        assert s.quantity_btc == 0.1
        assert s.quote_amount_eur is None

    def test_signal_creation_with_quote_amount(self) -> None:
        s = Signal(
            strategy_tag=StrategyTag.STRAT_06_BASELINE,
            timestamp_utc=TS,
            side=OrderSide.BUY,
            symbol="BTCUSDT",
            quote_amount_eur=500.0,
        )
        assert s.quote_amount_eur == 500.0
        assert s.quantity_btc is None

    def test_signal_has_auto_uuid(self) -> None:
        s = _base()
        assert isinstance(s.signal_id, UUID)

    def test_signal_ids_are_unique(self) -> None:
        s1 = _base()
        s2 = _base()
        assert s1.signal_id != s2.signal_id

    def test_signal_defaults(self) -> None:
        s = _base()
        assert s.stop_loss_price is None
        assert s.take_profit_prices is None
        assert s.take_profit_quantities_pct is None
        assert s.score is None
        assert s.reason_tag == ""
        assert s.metadata == {}

    def test_signal_with_stop_and_tp(self) -> None:
        s = _base(
            stop_loss_price=40_000.0,
            take_profit_prices=[50_000.0, 60_000.0],
            take_profit_quantities_pct=[0.5, 0.5],
            score=7,
            reason_tag="confluence",
            metadata={"regime": "bull"},
        )
        assert s.stop_loss_price == 40_000.0
        assert s.take_profit_prices == [50_000.0, 60_000.0]
        assert s.take_profit_quantities_pct == [0.5, 0.5]
        assert s.score == 7
        assert s.reason_tag == "confluence"
        assert s.metadata["regime"] == "bull"

    def test_signal_tp_sum_less_than_one_is_valid(self) -> None:
        # Partial exit — 30% at each TP, leaving 40% to trail
        s = _base(
            take_profit_prices=[50_000.0, 60_000.0],
            take_profit_quantities_pct=[0.3, 0.3],
        )
        assert sum(s.take_profit_quantities_pct) == pytest.approx(0.6)

    def test_signal_tp_sum_exactly_one_is_valid(self) -> None:
        s = _base(
            take_profit_prices=[50_000.0, 60_000.0],
            take_profit_quantities_pct=[0.5, 0.5],
        )
        assert sum(s.take_profit_quantities_pct) == pytest.approx(1.0)

    def test_signal_non_utc_zero_offset_accepted(self) -> None:
        """A tz-aware datetime with +00:00 offset (not timezone.utc) is accepted."""
        import datetime as dt_module
        fixed_utc = dt_module.timezone(dt_module.timedelta(0))
        ts = datetime(2024, 1, 1, tzinfo=fixed_utc)
        s = _base(timestamp_utc=ts)
        assert s.timestamp_utc == ts


class TestSignalValidation:
    def test_signal_both_quantity_and_quote_raises(self) -> None:
        with pytest.raises(ValueError, match="not both"):
            Signal(
                strategy_tag=StrategyTag.STRAT_07_TACTICAL,
                timestamp_utc=TS,
                side=OrderSide.BUY,
                symbol="BTCUSDT",
                quantity_btc=0.1,
                quote_amount_eur=500.0,
            )

    def test_signal_neither_quantity_nor_quote_raises(self) -> None:
        with pytest.raises(ValueError, match="One of"):
            Signal(
                strategy_tag=StrategyTag.STRAT_07_TACTICAL,
                timestamp_utc=TS,
                side=OrderSide.BUY,
                symbol="BTCUSDT",
            )

    def test_signal_naive_datetime_raises(self) -> None:
        with pytest.raises(ValueError, match="tz-aware"):
            _base(timestamp_utc=datetime(2024, 1, 15, 12, 0, 0))

    def test_signal_non_utc_offset_raises(self) -> None:
        ts_plus1 = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone(timedelta(hours=1)))
        with pytest.raises(ValueError, match="UTC"):
            _base(timestamp_utc=ts_plus1)

    def test_signal_zero_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity_btc must be > 0"):
            _base(quantity_btc=0.0)

    def test_signal_negative_quantity_raises(self) -> None:
        with pytest.raises(ValueError, match="quantity_btc must be > 0"):
            _base(quantity_btc=-0.1)

    def test_signal_zero_quote_raises(self) -> None:
        with pytest.raises(ValueError, match="quote_amount_eur must be > 0"):
            Signal(
                strategy_tag=StrategyTag.STRAT_06_BASELINE,
                timestamp_utc=TS,
                side=OrderSide.BUY,
                symbol="BTCUSDT",
                quote_amount_eur=0.0,
            )

    def test_signal_negative_stop_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="stop_loss_price must be > 0"):
            _base(stop_loss_price=-1.0)

    def test_signal_zero_stop_loss_raises(self) -> None:
        with pytest.raises(ValueError, match="stop_loss_price must be > 0"):
            _base(stop_loss_price=0.0)

    def test_signal_tp_lengths_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            _base(
                take_profit_prices=[50_000.0, 60_000.0],
                take_profit_quantities_pct=[0.5],
            )

    def test_signal_tp_sum_exceeds_one_raises(self) -> None:
        with pytest.raises(ValueError, match="<= 1.0"):
            _base(
                take_profit_prices=[50_000.0, 60_000.0],
                take_profit_quantities_pct=[0.6, 0.5],
            )

    def test_signal_tp_quantity_zero_raises(self) -> None:
        with pytest.raises(ValueError, match=r"must be in \(0, 1\]"):
            _base(
                take_profit_prices=[50_000.0],
                take_profit_quantities_pct=[0.0],
            )

    def test_signal_only_tp_prices_no_qtys_raises(self) -> None:
        with pytest.raises(ValueError, match="both be provided or both be None"):
            _base(take_profit_prices=[50_000.0])

    def test_signal_only_tp_qtys_no_prices_raises(self) -> None:
        with pytest.raises(ValueError, match="both be provided or both be None"):
            _base(take_profit_quantities_pct=[0.5])


class TestSignalImmutable:
    def test_signal_immutable_strategy_tag(self) -> None:
        s = _base()
        with pytest.raises(FrozenInstanceError):
            s.strategy_tag = StrategyTag.STRAT_06_BASELINE  # type: ignore[misc]

    def test_signal_immutable_quantity(self) -> None:
        s = _base()
        with pytest.raises(FrozenInstanceError):
            s.quantity_btc = 0.5  # type: ignore[misc]

    def test_signal_immutable_signal_id(self) -> None:
        s = _base()
        with pytest.raises(FrozenInstanceError):
            s.signal_id = s.signal_id  # type: ignore[misc]
