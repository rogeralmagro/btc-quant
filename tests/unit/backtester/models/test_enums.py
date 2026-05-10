"""Unit tests for backtester enums."""

from __future__ import annotations

import json

import pytest

from btc_quant.backtester.models.enums import (
    CircuitBreakerState,
    MarketRegime,
    OrderSide,
    OrderStatus,
    OrderType,
    StrategyTag,
    TradeOutcome,
)

ALL_ENUMS = [
    OrderSide,
    OrderType,
    OrderStatus,
    TradeOutcome,
    StrategyTag,
    CircuitBreakerState,
    MarketRegime,
]


class TestEnumValuesAreStrings:
    @pytest.mark.parametrize("enum_cls", ALL_ENUMS)
    def test_all_values_are_str(self, enum_cls) -> None:
        for member in enum_cls:
            assert isinstance(member.value, str), (
                f"{enum_cls.__name__}.{member.name}.value is not str"
            )

    @pytest.mark.parametrize("enum_cls", ALL_ENUMS)
    def test_member_is_str_subclass(self, enum_cls) -> None:
        """str-Enum members compare equal to their string value."""
        for member in enum_cls:
            assert member == member.value

    def test_order_side_values(self) -> None:
        assert OrderSide.BUY == "buy"
        assert OrderSide.SELL == "sell"

    def test_order_type_values(self) -> None:
        assert OrderType.MARKET == "market"
        assert OrderType.LIMIT == "limit"
        assert OrderType.STOP == "stop"
        assert OrderType.STOP_LIMIT == "stop_limit"

    def test_order_status_values(self) -> None:
        assert OrderStatus.PENDING == "pending"
        assert OrderStatus.FILLED == "filled"
        assert OrderStatus.PARTIALLY_FILLED == "partially_filled"
        assert OrderStatus.CANCELLED == "cancelled"
        assert OrderStatus.REJECTED == "rejected"

    def test_trade_outcome_values(self) -> None:
        assert TradeOutcome.WIN == "win"
        assert TradeOutcome.LOSS == "loss"
        assert TradeOutcome.BREAKEVEN == "breakeven"
        assert TradeOutcome.OPEN == "open"

    def test_strategy_tag_values(self) -> None:
        assert StrategyTag.STRAT_06_BASELINE == "strat06_baseline"
        assert StrategyTag.STRAT_06_BUFFER == "strat06_buffer"
        assert StrategyTag.STRAT_06_RESERVE == "strat06_reserve"
        assert StrategyTag.STRAT_07_TACTICAL == "strat07_tactical"
        assert StrategyTag.DCA_BENCHMARK == "dca_benchmark"
        assert StrategyTag.BUY_AND_HOLD_BENCHMARK == "buy_and_hold_benchmark"

    def test_circuit_breaker_state_values(self) -> None:
        assert CircuitBreakerState.NORMAL == "normal"
        assert CircuitBreakerState.PAUSED == "paused"
        assert CircuitBreakerState.HALTED == "halted"
        assert CircuitBreakerState.KILLED == "killed"

    def test_market_regime_enum_values(self) -> None:
        assert MarketRegime.NORMAL == "normal"
        assert MarketRegime.VOLATILE == "volatile"
        assert MarketRegime.STRESS == "stress"
        assert MarketRegime.CASCADE == "cascade"


class TestEnumSerializable:
    @pytest.mark.parametrize("enum_cls", ALL_ENUMS)
    def test_each_member_json_serializable(self, enum_cls) -> None:
        for member in enum_cls:
            serialized = json.dumps(member)
            assert json.loads(serialized) == member.value

    @pytest.mark.parametrize("enum_cls", ALL_ENUMS)
    def test_round_trip_from_string(self, enum_cls) -> None:
        """Each member can be reconstructed from its string value."""
        for member in enum_cls:
            reconstructed = enum_cls(member.value)
            assert reconstructed is member

    def test_strategy_tag_in_dict_key(self) -> None:
        """StrategyTag works as a dict key (used extensively in Portfolio)."""
        d = {StrategyTag.STRAT_06_BASELINE: 1000.0, StrategyTag.STRAT_07_TACTICAL: 500.0}
        assert d[StrategyTag.STRAT_06_BASELINE] == 1000.0

    def test_order_side_in_json_payload(self) -> None:
        payload = {"side": OrderSide.BUY, "qty": 0.1}
        serialized = json.dumps(payload)
        loaded = json.loads(serialized)
        assert loaded["side"] == "buy"
