"""Unit tests for StrategyBase abstract class."""

from __future__ import annotations

from typing import Any

import pytest

from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.backtester.models.signal import Signal
from btc_quant.backtester.strategy_base import StrategyBase

TAG = StrategyTag.STRAT_07_TACTICAL


# ---------------------------------------------------------------------------
# Concrete helpers (test-only, not production code)
# ---------------------------------------------------------------------------


class _MockStrategy(StrategyBase):
    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context, portfolio) -> list[Signal]:
        return []


class _IncompleteStrategy(StrategyBase):
    """Missing primary_timeframe and generate_signals."""

    def required_timeframes(self) -> list[str]:
        return ["1d"]


class _ValidatingStrategy(StrategyBase):
    """Tracks whether _validate_config was called."""

    def __init__(self, tag: StrategyTag, config: dict[str, Any]) -> None:
        self.validate_was_called = False
        super().__init__(tag, config)

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context, portfolio) -> list[Signal]:
        return []

    def _validate_config(self) -> None:
        self.validate_was_called = True


class _RejectingStrategy(StrategyBase):
    """Raises on invalid config key."""

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context, portfolio) -> list[Signal]:
        return []

    def _validate_config(self) -> None:
        if "invalid_key" in self.config:
            raise ValueError("invalid_key not allowed")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStrategyBaseAbstract:
    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            StrategyBase(TAG, {})  # type: ignore[abstract]

    def test_subclass_must_implement_all_abstract_methods(self) -> None:
        with pytest.raises(TypeError):
            _IncompleteStrategy(TAG, {})  # type: ignore[abstract]

    def test_concrete_subclass_works(self) -> None:
        s = _MockStrategy(TAG, {})
        assert s.strategy_tag == TAG
        assert s.config == {}

    def test_config_stored(self) -> None:
        cfg = {"lookback": 50, "threshold": 0.05}
        s = _MockStrategy(TAG, cfg)
        assert s.config["lookback"] == 50

    def test_required_timeframes_returns_list(self) -> None:
        s = _MockStrategy(TAG, {})
        result = s.required_timeframes()
        assert isinstance(result, list)
        assert result == ["1d"]

    def test_primary_timeframe_returns_string(self) -> None:
        s = _MockStrategy(TAG, {})
        assert s.primary_timeframe() == "1d"

    def test_generate_signals_returns_list(self) -> None:
        s = _MockStrategy(TAG, {})
        result = s.generate_signals(None, None)  # type: ignore[arg-type]
        assert isinstance(result, list)
        assert result == []


class TestValidateConfig:
    def test_validate_config_called_on_init(self) -> None:
        s = _ValidatingStrategy(TAG, {})
        assert s.validate_was_called is True

    def test_validate_config_receives_config(self) -> None:
        """Rejecting strategy raises when config contains forbidden key."""
        with pytest.raises(ValueError, match="invalid_key"):
            _RejectingStrategy(TAG, {"invalid_key": True})

    def test_validate_config_default_is_noop(self) -> None:
        """Default _validate_config does not raise for any config."""
        s = _MockStrategy(TAG, {"anything": "goes"})
        assert s.config["anything"] == "goes"


class TestAllowsPositionAccumulation:
    def test_default_returns_false(self) -> None:
        s = _MockStrategy(TAG, {})
        assert s.allows_position_accumulation() is False

    def test_override_can_return_true(self) -> None:
        class _AccumulatingStrategy(_MockStrategy):
            def allows_position_accumulation(self) -> bool:
                return True

        s = _AccumulatingStrategy(TAG, {})
        assert s.allows_position_accumulation() is True


class TestRepr:
    def test_repr_contains_class_name(self) -> None:
        s = _MockStrategy(TAG, {})
        assert "_MockStrategy" in repr(s)

    def test_repr_contains_tag(self) -> None:
        s = _MockStrategy(TAG, {})
        assert TAG.value in repr(s)
