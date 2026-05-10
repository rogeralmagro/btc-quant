"""Unit tests for DrawdownMultiplier and DrawdownThreshold."""

from __future__ import annotations

import pytest

from btc_quant.strategies.strat06.drawdown_multiplier import (
    DrawdownMultiplier,
    DrawdownThreshold,
)


class TestDrawdownThreshold:
    def test_threshold_with_multiplier_less_than_1_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1.0"):
            DrawdownThreshold(drawdown_pct=-0.10, multiplier=0.9)

    def test_threshold_positive_drawdown_raises(self) -> None:
        with pytest.raises(ValueError, match="<= 0"):
            DrawdownThreshold(drawdown_pct=0.10, multiplier=1.5)

    def test_threshold_zero_drawdown_is_valid(self) -> None:
        t = DrawdownThreshold(drawdown_pct=0.0, multiplier=1.0)
        assert t.drawdown_pct == 0.0

    def test_threshold_exactly_1x_multiplier_is_valid(self) -> None:
        t = DrawdownThreshold(drawdown_pct=-0.10, multiplier=1.0)
        assert t.multiplier == 1.0


class TestDrawdownMultiplierDefaults:
    def test_no_drawdown_returns_1x(self) -> None:
        dm = DrawdownMultiplier()
        assert dm.get_multiplier(0.0) == pytest.approx(1.0)

    def test_drawdown_minus_5pct_returns_1x(self) -> None:
        dm = DrawdownMultiplier()
        assert dm.get_multiplier(-0.05) == pytest.approx(1.0)

    def test_drawdown_minus_10pct_returns_125x(self) -> None:
        """Boundary at -10% is inclusive → 1.25×."""
        dm = DrawdownMultiplier()
        assert dm.get_multiplier(-0.10) == pytest.approx(1.25)

    def test_drawdown_minus_15pct_returns_125x(self) -> None:
        dm = DrawdownMultiplier()
        assert dm.get_multiplier(-0.15) == pytest.approx(1.25)

    def test_drawdown_minus_20pct_returns_15x(self) -> None:
        """Boundary at -20% is inclusive → 1.5×."""
        dm = DrawdownMultiplier()
        assert dm.get_multiplier(-0.20) == pytest.approx(1.50)

    def test_drawdown_minus_25pct_returns_15x(self) -> None:
        dm = DrawdownMultiplier()
        assert dm.get_multiplier(-0.25) == pytest.approx(1.50)

    def test_drawdown_minus_35pct_returns_2x(self) -> None:
        """Boundary at -35% is inclusive → 2.0×."""
        dm = DrawdownMultiplier()
        assert dm.get_multiplier(-0.35) == pytest.approx(2.00)

    def test_drawdown_minus_50pct_returns_25x(self) -> None:
        """Boundary at -50% is inclusive → 2.5×."""
        dm = DrawdownMultiplier()
        assert dm.get_multiplier(-0.50) == pytest.approx(2.50)

    def test_drawdown_minus_70pct_returns_25x(self) -> None:
        """Beyond the last threshold still returns the highest multiplier."""
        dm = DrawdownMultiplier()
        assert dm.get_multiplier(-0.70) == pytest.approx(2.50)

    def test_positive_drawdown_raises(self) -> None:
        dm = DrawdownMultiplier()
        with pytest.raises(ValueError, match="<= 0"):
            dm.get_multiplier(0.01)


class TestDrawdownMultiplierCustomThresholds:
    def test_custom_thresholds(self) -> None:
        thresholds = [
            DrawdownThreshold(drawdown_pct=-0.20, multiplier=2.0),
            DrawdownThreshold(drawdown_pct=-0.40, multiplier=3.0),
        ]
        dm = DrawdownMultiplier(thresholds=thresholds)
        assert dm.get_multiplier(-0.10) == pytest.approx(1.0)
        assert dm.get_multiplier(-0.20) == pytest.approx(2.0)
        assert dm.get_multiplier(-0.30) == pytest.approx(2.0)
        assert dm.get_multiplier(-0.40) == pytest.approx(3.0)
        assert dm.get_multiplier(-0.80) == pytest.approx(3.0)

    def test_unordered_thresholds_raises(self) -> None:
        thresholds = [
            DrawdownThreshold(drawdown_pct=-0.40, multiplier=2.0),
            DrawdownThreshold(drawdown_pct=-0.20, multiplier=1.5),  # wrong order
        ]
        with pytest.raises(ValueError, match="ordered"):
            DrawdownMultiplier(thresholds=thresholds)

    def test_duplicate_threshold_raises(self) -> None:
        thresholds = [
            DrawdownThreshold(drawdown_pct=-0.20, multiplier=1.5),
            DrawdownThreshold(drawdown_pct=-0.20, multiplier=2.0),  # duplicate
        ]
        with pytest.raises(ValueError, match="ordered"):
            DrawdownMultiplier(thresholds=thresholds)

    def test_single_threshold(self) -> None:
        dm = DrawdownMultiplier(thresholds=[
            DrawdownThreshold(drawdown_pct=-0.50, multiplier=3.0),
        ])
        assert dm.get_multiplier(-0.30) == pytest.approx(1.0)
        assert dm.get_multiplier(-0.50) == pytest.approx(3.0)

    def test_empty_thresholds_always_returns_1x(self) -> None:
        dm = DrawdownMultiplier(thresholds=[])
        assert dm.get_multiplier(-0.90) == pytest.approx(1.0)
