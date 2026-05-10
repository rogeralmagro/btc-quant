"""Unit tests for ATHTracker."""

from __future__ import annotations

import pytest

from btc_quant.strategies.strat06.ath_tracker import ATHTracker


class TestATHTrackerInitialState:
    def test_ath_starts_none_without_initial(self) -> None:
        tracker = ATHTracker()
        assert tracker.ath is None

    def test_ath_initialized_with_value(self) -> None:
        tracker = ATHTracker(initial_ath=50_000.0)
        assert tracker.ath == 50_000.0

    def test_initial_ath_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ATHTracker(initial_ath=0.0)

    def test_initial_ath_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            ATHTracker(initial_ath=-100.0)


class TestATHTrackerUpdate:
    def test_ath_only_goes_up(self) -> None:
        tracker = ATHTracker()
        tracker.update(100.0)
        tracker.update(120.0)
        assert tracker.ath == 120.0

    def test_ath_does_not_decrease_with_lower_prices(self) -> None:
        tracker = ATHTracker()
        tracker.update(100.0)
        tracker.update(80.0)
        tracker.update(60.0)
        assert tracker.ath == 100.0

    def test_first_update_sets_ath(self) -> None:
        tracker = ATHTracker()
        tracker.update(42.0)
        assert tracker.ath == 42.0

    def test_equal_price_does_not_change_ath(self) -> None:
        tracker = ATHTracker()
        tracker.update(100.0)
        tracker.update(100.0)
        assert tracker.ath == 100.0

    def test_update_negative_price_raises(self) -> None:
        tracker = ATHTracker()
        with pytest.raises(ValueError, match="positive"):
            tracker.update(-1.0)

    def test_update_zero_price_raises(self) -> None:
        tracker = ATHTracker()
        with pytest.raises(ValueError, match="positive"):
            tracker.update(0.0)

    def test_update_with_initial_ath_can_raise_it(self) -> None:
        tracker = ATHTracker(initial_ath=100.0)
        tracker.update(150.0)
        assert tracker.ath == 150.0

    def test_update_below_initial_ath_leaves_it_unchanged(self) -> None:
        tracker = ATHTracker(initial_ath=100.0)
        tracker.update(90.0)
        assert tracker.ath == 100.0


class TestATHTrackerDrawdown:
    def test_drawdown_at_ath_returns_zero(self) -> None:
        tracker = ATHTracker()
        tracker.update(100.0)
        assert tracker.get_drawdown(100.0) == pytest.approx(0.0)

    def test_drawdown_calculation_basic(self) -> None:
        tracker = ATHTracker()
        tracker.update(100.0)
        assert tracker.get_drawdown(80.0) == pytest.approx(-0.20)

    def test_drawdown_after_decline(self) -> None:
        tracker = ATHTracker()
        tracker.update(200.0)
        tracker.update(150.0)  # does not change ATH (200)
        assert tracker.get_drawdown(100.0) == pytest.approx(-0.50)

    def test_drawdown_is_non_positive(self) -> None:
        tracker = ATHTracker()
        tracker.update(100.0)
        dd = tracker.get_drawdown(90.0)
        assert dd <= 0.0

    def test_drawdown_without_update_raises(self) -> None:
        tracker = ATHTracker()
        with pytest.raises(RuntimeError, match="no ATH yet"):
            tracker.get_drawdown(50.0)

    def test_drawdown_negative_price_raises(self) -> None:
        tracker = ATHTracker()
        tracker.update(100.0)
        with pytest.raises(ValueError, match="positive"):
            tracker.get_drawdown(-10.0)

    def test_drawdown_zero_price_raises(self) -> None:
        tracker = ATHTracker()
        tracker.update(100.0)
        with pytest.raises(ValueError, match="positive"):
            tracker.get_drawdown(0.0)

    def test_drawdown_above_ath_returns_positive(self) -> None:
        tracker = ATHTracker(initial_ath=100.0)
        # price above historical ATH (before update) → positive, but update must be called first
        # With initial_ath=100, querying 110 gives positive — mathematically valid
        assert tracker.get_drawdown(110.0) == pytest.approx(0.10)
