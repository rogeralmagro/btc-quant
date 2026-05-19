"""Unit tests for the STRAT-07 confluence scorer."""

from math import isnan
from typing import Dict, Optional, Set

import pytest

from btc_quant.signals.types import SignalResult
from btc_quant.strategies.strat07 import (
    CATEGORIES,
    DEFAULT_MIN_SCORE,
    MANDATORY_CATEGORIES,
    TOTAL_SIGNALS,
    ConfluenceResult,
    evaluate_confluence,
)


def _make_signals(
    active_overrides: Optional[Dict[str, bool]] = None,
    nan_value_names: Optional[Set[str]] = None,
) -> Dict[str, SignalResult]:
    """Build a complete signals dict for testing.

    By default all 12 signals are inactive with value=0.0.
    active_overrides flips specific names to active=True.
    nan_value_names sets value=NaN for specific names.
    """
    active_overrides = active_overrides or {}
    nan_value_names = nan_value_names or set()
    signals: Dict[str, SignalResult] = {}
    for cat_sig_names in CATEGORIES.values():
        for name in cat_sig_names:
            is_active = active_overrides.get(name, False)
            value = float("nan") if name in nan_value_names else 0.0
            signals[name] = SignalResult(active=is_active, value=value)
    return signals


def test_all_twelve_active_full_data_enters():
    overrides = {name: True for cat in CATEGORIES.values() for name in cat}
    signals = _make_signals(active_overrides=overrides)
    result = evaluate_confluence(signals)

    assert result.enter is True
    assert result.score == 12
    assert result.threshold_met is True
    assert result.mandatory_met is True
    assert result.data_sufficient is True


def test_exactly_seven_active_with_majority_regime_and_momentum_enters():
    # 2/3 regime, 2/3 momentum, 1 mean_reversion, 1 volume, 1 structure = 7
    overrides = {
        "ema_trend": True, "atr_regime": True,             # regime 2/3
        "macd_bullish": True, "adx_trending": True,        # momentum 2/3
        "rsi_oversold": True,                              # mean_reversion 1
        "obv_trend": True,                                 # volume 1
        "market_structure": True,                          # structure 1
    }
    signals = _make_signals(active_overrides=overrides)
    result = evaluate_confluence(signals)

    assert result.score == 7
    assert result.threshold_met is True
    assert result.mandatory_met is True
    assert result.data_sufficient is True
    assert result.enter is True


def test_six_active_below_threshold_does_not_enter():
    # 2/3 regime + 2/3 momentum (mandatory met) + 2 others = 6, < 7
    overrides = {
        "ema_trend": True, "atr_regime": True,
        "macd_bullish": True, "adx_trending": True,
        "rsi_oversold": True,
        "obv_trend": True,
    }
    signals = _make_signals(active_overrides=overrides)
    result = evaluate_confluence(signals)

    assert result.score == 6
    assert result.threshold_met is False
    assert result.mandatory_met is True
    assert result.enter is False


def test_eight_active_but_only_one_third_regime_fails_mandatory():
    # regime 1/3, momentum 3/3, plus 4 others = 8
    overrides = {
        "ema_trend": True,                                            # regime 1/3
        "macd_bullish": True, "adx_trending": True, "roc_positive": True,  # momentum 3/3
        "rsi_oversold": True, "bb_lower": True, "far_below_ma": True,      # mean_reversion 3
        "obv_trend": True,                                            # volume 1
    }
    signals = _make_signals(active_overrides=overrides)
    result = evaluate_confluence(signals)

    assert result.score == 8
    assert result.threshold_met is True
    assert result.mandatory_met is False
    assert result.enter is False


def test_eight_active_but_only_one_third_momentum_fails_mandatory():
    # regime 3/3, momentum 1/3, plus 4 others = 8
    overrides = {
        "ema_trend": True, "atr_regime": True, "htf_alignment": True,  # regime 3/3
        "macd_bullish": True,                                          # momentum 1/3
        "rsi_oversold": True, "bb_lower": True, "far_below_ma": True,  # mean_reversion 3
        "obv_trend": True,                                             # volume 1
    }
    signals = _make_signals(active_overrides=overrides)
    result = evaluate_confluence(signals)

    assert result.score == 8
    assert result.threshold_met is True
    assert result.mandatory_met is False
    assert result.enter is False


def test_eight_active_one_third_in_both_mandatory_fails():
    # regime 1/3, momentum 1/3, others maxed = 1 + 1 + 3 + 2 + 1 = 8
    overrides = {
        "ema_trend": True,
        "macd_bullish": True,
        "rsi_oversold": True, "bb_lower": True, "far_below_ma": True,
        "obv_trend": True, "high_volume": True,
        "market_structure": True,
    }
    signals = _make_signals(active_overrides=overrides)
    result = evaluate_confluence(signals)

    assert result.score == 8
    assert result.threshold_met is True
    assert result.mandatory_met is False
    assert result.enter is False


def test_seven_active_mandatory_met_but_nan_value_blocks_entry():
    overrides = {
        "ema_trend": True, "atr_regime": True,
        "macd_bullish": True, "adx_trending": True,
        "rsi_oversold": True,
        "obv_trend": True,
        "market_structure": True,
    }
    # Inject NaN into an inactive signal -> still counts as data insufficient.
    signals = _make_signals(
        active_overrides=overrides,
        nan_value_names={"high_volume"},
    )
    result = evaluate_confluence(signals)

    assert result.score == 7
    assert result.threshold_met is True
    assert result.mandatory_met is True
    assert result.data_sufficient is False
    assert result.enter is False


def test_nan_value_allowed_when_require_full_data_false():
    overrides = {
        "ema_trend": True, "atr_regime": True,
        "macd_bullish": True, "adx_trending": True,
        "rsi_oversold": True,
        "obv_trend": True,
        "market_structure": True,
    }
    signals = _make_signals(
        active_overrides=overrides,
        nan_value_names={"high_volume"},
    )
    result = evaluate_confluence(signals, require_full_data=False)

    assert result.data_sufficient is False
    assert result.threshold_met is True
    assert result.mandatory_met is True
    assert result.enter is True


def test_custom_min_score_five_enters_with_five_active():
    # 2/3 regime, 2/3 momentum, 1 mean_reversion = 5
    overrides = {
        "ema_trend": True, "atr_regime": True,
        "macd_bullish": True, "adx_trending": True,
        "rsi_oversold": True,
    }
    signals = _make_signals(active_overrides=overrides)
    result = evaluate_confluence(signals, min_score=5)

    assert result.score == 5
    assert result.threshold_met is True
    assert result.mandatory_met is True
    assert result.enter is True


def test_custom_min_score_ten_blocks_with_seven_active():
    overrides = {
        "ema_trend": True, "atr_regime": True,
        "macd_bullish": True, "adx_trending": True,
        "rsi_oversold": True,
        "obv_trend": True,
        "market_structure": True,
    }
    signals = _make_signals(active_overrides=overrides)
    result = evaluate_confluence(signals, min_score=10)

    assert result.score == 7
    assert result.threshold_met is False
    assert result.mandatory_met is True
    assert result.enter is False


def test_missing_signal_raises_value_error():
    signals = _make_signals()
    del signals["ema_trend"]

    with pytest.raises(ValueError) as exc_info:
        evaluate_confluence(signals)
    assert "ema_trend" in str(exc_info.value)
    assert "Missing" in str(exc_info.value)


def test_extra_signal_raises_value_error():
    signals = _make_signals()
    signals["bogus_extra_signal"] = SignalResult(active=False, value=0.0)

    with pytest.raises(ValueError) as exc_info:
        evaluate_confluence(signals)
    assert "bogus_extra_signal" in str(exc_info.value)
    assert "Unexpected" in str(exc_info.value)


def test_mandatory_category_not_in_categories_raises():
    signals = _make_signals()

    with pytest.raises(ValueError) as exc_info:
        evaluate_confluence(
            signals,
            mandatory=("regime", "nonexistent_cat"),
        )
    assert "nonexistent_cat" in str(exc_info.value)


def test_active_by_category_breakdown_is_correct():
    # 3 active in regime, 1 in momentum, 0 in the others.
    overrides = {
        "ema_trend": True, "atr_regime": True, "htf_alignment": True,
        "macd_bullish": True,
    }
    signals = _make_signals(active_overrides=overrides)
    result = evaluate_confluence(signals)

    assert result.active_by_category == {
        "regime": 3,
        "mean_reversion": 0,
        "momentum": 1,
        "volume": 0,
        "structure": 0,
    }


def test_active_signals_are_sorted_alphabetically():
    overrides = {
        "ema_trend": True,
        "macd_bullish": True,
        "atr_regime": True,
        "obv_trend": True,
        "rsi_oversold": True,
    }
    signals = _make_signals(active_overrides=overrides)
    result = evaluate_confluence(signals)

    assert result.active_signals == tuple(sorted(result.active_signals))
    assert result.active_signals == (
        "atr_regime",
        "ema_trend",
        "macd_bullish",
        "obv_trend",
        "rsi_oversold",
    )


def test_all_twelve_inactive_full_data():
    signals = _make_signals()
    result = evaluate_confluence(signals)

    assert result.score == 0
    assert result.threshold_met is False
    assert result.mandatory_met is False
    assert result.data_sufficient is True
    assert result.active_signals == ()
    assert result.enter is False


def test_empty_mandatory_tuple_is_vacuously_true():
    # 7 active but neither mandatory category in majority -> still enters
    # because mandatory=() makes the gate vacuously True.
    overrides = {
        "ema_trend": True,                                  # regime 1/3
        "macd_bullish": True,                               # momentum 1/3
        "rsi_oversold": True, "bb_lower": True, "far_below_ma": True,
        "obv_trend": True, "high_volume": True,
    }
    signals = _make_signals(active_overrides=overrides)
    result = evaluate_confluence(signals, mandatory=())

    assert result.score == 7
    assert result.threshold_met is True
    assert result.mandatory_met is True
    assert result.enter is True


def test_custom_categories_single_signal_mandatory():
    custom_cats = {
        "alpha": ("sig_a",),
        "beta": ("sig_b",),
        "gamma": ("sig_c",),
    }
    # Active sig_a only, mandatory=("alpha",), min_score=1.
    signals = {
        "sig_a": SignalResult(active=True, value=0.0),
        "sig_b": SignalResult(active=False, value=0.0),
        "sig_c": SignalResult(active=False, value=0.0),
    }
    result = evaluate_confluence(
        signals,
        min_score=1,
        categories=custom_cats,
        mandatory=("alpha",),
    )

    assert result.score == 1
    assert result.active_by_category == {"alpha": 1, "beta": 0, "gamma": 0}
    assert result.mandatory_met is True  # 1 * 2 > 1
    assert result.threshold_met is True
    assert result.enter is True

    # Same custom cats but mandatory=("beta",) -> 0 * 2 > 1 is False.
    result2 = evaluate_confluence(
        signals,
        min_score=1,
        categories=custom_cats,
        mandatory=("beta",),
    )
    assert result2.mandatory_met is False
    assert result2.enter is False


def test_constants_have_expected_shape():
    # Sanity check on the locked v1 constants.
    assert TOTAL_SIGNALS == 12
    assert DEFAULT_MIN_SCORE == 7
    assert MANDATORY_CATEGORIES == ("regime", "momentum")
    total = sum(len(v) for v in CATEGORIES.values())
    assert total == TOTAL_SIGNALS
