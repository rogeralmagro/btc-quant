"""Confluence scorer for STRAT-07 multi-signal long entries.

Orchestrates the 12 individual signals from src/btc_quant/signals/
into a single binary entry decision based on:
- A minimum count of active signals (default 7/12)
- Mandatory majority (>50%) in specified categories
- Optional requirement that all signals have non-NaN values
"""

from dataclasses import dataclass
from math import isnan
from typing import Dict, Optional, Tuple

from btc_quant.signals.types import SignalResult


CATEGORIES: Dict[str, Tuple[str, ...]] = {
    "regime":         ("ema_trend", "atr_regime", "htf_alignment"),
    "mean_reversion": ("rsi_oversold", "bb_lower", "far_below_ma"),
    "momentum":       ("macd_bullish", "adx_trending", "roc_positive"),
    "volume":         ("obv_trend", "high_volume"),
    "structure":      ("market_structure",),
}

MANDATORY_CATEGORIES: Tuple[str, ...] = ("regime", "momentum")

DEFAULT_MIN_SCORE: int = 7
TOTAL_SIGNALS: int = 12


@dataclass(frozen=True)
class ConfluenceResult:
    """Result of evaluating the confluence scorer.

    Attributes:
        enter: Final entry decision. True only if all gates pass:
               threshold_met AND mandatory_met AND
               (data_sufficient OR not require_full_data).
        score: Number of signals with active=True (0..12).
        threshold_met: score >= min_score.
        mandatory_met: All mandatory categories have majority active.
        data_sufficient: All signals have non-NaN value.
        active_by_category: Mapping category name -> count of active.
        active_signals: Names of signals with active=True.
    """
    enter: bool
    score: int
    threshold_met: bool
    mandatory_met: bool
    data_sufficient: bool
    active_by_category: Dict[str, int]
    active_signals: Tuple[str, ...]


def evaluate_confluence(
    signals: Dict[str, SignalResult],
    min_score: int = DEFAULT_MIN_SCORE,
    categories: Optional[Dict[str, Tuple[str, ...]]] = None,
    mandatory: Optional[Tuple[str, ...]] = None,
    require_full_data: bool = True,
) -> ConfluenceResult:
    """Evaluate the confluence of STRAT-07 signals.

    Args:
        signals: Dict mapping signal name to SignalResult. Must
                 contain EXACTLY the union of all signal names
                 listed in `categories`.
        min_score: Minimum count of active signals required.
                   Default DEFAULT_MIN_SCORE (7).
        categories: Override the category structure. Default uses
                    CATEGORIES (STRAT-07 v1).
        mandatory: Override mandatory categories. Default uses
                   MANDATORY_CATEGORIES (regime, momentum).
        require_full_data: If True, enter requires data_sufficient.
                           If False, enter ignores data_sufficient
                           (but the flag is still reported).

    Returns:
        ConfluenceResult with the decision and breakdown.

    Raises:
        ValueError: If signals dict is missing any name listed in
                    categories, or contains unexpected names.
        ValueError: If any name in mandatory is not in categories.
    """
    cats = categories if categories is not None else CATEGORIES
    mand = mandatory if mandatory is not None else MANDATORY_CATEGORIES

    expected_signals = set()
    for cat_signals in cats.values():
        expected_signals.update(cat_signals)

    provided_signals = set(signals.keys())
    missing = expected_signals - provided_signals
    extra = provided_signals - expected_signals
    if missing:
        raise ValueError(f"Missing signal(s): {sorted(missing)}")
    if extra:
        raise ValueError(f"Unexpected signal(s): {sorted(extra)}")

    for cat_name in mand:
        if cat_name not in cats:
            raise ValueError(
                f"Mandatory category '{cat_name}' not in categories"
            )

    active_signals = tuple(
        sorted(name for name, res in signals.items() if res.active)
    )
    score = len(active_signals)

    active_by_category: Dict[str, int] = {}
    for cat_name, cat_sig_names in cats.items():
        active_by_category[cat_name] = sum(
            1 for name in cat_sig_names if signals[name].active
        )

    threshold_met = score >= min_score

    # Majority (>50%) per mandatory category: active * 2 > size.
    mandatory_met = all(
        active_by_category[cat_name] * 2 > len(cats[cat_name])
        for cat_name in mand
    )

    data_sufficient = all(
        not isnan(res.value) for res in signals.values()
    )

    data_gate = data_sufficient or not require_full_data
    enter = threshold_met and mandatory_met and data_gate

    return ConfluenceResult(
        enter=enter,
        score=score,
        threshold_met=threshold_met,
        mandatory_met=mandatory_met,
        data_sufficient=data_sufficient,
        active_by_category=active_by_category,
        active_signals=active_signals,
    )
