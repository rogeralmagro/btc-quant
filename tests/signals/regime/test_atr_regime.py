import math

import numpy as np
import pandas as pd
import pytest

from btc_quant.signals.regime.atr_regime import (
    AtrRegime,
    AtrRegimeResult,
    classify_atr_regime,
    is_atr_regime_tradeable,
)


def _ohlc(n: int, base: float, spread: float):
    """Synthetic OHLC with constant spread (ATR ≈ spread)."""
    close = pd.Series(np.full(n, base), dtype=float)
    high = close + spread
    low = close - spread
    return high, low, close


def _ohlc_volatile_tail(n: int, base: float, calm_spread: float, spike_spread: float, spike_len: int = 14):
    """Most bars are calm; last spike_len bars have a much larger spread."""
    close = pd.Series(np.full(n, base), dtype=float)
    spread = pd.Series(np.full(n, calm_spread), dtype=float)
    spread.iloc[-spike_len:] = spike_spread
    high = close + spread
    low = close - spread
    return high, low, close


class TestAtrRegimeBullishSynthetic:
    """Normal-regime series should be active (tradeable)."""

    def test_normal_regime_is_active(self):
        # Uniform spread of 1 throughout → all ATR values equal → percentile ~50
        high, low, close = _ohlc(300, base=100, spread=1.0)
        result = classify_atr_regime(high, low, close)
        assert result.active is True
        assert result.regime == AtrRegime.NORMAL


class TestAtrRegimeBearishSynthetic:
    """Extreme-volatility tail should produce active=False."""

    def test_extreme_regime_blocks_entry(self):
        # Last 14 bars have 200× the spread of the prior 252 bars.
        high, low, close = _ohlc_volatile_tail(280, 100, calm_spread=1.0, spike_spread=200.0)
        result = classify_atr_regime(high, low, close)
        assert result.active is False
        assert result.regime == AtrRegime.EXTREME


class TestAtrRegimeSideways:
    """Gently oscillating series stays in NORMAL or CALM — active in either case."""

    def test_oscillating_is_active(self):
        t = np.arange(300)
        base = 100.0
        spread = 1.0 + 0.1 * np.sin(t * 0.1)  # tiny variation
        close = pd.Series(np.full(300, base), dtype=float)
        high = close + pd.Series(spread)
        low = close - pd.Series(spread)
        result = classify_atr_regime(high, low, close)
        assert result.active is True
        assert result.regime in (AtrRegime.CALM, AtrRegime.NORMAL, AtrRegime.VOLATILE)


class TestAtrRegimeInsufficientData:
    def test_too_short(self):
        # Need atr_period + rolling_window = 14 + 252 = 266 bars minimum
        high, low, close = _ohlc(265, base=100, spread=1.0)
        result = classify_atr_regime(high, low, close)
        assert result.active is False
        assert math.isnan(result.percentile)
        assert math.isnan(result.atr_value)

    def test_exactly_min_minus_one(self):
        high, low, close = _ohlc(265, base=100, spread=1.0)
        result = classify_atr_regime(high, low, close, atr_period=14, rolling_window=252)
        assert result.active is False
        assert math.isnan(result.percentile)


class TestAtrRegimeBadInput:
    def test_nan_in_close(self):
        high, low, close = _ohlc(300, base=100, spread=1.0)
        close.iloc[150] = float("nan")
        result = classify_atr_regime(high, low, close)
        # Either falls back to insufficient or handles gracefully — must not crash
        # and must not return a percentile outside [0, 100] when defined
        if math.isfinite(result.percentile):
            assert 0.0 <= result.percentile <= 100.0
        assert isinstance(result.active, bool)

    def test_inf_in_high(self):
        high, low, close = _ohlc(300, base=100, spread=1.0)
        high.iloc[10] = float("inf")
        result = classify_atr_regime(high, low, close)
        assert isinstance(result.active, bool)


class TestAtrRegimeBands:
    """One test per regime band using calibrated synthetic series."""

    def test_calm_band(self):
        # All bars have identical spread → current bar is NOT higher than any
        # prior bar → percentile = 0 → CALM.
        # We shrink the last bar's spread to guarantee it falls below all prior.
        n = 280
        close = pd.Series(np.full(n, 100.0))
        high = close + 2.0
        low = close - 2.0
        # Make last bar's spread tiny so it ranks below the entire window
        high.iloc[-1] = 100.0 + 0.01
        low.iloc[-1] = 100.0 - 0.01
        result = classify_atr_regime(high, low, close)
        # ATR is a rolling SMA so the last value still partially reflects prior
        # bars; percentile should be very low → CALM
        assert result.regime == AtrRegime.CALM
        assert result.percentile < 25

    def test_normal_band(self):
        # Uniform spread throughout → percentile ≈ 50 → NORMAL
        high, low, close = _ohlc(300, base=100, spread=1.0)
        result = classify_atr_regime(high, low, close)
        assert result.regime == AtrRegime.NORMAL
        assert 25 <= result.percentile < 75

    def test_volatile_band(self):
        # Current ATR should rank around the 85th percentile.
        # Strategy: calm base, then a moderate spike in the last few bars
        # (less extreme than extreme_band).
        n = 280
        close = pd.Series(np.full(n, 100.0), dtype=float)
        spread = np.full(n, 1.0)
        # Spike the last 14 bars to ~4× the calm spread
        spread[-14:] = 4.0
        high = close + pd.Series(spread)
        low = close - pd.Series(spread)
        result = classify_atr_regime(high, low, close)
        # ATR SMA averages the spike across 14 bars; percentile should be high
        # but not necessarily ≥95 with only a 4× spike.
        assert result.regime in (AtrRegime.VOLATILE, AtrRegime.EXTREME)
        assert result.percentile >= 75
        assert result.active is (result.regime != AtrRegime.EXTREME)

    def test_extreme_band(self):
        # A 200× spike in the last 14 bars guarantees percentile ≥ 95 → EXTREME.
        high, low, close = _ohlc_volatile_tail(280, 100, calm_spread=1.0, spike_spread=200.0)
        result = classify_atr_regime(high, low, close)
        assert result.regime == AtrRegime.EXTREME
        assert result.percentile >= 95
        assert result.active is False


class TestIsAtrRegimeTradeable:
    def test_wrapper_returns_signal_result(self):
        high, low, close = _ohlc(300, base=100, spread=1.0)
        result = is_atr_regime_tradeable(high, low, close)
        assert isinstance(result.active, bool)
        assert math.isfinite(result.value)

    def test_wrapper_extreme_is_inactive(self):
        high, low, close = _ohlc_volatile_tail(280, 100, calm_spread=1.0, spike_spread=200.0)
        result = is_atr_regime_tradeable(high, low, close)
        assert result.active is False

    def test_wrapper_insufficient_data(self):
        high, low, close = _ohlc(265, base=100, spread=1.0)
        result = is_atr_regime_tradeable(high, low, close)
        assert result.active is False
        assert math.isnan(result.value)
