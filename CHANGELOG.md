# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- F7.1: Category 1 (Regime) signal modules in
  `src/btc_quant/signals/regime/`:
  - `is_bullish_ema_trend(close, fast=50, slow=200) -> SignalResult`
  - `classify_atr_regime(high, low, close, atr_period=14,
    rolling_window=252) -> AtrRegimeResult` and wrapper
    `is_atr_regime_tradeable(...) -> SignalResult`
  - `is_htf_aligned_bullish(close_1d, close_1w, daily_fast=50,
    daily_slow=200, weekly_fast=10, weekly_slow=40) -> SignalResult`
- F7.1: Shared `SignalResult` dataclass in
  `src/btc_quant/signals/types.py` (frozen, active: bool +
  value: float)
- F7.1: `AtrRegime` enum and `AtrRegimeResult` dataclass for rich
  regime classification output
- F7.1: 33 unit tests on synthetic data and edge cases
  (948 passing total)
- F7.2: Category 2 (Mean Reversion) signal modules in
  `src/btc_quant/signals/mean_reversion/`. All three signals are
  long-only (no overbought / upper-band / above-MA branches):
  - `is_rsi_oversold(close, period=14, threshold=30.0) -> SignalResult`
    using Wilder's smoothing
  - `is_at_bb_lower(close, period=20, num_std=2.0) -> SignalResult`
    returning Bollinger %B as value
  - `is_far_below_ma(close, period=50, threshold_pct=-10.0) -> SignalResult`
    using SMA(50) with BTC-calibrated default threshold
- F7.2: 32 unit tests on synthetic data and edge cases
  (980 passing total)
- F7.3a: Category 3 (Momentum) signal modules in
  `src/btc_quant/signals/momentum/`. All long-only:
  - `is_macd_bullish(close, fast=12, slow=26, signal=9) -> SignalResult`
    with active=True when MACD > Signal at last bar
  - `is_bullish_trending(high, low, close, period=14, adx_threshold=25.0)
    -> SignalResult` with active=True only when both ADX > threshold
    AND +DI > -DI (combined trend strength + bullish direction)
  - `is_roc_positive(close, period=10, threshold_pct=0.0) -> SignalResult`
- F7.3a: 28 unit tests on synthetic data and edge cases
  (1008 passing total)
- F7.3b: Category 4 (Volume) signal modules in
  `src/btc_quant/signals/volume/`:
  - `is_obv_bullish_trend(close, volume, ma_period=20) -> SignalResult`
    with active=True when OBV > SMA(OBV, N)
  - `is_high_volume(volume, rolling_window=252, threshold_percentile=75.0)
    -> SignalResult`, direction-agnostic by design (high volume regime
    as confluence context, directional confirmation from Cat 1 + Cat 3)
- F7.3b: Category 5 (Structure) signal module in
  `src/btc_quant/signals/structure/`:
  - `is_bullish_market_structure(high, low, swing_lookback=5) -> SignalResult`
    using symmetric pivot detection, active=True when both higher highs
    AND higher lows on most recent confirmed swings
- F7.3b: 32 unit tests on synthetic data and edge cases (1040 passing total)
- F7 block complete: 12 individual signals across 5 categories implemented,
  ~130 new tests vs F6 baseline. Ready for F8 (ConfluenceScorer).

### Design notes
- ATR regime percentile uses midpoint convention: ties contribute
  0.5 each, so a constant historical window yields percentile=50
  (NORMAL), not 0 (CALM)
- All signal modules cast np.bool_/np.float64 to native bool/float
  for serialization safety
- F7.2 signals are strictly long-only by design (per
  STRAT_07_SCOPE_DECISION). Enforced via tests that assert
  active=False on bearish-extended inputs (overbought RSI,
  price above upper BB, price above SMA).
- Bollinger Bands use ddof=0 (population std) per industry TA
  convention for reproducibility with TradingView and other
  TA libraries.
- SMA includes the current bar in rolling window (pandas
  .rolling(N).mean() default behavior).
- threshold_pct=-10% in distance_from_ma is a BTC-calibrated
  default. Other markets may require recalibration.
- F7.3a ADX uses Wilder's smoothing throughout (DI+/DI-/ATR/ADX
  all use `ewm(alpha=1/period, adjust=False, min_periods=period)`),
  not standard EMA. This is the original Wilder formulation and
  matches reference TA implementations.
- F7.3a ADX requires ~2*period bars of data due to cascaded
  smoothing (DM/ATR → DX → ADX). Guard `len < period*2` returns
  NaN result.
- F7.3a ADX guard against `di_sum == 0` (perfectly flat price)
  to avoid NaN propagation from division by zero in DX.
- F7.3a ADX threshold test uses random walk with seed=42 plus
  dynamic threshold (adx ± 1) instead of a hardcoded value,
  to avoid degenerate ADX=100 from a perfect linear input.
  Test depends on numpy RNG implementation (rarely changes).
- F7.3b OBV uses np.sign(close.diff()) with iloc[0]=0 to handle
  first bar, then cumulative sum for vectorized computation.
- F7.3b high_volume reuses the midpoint percentile convention
  from F7.1 ATR regime: (n_below + 0.5*n_equal)/n_total*100,
  with trailing window that excludes the current bar.
- F7.3b market structure uses symmetric pivot detection
  (swing_lookback bars before AND after). Most recent confirmed
  swing is at least swing_lookback bars old (structural lag, by
  design). Strict > comparison for both pivot detection and HH/HL
  check (equal-level swings do NOT count).
- F7.3b structure tests use explicitly constructed synthetic data
  rather than transformations of bullish data: reversing a bullish
  series with [::-1] does not produce a clean bearish pattern in
  pivot detection (the logic is direction-agnostic over time).

## [0.5.0] - 2026-05-15 — STRAT-07 scope decided

### Decided

- F6: STRAT-07 v1 scope locked to 12 technical signals from SYSTEM_FINAL §2.3.3,
  organized in 5 categories (regime, mean reversion, momentum, volume, structure).
  On-chain signals (MVRV, NUPL, SOPR, exchange flows) deferred to v2, evaluated
  once after F14 paper trading completes. See
  `docs/decisions/STRAT_07_SCOPE_DECISION.md` for v1 scope, v2 activation criteria,
  and budget implications.

### Added

- `docs/decisions/` directory for binding architectural decisions
- `docs/decisions/STRAT_07_SCOPE_DECISION.md` — Option C approval, v1 scope lock,
  v2 activation criteria, out-of-scope list, budget implications, next phase (F7.1)

## [0.4.0] - 2026-05-15 — STRAT-06 validated

### Added
- F4: STRAT-06 (Drawdown-Modulated DCA + Deep Value Reserve)
  full implementation under `src/btc_quant/strategies/strat06/`
  (`ATHTracker`, `DrawdownMultiplier`, `ReserveManager`,
  `DCAModulatedStrategy`, `DCAModulatedConfig`)
- F4: BAH and DCA benchmark strategies for comparative analysis
- F4: Comparative backtest script
  (`scripts/run_strat06_comparative_backtest.py`)
- F4: Visual comparison notebook
  (`notebooks/04_strat06_visual_comparison.ipynb`)
- F4: Pine Script visualizations of BAH, DCA, STRAT-06 for
  TradingView cross-validation (`docs/tradingview/`)
- F5.1: Cross-validation report (`docs/reports/STRAT_06_CROSS_VALIDATION.md`)
- F5.1: Three regression tests for cost basis calculation
  with inter-pool transfers

### Changed
- F4: `ExecutionSimulator` upgraded to V2 with regime-aware slippage
- F4: `max_concentration_pct` default changed from 0.70 to 1.0
  for STRAT-06 (see `STRAT_06_DESIGN_REFINEMENT_001.md`)
- F5.1: STRAT-06 cost basis reporting in `MetricsCalculator` now
  uses `cost_basis_eur` (includes inter-pool transfers) instead of
  `total_invested_eur` (see `STRAT_06_DESIGN_REFINEMENT_002.md`).
  Corrected cost basis: €14,389/BTC (was reported €8,775/BTC).
  Strategy behavior unchanged; only reporting was affected.

### Fixed
- F4: `ZeroDivisionError` in `total_return_pct`, `excess_return_vs_bah_pct`,
  `excess_return_vs_dca_pct` for strategies starting with zero capital
- F4: Sharpe NaN for strategies with inflows
  (`calculate_returns` now skips leading zeros)
- F4: `max_drawdown_guard` division by zero when `running_peak == 0`
- F5.1: Cost basis calculation for strategies with inter-pool
  transfers (was using direct inflows only)

### Validated
- STRAT-06 historical backtest 2017-08-17 → 2026-05-06
  (8.5 years, 3,185 daily bars)
- 8 reserve deployments in 2018 and 2022 bear markets as designed
- Pine Script ↔ Python cross-validation passes within ±5% on all
  strategies after F5.1 bug fix
- 915 unit tests passing, 0 failures

### Moved
- `scripts/diagnose_strat06_deployment.py` → `archive/`
  (F4 development tool, no longer in active scripts)

## [0.1.0] — 2026-05-07

### Added
- Initial project scaffold: folder structure, pyproject.toml, .gitignore
- Package skeleton: `data`, `indicators`, `strategies`, `risk`, `backtester`, `execution`, `monitoring`
- `scripts/hello_binance.py`: fetches latest BTCUSDT 1D candle from Binance public API
- `tests/unit/test_setup.py`: smoke test for package import
