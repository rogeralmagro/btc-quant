# Changelog

All notable changes to this project will be documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
