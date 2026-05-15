# STRAT-06 F5.1 Cross-Validation: Pine Script vs Python

## Date

2026-05-15

## Methodology

Three Pine Scripts applied to BINANCE:BTCUSDT 1D chart over the historical range
2017-08-17 → 2026-05-06. End-state numerical comparison against Python backtest
with ±5% tolerance.

| Script | File |
|---|---|
| BAH Benchmark | `docs/tradingview/strat06_bah.pine` |
| DCA Benchmark | `docs/tradingview/strat06_dca_benchmark.pine` |
| STRAT-06 | `docs/tradingview/strat06_main.pine` |

Python reference: `data/reports/strat06_comparative_20260515_140501/comparative_summary.json`

---

## Results

### Buy-and-Hold

| Metric | TradingView | Python | Δ% | ±5%? |
|---|---|---|---|---|
| Final portfolio | ~€986k | €986,072 | <0.1% | ✓ |
| BTC held | ~12.10 | 12.1063 | <0.1% | ✓ |
| Cost basis €/BTC | €4,289 | €4,337* | −1.1% | ✓ |

*Python BAH cost basis derived as total_invested / btc_accumulated = 52500 / 12.1063.

### DCA Benchmark

| Metric | TradingView | Python | Δ% | ±5%? |
|---|---|---|---|---|
| BTC held | ~3.51 | 3.5063 | <0.1% | ✓ |
| Cost basis €/BTC | €14,659 | €14,852* | −1.3% | ✓ |

*Python DCA cost basis derived as (total_invested − cash_remaining) / btc_accumulated
= (52,500 − 424) / 3.5063.

### STRAT-06

| Metric | TradingView | Python (fixed) | Δ% | ±5%? |
|---|---|---|---|---|
| BTC held | ~3.29 | 3.2905 | <0.1% | ✓ |
| Cost basis €/BTC | €14,165 | €14,389 | −1.6% | ✓ |
| BASELINE cash | ~€138 | €138 | <1% | ✓ |
| BUFFER cash | ~€3,464 | €3,468 | <0.1% | ✓ |
| RESERVE cash | €1,500 | €1,500 | 0.0% | ✓ |

---

## Findings

### BAH and DCA

All verifiable metrics pass within ±5%. Small residual differences (~1-2%) are
consistent with execution model differences: Pine uses theoretical close price,
Python uses `ExecutionSimulatorV2` with regime-aware slippage.

### STRAT-06

**Reserve deployment dates (Verification 1):** 8 red triangle markers (▲) appeared
in the Pine Script at the correct bear-market dates. Visual confirmation:

| Date | Expected DD | Expected deploy | TV marker |
|---|---|---|---|
| 2018-02-07 | ≈ −60% | €150 | ✓ |
| 2018-04-04 | ≈ −64% | €213 | ✓ |
| 2018-11-22 | ≈ −77% | €375 | ✓ |
| 2018-11-26 | ≈ −76% | €338 | ✓ |
| 2022-05-15 | ≈ −54% | €300 | ✓ |
| 2022-06-18 | ≈ −72% | €331 | ✓ |
| 2022-07-05 | ≈ −70% | €280 | ✓ |
| 2022-11-22 | ≈ −76% | €402 | ✓ |

**Pool balances (Verification 3):** RESERVE cap (€1,500), BUFFER dry powder (€3,468),
and BASELINE residual (€138) all match within 1%.

**Cost basis (Verification 2):** Initially FAILED (+61.4% divergence). Root cause
identified as a reporting bug in `MetricsCalculator` — the strategy's actual execution
was correct. See `STRAT_06_DESIGN_REFINEMENT_002.md`. After fix: −1.6% ✓ PASS.

---

## Residual Differences (1–2% across all strategies)

All remaining small differences trace to a single systematic cause: Pine Script
executes buys at the theoretical daily close price (no slippage), while
`ExecutionSimulatorV2` applies regime-aware slippage on every order. Over 8.5 years
and ~400+ buy events, this compounds to a ~1–2% higher cost basis in Python vs Pine.
This is expected and correct behavior.

---

## Verdict

**F5.1 PASSED** (after cost basis bug fix).

- 3/3 strategies: BTC accumulation within ±0.1% ✓
- 3/3 strategies: cost basis within ±2% ✓
- STRAT-06 reserve deployment dates: 8/8 correct ✓
- STRAT-06 pool balances: 3/3 within ±1% ✓

The cross-validation confirms that the Python backtest engine and the Pine Script
simulation agree on strategy behavior. Differences are fully explained by the
slippage model and are within expected bounds.
