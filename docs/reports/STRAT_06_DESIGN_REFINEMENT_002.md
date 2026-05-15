# STRAT-06 Design Refinement #002: Cost Basis Reporting Fix

## Status

APPROVED — reporting bug fix, not strategy change.

## Date

2026-05-15

## Discovery Context

Detected during F5.1 cross-validation: Pine Script (BINANCE:BTCUSDT 1D, 2017-08-17 →
2026-05-06) vs Python backtest.

| Metric | Pine Script | Python (buggy) | Δ% |
|---|---|---|---|
| BAH cost basis | €4,289 | N/A | — |
| DCA cost basis | €14,659 | N/A | — |
| STRAT-06 cost basis | **€14,165** | **€8,775** | **+61.4%** |

BAH and DCA cost basis figures matched within 2%. Discrepancy isolated to STRAT-06
as the only strategy with inter-pool capital transfers (BUFFER→BASELINE,
RESERVE→BASELINE).

---

## Root Cause

`MetricsCalculator.calculate()` (lines 153–157 before fix) computed cost basis as:

```python
invested = strat06.total_invested_eur          # ← bug
avg_cost = invested / strat06.btc_held
```

`CapitalPool.total_invested_eur` is incremented **only** by `add_cash()`:

```python
# pool.py line 65
def add_cash(self, amount: float, source: str) -> None:
    self.cash_eur += amount
    self.total_invested_eur += amount              # ← only add_cash updates this
```

Inter-pool transfers use `Portfolio.transfer_cash()`, which intentionally bypasses
`add_cash()` to avoid double-counting capital in the portfolio-level total:

```python
# portfolio.py line 116
to_pool.cash_eur += amount  # bypass add_cash to not inflate invested_eur
```

Result: `BASELINE.total_invested_eur` = only BASELINE's direct monthly inflows (55%
of €52,500 = **€28,875**), missing:

| Capital source | EUR | Captured by total_invested_eur? |
|---|---|---|
| Direct inflows (55% × €52,500) | €28,875 | ✓ yes |
| BUFFER → BASELINE (modulation) | €16,270 | ✗ no |
| RESERVE → BASELINE (tranches) | €2,388 | ✗ no |
| **Total capital used to buy BTC** | **€47,395** | |

Buggy cost basis: €28,875 / 3.2905 BTC = **€8,775/BTC**
(61% below the actual average purchase price)

---

## Why This Is a Reporting Bug, Not Overfitting

Per Manual §6.4 (Design Refinement criteria):

1. The fix corrects a **metric calculation error**, not a parameter choice. No
   threshold, percentage, or model parameter is changed.
2. The fix **worsens** the reported cost basis metric from €8,775 to €14,389 —
   unambiguously not an optimization.
3. The strategy's actual execution (buys, amounts, dates, BTC accumulated) is
   completely unchanged. All other metrics (Sharpe, Sortino, MaxDD, final value,
   BTC held) are identical.
4. The root cause is structural: `total_invested_eur` was designed to prevent
   double-counting at the portfolio level, and the MetricsCalculator incorrectly
   used it as a cost-basis numerator.

---

## Fix

`src/btc_quant/backtester/metrics/calculator.py` lines 153–163 (after fix):

```python
# Before (buggy):
invested = strat06.total_invested_eur
avg_cost = invested / strat06.btc_held
btc_per_eur = strat06.btc_held / invested if invested > 0 else None

# After (correct):
avg_cost = strat06.average_entry_price()
# cost_basis_eur / btc_held — includes inter-pool transfers via add_btc()
btc_per_eur = (
    strat06.btc_held / strat06.cost_basis_eur
    if strat06.cost_basis_eur > 0
    else None
)
```

`CapitalPool.average_entry_price()` returns `cost_basis_eur / btc_held`.
`cost_basis_eur` is updated by `add_btc(qty, cost_eur)` every time a BTC purchase
executes, regardless of which pool originally funded the buy. It therefore captures
the full economic cost.

---

## Regression Test

`tests/unit/backtester/metrics/test_calculator.py`
→ class `TestAvgCostBasisInterPoolTransfers` (3 tests)

Key scenario:
- Pool receives €1,000 direct inflow + €500 inter-pool transfer
- Buys 0.01 BTC with all €1,500
- Correct cost basis: €1,500 / 0.01 = **€150,000/BTC** ← asserted
- Buggy cost basis:   €1,000 / 0.01 = €100,000/BTC ← would fail on old code

---

## Impact on Strategy Narrative

**Previous claim (incorrect):** "STRAT-06 achieves 41% better cost basis than DCA
(€8,775 vs €14,973), demonstrating the anti-cyclical modulation effect."

**Corrected (2026-05-15):** "STRAT-06 achieves ~3.8% better cost basis than DCA
(€14,389 vs €14,973), a modest improvement consistent with marginally more BTC
purchased at depressed prices via the multiplier and reserve deployments."

The strategy still validates on risk-adjusted metrics — unchanged by this fix:
- Sharpe: 1.34 vs DCA 1.32 (+0.02)
- Sortino: 2.17 vs DCA 2.07 (+0.10)
- Max Drawdown: −73.9% vs DCA −74.8% (+0.9 pp shallower)

---

## Affected Metrics

Only `avg_cost_basis_eur_per_btc` for STRAT-06. No other strategy or metric
is affected by this fix.

**Critical path note:** STRAT-07 will introduce Profit Recycling — another
inter-pool transfer mechanism. MetricsCalculator's use of `cost_basis_eur`
(not `total_invested_eur`) is now validated and safe for that extension.

---

## Cross-Validation Result After Fix

| Metric | Pine Script | Python (fixed) | Δ% | ±5%? |
|---|---|---|---|---|
| STRAT-06 cost basis | €14,165 | €14,389 | −1.6% | ✓ PASS |
| STRAT-06 BTC held | ~3.29 | 3.2905 | <0.1% | ✓ PASS |
| STRAT-06 portfolio value | — | €273,105 | unchanged | ✓ |

The −1.6% gap between Pine Script and corrected Python is explained by regime-aware
slippage in `ExecutionSimulatorV2`: Python buys at a slightly worse price than Pine's
theoretical close, raising the cost basis by ~1.6% over 8.5 years.

---

## Reference Commits

Bug present in: `ba7edf138fd3915cd0c5aac1ced8382245553d75`
Fix in: *(commit to be created by Roger after reviewing diffs)*
