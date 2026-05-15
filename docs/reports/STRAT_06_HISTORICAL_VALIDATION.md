# STRAT-06 Historical Validation Report

**Version:** 1.1  
**Date:** 2026-05-11 (updated 2026-05-15)  
**Backtest run:** `data/reports/strat06_comparative_20260515_140501/`  
**Author:** btc-quant project

> **Note (2026-05-15):** Cost basis figures for STRAT-06 corrected following bug fix
> in `src/btc_quant/backtester/metrics/calculator.py` (see
> `docs/reports/STRAT_06_DESIGN_REFINEMENT_002.md`). Previously reported €8,775/BTC
> was incorrect — `total_invested_eur` only captured direct BASELINE inflows, ignoring
> €18,658 transferred from BUFFER and RESERVE. Corrected value: **€14,389/BTC**.
> Strategy execution and all other metrics are unchanged.

---

## 1. Executive Summary

STRAT-06 (Drawdown-Modulated DCA + Deep Value Reserve) was backtested against two
benchmarks over 8.5 years of BTC/USDT daily data (2017-08-17 → 2026-05-06).

| Strategy | Final Value | Return | Max Drawdown | Sharpe | Sortino |
|---|---|---|---|---|---|
| Buy-and-Hold | €986,072 | +1,778% | −83.2% | 0.84 | 1.13 |
| DCA Benchmark | €285,998 | +445% | −74.8% | 1.32 | 2.07 |
| **STRAT-06** | **€273,105** | **+420%** | **−73.9%** | **1.34** | **2.17** |

**Key finding:** STRAT-06 achieves marginally better risk-adjusted performance than
DCA (Sharpe +0.02, Sortino +0.10, MaxDD +0.9 pp shallower) at the cost of a 4.5%
lower final portfolio value. The strategy deployed 90.3% of injected capital
(vs 99.2% for DCA) — a deliberate trade-off reflecting the reserve design.

---

## 2. Methodology

### 2.1 Data

- **Source:** `data/processed/btcusdt_1d.parquet`
- **Timeframe:** 1-day (OHLCV), 3,185 bars
- **Period:** 2017-08-17 → 2026-05-06 (8.73 years)

### 2.2 Capital Fairness

All three strategies received identical capital: €52,500 (105 monthly inflows of €500).

- **Buy-and-Hold:** lump-sum €52,500 deployed on the first bar.
- **DCA Benchmark:** €500/month via `InflowScheduler.monthly()`, buying €115.38/week
  (= €500 × 12 / 52, the annualised weekly equivalent).
- **STRAT-06:** €500/month split 55% / 20% / 25% across BASELINE / BUFFER / RESERVE.

### 2.3 Execution Model

All three strategies used `ExecutionSimulatorV2` (regime-aware slippage, 0.1% fee per
side). The `RegimeDetector` uses the same 1-day dataset for slippage calibration.

### 2.4 Design Refinement Applied (pre-backtest)

Before running this backtest, `max_concentration_pct` was set to 1.0 (disabled) per
`docs/reports/STRAT_06_DESIGN_REFINEMENT_001.md`. The decision was made on diagnostic
grounds (a preliminary run revealed 69% idle capital), documented before any comparative
metrics were visible.

---

## 3. Strategy Results

### 3.1 Buy-and-Hold

- **Invested:** €52,500 (lump-sum, 2017-08-17)
- **Final value:** €986,072
- **BTC accumulated:** 12.1063
- **Return:** +1,778.2% | **CAGR:** 40.0%
- **Max drawdown:** −83.2% | **Sharpe:** 0.84 | **Sortino:** 1.13
- **Idle cash:** €52.50 (fee residual)

Buy-and-Hold benefits from the full 2017–2026 BTC appreciation cycle. Its inferior
risk-adjusted ratios (Sharpe 0.84) reflect the extreme volatility of holding a single
asset undiversified, including the −83% drawdown in 2018 and the −74% drawdown in 2022.

### 3.2 DCA Benchmark (Pure Weekly DCA)

- **Invested:** €52,500 (105 monthly inflows)
- **Final value:** €285,998
- **BTC accumulated:** 3.5063
- **Return:** +444.8% | **CAGR:** N/A (periodic inflows; IRR not implemented)
- **Max drawdown:** −74.8% | **Sharpe:** 1.32 | **Sortino:** 2.07
- **Idle cash:** €423.52 (pending-buy buffer, ~0.8%)

DCA reduces exposure to timing risk by averaging purchases across the full period.
Its Sharpe (1.32) and Sortino (2.07) substantially exceed Buy-and-Hold, confirming
the risk-reduction benefit of cost averaging over a volatile asset.

### 3.3 STRAT-06 (Drawdown-Modulated DCA + Reserve)

- **Invested:** €52,500 (105 monthly inflows)
- **Final value:** €273,105
- **BTC accumulated:** 3.2905 at avg cost basis €14,389/BTC
- **Return:** +420.2% | **CAGR:** N/A
- **Max drawdown:** −73.9% | **Sharpe:** 1.34 | **Sortino:** 2.17
- **Idle cash:** €5,105.35 (9.7%)

---

## 4. Capital Deployment Analysis

### 4.1 Pool Accounting

| Pool | Inflows received | Transfers in | Deployed to BTC | Transfers out | Final cash |
|---|---|---|---|---|---|
| BASELINE | €28,875 (55%) | €16,270 + €2,388 | €44,990 | — | €138 |
| BUFFER | €10,500 (20%) | €9,237 overflow | — | €16,270 | €3,468 |
| RESERVE | €13,125 (25%) | — | — | €11,625 | €1,500 |

### 4.2 Baseline Buy Execution

Of 455 eligible Mondays:

| Outcome | Count | % |
|---|---|---|
| Buy fired | 398 | 87.5% |
| Blocked (insufficient cash) | 57 | 12.5% |
| Blocked (concentration guard) | 0 | 0% |

Of the 398 executed baseline buys:
- Multiplier 1.0× (unmodulated): 85 (21.4%)
- Multiplier >1.0× (modulated): 313 (78.6%)
- Mean multiplier across all buys: 1.808×
- Mean multiplier for modulated buys: 2.028×
- Total EUR deployed: €45,203

The 57 cash-blocked Mondays occur when the BASELINE pool's accumulated inflows have
been consumed by prior weeks' purchases and the next monthly inflow has not yet arrived.
This is normal cash-flow timing, not a design flaw.

### 4.3 Reserve Deployments

8 tranche deployments across 2 bear cycles:

| Date | Drawdown | Deployed | BTC price |
|---|---|---|---|
| 2018-02-07 | −60.2% | €150.00 | €7,599 |
| 2018-04-04 | −64.4% | €212.50 | €6,796 |
| 2018-11-22 | −77.1% | €375.00 | €4,370 |
| 2018-11-26 | −79.8% | €337.50 | €3,862 |
| 2022-05-15 | −53.6% | €300.00 | €31,329 |
| 2022-06-18 | −71.9% | €331.25 | €18,971 |
| 2022-07-05 | −70.1% | €279.69 | €20,176 |
| 2022-11-22 | −76.0% | €401.72 | €16,227 |

**Total deployed by reserve:** €2,388 at deeply discounted prices (€3,862–€31,329/BTC).
The reserve correctly fired in both major bear markets and did not fire during the
non-qualifying sell-offs of 2019–2021.

---

## 5. Risk Analysis

### 5.1 Drawdown Profile

All three strategies experience deep drawdowns due to BTC's inherent volatility:

| Strategy | Max Drawdown | 2018 trough | 2022 trough |
|---|---|---|---|
| Buy-and-Hold | −83.2% | severe | severe |
| DCA Benchmark | −74.8% | moderate | moderate |
| STRAT-06 | **−73.9%** | moderate | moderate |

STRAT-06's slightly shallower drawdown (−73.9% vs −74.8% for DCA) is consistent with
the reserve design: capital deployed at the deepest drawdowns acquires BTC at lower
cost, slightly smoothing the portfolio's decline relative to the buying price.

### 5.2 Volatility and Risk-Adjusted Returns

STRAT-06's Sharpe ratio (1.34) exceeds both DCA (1.32) and Buy-and-Hold (0.84). The
Sortino ratio improvement (2.17 vs 2.07) indicates STRAT-06's downside deviation is
lower — the modulated buying reduces the severity of losses relative to gains.

### 5.3 Residual Idle Cash

The €5,105 idle cash (9.7%) breaks down as:
- **€138 BASELINE** — essentially zero; represents the sub-minimum-order residual
  from the last few buys before period end.
- **€3,468 BUFFER** — reserve overflow capital held as dry powder for the next
  bear cycle. The reserve cap (€1,500) redirected €9,237 of excess reserve inflows
  to BUFFER over 8.5 years, of which 82.4% (€16,270) was deployed via modulation.
  The residual €3,468 is intentional: it will deploy on the next deep drawdown.
- **€1,500 RESERVE** — at cap by design; the reserve holds this permanently as
  the base position for the next crisis cycle.

---

## 6. Comparative Conclusion

### 6.1 STRAT-06 vs DCA: Verdict

STRAT-06 delivers **marginally better risk-adjusted performance** than pure DCA at the
cost of **slightly lower absolute returns**:

| Metric | Direction | Magnitude |
|---|---|---|
| Final value | STRAT-06 lower | −€12,893 (−4.5%) |
| Total return | STRAT-06 lower | −24.6 pp |
| Max drawdown | STRAT-06 better | +0.9 pp shallower |
| Sharpe ratio | STRAT-06 better | +0.02 |
| Sortino ratio | STRAT-06 better | +0.10 |
| Avg cost basis | STRAT-06 better | €14,389 vs €14,973 (−3.9%) |
| Fees paid | STRAT-06 lower | €47 vs €52 |

The cost basis advantage (−3.9%) is modest: STRAT-06 acquires BTC at a marginally
lower average price than DCA through its modulated buying and reserve deployments,
but the anti-cyclical bias is insufficient to overcome the drag of €5,105 held as
dry powder over an 8.5-year bull market. The return gap is structural.

### 6.2 STRAT-06 vs Buy-and-Hold: Verdict

Buy-and-Hold dominates by 2.9× in final value and 4.2× in total return, but with a
much lower Sharpe (0.84 vs 1.34) and 9.3 pp deeper max drawdown. For investors who
require lower drawdowns or who cannot stomach prolonged periods of unrealised losses,
STRAT-06 offers a meaningful risk improvement at the cost of leaving significant returns
on the table.

---

## 7. Known Limitations and Open Questions

1. **Single asset, single period:** The backtest covers one asset (BTC) over one
   contiguous historical period. No walk-forward test has been performed. Parameters
   are not optimised but they have only been validated on the same period used to
   design the strategy.

2. **No intra-month timing variation:** All buys execute on Monday at bar close.
   Different weekday or intra-week timing could produce different results.

3. **CAGR not computed for STRAT-06 or DCA:** CAGR assumes lump-sum investment.
   For periodic-inflow strategies, IRR is the correct metric; it is not implemented
   in V1 of `MetricsCalculator`.

4. **Reserve overflow to BUFFER:** Over 8.5 years, €9,237 overflowed from RESERVE
   to BUFFER. The design assumption was that the reserve cap would be large enough
   to absorb inflows between bear markets. In practice the cap (€1,500 = 12 months
   × €125/month) was too small for an 8.5-year bull run with no drawdown exceeding
   −80%. Future calibration may benefit from a larger `reserve_cap_months`.

5. **STRAT-07 not yet active:** The system design calls for a tactical overlay
   (STRAT-07) that would provide additional entry signals for the BUFFER pool. This
   backtest measures STRAT-06 in isolation; the combined system is expected to deploy
   BUFFER capital more actively.

---

## 8. Data and Reproducibility

| Artifact | Path |
|---|---|
| Backtest script | `scripts/run_strat06_comparative_backtest.py` |
| Diagnostic script | `scripts/diagnose_strat06_deployment.py` |
| Notebook | `notebooks/03_strat06_comparative_analysis.ipynb` |
| Equity curves | `data/reports/strat06_comparative_20260511_193601/equity_curve_*.csv` |
| Summary JSON | `data/reports/strat06_comparative_20260511_193601/comparative_summary.json` |
| Design refinement | `docs/reports/STRAT_06_DESIGN_REFINEMENT_001.md` |

To reproduce:
```bash
source .venv/bin/activate
python scripts/run_strat06_comparative_backtest.py
```
