# STRAT-06 Design Refinement #001: Concentration Limit Removal

## Date
2026-05-11

## Trigger
First historical backtest (2017-09 to 2026-05, €52,500 capital, €500/month inflow)
revealed that STRAT-06 left €36,210 (69% of injected capital) idle in cash pools.

## Root Cause Analysis

Mechanism of failure:
1. RESERVE pool reached its €1,500 cap within ~12 months. Subsequent inflows
   to reserve (€9,237 total over the backtest) overflowed to BUFFER.
2. BTC concentration in portfolio exceeded 70% on 2017-11-27 (~3 months into
   the backtest) and never recovered to below 70% during sustained bull
   periods.
3. With max_concentration_pct=0.70 (from SYSTEM_FINAL §4), 363 of 455 eligible
   baseline-buy Mondays (79.8%) were blocked.
4. The BUFFER pool, which only deploys via modulation_transfer triggered
   during a baseline buy, could not contribute when baseline was blocked.
   Only 6.3% of BUFFER capital was eventually deployed.

Diagnostic data:
- BASELINE buys executed: 90 (of 455 eligible Mondays)
- Baseline buys blocked by concentration: 363
- BUFFER deployed via modulation_transfer: €1,252 (6.3% of €19,737 received)
- RESERVE deployed via tranches: 8 deployments totaling €2,388 (in 2018 and
  2022 bear markets, as designed)
- Final cash idle: €36,210 (69% of injected capital)

## Interpretation

The max_concentration_pct guard from SYSTEM_FINAL §4 was conceived for a
diversified portfolio context where limiting BTC exposure makes sense as a
risk control. Applied to STRAT-06 — a pure accumulation strategy with no
exit logic and a multi-year horizon — the guard becomes pathological:

- The user's INTENT in STRAT-06 is to accumulate BTC over years
- Cash held in pools is "capital pending deployment", not "diversification"
- BTC appreciation naturally drives concentration upward, but this is the
  desired state, not a risk to manage
- Once concentration exceeds threshold, recovery requires either BTC crash
  (uncommon in long-term BTC trajectory) or massive new inflows (limited
  by €500/month)

## Decision

Set max_concentration_pct default to 1.0 in DCAModulatedConfig (effectively
disabling the concentration check for STRAT-06). The field is retained in
case future variants of the strategy need it.

## Rationale Why This Is Not Overfitting

1. The change is NOT to optimize a backtest metric. It is to fix a
   pathological behavior where the strategy fails to invest its stated
   capital.
2. The change does not optimize any threshold parameter on observed historical
   data. It removes a guard that prevented the strategy from functioning as
   designed.
3. The decision is made BEFORE seeing comparative metrics vs benchmarks. We
   do not yet know whether the corrected STRAT-06 will outperform DCA or
   BuyAndHold. The change is a correctness fix, not a performance optimization.
4. Documented before code change, in compliance with no-override and process
   disciplines.

## Implications for SYSTEM_FINAL

SYSTEM_FINAL §4 (Concentration Limits) needs clarification:
- The "70% portfolio concentration limit" applies to SYSTEM-WIDE allocation
  decisions, NOT to individual strategy execution
- When STRAT-07 (tactical layer) is active alongside STRAT-06, system-level
  concentration management can be reintroduced via the Tactical strategy's
  sizing decisions, not via STRAT-06's internal guards

This clarification will be incorporated into SYSTEM_FINAL when STRAT-07 is
implemented (Phase F6+).

## Alternative Options Considered

Option B — Recompute concentration over cumulative capital invested
(BTC_value / capital_invested):
Rejected. Changes semantic meaning of the field and may confuse future
maintainers. The metric is also less intuitive than market-value concentration.

Option C — Bypass concentration check for small baseline buys:
Rejected. Adds complexity without addressing root cause. Pathological behavior
would persist for larger modulated buys.

## Validation Plan

After implementing this change:
1. Re-run scripts/run_strat06_comparative_backtest.py
2. Verify total injected capital is fully deployed (or near-fully, with only
   minor residual cash for the next pending buy)
3. Compare STRAT-06 metrics vs DCA Benchmark and Buy-and-Hold
4. Document the new comparative results in STRAT_06_HISTORICAL_VALIDATION.md

## Status

DECISION APPROVED — implementation pending.
