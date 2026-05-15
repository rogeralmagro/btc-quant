# STRAT-07 Scope Decision: Technical Signals v1, On-Chain Deferred to v2

## Status

APPROVED — Option C.

## Date

2026-05-15

## The Question

Should STRAT-07 v1 include on-chain signals (MVRV, NUPL, SOPR, exchange
flows, etc.) in addition to the 12 technical signals already specified
in SYSTEM_FINAL, or should it be limited to technical signals only with
on-chain deferred to a v2 iteration?

## Options Considered

| Dimension | A: Technical only | B: Technical + on-chain | C: Technical first, on-chain v2 |
|---|---|---|---|
| Sessions to F10 | 4-5 | 6-8 | 4-5 |
| New monthly cost | €0 | €30-50 (Glassnode) | €0 now, €30-50 if v2 activates |
| Parameters to calibrate | ~25 | ~40 | ~25 initial |
| Overfitting risk (Manual sec 7.1) | Medium | High | Medium |
| New failure modes | 0 | On-chain API outage, data gaps, latency | 0 now |
| Alignment with contrarian/value preference | Partial | Full | Full, deferred |
| Reproducibility | High | Medium (external provider dependency) | High |
| Risk that "v2 never happens" | N/A | N/A | Real if not gated |

## Decision

**Option C**, with explicit v2 activation criteria.

## Reasoning

1. **Project discipline.** Manual section 7.1 lists "overfitting in STRAT-07"
   as High × High risk. Adding 4 signals from a new category to a system
   that already has 12 multiplies the parameter space and the overfitting
   surface. F10's OOS walk-forward becomes harder, not easier.

2. **Principled phasing, not lazy deferral.** The only honest way to measure
   whether on-chain adds marginal value over technical-only is to compare
   against a working v1. Starting with both prevents isolating the
   contribution of each block. If v1 yields Sharpe 1.4 OOS and v2 yields
   Sharpe 1.45 OOS, we know on-chain contributes 0.05. Starting with both
   makes attribution impossible.

3. **Capital discipline.** €30-50/month for Glassnode during 4-8 months of
   development plus 3-6 months of paper trading equals €210-700 spent
   before a single real euro is traded. On an initial real-money cap of
   €300-500, this is not trivial.

4. **v2 must be real, not aspirational.** The activation criteria below
   make this concrete.

## v1 Scope (LOCKED — not renegotiable during F7-F10)

- **Signals:** the 12 technical signals from SYSTEM_FINAL §2.3.3, organized
  in 5 categories (regime, mean reversion, momentum, volume, structure).
- **Score threshold:** 7/12 minimum for entry.
- **Mandatory categories:** Category 1 (regime favorable) AND Category 3
  (momentum confirms).
- **Risk management:** as specified in SYSTEM_FINAL §2.3.4
  (sizing 2% capital, SL 2x ATR(14), TPs at 1R/2R/trailing).
- **Single concurrent position** (max_concurrent = 1).

## v2 Activation Criteria

v2 is evaluated **once**, in a specific window: **after F14 (paper trading
period) completes, before F15 (pre-real-money audit)**. Not earlier.

Rationale:

- If v1 fails F10 walk-forward → no v2 without first redesigning v1.
- If v1 passes F10 but fails F14 paper → reopen design before adding
  complexity.
- If v1 passes both F10 and F14 → then the question of whether on-chain
  adds incremental value can be answered meaningfully.

Concrete criteria when the window opens:

| v1 Result | v2 Decision |
|---|---|
| v1 exceeds targets clearly (Sharpe OOS > 1.5, win rate > 50%) | v2 OPTIONAL — system already competitive, on-chain is enhancement |
| v1 meets targets marginally (Sharpe OOS 1.0-1.5, win rate 40-50%) | v2 RECOMMENDED — on-chain likely improves robustness |
| v1 fails targets | v2 SUSPENDED — redesign v1 before stacking signals |

## Explicitly Out of Scope for v1

The following are not used, not implemented, not integrated in v1:

- Glassnode, CryptoQuant, Coin Metrics, blockchain.com APIs
- MVRV, NUPL, SOPR, Puell Multiple, exchange flows, miner flows
- Any data source other than OHLCV from Binance already stored in
  `data/processed/`
- Any modification to SYSTEM_FINAL §2.3.3 signal categories or weights

## Budget Implications

- F7-F10 (v1 implementation): €0 new costs. Uses existing data pipeline.
- F11-F14 (system integration + paper trading): €0 new costs.
- F15-F16 (real money): capital only (€300-500 initial as agreed).
- v2 budget reserved: €30-50/month if activated post-F14, for Glassnode
  or equivalent on-chain data provider.

## Next Phase

**F7.1 — Implementation of Category 1 signals (Regime detection).**

Per Manual section 3.4: EMA trend signal, ATR regime signal, higher
timeframe alignment. Implemented as pure functions in
`src/btc_quant/signals/regime/` with unit tests covering synthetic
bullish/bearish/sideways series and edge cases.

## Sign-Off

This decision is binding for the duration of F7 through F14. Any
proposed change to v1 scope before F14 completes requires a new
decision document with explicit reasoning and Roger's approval,
following the same pattern as DESIGN_REFINEMENT_001 and
DESIGN_REFINEMENT_002.
