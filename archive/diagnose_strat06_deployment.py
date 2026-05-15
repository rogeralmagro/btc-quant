"""STRAT-06 Deployment Diagnostic.

Answers WHY STRAT-06 left €36,210 (69% of capital) undeployed.

Runs STRAT-06 with an instrumented wrapper that logs per-bar decisions
and produces a full accounting of capital flows across the three pools.

Output: printed report covering all 5 diagnostic questions.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulatorV2
from btc_quant.backtester.inflow_scheduler import InflowScheduler
from btc_quant.backtester.market_context import MarketContext
from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.backtester.models.signal import Signal
from btc_quant.backtester.regime_detector import RegimeDetector
from btc_quant.strategies.strat06.config import DCAModulatedConfig
from btc_quant.strategies.strat06.strategy import DCAModulatedStrategy

UTC = timezone.utc
_BASELINE = StrategyTag.STRAT_06_BASELINE
_BUFFER = StrategyTag.STRAT_06_BUFFER
_RESERVE = StrategyTag.STRAT_06_RESERVE

MONTHLY_INFLOW_EUR = 500.0


# ── Instrumented strategy wrapper ─────────────────────────────────────────────


class _InstrumentedDCAModulated(DCAModulatedStrategy):
    """Subclass that records per-bar diagnostic data without altering logic."""

    def __init__(self, config: DCAModulatedConfig) -> None:
        super().__init__(config)
        self.bar_log: list[dict] = []

    def generate_signals(
        self, context: MarketContext, portfolio: Portfolio
    ) -> list[Signal]:
        t = context.current_time
        price = context.get_current_btc_price()

        # Capture pool state BEFORE any transfers inside super()
        bl_cash_before = portfolio.get_pool(_BASELINE).cash_eur
        buf_cash_before = portfolio.get_pool(_BUFFER).cash_eur
        res_cash_before = portfolio.get_pool(_RESERVE).cash_eur

        # Compute conditions BEFORE super() mutates internal state
        is_time = self._is_baseline_time(t)
        concentration = self._calculate_concentration(portfolio, price)
        concentration_blocked = is_time and (
            concentration > self._config.max_concentration_pct
        )

        # Run normal logic (may transfer cash between pools; also calls
        # _ath_tracker.update() so get_drawdown() is safe to call after).
        signals = super().generate_signals(context, portfolio)

        # Drawdown is computed after super() has updated the ATH tracker.
        drawdown = self._ath_tracker.get_drawdown(price)

        # Capture pool state AFTER transfers (before engine executes BTC buys)
        bl_cash_after = portfolio.get_pool(_BASELINE).cash_eur
        buf_cash_after = portfolio.get_pool(_BUFFER).cash_eur
        res_cash_after = portfolio.get_pool(_RESERVE).cash_eur

        # Identify signal types
        baseline_signal = next(
            (s for s in signals if "baseline" in s.reason_tag), None
        )
        reserve_signal = next(
            (s for s in signals if "reserve_tranche" in s.reason_tag), None
        )

        # Extract multiplier from reason_tag (e.g. "baseline_modulated_x2.0" → 2.0)
        multiplier: float | None = None
        if baseline_signal is not None:
            if "modulated_x" in baseline_signal.reason_tag:
                try:
                    multiplier = float(baseline_signal.reason_tag.split("_x")[1])
                except (IndexError, ValueError):
                    multiplier = 1.0
            else:
                multiplier = 1.0

        # Decompose the three possible transfers using strategy execution order:
        #   1. _maybe_redirect_reserve_overflow: RESERVE → BUFFER  (if reserve > cap)
        #   2. _check_reserve_deployment:        RESERVE → BASELINE (if tranche fires)
        #   3. _check_baseline_buy:              BUFFER  → BASELINE (if modulation deficit)
        # Solved analytically (all three amounts are >= 0):
        cap = self._config.reserve_cap_eur
        res_to_buf = max(0.0, res_cash_before - cap)
        res_after_overflow = res_cash_before - res_to_buf
        res_to_bl = max(0.0, res_after_overflow - res_cash_after)
        buf_net = buf_cash_after - buf_cash_before  # +overflow - transfer_to_bl
        buf_to_bl = max(0.0, res_to_buf - buf_net)

        self.bar_log.append(
            {
                "date": t,
                "price": price,
                "drawdown": drawdown,
                "is_baseline_time": is_time,
                "concentration": concentration,
                "concentration_blocked": concentration_blocked,
                "baseline_fired": baseline_signal is not None,
                "baseline_amount_eur": (
                    baseline_signal.quote_amount_eur if baseline_signal else 0.0
                ),
                "multiplier": multiplier,
                "reserve_fired": reserve_signal is not None,
                "reserve_amount_eur": (
                    reserve_signal.quote_amount_eur if reserve_signal else 0.0
                ),
                "buf_to_bl_transfer_eur": buf_to_bl,
                "res_to_buf_transfer_eur": res_to_buf,
                "res_to_bl_transfer_eur": res_to_bl,
                "bl_cash_after_transfer": bl_cash_after,
                "buf_cash_after_transfer": buf_cash_after,
                "res_cash_after_transfer": res_cash_after,
            }
        )

        return signals


# ── Inflow helpers ────────────────────────────────────────────────────────────


def _make_strat06_inflows(
    cfg: DCAModulatedConfig, start: datetime, end: datetime
) -> list:
    baseline = InflowScheduler.monthly(
        amount_eur=cfg.monthly_inflow_eur * cfg.baseline_pct,
        target_strategy=_BASELINE,
        start_date=start,
        end_date=end,
    )
    buffer_ = InflowScheduler.monthly(
        amount_eur=cfg.monthly_inflow_eur * cfg.buffer_pct,
        target_strategy=_BUFFER,
        start_date=start,
        end_date=end,
    )
    reserve = InflowScheduler.monthly(
        amount_eur=cfg.monthly_inflow_eur * cfg.reserve_pct,
        target_strategy=_RESERVE,
        start_date=start,
        end_date=end,
    )
    return sorted(baseline + buffer_ + reserve, key=lambda x: x.timestamp_utc)


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    data_path = ROOT / "data" / "processed" / "btcusdt_1d.parquet"
    print(f"Loading {data_path} …")
    df = pd.read_parquet(data_path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["close_time_utc"] = pd.to_datetime(df["close_time_utc"], utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    start = df["timestamp_utc"].iloc[0].to_pydatetime()
    end = df["timestamp_utc"].iloc[-1].to_pydatetime()
    print(f"Period: {start.date()} → {end.date()}  ({len(df)} bars)\n")

    cfg = DCAModulatedConfig(monthly_inflow_eur=MONTHLY_INFLOW_EUR)
    strategy = _InstrumentedDCAModulated(cfg)
    inflows = _make_strat06_inflows(cfg, start, end)

    engine = BacktestEngine(
        strategies=[strategy],
        data={"1d": df},
        initial_capital_per_strategy={_BASELINE: 0.0, _BUFFER: 0.0, _RESERVE: 0.0},
        execution_simulator=ExecutionSimulatorV2(
            regime_detector=RegimeDetector(), fee_rate=0.001
        ),
        profit_recycle_pct=0.0,
        inflow_schedule=inflows,
    )

    print("Running STRAT-06 (instrumented) …")
    result = engine.run()
    print("Done.\n")

    # ── Post-process bar log ──────────────────────────────────────────────────
    df_log = pd.DataFrame(strategy.bar_log)

    final_portfolio = result.final_portfolio
    bl_pool = final_portfolio.get_pool(_BASELINE)
    buf_pool = final_portfolio.get_pool(_BUFFER)
    res_pool = final_portfolio.get_pool(_RESERVE)
    final_price = df["close"].iloc[-1]  # last bar close price

    n_months = len(inflows) // 3  # 3 pools, each gets monthly inflow
    total_injected = n_months * MONTHLY_INFLOW_EUR

    # ── Inflow accounting per pool ────────────────────────────────────────────
    bl_inflows_total = sum(
        i.amount_eur for i in result.processed_inflows if i.target_strategy == _BASELINE
    )
    buf_inflows_total = sum(
        i.amount_eur for i in result.processed_inflows if i.target_strategy == _BUFFER
    )
    res_inflows_total = sum(
        i.amount_eur for i in result.processed_inflows if i.target_strategy == _RESERVE
    )

    # ── Transfer totals from bar log ─────────────────────────────────────────
    total_buf_to_bl = df_log["buf_to_bl_transfer_eur"].sum()
    total_res_to_buf = df_log["res_to_buf_transfer_eur"].sum()
    total_res_to_bl = df_log["res_to_bl_transfer_eur"].sum()

    # ── BTC purchase cost by pool ─────────────────────────────────────────────
    # All BTC buys debit BASELINE. cost_basis_eur = total EUR spent on BTC held.
    bl_btc_cost = bl_pool.cost_basis_eur

    # ── Total fees from all open positions' entry orders ─────────────────────
    # (STRAT-06 never closes, so closed_trades is empty)
    total_fees = sum(
        order.fees_eur
        for pos in result.final_portfolio.open_positions.values()
        for order in pos.entry_orders
    )

    # ── Baseline buy statistics ───────────────────────────────────────────────
    baseline_days = df_log[df_log["is_baseline_time"]]
    baseline_fired = baseline_days[baseline_days["baseline_fired"]]
    concentration_blocked_days = baseline_days[baseline_days["concentration_blocked"]]

    n_baseline_days = len(baseline_days)
    n_baseline_fired = len(baseline_fired)
    n_concentration_blocked = len(concentration_blocked_days)
    n_other_blocked = n_baseline_days - n_baseline_fired - n_concentration_blocked

    multiplier_all = baseline_fired["multiplier"]
    multiplier_gt1 = baseline_fired[baseline_fired["multiplier"] > 1.0]["multiplier"]

    # ── Reserve deployment details ────────────────────────────────────────────
    reserve_deploys = df_log[df_log["reserve_fired"]]

    # ── Buffer usage ──────────────────────────────────────────────────────────
    buf_to_bl_events = df_log[df_log["buf_to_bl_transfer_eur"] > 0.01]

    # ─────────────────────────────────────────────────────────────────────────
    # REPORT
    # ─────────────────────────────────────────────────────────────────────────
    sep = "=" * 65

    print(sep)
    print("STRAT-06 DEPLOYMENT DIAGNOSTIC REPORT")
    print(sep)

    # ── 1. Final pool states ──────────────────────────────────────────────────
    print("\n[1] FINAL POOL STATES (at last bar close, BTC @"
          f" €{final_price:,.0f})")
    print(f"  {'Pool':<12} {'Cash EUR':>12} {'BTC held':>14} {'BTC value EUR':>14} {'Total EUR':>12}")
    print("  " + "-" * 68)
    for tag, pool in [(_BASELINE, bl_pool), (_BUFFER, buf_pool), (_RESERVE, res_pool)]:
        btc_val = pool.btc_held * final_price
        total = pool.cash_eur + btc_val
        print(
            f"  {tag.value:<12} "
            f"€{pool.cash_eur:>10,.2f} "
            f"{pool.btc_held:>14.8f} "
            f"€{btc_val:>12,.2f} "
            f"€{total:>10,.2f}"
        )
    total_cash = bl_pool.cash_eur + buf_pool.cash_eur + res_pool.cash_eur
    total_btc = bl_pool.btc_held + buf_pool.btc_held + res_pool.btc_held
    total_btc_val = total_btc * final_price
    print("  " + "-" * 68)
    print(
        f"  {'TOTAL':<12} "
        f"€{total_cash:>10,.2f} "
        f"{total_btc:>14.8f} "
        f"€{total_btc_val:>12,.2f} "
        f"€{total_cash + total_btc_val:>10,.2f}"
    )

    # ── 2. Capital accounting ─────────────────────────────────────────────────
    print(f"\n[2] CAPITAL ACCOUNTING (total injected: €{total_injected:,.2f})")
    print(f"\n  Inflows received:")
    print(f"    BASELINE inflows  : €{bl_inflows_total:>10,.2f}  ({bl_inflows_total/total_injected*100:.1f}%)")
    print(f"    BUFFER  inflows   : €{buf_inflows_total:>10,.2f}  ({buf_inflows_total/total_injected*100:.1f}%)")
    print(f"    RESERVE inflows   : €{res_inflows_total:>10,.2f}  ({res_inflows_total/total_injected*100:.1f}%)")
    print(f"    Total             : €{bl_inflows_total+buf_inflows_total+res_inflows_total:>10,.2f}")

    print(f"\n  Inter-pool transfers:")
    print(f"    BUFFER  → BASELINE (modulation)   : €{total_buf_to_bl:>10,.2f}")
    print(f"    RESERVE → BASELINE (deployments)  : €{total_res_to_bl:>10,.2f}")
    print(f"    RESERVE → BUFFER   (overflow cap) : €{total_res_to_buf:>10,.2f}")

    print(f"\n  BASELINE pool balance check:")
    bl_in = bl_inflows_total + total_buf_to_bl + total_res_to_bl
    bl_spent_btc = bl_btc_cost
    bl_fees_approx = total_fees
    bl_cash_final = bl_pool.cash_eur
    print(f"    In  (inflows + transfers)  : €{bl_in:>10,.2f}")
    print(f"    Out (BTC cost basis)       : €{bl_spent_btc:>10,.2f}")
    print(f"    Out (fees)                 : €{bl_fees_approx:>10,.2f}")
    print(f"    Cash remaining (computed)  : €{bl_in - bl_spent_btc - bl_fees_approx:>10,.2f}")
    print(f"    Cash remaining (actual)    : €{bl_cash_final:>10,.2f}")

    print(f"\n  BUFFER pool balance check:")
    buf_in = buf_inflows_total + total_res_to_buf
    buf_out = total_buf_to_bl
    print(f"    In  (inflows + overflow)   : €{buf_in:>10,.2f}")
    print(f"    Out (modulation transfers) : €{buf_out:>10,.2f}")
    print(f"    Cash remaining (computed)  : €{buf_in - buf_out:>10,.2f}")
    print(f"    Cash remaining (actual)    : €{buf_pool.cash_eur:>10,.2f}")

    print(f"\n  RESERVE pool balance check:")
    res_in = res_inflows_total
    res_out_deploy = total_res_to_bl
    res_out_overflow = total_res_to_buf
    print(f"    In  (inflows)              : €{res_in:>10,.2f}")
    print(f"    Out (deployments → BL)     : €{res_out_deploy:>10,.2f}")
    print(f"    Out (overflow → BUF)       : €{res_out_overflow:>10,.2f}")
    print(f"    Cash remaining (computed)  : €{res_in - res_out_deploy - res_out_overflow:>10,.2f}")
    print(f"    Cash remaining (actual)    : €{res_pool.cash_eur:>10,.2f}")

    # ── 3. Baseline execution analysis ───────────────────────────────────────
    print(f"\n[3] BASELINE EXECUTION ANALYSIS")
    print(f"  Total bars where _is_baseline_time() = True : {n_baseline_days}")
    print(f"  → Baseline buy FIRED                        : {n_baseline_fired}")
    print(f"  → Blocked by CONCENTRATION > {cfg.max_concentration_pct*100:.0f}%          : {n_concentration_blocked}")
    print(f"  → Blocked by other reasons (low cash, etc.) : {n_other_blocked}")
    print()
    print(f"  Of {n_baseline_fired} executed baseline buys:")
    print(f"    Multiplier = 1.0 (no modulation)   : {(multiplier_all == 1.0).sum()}")
    print(f"    Multiplier > 1.0 (modulated)        : {(multiplier_all > 1.0).sum()}")
    print(f"    Mean multiplier (all buys)          : {multiplier_all.mean():.4f}")
    if len(multiplier_gt1) > 0:
        print(f"    Mean multiplier (modulated only)    : {multiplier_gt1.mean():.4f}")
    print(f"    Total EUR deployed via baseline     : €{df_log['baseline_amount_eur'].sum():,.2f}")

    # When did concentration blocking start/end?
    if n_concentration_blocked > 0:
        conc_block_dates = concentration_blocked_days["date"]
        print(f"\n  Concentration-blocked periods:")
        print(f"    First block : {conc_block_dates.iloc[0].date()}")
        print(f"    Last block  : {conc_block_dates.iloc[-1].date()}")
        # Group into contiguous periods
        block_dates_sorted = conc_block_dates.sort_values().reset_index(drop=True)
        gaps = block_dates_sorted.diff().dt.days.fillna(0)
        period_starts = block_dates_sorted[gaps > 14].tolist()
        period_starts.insert(0, block_dates_sorted.iloc[0])
        print(f"    Approximate block episodes (gap > 14 days between): {len(period_starts)}")
        for ps in period_starts[:10]:  # show up to 10
            print(f"      {ps.date()}")

    # Distribution of concentration on baseline days
    print(f"\n  Concentration distribution on baseline days:")
    conc_bins = [0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    conc_labels = ["<50%", "50-60%", "60-70%", "70-80%", "80-90%", ">90%"]
    conc_vals = baseline_days["concentration"]
    for i, label in enumerate(conc_labels):
        lo, hi = conc_bins[i], conc_bins[i + 1]
        count = ((conc_vals >= lo) & (conc_vals < hi)).sum()
        print(f"    {label:>8}: {count:4d} baseline days")

    # ── 4. Reserve deployment details ────────────────────────────────────────
    print(f"\n[4] RESERVE DEPLOYMENT DETAILS ({len(reserve_deploys)} deployments)")
    print(f"  {'Date':<12} {'Drawdown':>10} {'Deployed EUR':>14} {'Est BTC@price':>15} {'Price EUR':>12}")
    print("  " + "-" * 68)
    for _, row in reserve_deploys.iterrows():
        est_btc = row["reserve_amount_eur"] / row["price"] if row["price"] > 0 else 0
        print(
            f"  {str(row['date'].date()):<12} "
            f"{row['drawdown']*100:>9.1f}% "
            f"€{row['reserve_amount_eur']:>12,.2f} "
            f"  {est_btc:>12.6f} BTC "
            f"€{row['price']:>10,.0f}"
        )

    # Tranche IDs from reserve_signal reason tags
    reserve_bars = df_log[df_log["reserve_fired"]].copy()

    # ── 5. Buffer usage ────────────────────────────────────────────────────────
    print(f"\n[5] BUFFER USAGE (modulation transfers BUFFER → BASELINE)")
    print(f"  Times buffer transferred to baseline : {len(buf_to_bl_events)}")
    print(f"  Total transferred                    : €{total_buf_to_bl:,.2f}")
    print(f"  Total received by BUFFER (inflows)   : €{buf_inflows_total:,.2f}")
    print(f"  Total received by BUFFER (overflow)  : €{total_res_to_buf:,.2f}")
    print(f"  Total in to BUFFER                   : €{buf_inflows_total+total_res_to_buf:,.2f}")
    print(f"  Fraction of BUFFER capital deployed  : "
          f"{total_buf_to_bl/(buf_inflows_total+total_res_to_buf)*100:.1f}%"
          if (buf_inflows_total + total_res_to_buf) > 0 else "  N/A")
    print(f"  BUFFER cash remaining                : €{buf_pool.cash_eur:,.2f}")

    if len(buf_to_bl_events) > 0:
        print(f"\n  First 5 BUFFER→BASELINE transfers:")
        for _, row in buf_to_bl_events.head(5).iterrows():
            print(
                f"    {str(row['date'].date())}  "
                f"DD={row['drawdown']*100:5.1f}%  "
                f"mult={row['multiplier']}  "
                f"transfer=€{row['buf_to_bl_transfer_eur']:,.2f}"
            )

    # ── Summary hypothesis ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("DIAGNOSIS SUMMARY")
    print(sep)
    print(f"\n  Total undeployed cash        : €{total_cash:,.2f}")
    print(f"    BASELINE pool cash         : €{bl_pool.cash_eur:,.2f}")
    print(f"    BUFFER  pool cash          : €{buf_pool.cash_eur:,.2f}  ← primary suspect")
    print(f"    RESERVE pool cash          : €{res_pool.cash_eur:,.2f}")
    print()
    print(f"  Concentration-blocked baseline days  : {n_concentration_blocked}")
    if n_concentration_blocked > 0:
        eur_not_deployed_due_to_conc = n_concentration_blocked * cfg.baseline_per_week_eur
        print(f"  Approx. EUR NOT bought due to block  : "
              f"€{eur_not_deployed_due_to_conc:,.0f}  "
              f"(= {n_concentration_blocked} × €{cfg.baseline_per_week_eur:.2f})")
    print(f"  max_concentration_pct threshold      : {cfg.max_concentration_pct*100:.0f}%")
    print(f"  baseline_per_week_eur                : €{cfg.baseline_per_week_eur:.2f}")
    print(f"  reserve_cap_eur                      : €{cfg.reserve_cap_eur:,.2f}")
    print()


if __name__ == "__main__":
    main()
