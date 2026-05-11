"""Comparative backtest: Buy-and-Hold vs DCA Benchmark vs STRAT-06.

Runs three backtests over the same BTC/USDT 1-day dataset and emits
structured output to data/reports/strat06_comparative_<timestamp>/.

Capital fairness
================
All three strategies receive the same total EUR over the backtest period.
The number of monthly inflow periods (N) is determined from the data range;
total capital = N × €500.

  BuyAndHold  — deploys N×€500 on the first bar, no further inflows.
  DCA Bench   — receives €500/month via InflowScheduler; buys €115.38/week
                (= 500 × 12 / 52, the annualised weekly equivalent).
  STRAT-06    — receives €500/month split 55/20/25 across BASELINE/BUFFER/
                RESERVE; applies drawdown modulation + deep-value reserve.

DCA Benchmark weekly amount rationale
======================================
DCA Benchmark receives the same inflow schedule as STRAT-06 (€500/month
calendar to the same pool DCA_BENCHMARK). It buys €115.38 every Monday
(= 500 * 12 / 52), the annualised weekly equivalent.

This convention (weeks/year = 52) implies an investment rhythm of ~€6,000/year,
exactly matching STRAT-06's annual inflow. With ~4.33 weeks/month, the pool
accumulates ~€19 of cash residual at the end of each calendar month, which is
spent the following Monday.

If on any Monday the available cash is less than €115.38 (e.g. the next inflow
has not yet arrived), the DCA Benchmark buys the available cash without error,
staying in rhythm.

Executor
========
All three backtests use ExecutionSimulatorV2 (regime-aware slippage) with the
same RegimeDetector instance and 0.1 % fee per side.

Output structure
================
data/reports/strat06_comparative_<timestamp>/
    equity_curve_bah.csv
    equity_curve_dca.csv
    equity_curve_strat06.csv
    trades_bah.csv
    trades_dca.csv
    trades_strat06.csv
    comparative_summary.json

Usage
=====
    python scripts/run_strat06_comparative_backtest.py [--data PATH] [--output DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# ── project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulatorV2
from btc_quant.backtester.inflow_scheduler import InflowScheduler
from btc_quant.backtester.metrics.calculator import MetricsCalculator
from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.backtester.regime_detector import RegimeDetector
from btc_quant.strategies.buy_and_hold import BuyAndHoldStrategy
from btc_quant.strategies.dca_benchmark import DCABenchmarkStrategy
from btc_quant.strategies.strat06.config import DCAModulatedConfig
from btc_quant.strategies.strat06.strategy import DCAModulatedStrategy

UTC = timezone.utc

_BAH = StrategyTag.BUY_AND_HOLD_BENCHMARK
_DCA = StrategyTag.DCA_BENCHMARK
_BASELINE = StrategyTag.STRAT_06_BASELINE
_BUFFER = StrategyTag.STRAT_06_BUFFER
_RESERVE = StrategyTag.STRAT_06_RESERVE

MONTHLY_INFLOW_EUR = 500.0
WEEKLY_AMOUNT_EUR = round(MONTHLY_INFLOW_EUR * 12 / 52, 2)  # 115.38


# ── STRAT-06 inflow helper (mirrors tests/integration/conftest.py) ────────────


def _make_strat06_inflows(cfg: DCAModulatedConfig, start: datetime, end: datetime) -> list:
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


# ── Data loading ─────────────────────────────────────────────────────────────


def load_data(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df["close_time_utc"] = pd.to_datetime(df["close_time_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


# ── Inflow helpers ────────────────────────────────────────────────────────────


def _make_dca_inflows(start: datetime, end: datetime) -> list:
    return InflowScheduler.monthly(
        amount_eur=MONTHLY_INFLOW_EUR,
        target_strategy=_DCA,
        start_date=start,
        end_date=end,
    )


def _count_monthly_inflows(start: datetime, end: datetime) -> int:
    return len(
        InflowScheduler.monthly(
            amount_eur=MONTHLY_INFLOW_EUR,
            target_strategy=_DCA,
            start_date=start,
            end_date=end,
        )
    )


# ── Engine factories ──────────────────────────────────────────────────────────


def _make_simulator() -> ExecutionSimulatorV2:
    return ExecutionSimulatorV2(regime_detector=RegimeDetector(), fee_rate=0.001)


def run_bah(df: pd.DataFrame, total_capital_eur: float):
    engine = BacktestEngine(
        strategies=[BuyAndHoldStrategy()],
        data={"1d": df},
        initial_capital_per_strategy={_BAH: total_capital_eur},
        execution_simulator=_make_simulator(),
        profit_recycle_pct=0.0,
    )
    return engine.run()


def run_dca(df: pd.DataFrame, start: datetime, end: datetime):
    inflows = _make_dca_inflows(start, end)
    engine = BacktestEngine(
        strategies=[DCABenchmarkStrategy(weekly_amount_eur=WEEKLY_AMOUNT_EUR)],
        data={"1d": df},
        initial_capital_per_strategy={_DCA: 0.0},
        execution_simulator=_make_simulator(),
        profit_recycle_pct=0.0,
        inflow_schedule=inflows,
    )
    return engine.run()


def run_strat06(df: pd.DataFrame, start: datetime, end: datetime):
    cfg = DCAModulatedConfig(monthly_inflow_eur=MONTHLY_INFLOW_EUR)
    inflows = _make_strat06_inflows(cfg, start, end)
    engine = BacktestEngine(
        strategies=[DCAModulatedStrategy(cfg)],
        data={"1d": df},
        initial_capital_per_strategy={_BASELINE: 0.0, _BUFFER: 0.0, _RESERVE: 0.0},
        execution_simulator=_make_simulator(),
        profit_recycle_pct=0.0,
        inflow_schedule=inflows,
    )
    return engine.run()


# ── Output helpers ────────────────────────────────────────────────────────────


def _equity_curve_to_df(result) -> pd.DataFrame:
    rows = []
    for snap in result.equity_curve:
        row = {
            "timestamp_utc": snap.timestamp_utc.isoformat(),
            "total_value_eur": snap.total_value_eur,
            "open_positions_count": snap.open_positions_count,
        }
        for tag, cash in snap.cash_by_pool.items():
            row[f"cash_{tag.value}_eur"] = cash
        rows.append(row)
    return pd.DataFrame(rows)


def _trades_to_df(result) -> pd.DataFrame:
    rows = []
    for trade in result.closed_trades:
        rows.append(
            {
                "position_id": str(trade.position_id),
                "strategy_tag": trade.strategy_tag.value,
                "opened_at_utc": trade.opened_at_utc.isoformat(),
                "closed_at_utc": trade.closed_at_utc.isoformat(),
                "symbol": trade.symbol,
                "side": trade.side.value,
                "quantity_btc": trade.quantity_btc,
                "entry_price_avg": trade.entry_price_avg,
                "exit_price_avg": trade.exit_price_avg,
                "gross_pnl_eur": trade.gross_pnl_eur,
                "fees_paid_eur": trade.fees_paid_eur,
                "net_pnl_eur": trade.net_pnl_eur,
                "exit_reason": trade.exit_reason,
            }
        )
    return pd.DataFrame(rows)


def _metrics_summary(label: str, metrics, result) -> dict:
    final_portfolio = result.final_portfolio
    total_btc = sum(
        p.btc_held for p in final_portfolio.pools.values()
    )
    total_cash = sum(
        p.cash_eur for p in final_portfolio.pools.values()
    )
    total_invested = sum(
        p.total_invested_eur for p in final_portfolio.pools.values()
    )
    return {
        "strategy": label,
        "total_invested_eur": round(total_invested, 2),
        "final_value_eur": round(metrics.final_capital_eur, 2),
        "btc_accumulated": round(total_btc, 8),
        "cash_remaining_eur": round(total_cash, 2),
        "total_return_pct": round(metrics.total_return_pct * 100, 2),
        "cagr_pct": round(metrics.cagr * 100, 2) if metrics.cagr is not None else None,
        "max_drawdown_pct": round(metrics.max_drawdown_pct * 100, 2),
        "sharpe_ratio": round(metrics.sharpe_ratio, 4) if metrics.sharpe_ratio is not None else None,
        "sortino_ratio": round(metrics.sortino_ratio, 4) if metrics.sortino_ratio is not None else None,
        "total_fees_paid_eur": round(metrics.total_fees_paid_eur, 2),
        "num_closed_trades": metrics.num_trades,
        "avg_cost_basis_eur_per_btc": (
            round(metrics.avg_cost_basis_eur_per_btc, 2)
            if metrics.avg_cost_basis_eur_per_btc is not None
            else None
        ),
        "time_in_market_pct": round(metrics.time_in_market_pct * 100, 2),
        "duration_days": metrics.duration_days,
    }


def _strat06_reserve_deployments(result) -> list[dict]:
    rows = []
    for trade in result.closed_trades:
        pass  # STRAT-06 never closes trades

    # Reserve deployments appear in open_positions entry_orders with reason_tag
    # containing "reserve_tranche". Extract from the accumulated position.
    for pos in result.final_portfolio.open_positions.values():
        for order in pos.entry_orders:
            pass  # ExecutedOrder has no reason_tag

    # Read from processed_inflows — reserve deployments are internal transfers,
    # not visible as inflows. Instead, scan equity curve for RESERVE cash drops.
    reserve_drops = []
    prev_reserve = None
    for snap in result.equity_curve:
        reserve_cash = snap.cash_by_pool.get(_RESERVE, 0.0)
        if prev_reserve is not None and reserve_cash < prev_reserve - 0.01:
            reserve_drops.append(
                {
                    "timestamp_utc": snap.timestamp_utc.isoformat(),
                    "reserve_before_eur": round(prev_reserve, 2),
                    "reserve_after_eur": round(reserve_cash, 2),
                    "amount_deployed_eur": round(prev_reserve - reserve_cash, 2),
                }
            )
        prev_reserve = reserve_cash
    return reserve_drops


# ── Main ──────────────────────────────────────────────────────────────────────


def main(data_path: Path, output_dir: Path | None = None) -> Path:
    print(f"Loading data from {data_path} …")
    df = load_data(data_path)

    start = df["timestamp_utc"].iloc[0].to_pydatetime()
    end = df["timestamp_utc"].iloc[-1].to_pydatetime()
    print(f"Period: {start.date()} → {end.date()}  ({len(df)} bars)")

    n_months = _count_monthly_inflows(start, end)
    total_capital_eur = n_months * MONTHLY_INFLOW_EUR
    print(
        f"Monthly inflows: {n_months} × €{MONTHLY_INFLOW_EUR:.0f} = €{total_capital_eur:,.0f} total"
    )

    # ── Run backtests ─────────────────────────────────────────────────────────
    print("\n[1/3] Running Buy-and-Hold …")
    result_bah = run_bah(df, total_capital_eur)

    print("[2/3] Running DCA Benchmark …")
    result_dca = run_dca(df, start, end)

    print("[3/3] Running STRAT-06 …")
    result_strat06 = run_strat06(df, start, end)

    # ── Compute metrics ───────────────────────────────────────────────────────
    calc = MetricsCalculator()
    bah_curve = [s.total_value_eur for s in result_bah.equity_curve]
    dca_curve = [s.total_value_eur for s in result_dca.equity_curve]

    # total_invested for each benchmark (needed for correct excess-return calc)
    bah_total_invested = result_bah.initial_capital_eur  # lump-sum, no inflows
    dca_total_invested = sum(
        inflow.amount_eur for inflow in result_dca.processed_inflows
    )

    metrics_bah = calc.calculate(result_bah)
    metrics_dca = calc.calculate(result_dca, bah_equity_curve=bah_curve,
                                 bah_total_invested=bah_total_invested)
    metrics_strat06 = calc.calculate(
        result_strat06,
        bah_equity_curve=bah_curve,
        bah_total_invested=bah_total_invested,
        dca_equity_curve=dca_curve,
        dca_total_invested=dca_total_invested,
    )

    # ── Prepare output directory ──────────────────────────────────────────────
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = ROOT / "data" / "reports" / f"strat06_comparative_{ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Save equity curves ────────────────────────────────────────────────────
    _equity_curve_to_df(result_bah).to_csv(output_dir / "equity_curve_bah.csv", index=False)
    _equity_curve_to_df(result_dca).to_csv(output_dir / "equity_curve_dca.csv", index=False)
    _equity_curve_to_df(result_strat06).to_csv(output_dir / "equity_curve_strat06.csv", index=False)

    # ── Save trades ───────────────────────────────────────────────────────────
    _trades_to_df(result_bah).to_csv(output_dir / "trades_bah.csv", index=False)
    _trades_to_df(result_dca).to_csv(output_dir / "trades_dca.csv", index=False)
    _trades_to_df(result_strat06).to_csv(output_dir / "trades_strat06.csv", index=False)

    # ── Comparative summary JSON ──────────────────────────────────────────────
    reserve_deployments = _strat06_reserve_deployments(result_strat06)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "backtest_period": {
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "total_bars": len(df),
            "n_monthly_inflows": n_months,
            "total_capital_eur": total_capital_eur,
        },
        "strategies": [
            _metrics_summary("buy_and_hold", metrics_bah, result_bah),
            _metrics_summary("dca_benchmark", metrics_dca, result_dca),
            _metrics_summary("strat06", metrics_strat06, result_strat06),
        ],
        "strat06_reserve_deployments": reserve_deployments,
        "excess_returns": {
            "strat06_vs_bah_pct": (
                round(metrics_strat06.excess_return_vs_bah_pct * 100, 2)
                if metrics_strat06.excess_return_vs_bah_pct is not None
                else None
            ),
            "strat06_vs_dca_pct": (
                round(metrics_strat06.excess_return_vs_dca_pct * 100, 2)
                if metrics_strat06.excess_return_vs_dca_pct is not None
                else None
            ),
        },
        "output_files": {
            "equity_curve_bah": "equity_curve_bah.csv",
            "equity_curve_dca": "equity_curve_dca.csv",
            "equity_curve_strat06": "equity_curve_strat06.csv",
            "trades_bah": "trades_bah.csv",
            "trades_dca": "trades_dca.csv",
            "trades_strat06": "trades_strat06.csv",
        },
    }

    with open(output_dir / "comparative_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\n{'Strategy':<20} {'Invested':>12} {'Final':>12} {'BTC':>12} {'Return%':>10} {'CAGR%':>8} {'MaxDD%':>8} {'Sharpe':>8}")
    print("-" * 96)
    for s in summary["strategies"]:
        cagr = f"{s['cagr_pct']:.1f}" if s["cagr_pct"] is not None else "  N/A"
        sharpe = f"{s['sharpe_ratio']:.2f}" if s["sharpe_ratio"] is not None else "  N/A"
        print(
            f"{s['strategy']:<20} "
            f"€{s['total_invested_eur']:>10,.0f} "
            f"€{s['final_value_eur']:>10,.0f} "
            f"{s['btc_accumulated']:>12.4f} "
            f"{s['total_return_pct']:>9.1f}% "
            f"{cagr:>7}% "
            f"{s['max_drawdown_pct']:>7.1f}% "
            f"{sharpe:>8}"
        )
    print()

    if reserve_deployments:
        print(f"STRAT-06 reserve deployments: {len(reserve_deployments)}")
        for d in reserve_deployments:
            print(
                f"  {d['timestamp_utc'][:10]}  "
                f"€{d['reserve_before_eur']:>8,.2f} → €{d['reserve_after_eur']:>8,.2f}  "
                f"(deployed €{d['amount_deployed_eur']:,.2f})"
            )
    else:
        print("STRAT-06 reserve deployments: 0")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STRAT-06 comparative backtest")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "processed" / "btcusdt_1d.parquet",
        help="Path to btcusdt_1d.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: data/reports/strat06_comparative_<timestamp>)",
    )
    args = parser.parse_args()
    main(args.data, args.output)
