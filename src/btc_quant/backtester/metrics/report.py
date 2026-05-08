"""MetricsReport and StrategyMetrics: immutable backtest performance report."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from btc_quant.backtester.models.enums import StrategyTag


@dataclass(frozen=True)
class StrategyMetrics:
    """Performance metrics for a single strategy within the backtest."""

    strategy_tag: StrategyTag
    num_trades: int
    win_rate: float | None          # None if no closed trades
    profit_factor: float | None     # None if no losing trades
    total_pnl_eur: float
    avg_pnl_per_trade_eur: float | None  # None if no closed trades
    avg_r_multiple: float | None    # None if no trades with stop defined
    contribution_to_total_return_pct: float


@dataclass(frozen=True)
class MetricsReport:
    """Complete performance report for a backtest run. Immutable.

    Fields marked Optional[float] are None when the metric cannot be computed:
    - cagr: None if duration < 30 days (CAGR is meaningless on short samples)
    - sharpe_ratio / sortino_ratio: None if return volatility == 0
    - calmar_ratio: None if max_drawdown == 0
    - win_rate, profit_factor, avg_win_eur, avg_loss_eur, expectancy_eur,
      avg_r_multiple_realized, largest_losing_streak_*: None if no closed trades
    - average_recovery_time_days: None if no drawdowns recovered within the period
    - avg_cost_basis_eur_per_btc, total_btc_accumulated, btc_per_eur_invested:
      None unless a STRAT_06_BASELINE pool exists
    - excess_return_vs_bah_pct / dca_pct: None if no benchmark was supplied
    """

    # ── Backtest configuration ────────────────────────────────────────────────
    start_date_utc: datetime
    end_date_utc: datetime
    duration_days: int
    initial_capital_eur: float
    final_capital_eur: float

    # ── Return metrics ────────────────────────────────────────────────────────
    total_return_eur: float
    total_return_pct: float
    cagr: float | None
    time_in_market_pct: float       # fraction of bars with >= 1 open position

    # ── Risk metrics ─────────────────────────────────────────────────────────
    annualized_volatility: float
    max_drawdown_pct: float         # e.g. -0.25 for -25%
    max_drawdown_eur: float         # peak-to-trough absolute EUR loss
    max_drawdown_duration_days: int # bars from peak to recovery (or end of period)
    average_recovery_time_days: float | None
    largest_losing_streak_trades: int | None
    largest_losing_streak_eur: float | None
    var_95_daily: float             # 5th-percentile daily return (negative)
    expected_shortfall_95_daily: float  # mean of returns below VaR (negative)

    # ── Risk-adjusted metrics ─────────────────────────────────────────────────
    sharpe_ratio: float | None
    sortino_ratio: float | None
    calmar_ratio: float | None

    # ── Operational metrics (None when num_trades == 0) ───────────────────────
    num_trades: int
    win_rate: float | None
    profit_factor: float | None
    avg_win_eur: float | None
    avg_loss_eur: float | None      # negative value
    expectancy_eur: float | None
    avg_r_multiple_realized: float | None
    total_fees_paid_eur: float

    # ── Per-strategy breakdown ─────────────────────────────────────────────────
    metrics_by_strategy: dict[StrategyTag, StrategyMetrics]

    # ── STRAT-06 DCA specific ─────────────────────────────────────────────────
    avg_cost_basis_eur_per_btc: float | None
    total_btc_accumulated: float | None
    btc_per_eur_invested: float | None

    # ── Benchmark comparisons ─────────────────────────────────────────────────
    excess_return_vs_bah_pct: float | None
    excess_return_vs_dca_pct: float | None
