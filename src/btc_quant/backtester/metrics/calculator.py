"""MetricsCalculator: compute MetricsReport from BacktestResult."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from btc_quant.backtester.metrics.report import MetricsReport, StrategyMetrics
from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.backtester.models.portfolio import PortfolioSnapshot
from btc_quant.backtester.models.result import BacktestResult
from btc_quant.backtester.models.trade import Trade


class MetricsCalculator:
    """Compute MetricsReport from a BacktestResult.

    Static helper methods are individually testable and can be called
    standalone without a BacktestResult.

    Anualisation assumes daily bars (one PortfolioSnapshot per bar) with
    ``trading_days_per_year`` periods per year. For strategies running on the
    1-day timeframe this equals 365 (crypto trades 24/7).
    """

    def __init__(
        self,
        risk_free_rate_annual: float = 0.0,
        trading_days_per_year: int = 365,
        var_confidence: float = 0.95,
    ) -> None:
        self._risk_free_rate = risk_free_rate_annual
        self._trading_days_per_year = trading_days_per_year
        self._var_confidence = var_confidence

    # ── Main entry point ─────────────────────────────────────────────────────

    def calculate(
        self,
        result: BacktestResult,
        bah_equity_curve: list[float] | None = None,
        dca_equity_curve: list[float] | None = None,
    ) -> MetricsReport:
        """Compute all metrics and return an immutable MetricsReport.

        Args:
            result:            Full BacktestResult from BacktestEngine.run().
            bah_equity_curve:  Optional buy-and-hold equity curve (list of EUR
                               portfolio values) over the same period. When
                               provided, ``excess_return_vs_bah_pct`` is set.
            dca_equity_curve:  Optional pure-DCA equity curve. Same semantics.
        """
        equity_values = [s.total_value_eur for s in result.equity_curve]
        initial = result.initial_capital_eur
        final = equity_values[-1] if equity_values else initial
        duration_days = (result.end_date_utc - result.start_date_utc).days

        # Total capital deployed = seed capital + all inflows received.
        # Used as denominator for return metrics so DCA strategies (initial=0)
        # are not subject to ZeroDivisionError and returns are economically meaningful.
        total_invested = initial + sum(
            inflow.amount_eur for inflow in result.processed_inflows
        )

        returns = self.calculate_returns(equity_values)

        # ── Return metrics ────────────────────────────────────────────────────
        total_return_eur = final - total_invested
        total_return_pct = total_return_eur / total_invested if total_invested > 0 else 0.0

        # CAGR assumes a single lump-sum investment. When the strategy receives
        # periodic inflows (DCA pattern) the formula is misleading, so return None.
        # Also guard initial <= 0 to prevent ZeroDivisionError inside calculate_cagr.
        if initial <= 0.0 or result.processed_inflows:
            cagr = None
        else:
            cagr = self.calculate_cagr(initial, final, duration_days, self._trading_days_per_year)

        has_positions = [s.open_positions_count > 0 for s in result.equity_curve]
        time_in_market = self.calculate_time_in_market(has_positions)

        # ── Risk metrics ──────────────────────────────────────────────────────
        timestamps = [s.timestamp_utc for s in result.equity_curve]
        ann_vol = (
            float(np.std(returns, ddof=1) * np.sqrt(self._trading_days_per_year))
            if len(returns) > 1
            else 0.0
        )
        max_dd_pct, peak_idx, trough_idx = self.calculate_max_drawdown(equity_values)
        if equity_values and max_dd_pct != 0.0:
            max_dd_eur = equity_values[trough_idx] - equity_values[peak_idx]
        else:
            max_dd_eur = 0.0
        dd_duration = self.calculate_drawdown_duration(equity_values, timestamps)
        avg_recovery = self._calculate_average_recovery_time(result.equity_curve)

        var_95 = self.calculate_var(returns, self._var_confidence)
        es_95 = self.calculate_expected_shortfall(returns, self._var_confidence)

        # ── Risk-adjusted metrics ─────────────────────────────────────────────
        sharpe = self.calculate_sharpe(returns, self._risk_free_rate, self._trading_days_per_year)
        sortino = self.calculate_sortino(returns, self._risk_free_rate, self._trading_days_per_year)
        calmar = self.calculate_calmar(cagr, max_dd_pct)

        # ── Trade metrics ─────────────────────────────────────────────────────
        trades = result.closed_trades
        num_trades = len(trades)

        if num_trades > 0:
            winners = [t for t in trades if t.net_pnl_eur > 0]
            losers = [t for t in trades if t.net_pnl_eur < 0]
            win_rate: float | None = len(winners) / num_trades
            profit_factor = self.calculate_profit_factor(trades)
            avg_win: float | None = float(np.mean([t.net_pnl_eur for t in winners])) if winners else None
            avg_loss: float | None = float(np.mean([t.net_pnl_eur for t in losers])) if losers else None
            expectancy = self.calculate_expectancy(trades)
            r_mults = [t.r_multiple_realized for t in trades if t.r_multiple_realized is not None]
            avg_r: float | None = float(np.mean(r_mults)) if r_mults else None
            streak_n, streak_eur = self.calculate_largest_losing_streak(trades)
            largest_streak: int | None = streak_n if streak_n > 0 else None
            largest_streak_eur: float | None = streak_eur if streak_n > 0 else None
        else:
            win_rate = profit_factor = avg_win = avg_loss = None
            expectancy = avg_r = None
            largest_streak = largest_streak_eur = None

        # ── Fees ─────────────────────────────────────────────────────────────
        # Closed-trade fees + entry fees already paid for still-open positions.
        total_fees = sum(t.fees_eur for t in trades)
        for pos in result.final_portfolio.open_positions.values():
            total_fees += pos.fees_paid_eur

        # ── Per-strategy breakdown ─────────────────────────────────────────────
        metrics_by_strategy = self._calculate_strategy_metrics(result, result.equity_curve, total_invested)

        # ── STRAT-06 DCA metrics ───────────────────────────────────────────────
        strat06 = result.final_portfolio.pools.get(StrategyTag.STRAT_06_BASELINE)
        if strat06 is not None and strat06.btc_held > 0:
            total_btc: float | None = strat06.btc_held
            invested = strat06.total_invested_eur
            avg_cost: float | None = invested / strat06.btc_held
            btc_per_eur: float | None = strat06.btc_held / invested if invested > 0 else None
        else:
            total_btc = avg_cost = btc_per_eur = None

        # ── Benchmark comparisons ─────────────────────────────────────────────
        excess_bah: float | None = None
        excess_dca: float | None = None
        if bah_equity_curve and len(bah_equity_curve) >= 2:
            bah_ret = (bah_equity_curve[-1] - bah_equity_curve[0]) / bah_equity_curve[0]
            excess_bah = total_return_pct - bah_ret
        if dca_equity_curve and len(dca_equity_curve) >= 2:
            dca_ret = (dca_equity_curve[-1] - dca_equity_curve[0]) / dca_equity_curve[0]
            excess_dca = total_return_pct - dca_ret

        return MetricsReport(
            start_date_utc=result.start_date_utc,
            end_date_utc=result.end_date_utc,
            duration_days=duration_days,
            initial_capital_eur=initial,
            total_invested_eur=total_invested,
            final_capital_eur=final,
            total_return_eur=total_return_eur,
            total_return_pct=total_return_pct,
            cagr=cagr,
            time_in_market_pct=time_in_market,
            annualized_volatility=ann_vol,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_eur=max_dd_eur,
            max_drawdown_duration_days=dd_duration,
            average_recovery_time_days=avg_recovery,
            largest_losing_streak_trades=largest_streak,
            largest_losing_streak_eur=largest_streak_eur,
            var_95_daily=var_95,
            expected_shortfall_95_daily=es_95,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win_eur=avg_win,
            avg_loss_eur=avg_loss,
            expectancy_eur=expectancy,
            avg_r_multiple_realized=avg_r,
            total_fees_paid_eur=total_fees,
            metrics_by_strategy=metrics_by_strategy,
            avg_cost_basis_eur_per_btc=avg_cost,
            total_btc_accumulated=total_btc,
            btc_per_eur_invested=btc_per_eur,
            excess_return_vs_bah_pct=excess_bah,
            excess_return_vs_dca_pct=excess_dca,
        )

    # ── Public static helpers ─────────────────────────────────────────────────

    @staticmethod
    def calculate_returns(equity_curve: list[float]) -> np.ndarray:
        """Period-over-period returns: (v[i] - v[i-1]) / v[i-1]."""
        if len(equity_curve) < 2:
            return np.array([], dtype=float)
        values = np.array(equity_curve, dtype=float)
        return np.diff(values) / values[:-1]

    @staticmethod
    def calculate_cagr(
        initial: float,
        final: float,
        days: int,
        trading_days_per_year: int = 365,
    ) -> float | None:
        """CAGR = (final/initial)^(1/years) - 1. None if days < 30."""
        if days < 30:
            return None
        years = days / trading_days_per_year
        return (final / initial) ** (1.0 / years) - 1.0

    @staticmethod
    def calculate_max_drawdown(equity_curve: list[float]) -> tuple[float, int, int]:
        """Maximum peak-to-trough drawdown.

        Returns:
            (max_dd_pct, peak_idx, trough_idx)
            max_dd_pct is negative (or 0 if no drawdown).
        """
        if len(equity_curve) < 2:
            return 0.0, 0, 0
        max_dd = 0.0
        peak_i = 0
        trough_i = 0
        running_peak = equity_curve[0]
        running_peak_idx = 0
        for i in range(1, len(equity_curve)):
            if equity_curve[i] > running_peak:
                running_peak = equity_curve[i]
                running_peak_idx = i
            else:
                if running_peak == 0.0:
                    continue
                dd = (equity_curve[i] - running_peak) / running_peak
                if dd < max_dd:
                    max_dd = dd
                    peak_i = running_peak_idx
                    trough_i = i
        return max_dd, peak_i, trough_i

    @staticmethod
    def calculate_drawdown_duration(
        equity_curve: list[float],
        timestamps: list[datetime] | None = None,
    ) -> int:
        """Days (or bars if no timestamps) from the max-drawdown peak to recovery.

        When ``timestamps`` is provided the result is in calendar days, which is
        robust to any timeframe (1d, 4h, …). Without timestamps the result is in
        bars (legacy behaviour, useful for simple unit tests).

        If the drawdown is unrecovered at period end, the duration extends to the
        last bar / timestamp.
        """
        if len(equity_curve) < 2:
            return 0
        max_dd_pct, peak_idx, trough_idx = MetricsCalculator.calculate_max_drawdown(equity_curve)
        if max_dd_pct == 0.0:
            return 0
        peak_value = equity_curve[peak_idx]
        for i in range(trough_idx + 1, len(equity_curve)):
            if equity_curve[i] >= peak_value:
                if timestamps:
                    return (timestamps[i] - timestamps[peak_idx]).days
                return i - peak_idx
        # Unrecovered — duration to end of period
        if timestamps:
            return (timestamps[-1] - timestamps[peak_idx]).days
        return len(equity_curve) - 1 - peak_idx

    @staticmethod
    def calculate_sharpe(
        returns: np.ndarray,
        risk_free_rate: float,
        periods_per_year: int,
    ) -> float | None:
        """Annualised Sharpe ratio. None if std(returns) == 0 or returns empty."""
        if len(returns) < 2:
            return None
        std = float(np.std(returns, ddof=1))
        if std == 0.0:
            return None
        daily_rfr = risk_free_rate / periods_per_year
        excess = returns - daily_rfr
        return float(np.mean(excess) / std * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_sortino(
        returns: np.ndarray,
        risk_free_rate: float,
        periods_per_year: int,
    ) -> float | None:
        """Annualised Sortino ratio. None if no negative returns or returns empty."""
        if len(returns) < 2:
            return None
        daily_rfr = risk_free_rate / periods_per_year
        excess = returns - daily_rfr
        negative = returns[returns < 0]
        if len(negative) == 0:
            return None
        downside_std = float(np.std(negative, ddof=1)) if len(negative) > 1 else float(abs(negative[0]))
        if downside_std == 0.0:
            return None
        return float(np.mean(excess) / downside_std * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_calmar(cagr: float | None, max_dd_pct: float) -> float | None:
        """Calmar = CAGR / |max_dd_pct|. None if max_dd == 0 or cagr is None."""
        if cagr is None or max_dd_pct == 0.0:
            return None
        return cagr / abs(max_dd_pct)

    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Daily Value-at-Risk at given confidence level (e.g. 0.95 → 5th pct).

        Returns the (1-confidence)-th percentile of returns (negative value).
        """
        if len(returns) == 0:
            return 0.0
        return float(np.percentile(returns, (1.0 - confidence) * 100.0))

    @staticmethod
    def calculate_expected_shortfall(
        returns: np.ndarray,
        confidence: float = 0.95,
    ) -> float:
        """Conditional VaR: mean of returns that are <= VaR. Returns 0 if empty."""
        if len(returns) == 0:
            return 0.0
        var = MetricsCalculator.calculate_var(returns, confidence)
        tail = returns[returns <= var]
        if len(tail) == 0:
            return var
        return float(np.mean(tail))

    @staticmethod
    def calculate_largest_losing_streak(trades: list[Trade]) -> tuple[int, float]:
        """Largest consecutive losing streak.

        Returns:
            (n_consecutive_losses, total_loss_eur) — both 0 if no losing trades.
        """
        if not trades:
            return 0, 0.0
        max_streak = 0
        max_loss = 0.0
        cur_streak = 0
        cur_loss = 0.0
        for trade in trades:
            if trade.net_pnl_eur < 0:
                cur_streak += 1
                cur_loss += trade.net_pnl_eur
                if cur_streak > max_streak:
                    max_streak = cur_streak
                    max_loss = cur_loss
            else:
                cur_streak = 0
                cur_loss = 0.0
        return max_streak, max_loss

    @staticmethod
    def calculate_profit_factor(trades: list[Trade]) -> float | None:
        """Gross wins / |gross losses|. None if no losing trades (ratio is infinite)."""
        if not trades:
            return None
        wins = sum(t.net_pnl_eur for t in trades if t.net_pnl_eur > 0)
        losses = sum(t.net_pnl_eur for t in trades if t.net_pnl_eur < 0)
        if losses == 0.0:
            return None
        return wins / abs(losses)

    @staticmethod
    def calculate_expectancy(trades: list[Trade]) -> float | None:
        """Average net PnL per trade. None if no trades."""
        if not trades:
            return None
        return sum(t.net_pnl_eur for t in trades) / len(trades)

    @staticmethod
    def calculate_time_in_market(equity_curve_with_positions: list[bool]) -> float:
        """Fraction of bars with at least one open position."""
        if not equity_curve_with_positions:
            return 0.0
        return sum(equity_curve_with_positions) / len(equity_curve_with_positions)

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _calculate_average_recovery_time(snapshots: list[PortfolioSnapshot]) -> float | None:
        """Average calendar days from drawdown peak to recovery.

        Only drawdowns that are fully recovered within the backtest period are
        included. An unrecovered drawdown still in progress at the last bar is
        excluded (its true duration is unknown).

        Returns None if no drawdown was recovered within the period.
        """
        if len(snapshots) < 2:
            return None
        recovery_days: list[int] = []
        peak_val = snapshots[0].total_value_eur
        peak_ts = snapshots[0].timestamp_utc
        in_dd = False
        dd_peak_ts = peak_ts

        for snap in snapshots[1:]:
            val = snap.total_value_eur
            ts = snap.timestamp_utc
            if val >= peak_val:
                if in_dd:
                    recovery_days.append((ts - dd_peak_ts).days)
                    in_dd = False
                peak_val = val
                peak_ts = ts
            else:
                if not in_dd:
                    in_dd = True
                    dd_peak_ts = peak_ts

        if not recovery_days:
            return None
        return float(np.mean(recovery_days))

    def _calculate_strategy_metrics(
        self,
        result: BacktestResult,
        snapshots: list[PortfolioSnapshot],
        total_invested_eur: float,
    ) -> dict[StrategyTag, StrategyMetrics]:
        metrics: dict[StrategyTag, StrategyMetrics] = {}
        for tag in result.strategies_used:
            trades = result.trades_by_strategy.get(tag, [])
            n = len(trades)
            if n > 0:
                winners = [t for t in trades if t.net_pnl_eur > 0]
                losers = [t for t in trades if t.net_pnl_eur < 0]
                total_pnl = sum(t.net_pnl_eur for t in trades)
                win_rate: float | None = len(winners) / n
                loss_sum = sum(t.net_pnl_eur for t in losers)
                pf: float | None = (
                    sum(t.net_pnl_eur for t in winners) / abs(loss_sum)
                    if loss_sum != 0.0
                    else None
                )
                avg_pnl: float | None = total_pnl / n
                r_mults = [t.r_multiple_realized for t in trades if t.r_multiple_realized is not None]
                avg_r: float | None = float(np.mean(r_mults)) if r_mults else None
            else:
                total_pnl = 0.0
                win_rate = pf = avg_pnl = avg_r = None

            contribution = total_pnl / total_invested_eur if total_invested_eur > 0 else 0.0

            # Per-strategy time in market: fraction of snapshots where this
            # strategy had at least one open position.
            if snapshots:
                in_mkt = sum(
                    1 for s in snapshots if s.open_positions_by_strategy.get(tag, 0) > 0
                )
                strategy_time_in_mkt = in_mkt / len(snapshots)
            else:
                strategy_time_in_mkt = 0.0

            metrics[tag] = StrategyMetrics(
                strategy_tag=tag,
                num_trades=n,
                win_rate=win_rate,
                profit_factor=pf,
                total_pnl_eur=total_pnl,
                avg_pnl_per_trade_eur=avg_pnl,
                avg_r_multiple=avg_r,
                contribution_to_total_return_pct=contribution,
                time_in_market_pct=strategy_time_in_mkt,
            )
        return metrics
