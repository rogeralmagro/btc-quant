"""Integration test: engine output matches step-by-step manual calculation.

Uses a synthetic 10-candle dataset with known integer prices. A strategy buys
at bar[2] close (executes at bar[3].open=130) and sells at bar[6] close
(executes at bar[7].open=170).

Manual calculation (verified before writing this test):

  BUY execution (bar[3].open = 130):
    fill_buy      = 130 * 1.0005 = 130.065
    safe_quote    = 10000 * 0.999 = 9990        # engine SAFETY_FACTOR
    qty           = 9990 / (130.065 * 1.001) = 76.7310189522... BTC
    cash_residual = 10000 - 9990 = 10.00 EUR    # stays in pool after SAFETY_FACTOR

  SELL execution (bar[7].open = 170):
    fill_sell     = 170 * 0.9995 = 169.915
    notional_sell = qty * 169.915
    sell_fee      = notional_sell * 0.001
    cash_after    = 10.00 + notional_sell - sell_fee

  Trade:
    gross_pnl     = (fill_sell - fill_buy) * qty = 3057.731105
    fees_total    = buy_fee + sell_fee             = 23.017771
    net_pnl       = gross_pnl - fees_total         = 3034.713334

Tolerance: abs=0.01 EUR (1 cent). Any larger diff is a real engine bug.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulator
from btc_quant.backtester.market_context import MarketContext
from btc_quant.backtester.models.enums import OrderSide, StrategyTag
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.backtester.models.signal import Signal
from btc_quant.backtester.strategy_base import StrategyBase

UTC = timezone.utc
TAG = StrategyTag.BUY_AND_HOLD_BENCHMARK
INITIAL_CAPITAL = 10_000.0
FEE = 0.001
SLIP = 0.0005

# Pre-computed manual values (see module docstring)
_SAFETY_FACTOR = 0.999
_BUY_OPEN = 130.0
_SELL_OPEN = 170.0
_FILL_BUY = _BUY_OPEN * (1.0 + SLIP)            # 130.065
_FILL_SELL = _SELL_OPEN * (1.0 - SLIP)           # 169.915
_SAFE_QUOTE = INITIAL_CAPITAL * _SAFETY_FACTOR    # 9990
_QTY = _SAFE_QUOTE / (_FILL_BUY * (1.0 + FEE))   # 76.7310189522...
_CASH_RESIDUAL = INITIAL_CAPITAL - _SAFE_QUOTE    # 10.0
_NOTIONAL_SELL = _QTY * _FILL_SELL
_SELL_FEE = _NOTIONAL_SELL * FEE
_EXPECTED_FINAL_CASH = _CASH_RESIDUAL + _NOTIONAL_SELL - _SELL_FEE   # 13034.713334
_BUY_FEE = (_QTY * _FILL_BUY) * FEE
_GROSS_PNL = (_FILL_SELL - _FILL_BUY) * _QTY
_EXPECTED_NET_PNL = _GROSS_PNL - (_BUY_FEE + _SELL_FEE)              # 3034.713334


def _make_synthetic_df() -> pd.DataFrame:
    """10 daily candles with known integer open prices [100..190]."""
    opens = [100.0 + i * 10.0 for i in range(10)]
    start = datetime(2020, 1, 1, tzinfo=UTC)
    period = timedelta(days=1)
    ms = timedelta(milliseconds=1)
    timestamps = [start + i * period for i in range(10)]
    close_times = [ts + period - ms for ts in timestamps]
    return pd.DataFrame({
        "timestamp_utc": pd.to_datetime(timestamps, utc=True),
        "close_time_utc": pd.to_datetime(close_times, utc=True),
        "open":   opens,
        "high":   [o + 5.0 for o in opens],
        "low":    [o - 5.0 for o in opens],
        "close":  [o + 2.0 for o in opens],
        "volume": [1_000.0] * 10,
        "is_synthetic": [False] * 10,
    })


class _BuyBar2SellBar6Strategy(StrategyBase):
    """Buys at close of bar[2] (executes bar[3]) and sells at close of bar[6] (executes bar[7])."""

    def __init__(self) -> None:
        super().__init__(TAG, {})
        self._call = 0

    def required_timeframes(self) -> list[str]:
        return ["1d"]

    def primary_timeframe(self) -> str:
        return "1d"

    def generate_signals(self, context: MarketContext, portfolio: Portfolio) -> list[Signal]:
        self._call += 1

        if self._call == 3:  # close of bar[2] → buy executes at bar[3].open
            return [Signal(
                strategy_tag=self.strategy_tag,
                timestamp_utc=context.current_time,
                side=OrderSide.BUY,
                symbol=context.symbol,
                quote_amount_eur=INITIAL_CAPITAL,
            )]

        if self._call == 7:  # close of bar[6] → sell executes at bar[7].open
            pool = portfolio.get_pool(self.strategy_tag)
            return [Signal(
                strategy_tag=self.strategy_tag,
                timestamp_utc=context.current_time,
                side=OrderSide.SELL,
                symbol=context.symbol,
                quantity_btc=pool.btc_held,
            )]

        return []


@pytest.fixture(scope="module")
def engine_result():
    df = _make_synthetic_df()
    engine = BacktestEngine(
        strategies=[_BuyBar2SellBar6Strategy()],
        data={"1d": df},
        initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
        execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
        profit_recycle_pct=0.0,
    )
    return engine.run()


class TestEngineManualConsistency:
    def test_exactly_one_closed_trade(self, engine_result) -> None:
        """One BUY + one SELL → exactly one closed trade."""
        assert len(engine_result.closed_trades) == 1

    def test_entry_price_matches_manual(self, engine_result) -> None:
        """Entry price must equal bar[3].open * (1 + slippage) = 130.065."""
        trade = engine_result.closed_trades[0]
        assert trade.entry_price_avg == pytest.approx(_FILL_BUY, abs=0.01)

    def test_exit_price_matches_manual(self, engine_result) -> None:
        """Exit price must equal bar[7].open * (1 - slippage) = 169.915."""
        trade = engine_result.closed_trades[0]
        assert trade.exit_price_avg == pytest.approx(_FILL_SELL, abs=0.01)

    def test_net_pnl_matches_manual(self, engine_result) -> None:
        """Net P&L must match gross_pnl - (buy_fee + sell_fee) = 3034.71 EUR."""
        trade = engine_result.closed_trades[0]
        assert trade.net_pnl_eur == pytest.approx(_EXPECTED_NET_PNL, abs=0.01)

    def test_final_pool_cash_matches_manual(self, engine_result) -> None:
        """After sell, pool cash = cash_residual + sell_proceeds = 13034.71 EUR."""
        final_snap = engine_result.equity_curve[-1]
        final_cash = final_snap.cash_by_pool[TAG]
        assert final_cash == pytest.approx(_EXPECTED_FINAL_CASH, abs=0.01)
