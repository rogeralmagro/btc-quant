"""Integration test: BAH engine return tracks theoretical return within 0.3 pp.

Validates across 5 historical subperiods with well-known BTC behaviour:

  2018_bear    : ~-73%   (from ATH to crypto winter)
  2020_covid   : ~+302%  (post-covid recovery + institutional inflow)
  2021_bull    : ~+60%   (continuation bull, peaked Nov 2021)
  2022_bear    : ~-64%   (rate hikes + Luna/FTX collapse)
  2023_recovery: ~+156%  (ETF anticipation + halving narrative)

NOTA IMPORTANTE: el engine ejecuta órdenes al OPEN de la vela siguiente a la
señal (modelo anti-look-ahead). Por tanto el precio de fill real es bar[1].open,
NO bar[0].open. El raw_return calculado contra bar[0].open sobreestima
sistemáticamente el retorno del engine en períodos con gap overnight positivo
(y subestima con gap negativo).

Este test compara contra el retorno teórico calculado con bar[1].open como
referencia de fill, reproduciendo la lógica exacta del engine. Es
matemáticamente correcto y NO oculta bugs: una diferencia > 0.3 pp indica
un problema real en la lógica de fill o mark-to-market del engine.

Tolerancia 0.3 pp absorbe:
- Variabilidad de slippage si ExecutionSimulatorV2 detecta régimen distinto
- Rounding acumulado a lo largo del subperíodo
- Diferencias menores en el snapshot final (close_time_utc vs timestamp_utc)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from btc_quant.backtester.engine import BacktestEngine
from btc_quant.backtester.execution_simulator import ExecutionSimulator
from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.strategies.buy_and_hold import BuyAndHoldStrategy

DATA_DIR = Path(__file__).parents[2] / "data" / "processed"
DAILY_PARQUET = DATA_DIR / "btcusdt_1d.parquet"

TAG = StrategyTag.BUY_AND_HOLD_BENCHMARK
INITIAL_CAPITAL = 10_000.0
FEE = 0.001
SLIP = 0.0005
TOLERANCE = 0.003  # 0.3 pp

SUBPERIODS: dict[str, tuple[str, str]] = {
    "2018_bear":     ("2018-01-01", "2018-12-31"),
    "2020_covid":    ("2020-01-01", "2020-12-31"),
    "2021_bull":     ("2021-01-01", "2021-12-31"),
    "2022_bear":     ("2022-01-01", "2022-12-31"),
    "2023_recovery": ("2023-01-01", "2023-12-31"),
}


def _load_subperiod(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(DAILY_PARQUET)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.set_index("timestamp_utc").sort_index()
    return df.loc[start:end].reset_index()


_SAFETY_FACTOR = 0.999  # must match BacktestEngine._execute_signal


def _theoretical_return(sub: pd.DataFrame) -> float:
    """Compute the return the engine should produce, mirroring its exact logic.

    bar[0]: BAH strategy emits BUY with quote_amount_eur = pool.cash_eur
    bar[1]: engine applies SAFETY_FACTOR → safe_quote = capital * 0.999
            fill_price = bar[1].open * (1 + SLIP)
            qty = safe_quote / (fill_price * (1 + FEE))
            cash_residual = capital - safe_quote  (stays in pool)
    bar[-1]: total_value_eur = qty * last_close + cash_residual
    """
    safe_quote = INITIAL_CAPITAL * _SAFETY_FACTOR
    fill_price = float(sub.iloc[1]["open"]) * (1.0 + SLIP)
    qty = safe_quote / (fill_price * (1.0 + FEE))
    cash_residual = INITIAL_CAPITAL - safe_quote
    last_close = float(sub.iloc[-1]["close"])
    expected_final_value = qty * last_close + cash_residual
    return (expected_final_value / INITIAL_CAPITAL) - 1.0


def _bah_engine_return(sub: pd.DataFrame) -> float:
    """Run BAH engine on the subperiod and return (final_value / initial - 1)."""
    engine = BacktestEngine(
        strategies=[BuyAndHoldStrategy()],
        data={"1d": sub},
        initial_capital_per_strategy={TAG: INITIAL_CAPITAL},
        execution_simulator=ExecutionSimulator(fee_rate=FEE, slippage_rate=SLIP),
        profit_recycle_pct=0.0,
    )
    result = engine.run()
    final_value = result.equity_curve[-1].total_value_eur
    return (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL


@pytest.mark.parametrize("name,dates", list(SUBPERIODS.items()))
def test_bah_return_tracks_price_within_tolerance(
    name: str, dates: tuple[str, str]
) -> None:
    """BAH engine return must be within 0.3 pp of the theoretically expected return.

    The theoretical return is computed using bar[1].open as the fill price
    (matching the engine's pending-signal execution model) plus fee and slippage,
    marked to bar[-1].close. Any diff > 0.3 pp indicates a real engine bug.
    """
    start, end = dates
    sub = _load_subperiod(start, end)
    expected = _theoretical_return(sub)
    actual = _bah_engine_return(sub)

    diff = abs(actual - expected)
    assert diff <= TOLERANCE, (
        f"{name}: engine return {actual*100:.3f}% differs from theoretical "
        f"{expected*100:.3f}% (fill at bar[1].open={sub.iloc[1]['open']:.0f} "
        f"+ {SLIP*100:.2f}% slip, {FEE*100:.2f}% fee, last_close={sub.iloc[-1]['close']:.0f}). "
        f"Diff = {diff*100:.3f} pp, tolerance {TOLERANCE*100} pp."
    )
