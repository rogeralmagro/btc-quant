"""Shared fixtures for STRAT-06 unit tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from btc_quant.backtester.market_context import MarketContext
from btc_quant.backtester.models.enums import StrategyTag
from btc_quant.backtester.models.pool import CapitalPool
from btc_quant.backtester.models.portfolio import Portfolio
from btc_quant.strategies.strat06.config import DCAModulatedConfig

UTC = timezone.utc


@pytest.fixture
def default_config() -> DCAModulatedConfig:
    return DCAModulatedConfig(monthly_inflow_eur=500.0)


@pytest.fixture
def monday_noon_utc() -> datetime:
    """Monday 2024-01-08 12:00 UTC."""
    return datetime(2024, 1, 8, 12, 0, tzinfo=UTC)


@pytest.fixture
def mock_context_factory():
    """Factory: create a MockContext with a given timestamp and price."""
    def _factory(timestamp_utc: datetime, current_price: float) -> Mock:
        ctx = Mock(spec=MarketContext)
        ctx.current_time = timestamp_utc
        ctx.get_current_btc_price.return_value = current_price
        ctx.symbol = "BTCUSDT"
        return ctx
    return _factory


@pytest.fixture
def mock_portfolio_factory():
    """Factory: create a Portfolio with the 3 STRAT-06 pools."""
    def _factory(
        baseline_cash: float = 0.0,
        buffer_cash: float = 0.0,
        reserve_cash: float = 0.0,
        btc_held: float = 0.0,
    ) -> Portfolio:
        return Portfolio(
            pools={
                StrategyTag.STRAT_06_BASELINE: CapitalPool(
                    strategy_tag=StrategyTag.STRAT_06_BASELINE,
                    cash_eur=baseline_cash,
                    btc_held=btc_held,
                ),
                StrategyTag.STRAT_06_BUFFER: CapitalPool(
                    strategy_tag=StrategyTag.STRAT_06_BUFFER,
                    cash_eur=buffer_cash,
                ),
                StrategyTag.STRAT_06_RESERVE: CapitalPool(
                    strategy_tag=StrategyTag.STRAT_06_RESERVE,
                    cash_eur=reserve_cash,
                ),
            }
        )
    return _factory
