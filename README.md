# btc-quant

Bitcoin quantitative trading system built for capital preservation and systematic execution.

## Architecture

The system has two layers:

- **Core layer** — DCA modulated by drawdown. Accumulates BTC systematically; buy size scales with distance from all-time high.
- **Tactical overlay** — Multi-signal confluence strategy. Enters tactical positions only when several independent indicators align.

## Project status

| Week | Milestone |
|------|-----------|
| ✅ W1 | Repo scaffold + Binance data pipeline |
| ✅ W2 | Indicator library (RSI, MVRV-Z, Fear & Greed, NVT) |
| ⬜ W3 | Core DCA strategy + backtest harness |
| ⬜ W4 | Tactical overlay signals + confluence logic |
| ⬜ W5–6 | Full backtest + walk-forward validation |
| ⬜ W7–8 | Risk engine + position sizing |
| ⬜ W9–10 | Paper trading + monitoring |
| ⬜ W11–12 | Audit → first real trade |

## Setup

```bash
# Clone and enter the repo
git clone <repo-url>
cd btc-quant

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install all dependencies
uv sync --extra dev

# Copy env template
cp .env.example .env
# Edit .env with your credentials (never commit .env)

# Verify setup
pytest
python scripts/hello_binance.py
```

## Safety

`LIVE_TRADING_ENABLED=false` is the default and must be explicitly overridden to submit real orders.
No live execution code exists yet — this project is in the research/backtesting phase.

## Docs

- [`docs/strategies/`](docs/strategies/) — strategy design documents (Spanish)
- [`docs/data_known_issues.md`](docs/data_known_issues.md) — known data quality issues (4h gaps, forward-fill policy)
- [`CHANGELOG.md`](CHANGELOG.md) — version history

## License

All rights reserved. See [LICENSE](LICENSE).
