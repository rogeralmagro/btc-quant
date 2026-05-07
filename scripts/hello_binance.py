"""Fetch the latest BTCUSDT 1D candle from Binance public API and print it.

No API key required — uses public market data endpoint.
Run: python scripts/hello_binance.py
"""

from binance.client import Client
from datetime import datetime, timezone


SYMBOL = "BTCUSDT"
INTERVAL = Client.KLINE_INTERVAL_1DAY


def fetch_latest_candle(symbol: str, interval: str) -> dict:
    client = Client()  # no credentials needed for public endpoints
    raw = client.get_klines(symbol=symbol, interval=interval, limit=1)[0]

    return {
        "symbol": symbol,
        "open_time": datetime.fromtimestamp(raw[0] / 1000, tz=timezone.utc).isoformat(),
        "open": float(raw[1]),
        "high": float(raw[2]),
        "low": float(raw[3]),
        "close": float(raw[4]),
        "volume_btc": float(raw[5]),
        "close_time": datetime.fromtimestamp(raw[6] / 1000, tz=timezone.utc).isoformat(),
        "volume_usdt": float(raw[7]),
        "num_trades": int(raw[8]),
    }


def main() -> None:
    print(f"Fetching latest {INTERVAL} candle for {SYMBOL}...\n")
    candle = fetch_latest_candle(SYMBOL, INTERVAL)

    width = max(len(k) for k in candle) + 2
    for key, value in candle.items():
        label = f"{key}:".ljust(width)
        if isinstance(value, float):
            print(f"  {label} {value:,.2f}")
        else:
            print(f"  {label} {value}")


if __name__ == "__main__":
    main()
