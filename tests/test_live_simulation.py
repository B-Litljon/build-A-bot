import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, List
from zoneinfo import ZoneInfo

import polars as pl

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from data.market_provider import MarketDataProvider
from data.alpaca_provider import AlpacaProvider
from core.trading_bot import TradingBot
from strategies.concrete_strategies.rsi_bbands import RSIBBands
from core.ws_stream_simulator import simulate_ws_stream

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")


# --- MOCKS ---
class MockTradingClient:
    def __init__(self):
        self.orders = []

    def submit_order(self, order_request=None, **kwargs):
        order = order_request if order_request is not None else type("obj", (object,), kwargs)
        side = getattr(order, "side", kwargs.get("side"))
        qty = getattr(order, "qty", kwargs.get("qty"))
        symbol = getattr(order, "symbol", kwargs.get("symbol"))
        print(f"\n[MOCK API] ORDER RECEIVED: {side} {qty} {symbol}")
        self.orders.append(order)
        return type("obj", (object,), {"id": "mock_order_id"})

    def get_all_positions(self):
        return []


class MockProvider(MarketDataProvider):
    """
    A mock provider that stores pre-fetched historical data and
    records subscribe/run_stream calls without hitting the network.
    """

    def __init__(self, historical_data: dict[str, pl.DataFrame] | None = None):
        self._historical_data = historical_data or {}
        self._subscribed_callback: Callable | None = None
        self._subscribed_symbols: List[str] = []

    def get_active_symbols(self, limit: int = 10) -> List[str]:
        return list(self._historical_data.keys())[:limit]

    def get_historical_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        return self._historical_data.get(symbol, pl.DataFrame())

    def subscribe(self, symbols: List[str], callback: Callable) -> None:
        self._subscribed_symbols = symbols
        self._subscribed_callback = callback
        print(f"[MOCK PROVIDER] Subscribed to {symbols}")

    def run_stream(self) -> None:
        print("[MOCK PROVIDER] run_stream called (no-op)")


# --- RUNNER ---
async def run_simulation():
    print("--- Starting Historical Replay Simulation ---")

    # 1. Setup Environment
    api_key = os.getenv("alpaca_key")
    secret_key = os.getenv("alpaca_secret")
    if not api_key:
        print("Error: Alpaca keys not found in environment.")
        return

    # 2. Fetch REAL Historical Data via the Alpaca provider
    real_provider = AlpacaProvider(api_key, secret_key, paper=True)
    symbol = "SPY"
    print(f"Fetching real data for {symbol}...")

    end_date = datetime.now(ZoneInfo("America/New_York")) - timedelta(minutes=16)
    start_date = end_date - timedelta(days=2)

    df = real_provider.get_historical_bars(
        symbol=symbol,
        timeframe_minutes=1,
        start=start_date,
        end=end_date,
    )

    if df.is_empty():
        print("Error: No data fetched. Check API keys or market hours.")
        return

    print(f"Loaded {len(df)} candles. Starting replay...")

    # 3. Initialize Bot with MockProvider (holds the fetched data)
    mock_trade = MockTradingClient()
    mock_provider = MockProvider(historical_data={symbol: df})

    # Initialize RSIBBands with LOOSE parameters for testing
    strategy = RSIBBands(
        stage1_rsi_threshold=70,
        stage2_min_roc=0.0001,
        stage2_rsi_entry=10,
        stage2_rsi_exit=90,
    )
    if not hasattr(strategy, "timeframe"):
        strategy.timeframe = 5

    bot = TradingBot(
        strategy=strategy,
        capital=10000.0,
        trading_client=mock_trade,
        data_provider=mock_provider,
        symbols=[symbol],
    )

    # 4. Run Simulation Loop
    stream_generator = simulate_ws_stream(df, speed=1000000.0)

    for row in stream_generator:
        timestamp = row["timestamp"][0]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        # Build a provider-agnostic bar dict (same shape the real provider emits)
        bar = {
            "symbol": symbol,
            "timestamp": timestamp,
            "open": float(row["open"][0]),
            "high": float(row["high"][0]),
            "low": float(row["low"][0]),
            "close": float(row["close"][0]),
            "volume": float(row["volume"][0]),
        }

        # Feed to Bot
        await bot.handle_bar_update(bar)

    # 5. Summary
    print("\n--- Simulation Complete ---")
    print(f"Total Bars Processed: {len(df)}")
    print(f"Total Orders Placed: {len(mock_trade.orders)}")
    for order in mock_trade.orders:
        print(f" - {order.side} {order.symbol}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    asyncio.run(run_simulation())
