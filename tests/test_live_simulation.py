import sys
import os
import asyncio
import logging
from datetime import datetime, timedelta
from alpaca.data.models.bars import Bar
from alpaca.data.timeframe import TimeFrame

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from core.trading_bot import TradingBot
from data.api_requests import AlpacaClient
from strategies.concrete_strategies.sma_crossover import SMACrossover
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


class MockDataStream:
    def subscribe_bars(self, handler, *symbols):
        print(f"[MOCK STREAM] Subscribed to {symbols}")


# --- RUNNER ---
async def run_simulation():
    print("--- Starting Historical Replay Simulation ---")

    # 1. Setup Environment
    api_key = os.getenv("alpaca_key")
    secret_key = os.getenv("alpaca_secret")
    if not api_key:
        print("Error: Alpaca keys not found in environment.")
        return

    # 2. Fetch REAL Historical Data
    client = AlpacaClient(api_key, secret_key)
    symbol = "SPY"
    print(f"Fetching real data for {symbol}...")

    # Fetch last 2 days of 1-minute bars
    end_date = datetime.now()
    start_date = end_date - timedelta(days=2)

    df = client.get_historical_ohlcv(
        symbol=symbol,
        timeframe=TimeFrame.Minute,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    if df.is_empty():
        print("Error: No data fetched. Check API keys or market hours.")
        return

    print(f"Loaded {len(df)} candles. Starting replay...")

    # 3. Initialize Bot
    mock_trade = MockTradingClient()
    mock_stream = MockDataStream()
    strategy = SMACrossover(fast_period=10, slow_period=30)
    if not hasattr(strategy, "timeframe"):
        strategy.timeframe = 5

    bot = TradingBot(
        strategy=strategy,
        capital=10000.0,
        trading_client=mock_trade,
        live_stock_data=mock_stream,
        symbols=[symbol],
    )

    # 4. Run Simulation Loop
    # We set speed to 1000000 to run instantly (skip sleeps)
    stream_generator = simulate_ws_stream(df, speed=1000000.0)

    for row in stream_generator:
        timestamp = row["timestamp"][0]
        if hasattr(timestamp, "to_pydatetime"):
            timestamp = timestamp.to_pydatetime()

        raw_bar = {
            "t": timestamp,
            "o": float(row["open"][0]),
            "h": float(row["high"][0]),
            "l": float(row["low"][0]),
            "c": float(row["close"][0]),
            "v": float(row["volume"][0]),
            "n": 0.0,
            "vw": float(row["close"][0]),
            "S": symbol,
        }
        bar = Bar(symbol=symbol, raw_data=raw_bar)

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
