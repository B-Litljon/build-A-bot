import os
import sys
import asyncio
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from core.trading_bot import TradingBot
from strategies.concrete_strategies.rsi_bbands import RSIBBands

async def main():
    """
    The main asynchronous function to run the bot.
    """
    load_dotenv()
    API_KEY = os.getenv("alpaca_key")
    API_SECRET = os.getenv("alpaca_secret")

    if not API_KEY or not API_SECRET:
        print("Error: Make sure your API_KEY and API_SECRET are set in a .env file.")
        return

    # 1. Initialize the Strategy
    my_strategy = RSIBBands()

    # 2. Set up Alpaca Clients
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    live_stock_data = StockDataStream(API_KEY, API_SECRET)

    # 3. Create the Trading Bot
    bot = TradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        strategy=my_strategy,
        capital=100000,
        trading_client=trading_client,
        live_stock_data=live_stock_data,
        symbol="SPY"
    )

    print("Starting bot and data stream...")

    # 4. Create a task for our periodic status logger
    status_task = asyncio.create_task(bot.log_status_periodically(interval=30))

    # 5. Run the data stream
    # This will now run concurrently with the status logger
    await live_stock_data._run_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")