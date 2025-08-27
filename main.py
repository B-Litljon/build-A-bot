import os
import sys
import asyncio
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from core.trading_bot import TradingBot
from strategies.concrete_strategies.rsi_bbands import RSIBBands

def main():
    """
    The main function to set up and run the bot.
    """
    load_dotenv()
    API_KEY = os.getenv("alpaca_key")
    API_SECRET = os.getenv("alpaca_secret")

    if not API_KEY or not API_SECRET:
        print("Error: Make sure your alpaca_key and alpaca_secret are set in a .env file.")
        return

    # 1. Initialize the Strategy
    my_strategy = RSIBBands()

    # 2. Set up Alpaca Clients
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    live_stock_data = StockDataStream(API_KEY, API_SECRET)

    # 3. Create the Trading Bot
    bot = TradingBot(
        strategy=my_strategy,
        capital=100000,
        trading_client=trading_client,
        live_stock_data=live_stock_data,
        symbol="SPY"
    )

    print("Subscribing bot to data stream...")

    # 4. Subscribe the bot's async handler to the data stream.
    # The library will correctly schedule this on its event loop.
    live_stock_data.subscribe_bars(bot.handle_bar_update, bot.symbol)

    print("Starting data stream... (Press Ctrl+C to stop)")

    # 5. Run the data stream.
    # This is a blocking call that starts the asyncio event loop
    # and runs until the program is interrupted.
    live_stock_data.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
