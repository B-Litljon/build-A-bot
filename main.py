import os
import sys
import asyncio
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.api_requests import AlpacaClient
from core.ws_stream_simulator import simulate_ws_stream
from core.trading_bot import TradingBot
from strategies.concrete_strategies.rsi_bbands import RSIBBands

# def main():
#     """
#     The main function to set up and run the bot.
#     """
#     load_dotenv()
#     API_KEY = os.getenv("alpaca_key")
#     API_SECRET = os.getenv("alpaca_secret")

#     if not API_KEY or not API_SECRET:
#         print("Error: Make sure your alpaca_key and alpaca_secret are set in a .env file.")
#         return

#     # 1. Initialize the Strategy
#     my_strategy = RSIBBands()

#     # 2. Set up Alpaca Clients
#     trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
#     live_stock_data = StockDataStream(API_KEY, API_SECRET)

#     # 3. Create the Trading Bot
#     bot = TradingBot(
#         strategy=my_strategy,
#         capital=100000,
#         trading_client=trading_client,
#         live_stock_data=live_stock_data,
#         symbol="SPY"
#     )

#     print("Subscribing bot to data stream...")

#     # 4. Subscribe the bot's async handler to the data stream.
#     # The library will correctly schedule this on its event loop.
#     live_stock_data.subscribe_bars(bot.handle_bar_update, bot.symbol)

#     print("Starting data stream... (Press Ctrl+C to stop)")

#     # 5. Run the data stream.
#     # This is a blocking call that starts the asyncio event loop
#     # and runs until the program is interrupted.
#     live_stock_data.run()

# if __name__ == "__main__":
#     try:
#         main()
#     except KeyboardInterrupt:
#         print("\nBot stopped by user.")
#     except Exception as e:
#         print(f"An error occurred: {e}")

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

    # 1. Initialize the Alpaca Client and Strategy
    alpaca_client = AlpacaClient(API_KEY, API_SECRET)
    my_strategy = RSIBBands()

    # 2. Fetch Historical Data
    print("Fetching historical data...")
    historical_data = alpaca_client.get_historical_ohlcv(
        symbol="SPY",
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start_date="2023-01-01",
        end_date="2023-01-31"
    )

    if historical_data.is_empty():
        print("Could not fetch historical data. Exiting.")
        return
    
    print("Historical data fetched successfully.")

    # 3. Set up Alpaca Clients for the bot
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    live_stock_data = StockDataStream(API_KEY, API_SECRET)

    # 4. Create the Trading Bot
    bot = TradingBot(
        strategy=my_strategy,
        capital=100000,
        trading_client=trading_client,
        live_stock_data=live_stock_data,
        symbol="SPY"
    )

    # 5. Run the WebSocket Simulator for backtesting
    print("Starting WebSocket simulation for backtesting...")
    for bar in simulate_ws_stream(historical_data):
        # In a real application, you would now pass this 'bar' 
        bot.handle_bar_update(bar)
        print(bar)
    print("WebSocket simulation finished.")


    # The original live trading logic is commented out below.
    # You can uncomment it when you're ready to trade with live data.
    """
    print("Subscribing bot to data stream...")
    live_stock_data.subscribe_bars(bot.handle_bar_update, bot.symbol)
    print("Starting data stream... (Press Ctrl+C to stop)")
    live_stock_data.run()
    """

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")