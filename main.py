import os
import sys
import asyncio
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
# from alpaca.data.models.common import RawData

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.api_requests import AlpacaClient
from core.ws_stream_simulator import simulate_ws_stream
from core.trading_bot import TradingBot
from strategies.concrete_strategies.rsi_bbands import RSIBBands

# Updated main function for backtesting
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

    # 2. Fetch the most active stocks
    print("Fetching most active stocks...")
    most_active_stocks_df = alpaca_client.get_most_active_stocks()
    if most_active_stocks_df.is_empty():
        print("Could not fetch most active stocks. Exiting.")
        return
    
    active_symbols = most_active_stocks_df["ticker"].to_list()
    print(f"Most active stocks: {active_symbols}")

    # 3. Fetch Historical Data for all active stocks
    print("Fetching historical data...")
    historical_data_dict = {}
    for symbol in active_symbols:
        print(f"Fetching historical data for {symbol}...")
        historical_data = alpaca_client.get_historical_ohlcv(
            symbol=symbol,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start_date="2023-01-01",
            end_date="2023-01-31" 
        )
        if not historical_data.is_empty():
            historical_data_dict[symbol] = historical_data
            print(f"Historical data for {symbol} fetched successfully.")
        else:
            print(f"Could not fetch historical data for {symbol}.")

    if not historical_data_dict:
        print("No historical data fetched. Exiting.")
        return

    # 4. Set up Alpaca Clients for the bot
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    
    from unittest.mock import MagicMock
    live_stock_data_mock = MagicMock()

    # 5. Create the Trading Bot
    bot = TradingBot(
        strategy=my_strategy,
        capital=100000,
        trading_client=trading_client,
        live_stock_data=live_stock_data_mock,
        symbols=active_symbols 
    )

    # 6. Run the WebSocket Simulator for backtesting
    print("Starting WebSocket simulation for backtesting...")

    for symbol, historical_data in historical_data_dict.items():
        print(f"--- Simulating for {symbol} ---")
        for bar_row in historical_data.iter_rows(named=True):
            
            raw_bar_data = RawData({
                'S': symbol,
                'o': bar_row['open'],
                'h': bar_row['high'],
                'l': bar_row['low'],
                'c': bar_row['close'],
                'v': bar_row['volume'],
                't': bar_row['timestamp']
            })
            
            bar = Bar(raw_bar_data)

            asyncio.run(bot.handle_bar_update(bar))

    print("WebSocket simulation finished.")


# Updated main function for live trading
async def main_live():
    """
    The main function to set up and run the bot for live trading.
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

    # 2. Fetch the most active stocks
    print("Fetching most active stocks...")
    most_active_stocks_df = alpaca_client.get_most_active_stocks()
    if most_active_stocks_df.is_empty():
        print("Could not fetch most active stocks. Exiting.")
        return
    active_symbols = most_active_stocks_df["ticker"].to_list()
    print(f"Most active stocks: {active_symbols}")


    # 3. Set up Alpaca Clients
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    live_stock_data = StockDataStream(API_KEY, API_SECRET)

    # 4. Create the Trading Bot
    bot = TradingBot(
        strategy=my_strategy,
        capital=100000,
        trading_client=trading_client,
        live_stock_data=live_stock_data,
        symbols=active_symbols
    )

    print("Starting data stream... (Press Ctrl+C to stop)")
    live_stock_data.run()


if __name__ == "__main__":
    try:
        # Choose which main function to run.
        # For backtesting:
        main()
        # For live trading:
        # asyncio.run(main_live())
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
