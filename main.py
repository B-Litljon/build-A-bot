import os
import sys
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream

# --- This is the most important part ---
# It adds your 'src' folder to the list of places Python looks for code.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
# -----------------------------------------

# Now we can import your modules using absolute paths from 'src'
from core.trading_bot import TradingBot
from strategies.concrete_strategies.rsi_bbands import RSIBBands

# Load environment variables (like your API keys)
load_dotenv()
API_KEY = os.getenv("alpaca_key")
API_SECRET = os.getenv("alpaca_secret")

if not API_KEY or not API_SECRET:
    print("Error: Make sure your API_KEY and API_SECRET are set in a .env file.")
else:
    # 1. Initialize the Strategy
    # You can swap this out for any strategy you build
    my_strategy = RSIBBands()

    # 2. Set up Alpaca Clients
    trading_client = TradingClient(API_KEY, API_SECRET, paper=True)
    live_stock_data = StockDataStream(API_KEY, API_SECRET)

    # 3. Create and Run the Trading Bot
    bot = TradingBot(
        api_key=API_KEY,
        api_secret=API_SECRET,
        strategy=my_strategy,
        capital=100000,  # Example starting capital
        trading_client=trading_client,
        live_stock_data=live_stock_data,
        symbol="SPY" # The symbol to trade
    )

    # Run the bot's data stream
    # Note: The run() method in your bot has an infinite loop,
    # so we just start the stream here.
    print("Starting data stream...")
    live_stock_data.run()
