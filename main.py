import os
import asyncio
import logging
import sys
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data.api_requests import AlpacaClient
from core.trading_bot import TradingBot
from strategies.concrete_strategies.rsi_bbands import RSIBBands

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


async def main():
    """
    Entry point for the Live/Paper Trading Bot.
    """
    load_dotenv()
    api_key = os.getenv("alpaca_key")
    secret_key = os.getenv("alpaca_secret")

    if not api_key or not secret_key:
        logging.error("❌ Credentials missing. Please check your .env file.")
        return

    logging.info("--- 🚀 Initializing Build-A-Bot (Paper Mode) ---")

    # 1. Initialize Clients
    # Note: paper=True is hardcoded for safety. Change to False for Real Money.
    trading_client = TradingClient(api_key, secret_key, paper=True)
    data_stream = StockDataStream(api_key, secret_key)

    # 2. Strategy Setup
    # We use default parameters (Strict: RSI < 30) for "Production" simulation.
    # To use loose params for testing: RSIBBands(stage1_rsi_threshold=70, ...)
    strategy = RSIBBands()

    # 3. Dynamic Symbol Selection (Most Active)
    logging.info("Fetching market data to select active symbols...")
    data_client = AlpacaClient(api_key, secret_key)
    try:
        active_stocks_df = data_client.get_most_active_stocks()
        if active_stocks_df.is_empty():
            logging.error("No active stocks found. Exiting.")
            return

        # Select top 3 for safety/bandwidth management
        symbols = active_stocks_df["ticker"].head(3).to_list()
        logging.info(f"🎯 Target Symbols: {symbols}")

    except Exception as e:
        logging.error(f"Failed to fetch symbols: {e}")
        return

    # 4. Launch Bot
    bot = TradingBot(
        strategy=strategy,
        capital=100000.0,  # Paper money
        trading_client=trading_client,
        live_stock_data=data_stream,
        symbols=symbols
    )

    # 5. Run (Blocking)
    # This will maintain the WebSocket connection indefinitely
    await bot.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user.")
    except Exception as e:
        logging.critical(f"Fatal Error: {e}")
