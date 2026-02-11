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
from core.notification_manager import NotificationManager
from strategies.concrete_strategies.rsi_bbands import RSIBBands

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Synchronous Entry Point.
    Separates Async Setup (Warmup/Sync) from Blocking Stream execution.
    """
    load_dotenv()
    api_key = os.getenv("alpaca_key")
    secret_key = os.getenv("alpaca_secret")

    if not api_key or not secret_key:
        logging.error("❌ Credentials missing. Check .env file.")
        return

    logging.info("--- 🚀 Initializing Build-A-Bot (Paper Mode) ---")

    # 1. Initialize Objects (Safe in sync context)
    trading_client = TradingClient(api_key, secret_key, paper=True)
    data_stream = StockDataStream(api_key, secret_key)
    discord_url = os.getenv("discord_webhook_url")
    notifier = NotificationManager(discord_url)

    # Strategy: Strict (Default)
    strategy = RSIBBands() 

    # 2. Dynamic Symbol Selection
    logging.info("Fetching market data to select active symbols...")
    data_client = AlpacaClient(api_key, secret_key)
    try:
        # Note: On weekends, this might return low-volume/weird stocks
        active_stocks_df = data_client.get_most_active_stocks()
        if active_stocks_df.is_empty():
            logging.error("No active stocks found. Exiting.")
            return
        
        symbols = active_stocks_df["ticker"].head(3).to_list()
        logging.info(f"🎯 Target Symbols: {symbols}")

    except Exception as e:
        logging.error(f"Failed to fetch symbols: {e}")
        return

    # 3. Initialize Bot
    bot = TradingBot(
        strategy=strategy,
        capital=100000.0,
        trading_client=trading_client,
        live_stock_data=data_stream,
        symbols=symbols,
        notification_manager=notifier
    )

    # Notify Discord of startup
    notifier.notify_startup(symbols)

    # 4. Async Setup Phase (Warmup & Sync)
    async def setup_phase():
        # A. Warmup (Fetch History)
        # Note: On Sundays/Weekends, this will likely find 0 bars. That is OK.
        await bot.warmup(data_client)
        
        # B. Sync & Subscribe (Prepare Logic)
        await bot.run() 
    
    try:
        logging.info("⏳ Running Startup Sequence...")
        asyncio.run(setup_phase())
    except KeyboardInterrupt:
        return
    except Exception as e:
        logging.critical(f"Startup Failed: {e}")
        return

    # 5. Blocking Stream Execution (Main Thread)
    # We are now outside the asyncio loop, so Alpaca can manage the websocket safely.
    logging.info("✅ Startup Complete. Starting Data Stream (Ctrl+C to stop)...")
    try:
        data_stream.run()
    except KeyboardInterrupt:
        logging.info("🛑 Bot stopped by user.")
    except Exception as e:
        logging.critical(f"Stream Crashed: {e}")

if __name__ == "__main__":
    main()
