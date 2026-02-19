import os
import asyncio
import logging
import sys
from dotenv import load_dotenv

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from data.factory import get_market_provider
from data.alpaca_provider import AlpacaProvider
from core.trading_bot import TradingBot
from core.notification_manager import NotificationManager
from strategies.concrete_strategies.rsi_bbands import RSIBBands
from alpaca.trading.client import TradingClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def _get_execution_client(provider) -> "TradingClient":
    """
    Resolve the Alpaca TradingClient used for order execution.

    If the data provider is AlpacaProvider it already owns a
    TradingClient — reuse it.  Otherwise, create a standalone one
    from environment credentials (Alpaca is always the execution engine
    for now).
    """
    if isinstance(provider, AlpacaProvider):
        return provider.trading_client

    api_key = os.getenv("alpaca_key") or os.getenv("ALPACA_API_KEY")
    secret = os.getenv("alpaca_secret") or os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        raise ValueError(
            "Alpaca credentials are required for order execution even when "
            "using an alternate data source.  Set alpaca_key / alpaca_secret "
            "in your .env file."
        )
    is_paper = os.getenv("PAPER_MODE", "True").lower() == "true"
    return TradingClient(api_key, secret, paper=is_paper)


def main():
    """
    Synchronous Entry Point.
    Separates Async Setup (Warmup/Sync) from Blocking Stream execution.
    """
    load_dotenv()

    # 1. Initialize the data provider via factory (reads DATA_SOURCE env var)
    is_paper = os.getenv("PAPER_MODE", "True").lower() == "true"
    logging.info(
        f"--- Initializing Build-A-Bot (Mode: {'PAPER' if is_paper else 'LIVE'}) ---"
    )

    try:
        provider = get_market_provider()
    except ValueError as e:
        logging.error("Provider init failed: %s", e)
        return

    # Execution engine — always Alpaca for now, regardless of data source
    try:
        trading_client = _get_execution_client(provider)
    except ValueError as e:
        logging.error("Execution client init failed: %s", e)
        return

    discord_url = os.getenv("discord_webhook_url")
    notifier = NotificationManager(discord_url)

    # Strategy: Strict (Default)
    strategy = RSIBBands()

    # 2. Dynamic Symbol Selection
    logging.info("Fetching market data to select active symbols...")
    try:
        # Note: On weekends, this might return low-volume/weird stocks
        symbols = provider.get_active_symbols(limit=10)[:3]
        if not symbols:
            logging.error("No active stocks found. Exiting.")
            return

        logging.info(f"Target Symbols: {symbols}")

    except Exception as e:
        logging.error(f"Failed to fetch symbols: {e}")
        return

    # 3. Initialize Bot
    bot = TradingBot(
        strategy=strategy,
        capital=100000.0,
        trading_client=trading_client,
        data_provider=provider,
        symbols=symbols,
        notification_manager=notifier,
    )

    # Notify Discord of startup
    notifier.notify_startup(symbols)

    # 4. Async Setup Phase (Warmup & Sync)
    async def setup_phase():
        # A. Warmup (Fetch History)
        # Note: On Sundays/Weekends, this will likely find 0 bars. That is OK.
        await bot.warmup()

        # B. Sync & Subscribe (Prepare Logic)
        await bot.run()

    try:
        logging.info("Running Startup Sequence...")
        asyncio.run(setup_phase())
    except KeyboardInterrupt:
        return
    except Exception as e:
        logging.critical(f"Startup Failed: {e}")
        return

    # 5. Blocking Stream Execution (Main Thread)
    # We are now outside the asyncio loop, so the provider manages its own event loop.
    logging.info("Startup Complete. Starting Data Stream (Ctrl+C to stop)...")
    try:
        provider.run_stream()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    except Exception as e:
        logging.critical(f"Stream Crashed: {e}")


if __name__ == "__main__":
    main()
