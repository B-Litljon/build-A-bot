from typing import Dict,List, Any, Optional
from core.signal import Signal
from strategies.strategy import Strategy
from alpaca.trading.client import TradingClient
from alpaca.data.live import StockDataStream
from alpaca.data.models.bars import Bar 
from core.order_management import OrderManager
from utils.bar_aggregator import LiveBarAggregator as lba # Import BarAggregator
import asyncio 
import logging
import polars as pl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class TradingBot:
    """
    The main trading bot class.
    ...
    """
    def __init__(self,
                 strategy: Strategy,
                 capital: float,
                 trading_client: TradingClient,
                 live_stock_data: StockDataStream,
                 symbols: List[str],  # Changed from symbol: str
                 target_intervals: List[int] = [5, 15]):
        self.strategy = strategy
        self.capital = capital
        self.trading_client = trading_client
        self.order_manager = OrderManager(trading_client, strategy.get_order_params())
        self.live_stock_data = live_stock_data
        self.symbols = symbols
        self.target_intervals = target_intervals
        # Use a dictionary to store a bar aggregator for each symbol
        self.lba_dict = {
            symbol: lba(timeframe=strategy.timeframe, history_size=240)
            for symbol in self.symbols
        }
        logging.info(f"Subscribing to bar updates for symbols: {self.symbols}")
        # The '*' unpacks the list of symbols for the subscription
        self.live_stock_data.subscribe_bars(self.handle_bar_update, *self.symbols)

    async def handle_bar_update(self, raw_bar: Bar):
        """
        Async handler for incoming raw bar updates from the data stream.
        """
        symbol = raw_bar.symbol
        current_price = raw_bar.close

        # --- 1. EXIT LOGIC (Check every tick) ---
        # Create a mini market_data dict for the monitor
        self.order_manager.monitor_orders({symbol: current_price})

        # --- 2. EXISTING AGGREGATION LOGIC ---
        logging.info(
            f"Received raw bar for {symbol}:\n"
            f"  Open: {raw_bar.open}\n"
            f"  High: {raw_bar.high}\n"
            f"  Low: {raw_bar.low}\n"
            f"  Close: {raw_bar.close}\n"
            f"  Volume: {raw_bar.volume}"
        )
        try:
            formatted_bar = {
                "timestamp": raw_bar.timestamp,
                "open": raw_bar.open,
                "high": raw_bar.high,
                "low": raw_bar.low,
                "close": raw_bar.close,
                "volume": raw_bar.volume,
            }

            # Get the correct bar aggregator for the symbol
            symbol_lba = self.lba_dict.get(symbol)
            if not symbol_lba:
                logging.warning(f"No bar aggregator found for symbol: {symbol}")
                return

            is_new_agg_bar = symbol_lba.add_bar(formatted_bar)

            if is_new_agg_bar:
                logging.info(f"New aggregated bar created for {symbol}")
                candles = symbol_lba.history_df
                if len(candles) >= self.strategy.warmup_period:
                    signals = self.strategy.analyze({symbol: candles})
                    self.place_orders(signals)
                else:
                    logging.info(f"Warming up... {len(candles)}/{self.strategy.warmup_period} candles")

        except Exception as e:
            logging.error(f"Error handling bar update for {symbol}: {e}", exc_info=True)

    def place_orders(self, signals: List[Signal]):
        """Places orders based on the received signals."""
        for signal in signals:
            if signal.type == "BUY":
                self.order_manager.place_order(signal, self.capital)

    async def run(self):
        """
        Subscribes to the symbol and starts the data stream
        """
        logging.info(f"Subscribing to bar updates for symbols: {self.symbols}")
        self.live_stock_data.subscribe_bars(self.handle_bar_update, *self.symbols)

    async def log_status_periodically(self, interval: int = 30):
        """Logs the bot's status at regular intervals."""
        while True:
            self.log_status()
            await asyncio.sleep(interval)

    def log_status(self):
        """
        Logs the current status of the bot.
        """
        logging.info("--- Bot Status ---")
        logging.info(f"Current Capital: {self.capital}")
        logging.info(f"Active Orders: {self.order_manager.active_orders}")
        logging.info("------------------")
