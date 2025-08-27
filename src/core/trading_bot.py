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

    Attributes:
        strategy (Strategy): The trading strategy to use.
        capital (float): The initial trading capital.
        trading_client (TradingClient): The trading client for interacting with the exchange.
        order_manager (OrderManager): order manager created from strategy and trading client
        bar_aggregator (BarAggregator): Aggregates bar data into different timeframes.
    """
    def __init__(self,
        strategy: Strategy, capital: float,
        trading_client: TradingClient, live_stock_data: StockDataStream,
        symbol: str = "SPY",
        target_intervals: List[int] = [5, 15]
        ):
        self.strategy = strategy
        self.capital = capital
        self.trading_client = trading_client
        self.order_manager = OrderManager(trading_client, strategy.get_order_params())
        self.live_stock_data = live_stock_data
        self.symbol = symbol
        self.target_intervals = target_intervals
        self.lba = lba(
            timeframe=strategy.timeframe,
            history_size=240
        )
        logging.info(f"Subscribing to bar updates for symbol: {self.symbol}")
        self.live_stock_data.subscribe_bars(self.handle_bar_update, self.symbol)

    async def handle_bar_update(self, raw_bar: Bar):
        """
        Async handler for incoming raw bar updates from the data stream.
        """
        
        logging.info(
        f"Received raw bar for {raw_bar.symbol}:\n"
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

            is_new_agg_bar = self.lba.add_bar(formatted_bar)

            if is_new_agg_bar:
                logging.info(f"New aggregated bar created for {self.symbol}")
                candles = self.lba.history_df
                if len(candles) > self.strategy.rsi_period:
                    signals = self.strategy.analyze({self.symbol: candles})
                    self.place_orders(signals)

        except Exception as e:
            logging.error(f"Error handling bar update for {self.symbol}: {e}", exc_info=True)

    def place_orders(self, signals: List[Signal]):
        """Places orders based on the received signals."""
        for signal in signals:
            if signal.type == "BUY":
                self.order_manager.place_order(signal, self.capital)

    async def run(self):
        """
        Subscribes to the symbol and starts the data stream
        """
        logging.info(f"Subscribing to bar updates for symbol: {self.symbol}")
        self.live_stock_data.subscribe_bars(self.handle_bar_update, self.symbol)

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