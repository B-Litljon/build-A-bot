from typing import Dict,List, Any, Optional
from core.signal import Signal
from strategies.strategy import Strategy
from alpaca.trading.client import TradingClient
from alpaca.data import StockDataStream, Bar
from core.order_management import OrderManager
from utils.bar_aggregator import BarAggregator  # Import BarAggregator
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
    def __init__(self, api_key: str, api_secret: str, strategy: Strategy, capital: float,
                 trading_client: TradingClient, live_stock_data: StockDataStream,
                 symbol: str = "SPY",  # Hardcoded symbol as requested
                 target_intervals: List[int] = [5, 15]): # Example target intervals
        """
        Initializes the TradingBot.

        Args:
            api_key (str): Alpaca API key.
            api_secret (str): Alpaca API secret.
            strategy (Strategy): The trading strategy instance.
            capital (float): Initial trading capital.
            trading_client (TradingClient): Initialized Alpaca TradingClient.
            live_stock_data (StockDataStream): Initialized Alpaca StockDataStream.
            symbol (str, optional): The stock symbol to trade. Defaults to "SPY".
            target_intervals (List[int], optional): Target aggregation intervals in minutes. Defaults to [5, 15].
        """
        self.strategy = strategy
        self.capital = capital
        self.trading_client = trading_client
        # Assuming OrderManager is correctly initialized elsewhere or within strategy
        self.order_manager = OrderManager(trading_client, strategy.get_order_params())
        self.live_stock_data = live_stock_data # Ensure the stream instance is ready
        self.symbol = symbol # Assign the symbol
        self.target_intervals = target_intervals
        self.bar_aggregator = BarAggregator(
            base_interval=1,  # Assuming incoming bars are 1-minute
            target_intervals=self.target_intervals
        )
        self.completed_agg_bars: Dict[int, List[Dict[str, Any]]] = {interval: [] for interval in target_intervals} # Store completed bars


        # Subscribe to bar updates for the specified symbol
        logging.info(f"Subscribing to bar updates for symbol: {self.symbol}")
        self.live_stock_data.subscribe_bars(self.handle_bar_update, self.symbol)

    async def handle_bar_update(self, raw_bar: Bar):
        """
        Async handler for incoming raw bar updates from the data stream.
        Formats the bar, feeds it to the BarAggregator, retrieves completed
        aggregated bars, and stores them.

        Args:
            raw_bar (Bar): The raw bar object received from the Alpaca stream.
        """
        logging.debug(f"Received raw bar for {raw_bar.symbol}: {raw_bar}")

        try:
            # 1. Extract necessary data and format for BarAggregator
            formatted_bar = {
                "timestamp": raw_bar.timestamp,
                "open": raw_bar.open,
                "high": raw_bar.high,
                "low": raw_bar.low,
                "close": raw_bar.close,
                "volume": raw_bar.volume,
                # Optional: Add symbol if needed by aggregator or strategy later
                # "symbol": raw_bar.symbol
            }

            # 2. Add the 1-minute bar to the aggregator
            # The add_bar method handles the aggregation logic internally
            # and returns a completed bar if one is formed for a target interval.
            completed_bar: Optional[Dict[str, Any]] = self.bar_aggregator.add_bar(formatted_bar, interval=1)

            # 3. Check if a completed aggregated bar was returned
            if completed_bar:
                agg_interval = completed_bar.get("interval")
                if agg_interval and agg_interval in self.target_intervals:
                    logging.info(f"Completed {agg_interval}-minute bar for {self.symbol}: {completed_bar}")
                    # 4. Store the completed aggregated bar
                    self.completed_agg_bars[agg_interval].append(completed_bar)

                    # --- Placeholder for Strategy Analysis ---
                    # Here you would typically pass the completed bars (or recent history)
                    # to your strategy's analyze method.
                    # Example:
                    # recent_bars_for_strategy = self.get_recent_bars(agg_interval) # Method to get required history
                    # signals = self.strategy.analyze({self.symbol: recent_bars_for_strategy})
                    # self.place_orders(signals)
                    # --- End Placeholder ---

                else:
                     logging.warning(f"Aggregator returned a bar with unexpected interval: {completed_bar}")


            # --- Placeholder: Analyze raw 1-min bar if needed by strategy ---
            # signals_raw: List[Signal] = self.strategy.analyze({self.symbol: pl.DataFrame([formatted_bar])})
            # self.place_orders(signals_raw)
            # --- End Placeholder ---


        except Exception as e:
            logging.error(f"Error handling bar update for {self.symbol}: {e}", exc_info=True)


    # --- Placeholder methods (keep or remove based on your needs) ---
    # async def handle_trade_update(self, trade):
    #     """Handles trade updates from the WebSocket stream."""
    #     logging.info(f"Trade Update: {trade}")
    #     # Analyze the data and generate signals
    #     # signals: List[Signal] = self.strategy.analyze(trade)
    #     # Place orders based on signals
    #     # self.place_orders(signals)

    # async def handle_quote_update(self, quote):
    #     """Handles quote updates from the WebSocket stream."""
    #     logging.info(f"Quote Update: {quote}")
    #     # Analyze the data and generate signals
    #     # signals: List[Signal] = self.strategy.analyze(quote)
    #     # Place orders based on signals
    #     # self.place_orders(signals)

    async def handle_trade_update(self, trade):
        """Handles trade updates from the WebSocket stream."""
        print("Trade Update:", trade)
        # Analyze the data and generate signals
        signals: List[Signal] = self.strategy.analyze(trade)
        # Place orders based on signals
        self.place_orders(signals)

    async def handle_quote_update(self, quote):
        """Handles quote updates from the WebSocket stream."""
        print("Quote Update:", quote)
        # Analyze the data and generate signals
        signals: List[Signal] = self.strategy.analyze(quote)
        # Place orders based on signals
        self.place_orders(signals)

    def place_orders(self, signals: List[Signal]):
        """Places orders based on the received signals."""
        for signal in signals:
            if signal.type == "BUY":
                self.order_manager.place_order(signal, self.capital)

    def run(self):
        """
        The main trading loop.
        """
        while True:
            self.log_status()
            # Do other things here
            pass

    def log_status(self):
        """
        Logs the current status of the bot. Replace this with your actual logging mechanism.
        """
        print("Current Capital:", self.capital)
        print("Active Orders:", self.order_manager.active_orders)
        print("---")