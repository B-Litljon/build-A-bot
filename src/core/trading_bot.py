from typing import Dict, List, str
from core.signal import Signal
from strategies.strategy import Strategy
from alpaca.trading.client import TradingClient
from alpaca.data import StockDataStream
from core.order_management import OrderManager

class TradingBot:
    """
    The main trading bot class.

    Attributes:
        strategy (Strategy): The trading strategy to use.
        capital (float): The initial trading capital.
        trading_client (TradingClient): The trading client for interacting with the exchange.
        order_manager (OrderManager): order manager created from strategy and trading client
    """
    def __init__(self, api_key: str, api_secret: str,  strategy: Strategy, capital: float, 
                 trading_client: TradingClient, live_stock_data: StockDataStream, symbol: str):
        self.strategy = strategy
        self.capital = capital
        self.trading_client = trading_client
        self.order_manager = OrderManager(trading_client, strategy.get_order_params())
        self.live_stock_data = live_stock_data
        self.symbol = symbol  # Store the specified symbol

        # Subscribe to bar updates for the specified symbol
        self.live_stock_data.subscribe_bars(self.handle_bar_update, self.symbol)

    async def handle_bar_update(self, bar): # need to aggregate bars to various timeframes, alpaca only has 1m and 1d
        """Handles bar updates from the WebSocket stream."""
        print("Bar Update:, {bar}")
        # Analyze the data and generate signals
        signals: List[Signal] = self.strategy.analyze(bar)
        # Place orders based on signals
        self.place_orders(signals)

    async def handle_trade_update(self, trade):
        """Handles trade updates from the WebSocket stream."""
        print("Trade Update:, {trade}")
        # Analyze the data and generate signals
        signals: List[Signal] = self.strategy.analyze(trade)
        # Place orders based on signals
        self.place_orders(signals)

    async def handle_quote_update(self, quote):
        """Handles quote updates from the WebSocket stream."""
        print("Quote Update:, {quote}")
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