from typing import List
from core.signal import Signal
from strategies.strategy import Strategy
from alpaca.trading.client import TradingClient
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

    def __init__(self, strategy: Strategy, capital: float, trading_client: TradingClient):
        self.strategy = strategy
        self.capital = capital
        self.trading_client = trading_client
        self.order_manager = OrderManager(trading_client, strategy.get_order_params())

    def run(self):
        """
        The main trading loop.
        """
        while True:
            # 1. Fetch Market Data (replace with your DataHandler)
            market_data = self.fetch_market_data() # get data from alpaca

            # 2. Analyze Data and Generate Signals
            signals: List[Signal] = self.strategy.analyze(market_data)

            # 3. Place Orders
            for signal in signals:
                if signal.type == "BUY":
                    self.order_manager.place_order(signal, self.capital)

            # 4. Monitor Orders
            self.order_manager.monitor_orders(market_data)

            # 5. Logging/Tracking (replace with your logging mechanism)
            self.log_status()

            # 6. Sleep (adjust the sleep duration as needed)
            # time.sleep(60)  # Check every 60 seconds, for example

    def fetch_market_data(self):
        """
        Fetches market data. Replace this with your actual data fetching logic.
        """
        # You'll likely use a DataHandler component here
        # For now, let's simulate some data
        data = {
            "AAPL": {"close": 155.0},  # Example: Replace with actual data
            "MSFT": {"close": 280.0}
        }
        return data

    def log_status(self):
        """
        Logs the current status of the bot. Replace this with your actual logging mechanism.
        """
        print("Current Capital:", self.capital)
        print("Active Orders:", self.order_manager.active_orders)
        print("---")