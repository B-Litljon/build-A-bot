from typing import Dict, List
from core.signal import Signal
from alpaca.trading.client import TradingClient
from utils.order_params import OrderParams

class OrderManager:
    """
    Manages order execution and monitoring.

    Attributes:
        trading_client (TradingClient): An instance of a TradingClient.
        order_params (OrderParams): An OrderParams object.
        active_orders (Dict): A dictionary to track active orders.
    """

    def __init__(self, trading_client: TradingClient, order_params: OrderParams):
        self.trading_client = TradingClient
        self.order_params = order_params
        self.active_orders: Dict = {}  # {order_id: order_details} 

    def place_order(self, signal: Signal, current_capital: float) -> str:
        """
        Places an order based on a signal.

        Args:
            signal (Signal): The trading signal.
            current_capital (float): current capital of the account
        Returns:
            str: The order ID if the order was placed successfully, None otherwise.
        """
        if signal.type == "BUY":
            quantity = self.calculate_quantity(signal.entry_price, current_capital)
            # Calculate stop-loss and take-profit
            stop_loss = signal.entry_price * self.order_params.sl_multiplier
            take_profit = signal.entry_price * self.order_params.tp_multiplier

            # Place the order through the TradingClient
            order_id = self.trading_client.submit_order(
                symbol=signal.symbol,
                qty=quantity,
                side="buy",
                type="market",  # or "limit"
                time_in_force="gtc",
                limit_price=signal.entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            if order_id:
                self.active_orders[order_id] = {
                    "symbol": signal.symbol,
                    "entry_price": signal.entry_price,
                    "quantity": quantity,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": "current_time",  # You'll want to track the actual time
                }
            return order_id
        return None
    
    def calculate_quantity(self, entry_price: float, current_capital: float) -> float:
        """
        Calculates the quantity to buy based on risk percentage and current capital.
        """
        risk_amount = current_capital * self.order_params.risk_percentage
        quantity = risk_amount / entry_price  # This is a simple calculation
        return quantity

    def monitor_orders(self, market_data):
        """
        Monitors active orders and implements stop-loss/take-profit/trailing stop logic.
        """
        for order_id, order_details in list(self.active_orders.items()):  # Iterate over a copy to allow modification
            symbol = order_details["symbol"]
            if symbol not in market_data:
                continue

            current_price = market_data[symbol]["close"] # Assuming market data has a "close" price
            
            # Check for stop-loss or take-profit trigger
            if current_price <= order_details["stop_loss"]:
                self.trading_client.submit_order(
                    symbol=symbol,
                    qty=order_details["quantity"],
                    side="sell",  # Closing the position
                    type="market",
                    time_in_force="gtc",
                )
                print(f"Stop-loss triggered for {symbol} at {current_price}")
                del self.active_orders[order_id]

            elif current_price >= order_details["take_profit"]:
                self.trading_client.submit_order(
                    symbol=symbol,
                    qty=order_details["quantity"],
                    side="sell",  # Closing the position
                    type="market",
                    time_in_force="gtc",
                )                
                print(f"Take-profit triggered for {symbol} at {current_price}")
                del self.active_orders[order_id]

            elif self.order_params.use_trailing_stop:
                # Trailing stop logic (simplified)
                if (
                    self.order_params.sma_short_period
                    and self.order_params.sma_long_period
                    and self.order_params.sma_crossover_type
                ):
                    short_sma = self.calculate_sma(
                        market_data[symbol], self.order_params.sma_short_period
                    )
                    long_sma = self.calculate_sma(
                        market_data[symbol], self.order_params.sma_long_period
                    )

                    if self.order_params.sma_crossover_type == "long":
                        if short_sma > long_sma:
                            new_stop_loss = current_price * self.order_params.sl_multiplier
                            self.active_orders[order_id]["stop_loss"] = max(
                                new_stop_loss, order_details["stop_loss"]
                            )
                    elif self.order_params.sma_crossover_type == "short":
                        if short_sma < long_sma:
                            new_stop_loss = current_price * self.order_params.sl_multiplier
                            self.active_orders[order_id]["stop_loss"] = max(
                                new_stop_loss, order_details["stop_loss"]
                            )

    def calculate_sma(self, data, period):
        """Calculates the Simple Moving Average (replace with your preferred method)."""
        # You might need to adjust this based on your market_data format
        closing_prices = [data["close"]]  # Assuming you have a list of closing prices
        if len(closing_prices) >= period:
            return sum(closing_prices[-period:]) / period
        else:
            return None