from typing import Dict, List, Optional
from core.signal import Signal
from alpaca.trading.client import TradingClient


class OrderParams: # V
    """
    Defines parameters for order calculation and risk management.

    Attributes:
        risk_percentage (float): Percentage of capital to risk per trade.
        tp_multiplier (float): Multiplier to calculate take-profit level from entry price.
        sl_multiplier (float): Multiplier to calculate stop-loss level from entry price.
        sma_short_period (int, optional): Period for short-term SMA (trailing stop).
        sma_long_period (int, optional): Period for long-term SMA (trailing stop).
        sma_crossover_type (str, optional): "long" or "short" for trailing stop type.
        use_trailing_stop (bool, optional): Whether to use trailing stop-loss. Defaults to False.
        **kwargs: For adding other custom parameters.
    """
    def __init__(self, risk_percentage: float, tp_multiplier: float, sl_multiplier: float,
                 sma_short_period: int = None, sma_long_period: int = None,
                 sma_crossover_type: str = None, use_trailing_stop: bool = False, **kwargs):
        self.risk_percentage = risk_percentage
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.sma_short_period = sma_short_period
        self.sma_long_period = sma_long_period
        self.sma_crossover_type = sma_crossover_type
        self.use_trailing_stop = use_trailing_stop
        self.kwargs = kwargs

    def __str__(self):
        return f"OrderParams(risk_percentage={self.risk_percentage}, tp_multiplier={self.tp_multiplier}, sl_multiplier={self.sl_multiplier}, use_trailing_stop={self.use_trailing_stop}, ...)"
    
class OrderCalculator:
    def __init__(self, order_params: OrderParams):
        self.order_params = order_params

    def calculate_quantity(self, entry_price: float, current_capital: float) -> float:
        """Calculates quantity, handling potential division by zero."""
        if entry_price == 0:
            raise ValueError("Entry price cannot be zero.")  # Raise an exception
        risk_amount = current_capital * self.order_params.risk_percentage
        return risk_amount / entry_price

    def calculate_stop_loss(self, entry_price: float) -> float:
        """Calculates stop-loss level."""
        return entry_price * self.order_params.sl_multiplier

    def calculate_take_profit(self, entry_price: float) -> float:
        """Calculates take-profit level."""
        return entry_price * self.order_params.tp_multiplier


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