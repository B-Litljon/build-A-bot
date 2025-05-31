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
    # can modify the calculate_quantity method to account for fees
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
    def __init__(self, trading_client: TradingClient, order_params: OrderParams):
        self.trading_client = trading_client
        self.order_params = order_params
        self.active_orders: Dict[str, Dict] = {}  # {order_id: order_details}
        self.order_calculator = OrderCalculator(self.order_params)  # Initialize calculator

    def place_order(self, signal: Signal, current_capital: float) -> Optional[str]:
        """Places an order, using OrderCalculator and handling errors."""
        if signal.type == "BUY":
            try:
                quantity = self.order_calculator.calculate_quantity(signal.entry_price, current_capital)
                stop_loss = self.order_calculator.calculate_stop_loss(signal.entry_price)
                take_profit = self.order_calculator.calculate_take_profit(signal.entry_price)

                order_id = self.trading_client.submit_order(
                    symbol=signal.symbol,
                    qty=quantity,
                    side="buy",
                    type="market",
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
                        "entry_time": "current_time",  # Replace with actual time tracking
                    }
                    return order_id
                else:
                    print("Order submission failed.")
                    return None

            except ValueError as e:
                print(f"Error calculating order: {e}")
                return None
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None
        return None

    def monitor_orders(self, market_data: Dict[str, Dict]):  # Type hint market_data
        """Monitors active orders and implements stop-loss/take-profit/trailing stop."""
        for order_id, order_details in list(self.active_orders.items()):
            symbol = order_details["symbol"]
            if symbol not in market_data:
                continue

            current_price = market_data[symbol]["close"]

            if current_price <= order_details["stop_loss"]:
                self.trading_client.submit_order(
                    symbol=symbol,
                    qty=order_details["quantity"],
                    side="sell",
                    type="market",
                    time_in_force="gtc",
                )
                print(f"Stop-loss triggered for {symbol} at {current_price}")
                del self.active_orders[order_id]

            elif current_price >= order_details["take_profit"]:
                self.trading_client.submit_order(
                    symbol=symbol,
                    qty=order_details["quantity"],
                    side="sell",
                    type="market",
                    time_in_force="gtc",
                )
                print(f"Take-profit triggered for {symbol} at {current_price}")
                del self.active_orders[order_id]

            elif self.order_params.use_trailing_stop:
                if (
                    self.order_params.sma_short_period
                    and self.order_params.sma_long_period
                    and self.order_params.sma_crossover_type
                ):
                    short_sma = self.calculate_sma(market_data[symbol], self.order_params.sma_short_period)
                    long_sma = self.calculate_sma(market_data[symbol], self.order_params.sma_long_period)

                    if short_sma is None or long_sma is None: # Check for None values
                        continue # Skip if SMA calculation failed

                    if self.order_params.sma_crossover_type == "long" and short_sma > long_sma:
                        new_stop_loss = current_price * self.order_params.sl_multiplier
                        self.active_orders[order_id]["stop_loss"] = max(new_stop_loss, order_details["stop_loss"])
                    elif self.order_params.sma_crossover_type == "short" and short_sma < long_sma:
                        new_stop_loss = current_price * self.order_params.sl_multiplier
                        self.active_orders[order_id]["stop_loss"] = max(new_stop_loss, order_details["stop_loss"])

    def calculate_sma(self, data: Dict, period: int) -> Optional[float]: # Type hints and optional return
        """Calculates the Simple Moving Average."""
        closing_prices = [data["close"]]  # Assuming you have a list of closing prices
        if len(closing_prices) >= period:
            return sum(closing_prices[-period:]) / period
        else:
            return None  # Return None if not enough data