from typing import Dict, Optional
import logging
from core.signal import Signal
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


class OrderParams:
    """
    Defines parameters for order calculation and risk management.
    """

    def __init__(
        self,
        risk_percentage: float,
        tp_multiplier: float,
        sl_multiplier: float,
        use_trailing_stop: bool = False,
        **kwargs,
    ):
        self.risk_percentage = risk_percentage
        self.tp_multiplier = tp_multiplier
        self.sl_multiplier = sl_multiplier
        self.use_trailing_stop = use_trailing_stop
        self.kwargs = kwargs


class OrderCalculator:
    def __init__(self, order_params: OrderParams):
        self.order_params = order_params

    def calculate_quantity(self, entry_price: float, current_capital: float) -> float:
        if entry_price == 0:
            return 0.0
        risk_amount = current_capital * self.order_params.risk_percentage
        return float(risk_amount / entry_price)

    def calculate_stop_loss(self, entry_price: float) -> float:
        return entry_price * self.order_params.sl_multiplier

    def calculate_take_profit(self, entry_price: float) -> float:
        return entry_price * self.order_params.tp_multiplier


class OrderManager:
    def __init__(self, trading_client: TradingClient, order_params: OrderParams, notification_manager=None):
        self.trading_client = trading_client
        self.order_params = order_params
        self.notification_manager = notification_manager
        self.active_orders: Dict[str, Dict] = {}
        self.order_calculator = OrderCalculator(self.order_params)

    def place_order(self, signal: Signal, current_capital: float) -> Optional[str]:
        if signal.type == "BUY":
            try:
                qty = self.order_calculator.calculate_quantity(signal.price, current_capital)
                if qty <= 0:
                    return None

                stop_loss = self.order_calculator.calculate_stop_loss(signal.price)
                take_profit = self.order_calculator.calculate_take_profit(signal.price)

                logging.info(
                    f"Placing BUY for {signal.symbol}: Qty={qty:.4f}, SL={stop_loss:.2f}, TP={take_profit:.2f}"
                )

                req = MarketOrderRequest(
                    symbol=signal.symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                )

                order = self.trading_client.submit_order(req)
                order_id = getattr(order, "id", None)

                if order_id:
                    self.active_orders[str(order_id)] = {
                        "symbol": signal.symbol,
                        "entry_price": signal.price,
                        "quantity": qty,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                    }
                    logging.info(f"Order {order_id} placed.")
                    if self.notification_manager:
                        self.notification_manager.notify_trade("BUY", signal.symbol, signal.price, qty, "Signal Triggered")
                    return str(order_id)

            except Exception as e:
                logging.error(f"Order Placement Failed: {e}", exc_info=True)
        return None

    def monitor_orders(self, market_data: Dict[str, float]):
        """
        Checks active orders against current market price.
        Args:
            market_data: Dict { "AAPL": 150.23, "TSLA": 200.50 }
        """
        for order_id, details in list(self.active_orders.items()):
            symbol = details["symbol"]
            if symbol not in market_data:
                continue

            current_price = market_data[symbol]
            action = None
            reason = ""

            if current_price <= details["stop_loss"]:
                action = OrderSide.SELL
                reason = f"Stop Loss ({current_price} <= {details['stop_loss']})"
            elif current_price >= details["take_profit"]:
                action = OrderSide.SELL
                reason = f"Take Profit ({current_price} >= {details['take_profit']})"

            if action:
                logging.info(f"Triggering Exit for {symbol}: {reason}")
                try:
                    req = MarketOrderRequest(
                        symbol=symbol,
                        qty=details["quantity"],
                        side=action,
                        time_in_force=TimeInForce.GTC,
                    )
                    self.trading_client.submit_order(req)
                    if self.notification_manager:
                        self.notification_manager.notify_trade("SELL", symbol, current_price, details["quantity"], reason)
                    del self.active_orders[order_id]
                except Exception as e:
                    logging.error(f"Failed to exit {symbol}: {e}")

    def sync_positions(self):
        """
        Reconciles memory with actual Alpaca positions on startup.
        """
        try:
            positions = self.trading_client.get_all_positions()
            for pos in positions:
                # Check if we are already managing this symbol
                is_managed = False
                for details in self.active_orders.values():
                    if details["symbol"] == pos.symbol:
                        is_managed = True
                        break

                if not is_managed:
                    logging.warning(f"⚠️ Found unmanaged position for {pos.symbol} (Qty: {pos.qty}). Adopting it.")

                    # Reconstruct thresholds based on Average Entry Price
                    avg_entry = float(pos.avg_entry_price)
                    sl = self.order_calculator.calculate_stop_loss(avg_entry)
                    tp = self.order_calculator.calculate_take_profit(avg_entry)

                    # Create a synthetic Order ID (prefix 'sync_')
                    synthetic_id = f"sync_{pos.symbol}_{pos.asset_id}"

                    self.active_orders[synthetic_id] = {
                        "symbol": pos.symbol,
                        "entry_price": avg_entry,
                        "quantity": float(pos.qty),
                        "stop_loss": sl,
                        "take_profit": tp
                    }
                    logging.info(f"✅ Adopted {pos.symbol}: SL={sl:.2f}, TP={tp:.2f}")

        except Exception as e:
            logging.error(f"Failed to sync positions: {e}")
