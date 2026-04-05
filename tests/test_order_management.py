import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime
import sys
import os

# Ensure src is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.order_management import OrderManager, OrderParams
from core.signal import Signal

class TestOrderManager(unittest.TestCase):
    def setUp(self):
        self.mock_trading_client = MagicMock()
        self.order_params = OrderParams(
            risk_percentage=0.01,
            tp_multiplier=1.05,
            sl_multiplier=0.95
        )
        self.order_manager = OrderManager(
            trading_client=self.mock_trading_client,
            order_params=self.order_params
        )
        self.signal = Signal(
            symbol="AAPL",
            type="BUY",
            price=150.0,
            confidence=0.9,
            timestamp=datetime.now()
        )
        self.current_capital = 10000.0

    @patch('core.order_management.MarketOrderRequest')
    @patch('core.order_management.TimeInForce')
    @patch('core.order_management.OrderSide')
    def test_place_order_success(self, mock_order_side, mock_tif, mock_mor):
        # Mock successful order submission
        mock_order = MagicMock()
        mock_order.id = "test_order_id"
        self.mock_trading_client.submit_order.return_value = mock_order

        order_id = self.order_manager.place_order(self.signal, self.current_capital)

        self.assertEqual(order_id, "test_order_id")
        self.assertIn("test_order_id", self.order_manager.active_orders)
        self.assertEqual(self.order_manager.active_orders["test_order_id"]["symbol"], "AAPL")
        self.mock_trading_client.submit_order.assert_called_once()

    @patch('core.order_management.MarketOrderRequest')
    @patch('core.order_management.TimeInForce')
    @patch('core.order_management.OrderSide')
    def test_place_order_exception_handling(self, mock_order_side, mock_tif, mock_mor):
        # Mock exception during order submission
        self.mock_trading_client.submit_order.side_effect = Exception("API Error")

        # Suppress logging during test
        with self.assertLogs('root', level='ERROR') as cm:
            order_id = self.order_manager.place_order(self.signal, self.current_capital)

        self.assertIsNone(order_id)
        self.assertEqual(len(self.order_manager.active_orders), 0)
        self.assertTrue(any("Order Placement Failed: API Error" in output for output in cm.output))

if __name__ == '__main__':
    unittest.main()
