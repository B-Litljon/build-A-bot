import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.execution.oanda_order_manager import OandaOrderManager


class TestOandaEntry(unittest.TestCase):
    """Mocked unit tests for OandaOrderManager.submit_target_position()."""

    @patch("oandapyV20.API")
    def test_flat_to_long(self, mock_api_cls):
        """(a) Flat -> long: tradeOpened parsed, state updated."""
        manager = OandaOrderManager(api_key="fake", account_id="123")

        mock_req = MagicMock()
        mock_req.response = {
            "orderFillTransaction": {
                "type": "ORDER_FILL",
                "instrument": "EUR_USD",
                "units": "100",
                "price": "1.08500",
                "tradeOpened": {
                    "tradeID": "100",
                    "units": "100",
                    "price": "1.08500",
                },
            }
        }

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=mock_req
        ):
            result = manager.submit_target_position("EUR/USD", 100)

        self.assertEqual(result["filled"], 100)
        self.assertAlmostEqual(result["avg_price"], 1.08500)
        self.assertEqual(result["closed_units"], 0)
        self.assertEqual(result["opened_units"], 100)
        self.assertEqual(manager.get_net_position("EUR_USD"), 100)
        self.assertAlmostEqual(
            manager.get_average_entry_price("EUR_USD"), 1.08500
        )

    @patch("oandapyV20.API")
    def test_long_to_short_reversal(self, mock_api_cls):
        """(b) Long -> larger short reversal: tradesClosed + tradeOpened parsed."""
        manager = OandaOrderManager(api_key="fake", account_id="123")
        manager._net_positions["EUR_USD"] = 100
        manager._avg_entry_prices["EUR_USD"] = 1.08000

        mock_req = MagicMock()
        mock_req.response = {
            "orderFillTransaction": {
                "type": "ORDER_FILL",
                "instrument": "EUR_USD",
                "units": "-150",
                "price": "1.09000",
                "tradesClosed": [
                    {
                        "tradeID": "100",
                        "units": "-100",
                        "price": "1.08000",
                    }
                ],
                "tradeOpened": {
                    "tradeID": "101",
                    "units": "-50",
                    "price": "1.09000",
                },
            }
        }

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=mock_req
        ):
            result = manager.submit_target_position("EUR/USD", -50)

        self.assertEqual(result["filled"], 150)
        self.assertAlmostEqual(result["avg_price"], 1.09000)
        self.assertEqual(result["closed_units"], 100)
        self.assertEqual(result["opened_units"], 50)
        self.assertEqual(manager.get_net_position("EUR_USD"), -50)
        self.assertAlmostEqual(
            manager.get_average_entry_price("EUR_USD"), 1.09000
        )

    @patch("oandapyV20.API")
    def test_delta_zero_noop(self, mock_api_cls):
        """(c) delta == 0: no API call, state untouched."""
        manager = OandaOrderManager(api_key="fake", account_id="123")
        manager._net_positions["EUR_USD"] = 50

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate"
        ) as mock_order:
            result = manager.submit_target_position("EUR/USD", 50)
            mock_order.assert_not_called()

        self.assertEqual(result["filled"], 0)
        self.assertAlmostEqual(result["avg_price"], 0.0)
        self.assertEqual(result["closed_units"], 0)
        self.assertEqual(result["opened_units"], 0)
        self.assertEqual(manager.get_net_position("EUR_USD"), 50)

    @patch("oandapyV20.API")
    def test_api_error_leaves_state(self, mock_api_cls):
        """(d) API error: log ERROR, DO NOT mutate state."""
        manager = OandaOrderManager(api_key="fake", account_id="123")
        manager._net_positions["EUR_USD"] = 50
        manager._avg_entry_prices["EUR_USD"] = 1.08000

        mock_client = mock_api_cls.return_value
        mock_client.request.side_effect = Exception("Connection refused")

        result = manager.submit_target_position("EUR/USD", 100)

        self.assertEqual(result["filled"], 0)
        self.assertAlmostEqual(result["avg_price"], 0.0)
        self.assertEqual(result["closed_units"], 0)
        self.assertEqual(result["opened_units"], 0)
        self.assertEqual(manager.get_net_position("EUR_USD"), 50)
        self.assertAlmostEqual(
            manager.get_average_entry_price("EUR_USD"), 1.08000
        )

    @patch("oandapyV20.API")
    def test_add_to_existing_position_weighted_avg(self, mock_api_cls):
        """Add to long position: weighted average entry price computed."""
        manager = OandaOrderManager(api_key="fake", account_id="123")
        manager._net_positions["EUR_USD"] = 100
        manager._avg_entry_prices["EUR_USD"] = 1.08000

        mock_req = MagicMock()
        mock_req.response = {
            "orderFillTransaction": {
                "type": "ORDER_FILL",
                "instrument": "EUR_USD",
                "units": "100",
                "price": "1.09000",
                "tradeOpened": {
                    "tradeID": "101",
                    "units": "100",
                    "price": "1.09000",
                },
            }
        }

        with patch(
            "oandapyV20.endpoints.orders.OrderCreate", return_value=mock_req
        ):
            result = manager.submit_target_position("EUR/USD", 200)

        self.assertEqual(result["filled"], 100)
        self.assertAlmostEqual(result["avg_price"], 1.09000)
        self.assertEqual(result["closed_units"], 0)
        self.assertEqual(result["opened_units"], 100)
        self.assertEqual(manager.get_net_position("EUR_USD"), 200)
        # Weighted average: (100*1.08 + 100*1.09) / 200 = 1.085
        self.assertAlmostEqual(
            manager.get_average_entry_price("EUR_USD"), 1.08500
        )


if __name__ == "__main__":
    unittest.main()
