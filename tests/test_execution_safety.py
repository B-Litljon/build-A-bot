import unittest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root)) # for scripts

from scripts.portfolio_orchestrator import execute_rebalance
from src.execution.oanda_order_manager import OandaOrderManager

class TestExecutionSafety(unittest.TestCase):

    def test_alpaca_rebalance_gate(self):
        """Fix 1.5: Verify only UNIVERSE symbols are liquidated."""
        mock_trading = MagicMock()
        mock_data = MagicMock()
        
        # Mock account
        mock_account = MagicMock()
        mock_account.equity = 10000
        mock_account.buying_power = 20000
        mock_account.cash = 5000
        mock_account.status = "ACTIVE"
        mock_trading.get_account.return_value = mock_account
        
        # Mock positions: 
        # AAPL (in UNIVERSE, in top_k) -> keep
        # MSFT (in UNIVERSE, NOT in top_k) -> liquidate
        # TSLA (NOT in UNIVERSE, NOT in top_k) -> PROTECT (don't liquidate)
        p1 = MagicMock(symbol="AAPL", qty=10, market_value=1500)
        p2 = MagicMock(symbol="MSFT", qty=20, market_value=6000)
        p3 = MagicMock(symbol="TSLA", qty=5, market_value=1000)
        mock_trading.get_all_positions.return_value = [p1, p2, p3]
        
        # Mock UNIVERSE in the script
        with patch("scripts.portfolio_orchestrator.UNIVERSE", ["AAPL", "MSFT", "NVDA"]):
            with patch("scripts.portfolio_orchestrator.TOP_K", 1):
                execute_rebalance(
                    trading=mock_trading,
                    data_client=mock_data,
                    top_k=["AAPL"],
                    dry_run=False
                )
        
        # Should have called close_position for MSFT, but NOT for TSLA
        mock_trading.close_position.assert_any_call("MSFT")
        
        # Check that TSLA was NOT closed
        for call in mock_trading.close_position.call_args_list:
            self.assertNotEqual(call[0][0], "TSLA", "TSLA should be protected (not in UNIVERSE)")

    @patch("oandapyV20.API")
    def test_oanda_close_position_fill_parsing(self, mock_api):
        """Fix 1.7: Verify units_filled parsing in close_position."""
        manager = OandaOrderManager(api_key="fake", account_id="123")
        
        # Setup initial state
        manager._net_positions["EUR_USD"] = 100
        manager._avg_entry_prices["EUR_USD"] = 1.10
        
        # Mock response for PositionClose
        mock_response = {
            "longOrderFillTransaction": {"units": "-100"},
            "shortOrderFillTransaction": None
        }
        
        mock_req = MagicMock()
        mock_req.response = mock_response
        
        with patch("oandapyV20.endpoints.positions.PositionClose", return_value=mock_req):
            success = manager.close_position("EUR_USD")
            
        self.assertTrue(success)
        self.assertEqual(manager.get_net_position("EUR_USD"), 0)
        self.assertEqual(manager.get_average_entry_price("EUR_USD"), 0.0)

    @patch("oandapyV20.API")
    def test_oanda_partial_fill_behavior(self, mock_api):
        """Fix 1.7: Verify partial fill doesn't blindly zero state."""
        manager = OandaOrderManager(api_key="fake", account_id="123")
        
        # Setup initial state
        manager._net_positions["EUR_USD"] = 100
        manager._avg_entry_prices["EUR_USD"] = 1.10
        
        # Mock response for partial fill (e.g. only 40 closed)
        mock_response = {
            "longOrderFillTransaction": {"units": "-40"},
            "shortOrderFillTransaction": None
        }
        
        mock_req = MagicMock()
        mock_req.response = mock_response
        
        with patch("oandapyV20.endpoints.positions.PositionClose", return_value=mock_req):
            success = manager.close_position("EUR_USD")
            
        self.assertTrue(success)
        self.assertEqual(manager.get_net_position("EUR_USD"), 60)
        self.assertEqual(manager.get_average_entry_price("EUR_USD"), 1.10)

if __name__ == "__main__":
    unittest.main()
