import unittest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from execution.risk_manager import RiskManager, RiskProfile

class TestRiskManagerForex(unittest.TestCase):
    def test_equity_default_floor(self):
        """Verify fallback to min_sl_pct when no symbol is provided (equity mode)."""
        profile = RiskProfile(min_sl_pct=0.0015, sl_atr_multiplier=0.5)
        rm = RiskManager(profile)
        
        # entry=100.0, atr=0.2 -> sl_dist=0.10. Floor=100.0 * 0.0015 = 0.15.
        # sl_dist (0.10) < floor (0.15) -> Should reject (None)
        self.assertIsNone(rm.calculate_bracket(100.0, 0.2))
        
        # entry=100.0, atr=0.4 -> sl_dist=0.20 > floor (0.15) -> Should pass
        bracket = rm.calculate_bracket(100.0, 0.4)
        self.assertIsNotNone(bracket)
        self.assertEqual(bracket[0], 0.20)

    def test_forex_non_jpy_floor(self):
        """Verify pip floor for non-JPY forex pairs (pip_size = 0.0001)."""
        profile = RiskProfile(min_sl_pips=2.0, sl_atr_multiplier=0.5, round_precision=5)
        rm = RiskManager(profile)
        
        # Symbol is EUR_USD (forex) -> pip_size = 0.0001. Floor = 2.0 * 0.0001 = 0.0002.
        # entry_price=1.08000, atr=0.0003 -> sl_dist = 0.5 * 0.0003 = 0.00015.
        # sl_dist (0.00015) < floor (0.00020) -> Should reject (None)
        self.assertIsNone(rm.calculate_bracket(1.08000, 0.0003, symbol="EUR_USD"))
        
        # atr=0.0005 -> sl_dist = 0.00025 >= floor (0.00020) -> Should pass
        bracket = rm.calculate_bracket(1.08000, 0.0005, symbol="EUR_USD")
        self.assertIsNotNone(bracket)
        self.assertEqual(bracket[0], 0.00025)

    def test_forex_jpy_floor(self):
        """Verify pip floor for JPY forex pairs (pip_size = 0.01)."""
        profile = RiskProfile(min_sl_pips=2.0, sl_atr_multiplier=0.5, round_precision=3)
        rm = RiskManager(profile)
        
        # Symbol is USD_JPY (forex) -> pip_size = 0.01. Floor = 2.0 * 0.01 = 0.02.
        # entry_price=155.00, atr=0.03 -> sl_dist = 0.5 * 0.03 = 0.015.
        # sl_dist (0.015) < floor (0.02) -> Should reject (None)
        self.assertIsNone(rm.calculate_bracket(155.00, 0.03, symbol="USD_JPY"))
        
        # atr=0.05 -> sl_dist = 0.025 >= floor (0.02) -> Should pass
        bracket = rm.calculate_bracket(155.00, 0.05, symbol="USD_JPY")
        self.assertIsNotNone(bracket)
        self.assertEqual(bracket[0], 0.025)

    def test_metals_percent_floor(self):
        """XAU/XAG use a percent-of-price floor, not the meaningless pip floor."""
        profile = RiskProfile(
            min_sl_pips=2.0,
            min_sl_pct_metals=0.0001,
            sl_atr_multiplier=1.0,
            round_precision=5,
        )
        rm = RiskManager(profile)

        # XAU_USD at 2700: floor = 2700 * 0.0001 = 0.27.
        # atr=0.10 -> sl_dist = 0.10 < 0.27 -> reject. (The old pip floor of
        # 0.0002 would have passed this — the filter was a no-op on metals.)
        self.assertIsNone(rm.calculate_bracket(2700.0, 0.10, symbol="XAU_USD"))

        # atr=1.50 -> sl_dist = 1.50 >= 0.27 -> pass
        bracket = rm.calculate_bracket(2700.0, 1.50, symbol="XAU_USD")
        self.assertIsNotNone(bracket)
        self.assertEqual(bracket[0], 1.50)

        # Silver too: XAG_USD at 31.0, floor = 0.0031.
        self.assertIsNone(rm.calculate_bracket(31.0, 0.001, symbol="XAG_USD"))
        self.assertIsNotNone(rm.calculate_bracket(31.0, 0.05, symbol="XAG_USD"))

    def test_metal_detection_does_not_catch_fiat(self):
        """Fiat pairs still use the pip floor (XAU prefix only)."""
        profile = RiskProfile(
            min_sl_pips=2.0, min_sl_pct_metals=0.0001,
            sl_atr_multiplier=0.5, round_precision=5,
        )
        rm = RiskManager(profile)
        # EUR_USD must behave exactly as in test_forex_non_jpy_floor.
        self.assertIsNone(rm.calculate_bracket(1.08000, 0.0003, symbol="EUR_USD"))
        self.assertIsNotNone(rm.calculate_bracket(1.08000, 0.0005, symbol="EUR_USD"))

if __name__ == "__main__":
    unittest.main()
