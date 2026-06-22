import unittest
import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from execution.risk_manager import (
    COUPLING_LOOSEN,
    COUPLING_TIGHTEN,
    GATE_REGIME,
    GATE_SPREAD,
    RiskManager,
    RiskProfile,
    coupled_keff,
)


def _forex_profile(**overrides):
    """A forex-shaped profile for the dynamic-gate tests."""
    base = dict(
        sl_atr_multiplier=1.0,
        tp_atr_multiplier=2.0,
        spread_k_base=1.5,
        spread_k_coupling=0.0,
        spread_k_coupling_mode=COUPLING_TIGHTEN,
        regime_pctile=20.0,
        regime_window=260,
        regime_min_samples=60,
        spread_atr_alpha=0.15,
        round_precision=5,
    )
    base.update(overrides)
    return RiskProfile(**base)

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

class TestCoupledKeff(unittest.TestCase):
    """The shared k_eff kernel used by both live execution and the retrainer."""

    def test_no_coupling_below_median_is_base(self):
        # scale = 0 for rank <= 0.5 → k_eff == base (clipped to >= 1.0).
        self.assertAlmostEqual(
            float(coupled_keff(1.5, 0.5, COUPLING_TIGHTEN, 0.3)), 1.5
        )
        self.assertAlmostEqual(
            float(coupled_keff(1.5, 0.5, COUPLING_LOOSEN, 0.5)), 1.5
        )

    def test_tighten_raises_with_vol(self):
        # rank 1.0 → scale 1.0 → base * (1 + coupling).
        self.assertAlmostEqual(
            float(coupled_keff(1.5, 0.5, COUPLING_TIGHTEN, 1.0)), 2.25
        )

    def test_loosen_lowers_with_vol(self):
        self.assertAlmostEqual(
            float(coupled_keff(1.5, 0.2, COUPLING_LOOSEN, 1.0)), 1.2
        )

    def test_clipping_floor_at_one(self):
        # Aggressive loosen would push k_eff to 0.0 — must clip to 1.0 so the
        # spread can never exceed the stop distance.
        self.assertAlmostEqual(
            float(coupled_keff(1.5, 1.0, COUPLING_LOOSEN, 1.0)), 1.0
        )


class TestDynamicHybridFloor(unittest.TestCase):
    def test_regime_gate_vetoes_low_vol(self):
        """Current vol in the bottom P% of its window → Gate B veto."""
        rm = RiskManager(_forex_profile())
        series = [1.0] * 99 + [0.05]  # current is the window minimum
        res = rm.calculate_bracket(
            150.0, 0.5, symbol="GBP_JPY",
            spread=0.001, spread_fresh=True, regime_series=series,
        )
        self.assertIsNone(res)
        self.assertEqual(rm.last_veto_gate, GATE_REGIME)

    def test_cost_gate_vetoes_tight_stop_live_spread(self):
        """sl_dist below k_eff·spread → Gate A veto; comfortable stop passes."""
        rm = RiskManager(_forex_profile())
        series = [1.0] * 100  # rank 1.0, no regime veto; coupling 0 → k_eff 1.5
        # floor = 1.5 * 0.001 = 0.0015
        self.assertIsNone(
            rm.calculate_bracket(
                150.0, 0.001, symbol="GBP_JPY",
                spread=0.001, spread_fresh=True, regime_series=series,
            )
        )
        self.assertEqual(rm.last_veto_gate, GATE_SPREAD)
        self.assertIsNotNone(
            rm.calculate_bracket(
                150.0, 0.01, symbol="GBP_JPY",
                spread=0.001, spread_fresh=True, regime_series=series,
            )
        )

    def test_cost_gate_proxy_fallback_when_spread_stale(self):
        """No fresh spread → volatility-scaled proxy (alpha·baseline)."""
        rm = RiskManager(_forex_profile())
        series = [2.0] * 100  # baseline median 2.0% → proxy=0.15*2*150/100=0.45
        # floor = k_eff(1.5) * 0.45 = 0.675
        self.assertIsNone(
            rm.calculate_bracket(
                150.0, 0.5, symbol="GBP_JPY",
                spread=None, spread_fresh=False, regime_series=series,
            )
        )
        self.assertEqual(rm.last_veto_gate, GATE_SPREAD)
        self.assertIsNotNone(
            rm.calculate_bracket(
                150.0, 1.0, symbol="GBP_JPY",
                spread=None, spread_fresh=False, regime_series=series,
            )
        )

    def test_cold_start_bypasses_regime_gate(self):
        """Series shorter than min_samples → Gate B neutral (no veto)."""
        rm = RiskManager(_forex_profile())
        series = [0.01] * 10  # very low vol, but only 10 < 60 samples
        res = rm.calculate_bracket(
            150.0, 0.5, symbol="GBP_JPY",
            spread=None, spread_fresh=False, regime_series=series,
        )
        self.assertIsNotNone(res)
        self.assertEqual(rm.last_veto_gate, "none")

    def test_extreme_rollover_spread_vetoes_safely(self):
        """A blown-out spread vetoes via Gate A without crashing."""
        rm = RiskManager(_forex_profile())
        series = [1.0] * 100
        res = rm.calculate_bracket(
            150.0, 0.5, symbol="GBP_JPY",
            spread=100.0, spread_fresh=True, regime_series=series,
        )
        self.assertIsNone(res)
        self.assertEqual(rm.last_veto_gate, GATE_SPREAD)

    def test_tighten_vs_loosen_diverge_at_high_vol(self):
        """At high vol a borderline stop is vetoed under tighten, passes under loosen."""
        series = [1.0] * 100  # current rank 1.0 → scale 1.0
        spread = 0.1
        # tighten: k_eff = 1.5*(1+0.5)=2.25 → floor 0.225
        # loosen:  k_eff = 1.5*(1-0.5)=0.75 → clipped to 1.0 → floor 0.1
        sl = 0.15  # between the two floors
        rm_t = RiskManager(_forex_profile(spread_k_coupling=0.5, spread_k_coupling_mode=COUPLING_TIGHTEN))
        rm_l = RiskManager(_forex_profile(spread_k_coupling=0.5, spread_k_coupling_mode=COUPLING_LOOSEN))
        self.assertIsNone(
            rm_t.calculate_bracket(150.0, sl, symbol="GBP_JPY",
                                   spread=spread, spread_fresh=True, regime_series=series)
        )
        self.assertIsNotNone(
            rm_l.calculate_bracket(150.0, sl, symbol="GBP_JPY",
                                   spread=spread, spread_fresh=True, regime_series=series)
        )

    def test_no_regime_series_uses_static_floor(self):
        """Backward compat: without a regime series the legacy floor applies."""
        rm = RiskManager(_forex_profile(min_sl_pips=2.0))
        # JPY pip floor = 2.0 * 0.01 = 0.02; sl_dist 0.5*0.03=... uses mult 1.0 → 0.015 < 0.02
        self.assertIsNone(rm.calculate_bracket(155.0, 0.015, symbol="USD_JPY"))
        self.assertIsNotNone(rm.calculate_bracket(155.0, 0.05, symbol="USD_JPY"))


if __name__ == "__main__":
    unittest.main()
