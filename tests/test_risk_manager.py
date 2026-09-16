import unittest
import sys
from datetime import datetime, time, timezone
from pathlib import Path

"""
Tests for RiskManager -- the bracket floors and the three chop gates.

The most safety-critical test file in the suite: these pin the rules that decide
whether a trade is allowed at all. A regression here does not raise an error, it
silently starts taking trades the system was built to refuse.

Glossary:
    TestRiskManagerForex -- the static floors, one per instrument family.
        Verifies the minimum stop distance is measured in the right unit:
        percent for equities, pips for currency pairs, and percent again for
        metals (a 0.0001 pip on gold near 2700 would never fire).
    test_metal_detection_does_not_catch_fiat -- guards the string matching that
        separates XAU/XAG from ordinary six-letter currency pairs.
    TestCoupledKeff -- the volatility-coupled cost multiplier. Pins that it does
        nothing below median volatility, that "tighten" and "loosen" move in
        opposite directions above it, and that the result is clipped at 1.0 so
        a passing trade's cost can never exceed its own stop distance.
    TestDynamicHybridFloor -- the live gate combination.
    test_cost_gate_proxy_fallback_when_spread_stale -- when the live spread is
        too old to trust, the volatility-scaled estimate must be used instead.
    test_cold_start_bypasses_regime_gate -- with too little history the regime
        gate must stand down rather than veto everything on a half-filled
        buffer; a just-restarted bot must not be frozen.
    test_extreme_rollover_spread_vetoes_safely -- the rollover blowout case
        that Gate C exists for.
    test_no_regime_series_uses_static_floor -- without volatility context the
        manager falls back to the legacy floor rather than failing.
    TestProductionForexProfile -- pins the SHIPPED forex numbers (2.0/4.0
        bracket, k=3.0 cost gate). Everything else in this file builds its own
        profile to test mechanism, which meant no test noticed when the live
        values changed. These are the numbers the model is trained against;
        if they move, retrainer.get_asset_config feeds different labels to the
        Devil and train/serve skew follows silently.
    TestBarrierGeometry -- the learned-geometry substitution. A barrier payload
        must REPLACE the profile multipliers (not compound with them), the
        gates must be asked about the substituted distance rather than the
        static one, and any unusable payload must degrade to the static bracket
        instead of raising on the entry path.
"""

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from execution.risk_manager import (
    COUPLING_LOOSEN,
    COUPLING_TIGHTEN,
    GATE_NONE,
    GATE_REGIME,
    GATE_SPREAD,
    GEOMETRY_BARRIER,
    GEOMETRY_STATIC,
    PAUSE_DAILY_ROLLOVER,
    PAUSE_WEEKEND,
    RiskManager,
    RiskProfile,
    coupled_keff,
    scheduled_market_pause,
)
from strategies.base import BARRIER_GEOMETRY_KEY


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


class TestBarrierGeometry(unittest.TestCase):
    """
    Learned barrier geometry (ml.barriers sidecar) entering the bracket path.

    The contract under test: a payload replaces the profile's static
    multipliers for that bar, the three gates see the SUBSTITUTED stop (a
    learned stop too tight to pay the spread must still be refused), and a
    broken payload degrades to the static bracket rather than raising — a
    telemetry-shaped bug in a sidecar must never sit on the order path.
    """

    _SERIES = [1.0] * 100  # warm, flat: Gate B never fires, pctile rank 1.0

    def _payload(self, **overrides):
        base = {
            "source": "barrier",
            "sl_atr_mult": 1.0,
            "tp_atr_mult": 2.0,
            "rr": 0.29,
            "admissible": False,
            "tau_mae": 0.95,
            "tau_mfe": 0.50,
            "backend": "catboost",
        }
        base.update(overrides)
        return base

    def test_payload_replaces_static_multipliers(self):
        """sl/tp come from the payload, not from sl_atr_multiplier."""
        rm = RiskManager(_forex_profile())  # static 1.0 / 2.0
        res = rm.calculate_bracket(
            150.0, 0.5, symbol="GBP_JPY",
            spread=0.001, spread_fresh=True, regime_series=self._SERIES,
            barrier=self._payload(sl_atr_mult=3.0, tp_atr_mult=6.0),
        )
        self.assertEqual(res, (1.5, 3.0))
        self.assertEqual(rm.last_geometry_source, GEOMETRY_BARRIER)
        self.assertEqual(rm.last_veto_gate, GATE_NONE)

    def test_payload_does_not_compound_with_the_profile(self):
        """1.0x learned != 1.0x learned x 1.0x profile — substitution, not
        multiplication. A test that cannot tell those apart would pass on a
        silently doubled bracket."""
        rm = RiskManager(_forex_profile())
        res = rm.calculate_bracket(
            150.0, 0.5, symbol="GBP_JPY",
            spread=0.001, spread_fresh=True, regime_series=self._SERIES,
            barrier=self._payload(sl_atr_mult=1.0, tp_atr_mult=2.0),
        )
        self.assertEqual(res, (0.5, 1.0))

    def test_gate_a_asks_about_the_substituted_stop(self):
        """Same raw ATR and spread: the static stop passes, a learned stop
        tight enough to be eaten by the spread must be vetoed."""
        rm = RiskManager(_forex_profile())
        # static sl_dist = 0.5 * 1.0 = 0.5 > 1.5*0.001; learned = 0.5*0.001 = 0.0005
        self.assertIsNotNone(
            rm.calculate_bracket(
                150.0, 0.5, symbol="GBP_JPY",
                spread=0.001, spread_fresh=True, regime_series=self._SERIES,
            )
        )
        self.assertEqual(rm.last_geometry_source, GEOMETRY_STATIC)
        self.assertIsNone(
            rm.calculate_bracket(
                150.0, 0.5, symbol="GBP_JPY",
                spread=0.001, spread_fresh=True, regime_series=self._SERIES,
                barrier=self._payload(sl_atr_mult=0.001),
            )
        )
        self.assertEqual(rm.last_veto_gate, GATE_SPREAD)

    def test_gate_b_still_fires_under_learned_geometry(self):
        rm = RiskManager(_forex_profile())
        res = rm.calculate_bracket(
            150.0, 0.5, symbol="GBP_JPY",
            spread=0.001, spread_fresh=True,
            regime_series=[1.0] * 99 + [0.05],  # current is the window minimum
            barrier=self._payload(),
        )
        self.assertIsNone(res)
        self.assertEqual(rm.last_veto_gate, GATE_REGIME)

    def test_inadmissible_payload_is_not_a_veto(self):
        """admissible=False travels as telemetry only: the estimator's rr_floor
        scores Q_MFE(0.50)/Q_MAE(0.95), which is structurally below 1, so
        enforcing it live would refuse every bar. Measured rr 0.28-0.30 across
        all three evaluation folds on 2026-09-14."""
        rm = RiskManager(_forex_profile())
        res = rm.calculate_bracket(
            150.0, 0.5, symbol="GBP_JPY",
            spread=0.001, spread_fresh=True, regime_series=self._SERIES,
            barrier=self._payload(rr=0.10, admissible=False),
        )
        self.assertIsNotNone(res)
        self.assertEqual(rm.last_geometry_source, GEOMETRY_BARRIER)

    def test_unusable_payloads_degrade_to_static(self):
        """Every malformed shape must leave a tradeable static bracket."""
        rm = RiskManager(_forex_profile())
        for bad in (
            "not-a-mapping",
            {"tp_atr_mult": 2.0},                      # missing sl
            {"sl_atr_mult": "wide", "tp_atr_mult": 2.0},
            {"sl_atr_mult": float("nan"), "tp_atr_mult": 2.0},
            {"sl_atr_mult": float("inf"), "tp_atr_mult": 2.0},
            {"sl_atr_mult": 0.0, "tp_atr_mult": 2.0},  # zero-width stop
            {"sl_atr_mult": -3.0, "tp_atr_mult": 2.0}, # inverted bracket
        ):
            res = rm.calculate_bracket(
                150.0, 0.5, symbol="GBP_JPY",
                spread=0.001, spread_fresh=True, regime_series=self._SERIES,
                barrier=bad,
            )
            self.assertEqual(
                res, (0.5, 1.0), f"payload {bad!r} should have been ignored"
            )
            self.assertEqual(rm.last_geometry_source, GEOMETRY_STATIC)

    def test_provenance_resets_each_call(self):
        """A barrier bar followed by a payload-less bar must not keep claiming
        learned provenance — the orchestrator logs this per entry."""
        rm = RiskManager(_forex_profile())
        rm.calculate_bracket(
            150.0, 0.5, symbol="GBP_JPY", spread=0.001, spread_fresh=True,
            regime_series=self._SERIES, barrier=self._payload(),
        )
        self.assertEqual(rm.last_geometry_source, GEOMETRY_BARRIER)
        rm.calculate_bracket(
            150.0, 0.5, symbol="GBP_JPY", spread=0.001, spread_fresh=True,
            regime_series=self._SERIES,
        )
        self.assertEqual(rm.last_geometry_source, GEOMETRY_STATIC)

    def test_payload_key_matches_the_signal_contract(self):
        """The metadata key is defined once (strategies.base); execution reads
        whatever the caller hands it. Pin the string so a rename cannot leave
        an orchestrator looking up a key nobody writes."""
        self.assertEqual(BARRIER_GEOMETRY_KEY, "barrier_geometry")

    def test_no_regime_series_still_honours_a_payload(self):
        """Equities-style path (static floor, no vol context): the learned
        stop is still the one floored."""
        rm = RiskManager(_forex_profile(min_sl_pips=2.0))
        # JPY pip floor 0.02; learned 0.015 < floor -> static floor veto
        self.assertIsNone(
            rm.calculate_bracket(
                155.0, 0.015, symbol="USD_JPY",
                barrier=self._payload(sl_atr_mult=1.0),
            )
        )
        self.assertIsNotNone(
            rm.calculate_bracket(
                155.0, 0.05, symbol="USD_JPY",
                barrier=self._payload(sl_atr_mult=1.0),
            )
        )


class TestProductionForexProfile(unittest.TestCase):
    """The live forex numbers, pinned.

    These are NOT arbitrary: `retrainer.get_asset_config` reads the same
    profile to build the Devil's training labels ("was the target reached
    before the stop"). Changing a multiplier here without retraining puts the
    model's labels out of step with the brackets it is scored against.
    """

    def test_forex_bracket_is_two_and_four_atr(self):
        p = RiskProfile.for_asset_class("forex")
        self.assertEqual(p.sl_atr_multiplier, 2.0)
        self.assertEqual(p.tp_atr_multiplier, 4.0)

    def test_forex_payoff_ratio_is_two_to_one(self):
        """Widening the stop on 2026-08-08 held the payoff ratio deliberately;
        the change was about diluting the spread toll, not re-aiming the trade."""
        p = RiskProfile.for_asset_class("forex")
        self.assertEqual(p.tp_atr_multiplier / p.sl_atr_multiplier, 2.0)

    def test_forex_cost_gate_caps_the_toll_at_one_third(self):
        """Gate A is `sl_dist >= k * spread`, i.e. the spread may eat at most
        1/k of the stop. k=3.0 caps it at 33%; the old 1.5 allowed 67%."""
        p = RiskProfile.for_asset_class("forex")
        self.assertEqual(p.spread_k_base, 3.0)
        self.assertAlmostEqual(1.0 / p.spread_k_base, 1.0 / 3.0, places=6)

    def test_equities_profile_untouched_by_the_forex_change(self):
        p = RiskProfile.for_asset_class("equities")
        self.assertEqual(p.sl_atr_multiplier, 0.5)
        self.assertEqual(p.tp_atr_multiplier, 3.0)


class TestGateCBlackout(unittest.TestCase):
    """Gate C — the NY-rollover blackout window, DST-correct.

    Added 2026-09-09: this was the one timezone-sensitive gate with zero test
    coverage (a regression here silently admits trades into the 10x-spread
    rollover window). Pins: the window tracks America/New_York across DST
    (20:55Z summer / 21:55Z winter), start-inclusive/end-exclusive boundaries,
    midnight wrap, naive-timestamp-assumed-UTC, disabled-profile no-op, and
    the production default window.
    """

    def _rm(self, **overrides):
        return RiskManager(
            _forex_profile(blackout_start=time(16, 55), blackout_end=time(17, 30), **overrides)
        )

    def test_summer_window_edt(self):
        """Aug 18 2026: NY on EDT (UTC-4) → 16:55 NY == 20:55 UTC."""
        rm = self._rm()
        self.assertTrue(rm._in_blackout(datetime(2026, 8, 18, 20, 55, tzinfo=timezone.utc)))  # start inclusive
        self.assertTrue(rm._in_blackout(datetime(2026, 8, 18, 21, 0, tzinfo=timezone.utc)))
        self.assertTrue(rm._in_blackout(datetime(2026, 8, 18, 21, 29, tzinfo=timezone.utc)))
        self.assertFalse(rm._in_blackout(datetime(2026, 8, 18, 21, 30, tzinfo=timezone.utc)))  # end exclusive
        self.assertFalse(rm._in_blackout(datetime(2026, 8, 18, 15, 0, tzinfo=timezone.utc)))

    def test_winter_window_est(self):
        """Jan 13 2026: NY on EST (UTC-5) → 16:55 NY == 21:55 UTC."""
        rm = self._rm()
        self.assertTrue(rm._in_blackout(datetime(2026, 1, 13, 21, 55, tzinfo=timezone.utc)))
        self.assertTrue(rm._in_blackout(datetime(2026, 1, 13, 22, 29, tzinfo=timezone.utc)))
        self.assertFalse(rm._in_blackout(datetime(2026, 1, 13, 22, 30, tzinfo=timezone.utc)))
        # The summer edge must NOT read as in-window in winter: the window
        # tracks NY local time, so it shifts an hour across DST.
        self.assertFalse(rm._in_blackout(datetime(2026, 1, 13, 20, 55, tzinfo=timezone.utc)))

    def test_naive_timestamp_assumed_utc(self):
        rm = self._rm()
        self.assertTrue(rm._in_blackout(datetime(2026, 8, 18, 20, 55)))
        self.assertFalse(rm._in_blackout(datetime(2026, 8, 18, 21, 30)))

    def test_disabled_profile_never_blackouts(self):
        rm = RiskManager(_forex_profile(blackout_start=None, blackout_end=None))
        self.assertFalse(rm._in_blackout(datetime(2026, 8, 18, 21, 0, tzinfo=timezone.utc)))

    def test_midnight_wrap_window(self):
        rm = RiskManager(
            _forex_profile(blackout_start=time(23, 0), blackout_end=time(1, 0))
        )
        self.assertTrue(rm._in_blackout(datetime(2026, 8, 19, 3, 0, tzinfo=timezone.utc)))    # 23:00 NY
        self.assertTrue(rm._in_blackout(datetime(2026, 8, 19, 4, 59, tzinfo=timezone.utc)))   # 00:59 NY
        self.assertFalse(rm._in_blackout(datetime(2026, 8, 19, 5, 0, tzinfo=timezone.utc)))   # 01:00 NY
        self.assertFalse(rm._in_blackout(datetime(2026, 8, 19, 16, 0, tzinfo=timezone.utc)))  # 12:00 NY

    def test_production_profile_has_default_window(self):
        p = RiskProfile.for_asset_class("forex")
        self.assertEqual(p.blackout_start, time(16, 55))
        self.assertEqual(p.blackout_end, time(17, 30))


class TestScheduledMarketPause(unittest.TestCase):
    """Scheduled market pauses — the weekend and the daily NY rollover.

    Added 2026-09-15, after the soak's liveness watchdog spent the 2026-09-11..13
    weekend reporting the closed forex market as a dead feed: 12,674 CRITICAL
    lines, 14 alert incidents, 12 futile reconnects, and a price clock reading
    51,072s (14.2 hours) of "silence" — with every line correctly saying "no
    positions held". The same rule flattens open positions, which ARE legitimately
    held across the rollover (Gate C blocks only new entries).

    Pins the two windows, their exact boundaries, DST correctness, the
    unknown-timezone fail-safe, and the real incident timestamps replayed.
    """

    def test_the_two_real_incident_starts_are_recognised(self):
        """The exact moments this was written for: 09-14 and 09-15 at 21:00 UTC."""
        self.assertEqual(
            scheduled_market_pause(datetime(2026, 9, 15, 21, 0, 9, tzinfo=timezone.utc)),
            PAUSE_DAILY_ROLLOVER,
        )
        self.assertEqual(
            scheduled_market_pause(datetime(2026, 9, 14, 21, 0, 11, tzinfo=timezone.utc)),
            PAUSE_DAILY_ROLLOVER,
        )

    def test_weekend_closure_boundaries(self):
        """Fri 17:00 ET -> Sun 17:00 ET, and the Friday handover from rollover.

        September 2026 is EDT (UTC-4), so 17:00 ET == 21:00 UTC. Note the
        Friday sequence: trading ends at 16:55 ET when the *rollover* window
        opens, and at 17:00 ET that same window becomes the *weekend* — so the
        two pause kinds meet exactly at the weekly close.
        """
        self.assertIsNone(
            scheduled_market_pause(datetime(2026, 9, 11, 20, 50, tzinfo=timezone.utc))
        )  # Friday 16:50 ET — still trading
        self.assertEqual(
            scheduled_market_pause(datetime(2026, 9, 11, 20, 55, tzinfo=timezone.utc)),
            PAUSE_DAILY_ROLLOVER,
        )  # Friday 16:55 ET — rollover window opens
        self.assertEqual(
            scheduled_market_pause(datetime(2026, 9, 11, 21, 0, tzinfo=timezone.utc)),
            PAUSE_WEEKEND,
        )  # Friday 17:00 ET — weekly close
        self.assertEqual(
            scheduled_market_pause(datetime(2026, 9, 12, 16, 0, tzinfo=timezone.utc)),
            PAUSE_WEEKEND,
        )  # Saturday
        self.assertEqual(
            scheduled_market_pause(datetime(2026, 9, 13, 20, 59, tzinfo=timezone.utc)),
            PAUSE_WEEKEND,
        )  # Sunday 16:59 ET
        self.assertEqual(
            scheduled_market_pause(datetime(2026, 9, 13, 21, 0, tzinfo=timezone.utc)),
            PAUSE_DAILY_ROLLOVER,
        )  # Sunday 17:00 ET — the weekly reopen lands INSIDE the daily rollover
        # window (both are the 5pm-ET boundary), so the pause continues as a
        # rollover until 17:30 ET. That is the right answer: the reopen is when
        # spreads are widest and OANDA takes minutes to start ticking.
        self.assertIsNone(
            scheduled_market_pause(datetime(2026, 9, 13, 21, 30, tzinfo=timezone.utc))
        )  # Sunday 17:30 ET — both windows closed; trading resumes

    def test_rollover_window_and_boundaries(self):
        """Default 16:55-17:30 ET, start-inclusive / end-exclusive."""
        self.assertIsNone(
            scheduled_market_pause(datetime(2026, 8, 18, 20, 54, tzinfo=timezone.utc))
        )
        self.assertEqual(
            scheduled_market_pause(datetime(2026, 8, 18, 20, 55, tzinfo=timezone.utc)),
            PAUSE_DAILY_ROLLOVER,
        )
        self.assertEqual(
            scheduled_market_pause(datetime(2026, 8, 18, 21, 29, tzinfo=timezone.utc)),
            PAUSE_DAILY_ROLLOVER,
        )
        self.assertIsNone(
            scheduled_market_pause(datetime(2026, 8, 18, 21, 30, tzinfo=timezone.utc))
        )

    def test_normal_trading_hours_are_not_a_pause(self):
        """The watchdog must stay armed when prices are actually expected."""
        for ts in (
            datetime(2026, 9, 15, 14, 0, tzinfo=timezone.utc),   # Tue 10:00 ET
            datetime(2026, 9, 15, 3, 30, tzinfo=timezone.utc),   # Mon 23:30 ET (Tokyo)
            datetime(2026, 9, 16, 6, 0, tzinfo=timezone.utc),    # Wed 02:00 ET
        ):
            self.assertIsNone(scheduled_market_pause(ts), ts)

    def test_dst_correctness(self):
        """Jan 2026 is EST (UTC-5): the rollover is an hour later in UTC."""
        self.assertEqual(
            scheduled_market_pause(datetime(2026, 1, 13, 22, 5, tzinfo=timezone.utc)),
            PAUSE_DAILY_ROLLOVER,
        )
        self.assertIsNone(
            scheduled_market_pause(datetime(2026, 1, 13, 21, 5, tzinfo=timezone.utc))
        )  # 16:05 ET — the summer window's hour is NOT the winter window

    def test_naive_timestamp_assumed_utc(self):
        self.assertEqual(
            scheduled_market_pause(datetime(2026, 9, 15, 21, 0)), PAUSE_DAILY_ROLLOVER
        )
        self.assertIsNone(scheduled_market_pause(datetime(2026, 9, 15, 14, 0)))

    def test_spec_override_moves_the_rollover_window(self):
        self.assertIsNone(
            scheduled_market_pause(
                datetime(2026, 9, 15, 21, 0, tzinfo=timezone.utc), spec="18:00-18:30"
            )
        )
        self.assertEqual(
            scheduled_market_pause(
                datetime(2026, 9, 15, 22, 5, tzinfo=timezone.utc), spec="18:00-18:30"
            ),
            PAUSE_DAILY_ROLLOVER,
        )

    def test_bad_spec_disables_only_the_rollover_not_the_weekend(self):
        """A malformed window must not take the weekend pause down with it."""
        self.assertIsNone(
            scheduled_market_pause(
                datetime(2026, 9, 15, 21, 0, tzinfo=timezone.utc), spec="not-a-window"
            )
        )
        self.assertEqual(
            scheduled_market_pause(
                datetime(2026, 9, 12, 16, 0, tzinfo=timezone.utc), spec="not-a-window"
            ),
            PAUSE_WEEKEND,
        )


class TestCalculateForexUnits(unittest.TestCase):
    """
    Forex position unit scaling inversely with stop width.

    Preserves constant dollar risk per stop-out when learned barrier geometry
    predicts wider or narrower stops than the static 2.0x ATR baseline.
    """

    def setUp(self):
        self.rm = RiskManager(_forex_profile())

    def test_identical_sl_retains_base_units(self):
        units = self.rm.calculate_forex_units(1000, 0.50, 0.50)
        self.assertEqual(units, 1000)

    def test_double_width_halves_units(self):
        units = self.rm.calculate_forex_units(1000, 0.50, 1.00)
        self.assertEqual(units, 500)

    def test_wider_stop_preserves_dollar_risk(self):
        base_units = 1000
        static_sl = 0.20
        actual_sl = 0.745
        units = self.rm.calculate_forex_units(base_units, static_sl, actual_sl)
        self.assertEqual(units, 268)
        dollar_risk_static = base_units * static_sl
        dollar_risk_barrier = units * actual_sl
        self.assertAlmostEqual(dollar_risk_barrier, dollar_risk_static, delta=1.0)

    def test_narrower_stop_increases_units_capped_at_max(self):
        units = self.rm.calculate_forex_units(1000, 0.50, 0.25)
        self.assertEqual(units, 2000)
        units = self.rm.calculate_forex_units(1000, 0.50, 0.125)
        self.assertEqual(units, 2000)

    def test_very_wide_stop_clamped_to_min_units(self):
        units = self.rm.calculate_forex_units(1000, 0.10, 10.0, min_units=100)
        self.assertEqual(units, 100)

    def test_invalid_or_zero_sl_returns_base_units(self):
        self.assertEqual(self.rm.calculate_forex_units(1000, 0.0, 0.50), 1000)
        self.assertEqual(self.rm.calculate_forex_units(1000, 0.50, 0.0), 1000)
        self.assertEqual(self.rm.calculate_forex_units(1000, -0.50, 0.50), 1000)
        self.assertEqual(self.rm.calculate_forex_units(1000, 0.50, -0.50), 1000)

    def test_calculate_quantity_short_direction(self):
        qty = self.rm.calculate_quantity(equity=10000.0, buying_power=10000.0, entry_price=100.0, sl_price=105.0)
        self.assertGreater(qty, 0.0)


if __name__ == "__main__":
    unittest.main()
