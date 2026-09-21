"""
Tests for the 2026-08-29 gate rebuild: the OOF-calibrated Angel threshold,
the Devil min_child auto-scaler, and the Clopper-Pearson evidence instrument
that replaced the flat 300-trade fold floor.

Why these are pinned: each fix exists because this project already shipped
its failure mode once — a fixed 0.40 Angel bar starving a compressed model
(11 proposals per fold), a Devil degenerated to a constant by an oversized
min_child (100% approval, zero signal), and a trade-count cliff the shipped
config itself cleared only once in three pins. A silent regression in any of
the three would look like a normal retrain run, so they are tested here as
units rather than rediscovered in a gate log.

Glossary:
    compressed distribution -- an Angel score distribution whose mass sits
        far below the old fixed bar (e.g. p99 < 0.40); the failure shape the
        calibration was built for.
    fallback (sweep) -- when no candidate yields MIN_ANGEL_PROPOSALS, the
        sweep returns propose-everything (min observed score) rather than a
        bar that starves the Devil.
    reference values -- CP lower-bound numbers computed once from the
        instrument and pinned, so a formula change shows up as a test diff.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# NB: do NOT put this file's own dir on sys.path to reach sibling test
# modules — tests/execution/ then shadows the real execution package and
# `from execution.risk_manager import ...` dies mid-collection. tests/ is a
# package: import siblings as tests.test_* instead.

from core import retrainer as R  # noqa: E402
from core.retrainer import _train as train  # noqa: E402  (refit_models lives here post-2026-09-16 split)
from core.retrainer import (  # noqa: E402
    _devil_min_child,
    _find_optimal_angel_threshold,
    _holdout_pf_lower_bound,
)


class TestAngelThresholdSweep(unittest.TestCase):
    """_find_optimal_angel_threshold on synthetic score distributions."""

    def _compressed(self, n=200_000, seed=7):
        """The trim-model failure shape: mass near 0.20, tail to ~0.45."""
        rng = np.random.default_rng(seed)
        probs = np.clip(rng.normal(0.20, 0.05, n), 0.01, 0.6)
        # Learnable signal: higher scores genuinely win more.
        macro = (rng.random(n) < np.clip((probs - 0.10) * 2.2, 0.05, 0.9)).astype(float)
        return probs, macro

    def test_compressed_distribution_finds_a_real_bar(self):
        probs, macro = self._compressed()
        t, ev, n = _find_optimal_angel_threshold(
            probs, macro, sl_mult=2.0, tp_mult=4.0, min_proposals=300
        )
        # Bar lives inside the observed distribution (a fixed 0.40 would
        # starve this shape), respects the proposal floor, and picks a
        # positive-EV region of the score spectrum.
        self.assertGreater(t, 0.10)
        self.assertLess(t, probs.max())
        self.assertGreaterEqual(n, 300)
        self.assertGreater(ev, 0.0)

    def test_prefers_the_higher_ev_plateau(self):
        # Two clusters: mediocre scores with 40% wins, strong scores with
        # 80% wins. The sweep must choose the strong cluster's bar even
        # though the mediocre one yields MORE proposals.
        probs = np.concatenate([np.full(50_000, 0.30), np.full(10_000, 0.45)])
        macro = np.concatenate(
            [
                np.rint(np.linspace(0, 1, 50_000) < 0.4).astype(float)[::-1],
                np.rint(np.linspace(0, 1, 10_000) < 0.8).astype(float)[::-1],
            ]
        )
        t, ev, n = _find_optimal_angel_threshold(
            probs, macro, sl_mult=2.0, tp_mult=4.0, min_proposals=300
        )
        self.assertAlmostEqual(t, 0.45, places=6)
        self.assertEqual(n, 10_000)

    def test_constant_distribution_proposes_everything(self):
        probs = np.full(5_000, 0.22)
        macro = np.tile([0.0, 1.0], 2_500)
        t, _ev, n = _find_optimal_angel_threshold(
            probs, macro, sl_mult=2.0, tp_mult=4.0, min_proposals=300
        )
        self.assertEqual(t, 0.22)
        self.assertEqual(n, 5_000)

    def test_tiny_frame_falls_back_to_propose_everything(self):
        rng = np.random.default_rng(1)
        probs = rng.random(100)  # fewer rows than min_proposals
        macro = np.rint(probs).astype(float)
        t, _ev, n = _find_optimal_angel_threshold(
            probs, macro, sl_mult=2.0, tp_mult=4.0, min_proposals=300
        )
        self.assertAlmostEqual(t, float(probs.min()), places=12)
        self.assertEqual(n, 100)

    def test_empty_frame_is_safe(self):
        t, ev, n = _find_optimal_angel_threshold(
            np.array([]), np.array([]), sl_mult=2.0, tp_mult=4.0
        )
        self.assertEqual((t, n), (0.5, 0))
        self.assertEqual(ev, -float("inf"))

    def test_deterministic(self):
        probs, macro = self._compressed(n=5_000)
        a = _find_optimal_angel_threshold(probs, macro, 2.0, 4.0, 300)
        b = _find_optimal_angel_threshold(probs, macro, 2.0, 4.0, 300)
        self.assertEqual(a, b)


class TestDevilMinChildScaling(unittest.TestCase):
    """_devil_min_child: the auto-scaler behind the Devil's revival."""

    def test_scales_to_a_tenth_of_the_population(self):
        # 310 approved rows -> 31, the value the 2026-08-29 verification run
        # logged for exactly this population.
        self.assertEqual(_devil_min_child(80, 310), 31)

    def test_capped_at_configured_when_population_is_large(self):
        # 6000 approved rows -> the configured 80 wins unchanged (old
        # behaviour preserved for chatty configurations).
        self.assertEqual(_devil_min_child(80, 6_000), 80)

    def test_floored_at_five(self):
        self.assertEqual(_devil_min_child(80, 30), 5)
        self.assertEqual(_devil_min_child(80, 0), 5)

    def test_a_split_is_always_possible(self):
        # The whole point of the fix: with the scaled value, the population
        # always satisfies the 2x-min_child split requirement.
        for n in (50, 123, 158, 583, 739, 10_000):
            self.assertGreaterEqual(n, 2 * _devil_min_child(80, n))


class TestCPLowerBoundReferences(unittest.TestCase):
    """Pinned reference values for the fold gate's evidence instrument."""

    def test_reference_points(self):
        # (wins, trades) -> the bound's value at the forex 2.0x/4.0x bracket.
        # 20/24: the 5yr trim run's pooled evidence — a strong bound.
        self.assertAlmostEqual(
            _holdout_pf_lower_bound(20, 24, 2.0, 4.0), 3.851, places=2
        )
        # 4/11: that run's Fold-3 evidence alone — correctly below the bar.
        self.assertAlmostEqual(
            _holdout_pf_lower_bound(4, 11, 2.0, 4.0), 0.312, places=2
        )

    def test_perfect_records_at_tiny_n(self):
        # The audit's canonical shape: 3/3 must NOT clear the 1.2 bar,
        # 4/4 must. (Break-even at 2:1 payoff is WR 1/3; CP at 95% needs
        # four clean trades to exclude it.)
        self.assertLess(_holdout_pf_lower_bound(3, 3, 2.0, 4.0), 1.2)
        self.assertGreater(_holdout_pf_lower_bound(4, 4, 2.0, 4.0), 1.2)

    def test_degenerate_inputs(self):
        self.assertEqual(_holdout_pf_lower_bound(0, 0, 2.0, 4.0), 0.0)
        self.assertEqual(_holdout_pf_lower_bound(0, 50, 2.0, 4.0), 0.0)


class TestGateReportCarriesThresholdAndBounds(unittest.TestCase):
    """validate_candidate threads the calibrated bar + CP bounds to the report."""

    def test_report_fields(self):
        from tests.test_holdout_gate import (  # noqa: PLC0415 — tests pkg sibling
            TestPermanentLeakGuard,
            _make_raw_frame,
        )
        from unittest import mock

        raw = _make_raw_frame(n_per_day=24, days=30)
        rem_features, feature_cols, _ = R.engineer_features_and_labels(
            raw,
            sl_mult=2.0,
            tp_mult=4.0,
            max_hold=45,
            survival_bars=5,
            htf_timeframe="5m",
        )
        recorder = TestPermanentLeakGuard._Recorder()
        with mock.patch.object(train, "refit_models", recorder):
            report = R.validate_candidate(
                rem_features, feature_cols, sl_mult=2.0, tp_mult=4.0, n_folds=3
            )[0]

        # The recorder stands in at the pre-calibration fixed bar of 0.40;
        # the report must carry exactly that through.
        self.assertEqual(report.production_angel_threshold, 0.4)
        # The CP instrument's outputs exist and are sane under the mock.
        self.assertGreaterEqual(report.pooled_pf_lower_bound, 0.0)
        self.assertGreaterEqual(report.fold3_pf_lower_bound, 0.0)
        self.assertIsInstance(report.rejection_reasons, list)


if __name__ == "__main__":
    unittest.main()
