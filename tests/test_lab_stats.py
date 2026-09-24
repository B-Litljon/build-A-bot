"""Shared falsification statistics — the cross-lane contract.

Pins `src/lab/stats.py`'s three entry-point signatures (Lanes 2-5 import
them), the DSR/PBO/HLZ definitions against hand-computable cases, and the
guard rails (degenerate inputs raise or short-circuit rather than returning
a quiet nan that a report could read as a number).
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from lab.stats import (  # noqa: E402
    BLOCKS,
    cscv_pbo,
    deflated_sharpe_ratio,
    hlz_haircut_sharpe,
    hlz_t_stat,
)


class TestDeflatedSharpeRatio(unittest.TestCase):
    def test_signature_defaults(self):
        # var_sr is keyword-only and optional.
        p = deflated_sharpe_ratio(1.0, 10, 250, 0.0, 3.0)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_probability_bounds(self):
        for sr in (-2.0, 0.0, 0.5, 3.0):
            p = deflated_sharpe_ratio(sr, 64, 500, -0.1, 4.0)
            self.assertTrue(0.0 <= p <= 1.0, sr)

    def test_strong_series_beats_the_null(self):
        # A Sharpe of 3 over 1000 daily observations with Normal shape is far
        # beyond the expected max of 50 null trials.
        p = deflated_sharpe_ratio(3.0, 50, 1000, 0.0, 3.0)
        self.assertGreater(p, 0.99)

    def test_mediocre_series_with_many_trials_deflates(self):
        # SR 0.2 with 5000 trials is exactly the "best of the lucky nulls"
        # the statistic exists to catch.
        p = deflated_sharpe_ratio(0.2, 5000, 250, 0.0, 3.0)
        self.assertLess(p, 0.5)

    def test_monotone_in_trials(self):
        ps = [deflated_sharpe_ratio(1.0, n, 500, 0.0, 3.0)
              for n in (2, 10, 100, 1000)]
        self.assertTrue(all(ps[i] >= ps[i + 1] for i in range(len(ps) - 1)))

    def test_single_trial_returns_zero(self):
        # With no selection there is nothing to deflate; the pinned form
        # returns 0.0 rather than Phi of an unadjusted z.
        self.assertEqual(deflated_sharpe_ratio(2.0, 1, 500, 0.0, 3.0), 0.0)

    def test_hand_computed_reference(self):
        # n_trials=2, var_sr=1/n_obs=1/100, skew=0, kurt=3 (Normal):
        #   SR0 = 0.1*((1-g)*Phi^-1(1/2) + g*Phi^-1(1 - 1/(2e)))
        # with Phi^-1(1/2)=0 the only term is the gamma one; check the
        # probability lands where an independent hand computation puts it.
        gamma = 0.5772156649015329
        from scipy.stats import norm
        sr0 = 0.1 * gamma * float(norm.ppf(1.0 - 1.0 / (2.0 * math.e)))
        denom = math.sqrt(1.0 - 0.0 + ((3.0 - 1.0) / 4.0) * 0.25)
        z = (0.5 - sr0) * math.sqrt(99) / denom
        expected = float(norm.cdf(z))
        got = deflated_sharpe_ratio(0.5, 2, 100, 0.0, 3.0)
        self.assertAlmostEqual(got, expected, places=12)

    def test_degenerate_denominator_raises(self):
        with self.assertRaises(ValueError):
            deflated_sharpe_ratio(10.0, 5, 100, 3.0, 1.0)

    def test_too_few_obs_raises(self):
        with self.assertRaises(ValueError):
            deflated_sharpe_ratio(1.0, 5, 1, 0.0, 3.0)


class TestCscvPbo(unittest.TestCase):
    def test_returns_scalar_in_unit_interval(self):
        rng = np.random.default_rng(7)
        m = rng.normal(0, 0.01, size=(400, 6))
        pbo = cscv_pbo(m)
        self.assertTrue(0.0 <= pbo <= 1.0)

    def test_identical_columns_are_all_overfit_by_construction(self):
        # All strategies identical: the train argmax is column 0, its test
        # rank is the average rank (N+1)/2, omega = 0.5, lambda = logit(0.5)
        # = 0, and lambda <= 0 counts as overfit on every one of the 70
        # splits. A perfectly tie-ridden matrix reads fully overfit — the
        # statistic's honest answer to "the rank carries no information".
        col = np.linspace(-0.01, 0.02, 200)
        m = np.tile(col.reshape(-1, 1), (1, 4))
        self.assertEqual(cscv_pbo(m), 1.0)

    def test_clearly_dominant_strategy_never_overfits(self):
        # Strategy 0 beats the others on EVERY contiguous half, so the
        # train-best never falls below the test median: PBO = 0.
        rng = np.random.default_rng(11)
        base = rng.normal(0, 0.005, size=(800, 4))
        base[:, 0] += 0.01  # persistent edge, every block
        self.assertEqual(cscv_pbo(base), 0.0)

    def test_pure_noise_is_high(self):
        # Iid noise with no persistent winner: the train argmax is a coin
        # flip, so roughly half the splits should read overfit. Bound loosely
        # (this is stochastic machinery on a fixed seed, but the seed is
        # pinned so the assertion is deterministic).
        rng = np.random.default_rng(3)
        m = rng.normal(0, 0.01, size=(640, 8))
        pbo = cscv_pbo(m)
        self.assertGreaterEqual(pbo, 0.30)
        self.assertLessEqual(pbo, 0.75)

    def test_shape_guards(self):
        with self.assertRaises(ValueError):
            cscv_pbo(np.zeros((4, 3)))          # too few rows for 8 blocks
        with self.assertRaises(ValueError):
            cscv_pbo(np.zeros((100, 1)))        # need >= 2 strategies
        with self.assertRaises(ValueError):
            cscv_pbo(np.zeros(10))              # 1-D is not a matrix

    def test_block_count_is_the_papers(self):
        self.assertEqual(BLOCKS, 8)


class TestHlzHaircutSharpe(unittest.TestCase):
    def test_zero_haircut_for_one_trial(self):
        self.assertEqual(hlz_haircut_sharpe(1.7, 1), 1.7)

    def test_monotone_in_trials(self):
        vals = [hlz_haircut_sharpe(1.0, n) for n in (2, 5, 20, 100)]
        self.assertTrue(all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)))
        self.assertTrue(all(v < 1.0 for v in vals))

    def test_hand_computed_reference(self):
        from scipy.stats import norm
        z = float(norm.ppf(1.0 - 1.0 / (2.0 * 10)))
        expected = 1.2 - z
        self.assertAlmostEqual(hlz_haircut_sharpe(1.2, 10), expected, places=12)

    def test_high_trial_count_can_deflate_below_zero(self):
        # At 50 trials the unit-information haircut exceeds a Sharpe of 0.8
        # entirely: the adjusted number going negative IS the statistic
        # saying the point estimate is explainable by selection.
        self.assertLess(hlz_haircut_sharpe(0.8, 50), 0.0)
        # ...while at a small trial count it stays positive.
        self.assertGreater(hlz_haircut_sharpe(0.8, 2), 0.0)

    def test_t_stat_helper_scales_with_root_n(self):
        t = hlz_t_stat(0.5, 500, 10)
        expected = hlz_haircut_sharpe(0.5, 10) * math.sqrt(499)
        self.assertAlmostEqual(t, expected, places=9)
        with self.assertRaises(ValueError):
            hlz_t_stat(0.5, 1, 10)


if __name__ == "__main__":
    unittest.main()
