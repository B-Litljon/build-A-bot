"""
The gate's edge-over-random benchmark: `_macro_base_rate` and its reporting.

A gate scoring ABSOLUTE win rate and PF cannot distinguish skill from weather.
Measured on the H4 CatBoost candidate (2026-09-14): its metals approvals won 57.1%
against their period's base rate of 56.6% (+0.005, p=0.50) — no selectivity at all —
inside a regime where a 2:1 long bracket won 56.6% against a 33.3% break-even, so the
regime alone cleared a 1.2 PF lower bound the gate was reading as evidence.

These tests pin the benchmark itself: it is the macro outcome's mean on the population
it is given, it refuses to invent a number for an unlabelled frame, and the report
carries it per fold and pooled.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import core.retrainer as R  # noqa: E402


class TestMacroBaseRate(unittest.TestCase):
    def test_mean_of_the_macro_outcome(self):
        df = pl.DataFrame({"devil_target_macro": [1.0, 0.0, 1.0, 0.0]})
        self.assertAlmostEqual(R._macro_base_rate(df), 0.5, places=9)

    def test_a_high_base_rate_is_reported_as_such(self):
        """The H4 case: 56.6% of random long entries won in that period."""
        df = pl.DataFrame({"devil_target_macro": [1.0] * 566 + [0.0] * 434})
        self.assertAlmostEqual(R._macro_base_rate(df), 0.566, places=3)

    def test_non_finite_outcomes_are_dropped_not_counted_as_losses(self):
        df = pl.DataFrame({"devil_target_macro": [1.0, 1.0, None, float("nan")]})
        self.assertAlmostEqual(R._macro_base_rate(df), 1.0, places=9)

    def test_unlabelled_frame_returns_nan_not_zero(self):
        """A 0.0 base rate would make every model look like it has edge."""
        self.assertTrue(np.isnan(R._macro_base_rate(pl.DataFrame({"close": [1.0]}))))
        self.assertTrue(np.isnan(R._macro_base_rate(pl.DataFrame({"devil_target_macro": []}))))
        self.assertTrue(np.isnan(R._macro_base_rate(pl.DataFrame({"devil_target_macro": [None, None]}))))

    def test_report_fields_exist_with_nan_defaults(self):
        fm = R.FoldMetrics(
            fold_number=1, train_size=10, val_size=5, brier_score=0.2,
            expected_value=0.1, angel_proposed_trades=3, devil_approved_trades=2,
            win_rate=0.5,
        )
        self.assertTrue(np.isnan(fm.base_rate))
        rep = R.ValidationReport(
            fold_metrics=[fm], mean_brier=0.2, mean_ev=0.1, final_profit_factor=1.0,
            final_win_rate=0.5, final_total_trades=2, gate_passed=False,
        )
        self.assertTrue(np.isnan(rep.pooled_base_rate))
        self.assertTrue(np.isnan(rep.edge_over_random))

    def test_validate_candidate_reports_the_benchmark(self):
        """End-to-end on a small synthetic frame: the gate must carry the base rate
        and the edge into the report it returns."""
        n = 600
        ts = pl.datetime_range(
            __import__("datetime").datetime(2026, 1, 1),
            __import__("datetime").datetime(2026, 1, 1)
            + __import__("datetime").timedelta(minutes=15 * (n - 1)),
            interval="15m", eager=True,
        )
        rng = np.random.default_rng(5)
        close = 150.0 + np.cumsum(rng.normal(0, 0.1, n))
        raw = pl.DataFrame({
            "timestamp": ts, "open": close,
            "high": close + 0.05, "low": close - 0.05, "close": close,
            "volume": np.full(n, 10.0), "symbol": ["GBP_JPY"] * n,
        })
        feats, cols, chop = R.engineer_features_and_labels(
            raw, sl_mult=2.0, angel_mult=1.0, tp_mult=4.0, max_hold=30,
            survival_bars=5, htf_timeframe="1h",
            risk_profile=R.RiskProfile.for_asset_class("forex"), alpha_table=None,
        )
        if feats.height < 150:
            self.skipTest("synthetic frame too small after cleaning for 3 folds")
        ap, dp = R.get_hyperparameters("forex")
        rep, *_ = R.validate_candidate(
            feats, cols, sl_mult=2.0, tp_mult=4.0, n_folds=3,
            angel_params=ap, devil_params=dp, chop_veto_rate=chop,
        )
        for fm in rep.fold_metrics:
            self.assertTrue(np.isfinite(fm.base_rate), fm)
        self.assertTrue(np.isfinite(rep.pooled_base_rate))
        self.assertTrue(np.isfinite(rep.edge_over_random))


if __name__ == "__main__":
    unittest.main()
