"""
Tests for the artifact-level holdout gate.

The core property: the served model must never have seen the holdout rows.
These tests prove that property at the data-flow level by checking that the
split is chronological and disjoint, that holdout metrics are recorded, and
that bypass states are declared in metadata.

Glossary:
    _MockLGBM -- a stand-in classifier that returns fixed probabilities so
        _evaluate_holdout can be unit-tested without training real models.
    _make_raw_frame -- builds a recognisable timestamp-ordered frame so the
        split boundary is easy to verify.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import retrainer as R  # noqa: E402
from core.retrainer import HoldoutMetrics, ValidationReport  # noqa: E402


class _MockLGBM:
    """Minimal LightGBM stand-in for holdout metric tests."""

    def __init__(self, class1_prob: float):
        self._class1_prob = float(class1_prob)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        n = len(X)
        p = np.zeros((n, 2), dtype=float)
        p[:, 1] = self._class1_prob
        p[:, 0] = 1.0 - self._class1_prob
        return p


def _make_raw_frame(n_per_day: int = 4, days: int = 10) -> pl.DataFrame:
    """Return a frame with one row per hour-ish bar over ``days`` days."""
    n = n_per_day * days
    timestamps = [
        R.datetime(2026, 1, 1, 0, 0, 0, tzinfo=R.timezone.utc)
        + R.timedelta(hours=i)
        for i in range(n)
    ]
    # Varying prices keep NATR non-zero so the chop veto and labels compute.
    close = [1.0 + 0.01 * (i % 7) + 0.001 * i for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["GBP_JPY"] * n,
            "open": [c - 0.005 for c in close],
            "high": [c + 0.015 for c in close],
            "low": [c - 0.015 for c in close],
            "close": close,
            "volume": [100] * n,
        }
    )


class TestSplitHoldout(unittest.TestCase):
    def test_holdout_is_chronologically_last(self):
        df = _make_raw_frame()
        remainder, holdout, holdout_range = R._split_holdout(df, 0.2)

        self.assertFalse(remainder.is_empty())
        self.assertFalse(holdout.is_empty())
        self.assertLess(
            remainder["timestamp"].max(), holdout["timestamp"].min()
        )
        self.assertIsNotNone(holdout_range)

    def test_split_is_disjoint(self):
        df = _make_raw_frame()
        remainder, holdout, _ = R._split_holdout(df, 0.25)

        rem_ts = set(remainder["timestamp"].to_list())
        hold_ts = set(holdout["timestamp"].to_list())
        self.assertFalse(rem_ts & hold_ts)

    def test_fraction_is_respected(self):
        df = _make_raw_frame(n_per_day=4, days=20)
        remainder, holdout, _ = R._split_holdout(df, 0.18)
        total = len(df)
        self.assertAlmostEqual(len(holdout) / total, 0.18, delta=0.05)

    def test_zero_fraction_disables_holdout(self):
        df = _make_raw_frame()
        remainder, holdout, holdout_range = R._split_holdout(df, 0.0)

        self.assertEqual(len(remainder), len(df))
        self.assertTrue(holdout.is_empty())
        self.assertIsNone(holdout_range)

    def test_empty_input_returns_empty_holdout(self):
        df = pl.DataFrame(
            {
                "timestamp": [],
                "symbol": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        )
        remainder, holdout, holdout_range = R._split_holdout(df, 0.2)
        self.assertTrue(remainder.is_empty())
        self.assertTrue(holdout.is_empty())
        self.assertIsNone(holdout_range)


class TestEvaluateHoldout(unittest.TestCase):
    def _frame(self, n: int = 20):
        return pl.DataFrame(
            {
                "timestamp": list(range(n)),
                "symbol": ["GBP_JPY"] * n,
                "angel_target": [1] * n,
                "devil_target": [1, 0] * (n // 2),
                "devil_target_macro": [1, 0] * (n // 2),
                "feat_a": [0.0] * n,
                "feat_b": [0.0] * n,
            }
        )

    def test_evaluates_tradeable_approvals(self):
        holdout = self._frame(20)
        angel = _MockLGBM(class1_prob=0.9)  # all proposed
        devil = _MockLGBM(class1_prob=0.9)  # all approved

        scores = R._evaluate_holdout(
            holdout,
            angel,
            devil,
            angel_features=["feat_a", "feat_b"],
            devil_features=["feat_a", "feat_b", "angel_prob"],
            threshold=0.5,
            sl_mult=2.0,
            tp_mult=4.0,
        )

        self.assertEqual(scores["angel_proposed_trades"], 20)
        self.assertEqual(scores["trades"], 20)
        self.assertAlmostEqual(scores["win_rate"], 0.5, delta=0.01)
        # At 2:1 payoff with 50% wins, EV = 0.5*2 - 0.5 = 0.5
        self.assertAlmostEqual(scores["expected_value"], 0.5, delta=0.01)

    def test_no_angel_proposals_returns_nan_metrics(self):
        holdout = self._frame(10)
        angel = _MockLGBM(class1_prob=0.1)  # no proposals at 0.40 bar
        devil = _MockLGBM(class1_prob=0.9)

        scores = R._evaluate_holdout(
            holdout,
            angel,
            devil,
            angel_features=["feat_a", "feat_b"],
            devil_features=["feat_a", "feat_b", "angel_prob"],
            threshold=0.5,
            sl_mult=2.0,
            tp_mult=4.0,
        )

        self.assertEqual(scores["trades"], 0)
        self.assertTrue(np.isnan(scores["brier_score"]))

    def test_untradeable_approvals_are_excluded_from_scoring(self):
        holdout = pl.DataFrame(
            {
                "timestamp": list(range(4)),
                "symbol": ["XAU_USD", "GBP_JPY", "XAG_USD", "EUR_JPY"],
                "angel_target": [1] * 4,
                "devil_target": [1, 1, 0, 0],
                "devil_target_macro": [1, 1, 0, 0],
                "feat_a": [0.0] * 4,
                "feat_b": [0.0] * 4,
            }
        )
        angel = _MockLGBM(class1_prob=0.9)  # all proposed
        devil = _MockLGBM(class1_prob=0.9)  # all approved

        scores = R._evaluate_holdout(
            holdout,
            angel,
            devil,
            angel_features=["feat_a", "feat_b"],
            devil_features=["feat_a", "feat_b", "angel_prob"],
            threshold=0.5,
            sl_mult=2.0,
            tp_mult=4.0,
        )

        # Two proposals are untradeable metals and should be dropped from scoring.
        self.assertEqual(scores["angel_proposed_trades"], 4)
        self.assertEqual(scores["trades"], 2)
        self.assertEqual(scores["devil_approved_raw"], 4)


class TestMetadataRecordsHoldout(unittest.TestCase):
    def test_metadata_records_holdout_metrics_on_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / "holdout_test"
            cfg = {
                "asset_class": "forex",
                "model_dir": str(override),
                "tickers": ["GBP_JPY"],
                "timeframe_minutes": 1,
            }
            report = ValidationReport(
                fold_metrics=[],
                mean_brier=0.10,
                mean_ev=0.01,
                final_profit_factor=1.5,
                final_win_rate=0.45,
                final_total_trades=100,
                gate_passed=True,
                holdout=HoldoutMetrics(
                    used=True,
                    fraction=0.18,
                    start_date="2026-01-10",
                    end_date="2026-01-12",
                    brier_score=0.12,
                    expected_value=0.02,
                    win_rate=0.40,
                    profit_factor=1.3,
                    trades=55,
                    angel_proposed_trades=80,
                ),
            )

            R.save_models({"angel": 1}, {"devil": 2}, cfg, report=report)

            meta = json.loads((override / "metadata.json").read_text())
            self.assertTrue(meta["holdout"]["used"])
            self.assertEqual(meta["holdout"]["fraction"], 0.18)
            self.assertEqual(meta["holdout"]["start_date"], "2026-01-10")
            self.assertEqual(meta["holdout"]["end_date"], "2026-01-12")
            self.assertEqual(meta["holdout"]["trades"], 55)
            self.assertIsNone(meta["holdout"]["bypass_reason"])

    def test_metadata_records_bypass_when_holdout_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / "holdout_disabled"
            cfg = {
                "asset_class": "forex",
                "model_dir": str(override),
                "tickers": ["GBP_JPY"],
                "timeframe_minutes": 1,
            }
            with mock.patch.object(R, "HOLDOUT_FRAC", 0.0):
                R.save_models({"angel": 1}, {"devil": 2}, cfg)

            meta = json.loads((override / "metadata.json").read_text())
            self.assertFalse(meta["holdout"]["used"])
            self.assertEqual(meta["holdout"]["bypass_reason"], "disabled")


class TestHoldoutNeverInTraining(unittest.TestCase):
    """
    The property the whole change exists to enforce. We cannot easily inspect
    the inside of LightGBM, but we can prove the data-flow invariant: the
    remainder passed to training contains no timestamps from the holdout slice.
    """

    def test_remainder_and_holdout_are_temporally_separated(self):
        # Need enough bars for HTF SMA-50 warm-up (~250 1m bars) plus headroom.
        raw = _make_raw_frame(n_per_day=24, days=30)
        remainder, holdout, _ = R._split_holdout(raw, 0.2)

        # engineer_features_and_labels preserves the timestamp column, so the
        # temporal separation survives the entire pipeline.
        rem_features, _, _ = R.engineer_features_and_labels(
            remainder,
            sl_mult=2.0,
            tp_mult=4.0,
            max_hold=45,
            survival_bars=5,
            htf_timeframe="5m",
        )
        hold_features, _, _ = R.engineer_features_and_labels(
            holdout,
            sl_mult=2.0,
            tp_mult=4.0,
            max_hold=45,
            survival_bars=5,
            htf_timeframe="5m",
        )

        self.assertGreater(rem_features.height, 0)
        self.assertGreater(hold_features.height, 0)
        self.assertLess(
            rem_features["timestamp"].max(),
            hold_features["timestamp"].min(),
        )


if __name__ == "__main__":
    unittest.main()
