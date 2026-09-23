"""
The lab's frame builder must be the production frame builder.

train/serve parity starts one level earlier than the models: if a candidate
feature set is scored on a frame that differs from the retrainer's — different
cleaning, different cleaning ORDER, a label built before vs after a veto — the
comparison to the served model's numbers is meaningless. So the headline test
here is exact frame equality between the lab and
``engineer_features_and_labels`` for the ``v3_base`` control, including the
Phase-3a unresolvable-tail purge that only exists in the lab because the lab
does not carve a holdout.

The other property pinned here: the tail purge is per symbol and exactly
``max_hold`` bars, because a wrong cutoff either trains on systematically
mislabelled rows or throws away resolvable evidence.
"""

import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from lab.frames import build_frame, stack_bars  # noqa: E402
from lab.spec import FeatureSpec  # noqa: E402

_SPREAD_TABLE = "config/spread_alphas_m15.json"


def synth_bars(symbols=("GBP_JPY", "EUR_JPY"), n=900, seed=7):
    rng = np.random.default_rng(seed)
    ts = pl.datetime_range(
        datetime(2026, 1, 1),
        datetime(2026, 1, 1) + __import__("datetime").timedelta(minutes=15 * (n - 1)),
        interval="15m",
        eager=True,
    )
    bars = {}
    for k, sym in enumerate(symbols):
        close = 150.0 + k + np.cumsum(rng.normal(0, 0.05, n))
        bars[sym] = pl.DataFrame(
            {
                "timestamp": ts,
                "open": close,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": np.full(n, 10.0),
            }
        )
    return bars


def production_frame(bars, *, alpha_table=None):
    """engineer_features_and_labels + the main() Phase-3a tail purge."""
    from core.retrainer import (
        RiskProfile,
        _purge_boundary_tail,
        _tail_cutoff_by_symbol,
        engineer_features_and_labels,
    )

    stacked = stack_bars(bars)
    feats, cols, chop = engineer_features_and_labels(
        stacked,
        sl_mult=2.0,
        angel_mult=1.0,
        tp_mult=4.0,
        max_hold=45,
        survival_bars=5,
        htf_timeframe="1h",
        risk_profile=RiskProfile.for_asset_class("forex"),
        alpha_table=alpha_table,
    )
    cutoffs = _tail_cutoff_by_symbol(stacked, 45)
    feats, purged = _purge_boundary_tail(feats, cutoffs)
    return feats, cols, chop, purged


class TestFrameParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bars = synth_bars()
        cls.spec = FeatureSpec(
            name="parity",
            symbols=("GBP_JPY", "EUR_JPY"),
            feature_sets=("v3_base",),
            use_spread_table=False,
        )
        cls.prod_df, cls.prod_cols, cls.prod_chop, cls.prod_purged = production_frame(
            cls.bars
        )
        cls.result = build_frame(cls.spec, cls.bars)

    def test_control_frame_is_identical_to_production(self):
        self.assertEqual(self.result.feature_cols, tuple(self.prod_cols))
        self.assertAlmostEqual(self.result.chop_veto_rate, self.prod_chop, places=12)
        self.assertEqual(self.result.purged_tail_rows, self.prod_purged)
        assert_frame_equal(
            self.result.df,
            self.prod_df,
            check_row_order=True,
            check_column_order=True,
        )

    def test_tail_purge_is_per_symbol_and_at_most_max_hold(self):
        for sym in self.spec.symbols:
            raw = self.bars[sym]["timestamp"].to_list()
            kept = self.result.df.filter(pl.col("symbol") == sym)["timestamp"].to_list()
            # The cutoff is the 45th bar from the end; every kept row is before it.
            self.assertEqual(kept[-1], raw[-46], f"{sym}: tail purge boundary wrong")
            self.assertTrue(all(ts < raw[-45] for ts in kept), sym)
        # Some of the last 45 bars per symbol were already dropped by the veto,
        # so the purge count is a ceiling, never zero on a warm frame.
        self.assertGreater(self.result.purged_tail_rows, 0)
        self.assertLessEqual(self.result.purged_tail_rows, 2 * 45)

    def test_feature_columns_are_complete_and_null_free(self):
        feats = self.result.features()
        self.assertEqual(feats.columns, list(self.result.feature_cols))
        self.assertEqual(feats.null_count().sum_horizontal()[0], 0)

    def test_content_hash_is_stable_across_builds(self):
        again = build_frame(self.spec, self.bars)
        self.assertEqual(again.content_hash, self.result.content_hash)
        self.assertEqual(again.df.height, self.result.df.height)


class TestSpreadTableParity(unittest.TestCase):
    def test_cost_ratio_and_veto_alphas_match_production(self):
        from core.retrainer import _load_spread_table

        table = _load_spread_table(_SPREAD_TABLE)["alphas"]
        bars = synth_bars(seed=11)
        spec = FeatureSpec(
            name="parity_table",
            symbols=("GBP_JPY", "EUR_JPY"),
            feature_sets=("v3_base",),
            use_spread_table=True,
            spread_table_path=_SPREAD_TABLE,
        )
        prod_df, prod_cols, prod_chop, prod_purged = production_frame(
            bars, alpha_table=table
        )
        result = build_frame(spec, bars)
        self.assertIn("cost_ratio", result.feature_cols)
        self.assertEqual(result.feature_cols, tuple(prod_cols))
        self.assertAlmostEqual(result.chop_veto_rate, prod_chop, places=12)
        self.assertEqual(result.purged_tail_rows, prod_purged)
        assert_frame_equal(
            result.df, prod_df, check_row_order=True, check_column_order=True
        )
        # The asymmetry this experiment exists to test: GBP_NZD is priced ~3x
        # AUD_JPY, so the veto must thin it more. Both symbols here are in the
        # table; assert the cost feature carries per-symbol values, not one flat
        # constant.
        per_symbol = result.df.group_by("symbol").agg(
            pl.col("cost_ratio").median().alias("med")
        )
        values = per_symbol["med"].to_list()
        self.assertGreater(len(set(round(v, 6) for v in values)), 1)


if __name__ == "__main__":
    unittest.main()
