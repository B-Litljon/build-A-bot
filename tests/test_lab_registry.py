"""
The lab's feature-family registry: explicit, ordered, and loud about unknowns.

The failure mode this file guards against is a silent one — a spec naming a
family that does not exist, or a family whose declared columns do not match what
its generators append, would score an experiment different from the one the spec
describes. Both must raise.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import lab.registry as registry  # noqa: E402
from lab.features import LabMicrostructureFeatures  # noqa: E402
from lab.spec import FeatureSpec  # noqa: E402
from ml.core.interfaces import BaseFeatureGenerator  # noqa: E402


class _OneCol(BaseFeatureGenerator):
    """A throwaway family for registry-mechanics tests."""

    feature_cols = ("one_col",)

    def generate(self, df):
        return df.with_columns(pl.lit(1.0).alias("one_col"))


class TestRegistryMechanics(unittest.TestCase):
    def test_duplicate_registration_raises(self):
        registry.register_feature("dup_test", columns=("x",))(type("Dup", (_OneCol,), {}))
        with self.assertRaises(ValueError):
            registry.register_feature("dup_test", columns=("x",))(type("Dup2", (_OneCol,), {}))
        registry.register_feature("dup_test", columns=("x",), replace=True)(
            type("Dup3", (_OneCol,), {})
        )

    def test_missing_family_raises_not_skips(self):
        spec = FeatureSpec(name="missing", feature_sets=("does_not_exist",))
        with self.assertRaises(KeyError):
            registry.get_generators(spec, None)
        with self.assertRaises(KeyError):
            registry.feature_columns(spec, None)

    def test_columns_come_from_the_declaration_not_the_instance(self):
        spec = FeatureSpec(name="unlisted", feature_sets=("unlisted_test",))
        registry.register_feature("unlisted_test", columns=("never_produced",))(_OneCol)
        self.assertEqual(registry.feature_columns(spec, None), ["never_produced"])

    def test_extra_generator_order_and_dedupe(self):
        registry.register_feature("dedupe_test", columns=("one_col", "shared"))(_OneCol)
        spec = FeatureSpec(
            name="dedupe",
            feature_sets=("dedupe_test",),
            extra_generators=(_ExtraShared(),),
        )
        cols = registry.feature_columns(spec, None)
        self.assertEqual(cols, ["one_col", "shared", "extra"])
        gens = registry.get_generators(spec, None)
        self.assertEqual(len(gens), 2)
        self.assertIsInstance(gens[1], _ExtraShared)


class _ExtraShared(BaseFeatureGenerator):
    feature_cols = ("shared", "extra")

    def generate(self, df):
        return df


class TestV3BaseFamily(unittest.TestCase):
    def test_columns_match_the_retrainer_contract(self):
        from core.retrainer._common import BASE_FEATURE_COLS

        spec = FeatureSpec(name="v3")
        self.assertEqual(registry.feature_columns(spec, None), list(BASE_FEATURE_COLS))
        with_table = registry.feature_columns(spec, {"GBP_JPY": 0.3})
        self.assertEqual(with_table, list(BASE_FEATURE_COLS) + ["cost_ratio"])

    def test_generators_append_cost_ratio_only_with_a_table(self):
        spec = FeatureSpec(name="v3")
        frame = pl.DataFrame(
            {
                "timestamp": pl.datetime_range(
                    __import__("datetime").datetime(2026, 1, 1),
                    __import__("datetime").datetime(2026, 1, 1, 2),
                    interval="15m", eager=True,
                ),
                "open": np.linspace(100, 101, 9),
                "high": np.linspace(100.1, 101.1, 9),
                "low": np.linspace(99.9, 100.9, 9),
                "close": np.linspace(100, 101, 9),
                "volume": np.full(9, 10.0),
            }
        ).with_columns(pl.lit("GBP_JPY").alias("symbol"))
        gens = registry.get_generators(spec, None)
        out = frame
        for gen in gens:
            out = gen.generate(out)
        self.assertNotIn("cost_ratio", out.columns)

        gens = registry.get_generators(spec, {"GBP_JPY": 0.3})
        out = frame
        for gen in gens:
            out = gen.generate(out)
        self.assertIn("cost_ratio", out.columns)
        self.assertTrue(out["cost_ratio"].drop_nulls().ge(0).all())


class TestMicrostructureSeed(unittest.TestCase):
    def test_declared_columns(self):
        self.assertEqual(
            LabMicrostructureFeatures.feature_cols,
            ("ms_close_pos_20", "ms_ret_z_10", "ms_updown_vol_10", "ms_autocorr_20"),
        )

    def test_requires_v3_features_first(self):
        bare = pl.DataFrame({"close": [1.0], "high": [1.0], "low": [1.0]})
        with self.assertRaises(ValueError):
            LabMicrostructureFeatures().generate(bare)

    def test_computes_all_declared_columns_per_symbol(self):
        n = 80
        rng = np.random.default_rng(3)
        frames = []
        for sym in ("A", "B"):
            close = 100 + np.cumsum(rng.normal(0, 0.1, n))
            frames.append(
                pl.DataFrame(
                    {
                        "close": close,
                        "high": close + 0.1,
                        "low": close - 0.1,
                        "log_return": np.concatenate([[0.0], np.diff(np.log(close))]),
                        "symbol": [sym] * n,
                    }
                )
            )
        out = LabMicrostructureFeatures().generate(pl.concat(frames))
        for col in LabMicrostructureFeatures.feature_cols:
            self.assertIn(col, out.columns)
            # The first row of each symbol's block must stay null (no lookahead,
            # warm-up honest) rather than being filled with an invented value.
            for sym in ("A", "B"):
                first = out.filter(pl.col("symbol") == sym)[col][0]
                self.assertIsNone(first, f"{col} leaked a value into {sym}'s first row")
        self.assertEqual(out.height, 2 * n)


if __name__ == "__main__":
    unittest.main()
