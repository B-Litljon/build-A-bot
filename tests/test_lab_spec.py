"""
The feature lab's FeatureSpec: frozen, validated, and content-hashed.

The hash is the frame cache key, so its contract is load-bearing: the same spec
must hash the same inside a process (and across processes), and any field that
changes the frame must change the hash. A stale hit would silently score a
different experiment than the spec names.

Light by design: this file must not import the training stack.
"""

import dataclasses
import sys
import unittest
from pathlib import Path

import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from lab.spec import (  # noqa: E402
    DEFAULT_TRADEABLE_6,
    FeatureSpec,
    GateConfig,
    GeometrySpec,
    LabelSpec,
)
from ml.core.interfaces import BaseFeatureGenerator  # noqa: E402


class _ExtraGen(BaseFeatureGenerator):
    feature_cols = ("extra_one",)

    def generate(self, df):
        return df.with_columns(pl.lit(1.0).alias("extra_one"))


class TestFeatureSpec(unittest.TestCase):
    def test_frozen(self):
        spec = FeatureSpec(name="frozen")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            spec.name = "changed"

    def test_defaults_are_the_served_configuration_shape(self):
        spec = FeatureSpec()
        self.assertEqual(spec.symbols, DEFAULT_TRADEABLE_6)
        self.assertEqual(spec.granularity, 15)
        self.assertEqual(spec.feature_sets, ("v3_base",))
        self.assertEqual(
            (spec.geometry.sl_mult, spec.geometry.tp_mult, spec.geometry.max_hold),
            (2.0, 4.0, 45),
        )
        self.assertEqual(spec.label.kind, "survival")
        self.assertFalse(spec.use_spread_table)
        self.assertEqual(spec.gate.model_family, "lightgbm")

    def test_same_spec_same_hash(self):
        a = FeatureSpec(name="same")
        b = FeatureSpec(name="same")
        self.assertEqual(a.content_hash(), b.content_hash())
        self.assertEqual(len(a.content_hash()), 16)

    def test_every_frame_affecting_change_moves_the_hash(self):
        base = FeatureSpec(name="base")
        variants = {
            "geometry": FeatureSpec(name="base", geometry=GeometrySpec(1.0, 2.0, 30)),
            "survival": FeatureSpec(name="base", label=LabelSpec(survival_bars=10)),
            "angel": FeatureSpec(name="base", label=LabelSpec(angel_mult=0.5)),
            "features": FeatureSpec(name="base", feature_sets=("other_family",)),
            "symbols": FeatureSpec(name="base", symbols=("GBP_JPY",)),
            "days": FeatureSpec(name="base", days_back=365),
            "granularity": FeatureSpec(name="base", granularity=60),
            "htf": FeatureSpec(name="base", htf_timeframe="4h"),
            "cost_table": FeatureSpec(name="base", use_spread_table=True),
            "name": FeatureSpec(name="other"),
            "extra": FeatureSpec(name="base", extra_generators=(_ExtraGen(),)),
        }
        baseline = base.content_hash()
        for label, spec in variants.items():
            with self.subTest(variant=label):
                self.assertNotEqual(
                    baseline, spec.content_hash(), f"{label} did not change the hash"
                )

    def test_schema_version_participates_in_the_hash(self):
        """A serialization change must invalidate every cached frame."""
        import lab.spec as spec_mod

        spec = FeatureSpec(name="hash-pin")
        before = spec.content_hash()
        original = spec_mod._SPEC_SCHEMA_VERSION
        try:
            spec_mod._SPEC_SCHEMA_VERSION = original + 1
            self.assertNotEqual(before, spec.content_hash())
        finally:
            spec_mod._SPEC_SCHEMA_VERSION = original
        self.assertEqual(before, spec.content_hash())

    def test_validation_rejects_empty_experiments(self):
        with self.assertRaises(ValueError):
            FeatureSpec(name="  ")
        with self.assertRaises(ValueError):
            FeatureSpec(name="x", symbols=())
        with self.assertRaises(ValueError):
            FeatureSpec(name="x", feature_sets=())
        with self.assertRaises(ValueError):
            FeatureSpec(name="x", days_back=0)
        with self.assertRaises(ValueError):
            FeatureSpec(name="x", label=LabelSpec(kind="bracket"))
        with self.assertRaises(ValueError):
            FeatureSpec(name="x", extra_generators=(_ExtraGenWithoutCols(),))

    def test_alpha_table_switch(self):
        off = FeatureSpec(name="off", use_spread_table=False)
        self.assertIsNone(off.alpha_table())
        on = FeatureSpec(
            name="on",
            use_spread_table=True,
            spread_table_path="config/spread_alphas_m15.json",
        )
        table = on.alpha_table()
        self.assertIn("GBP_NZD", table)
        self.assertAlmostEqual(table["GBP_NZD"], 0.8929, places=4)

    def test_gate_config_is_part_of_the_spec(self):
        a = FeatureSpec(name="x")
        b = FeatureSpec(name="x", gate=GateConfig(model_family="catboost"))
        self.assertNotEqual(a.content_hash(), b.content_hash())


class _ExtraGenWithoutCols(BaseFeatureGenerator):
    def generate(self, df):
        return df


if __name__ == "__main__":
    unittest.main()
