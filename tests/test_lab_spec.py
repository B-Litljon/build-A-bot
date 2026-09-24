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
        import lab.registry as registry

        # The unregistered-family variant needs the family to exist for the
        # version lookup; register it under a throwaway name for this test.
        registry.register_feature(
            "other_family", columns=("of_one",), version=1
        )(_ExtraGen)
        self.addCleanup(registry._REGISTRY.pop, "other_family", None)

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

    def test_estimator_config_is_excluded_from_the_frame_hash(self):
        """model_family/n_folds change the estimator, not the frame — W4's A/B
        must reuse the same cached frame by construction."""
        a = FeatureSpec(name="x")
        b = FeatureSpec(name="x", gate=GateConfig(model_family="catboost"))
        c = FeatureSpec(name="x", gate=GateConfig(n_folds=5))
        self.assertEqual(a.content_hash(), b.content_hash())
        self.assertEqual(a.content_hash(), c.content_hash())

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
        self.assertNotEqual(a, b)


class _ConfigurableGen(BaseFeatureGenerator):
    """Carries constructor state into generate() — the W1 hash contract."""

    feature_cols = ("cfg_one",)

    def __init__(self, scale: float, lookback: int = 5):
        self.scale = scale
        self.lookback = lookback

    def generate(self, df):
        return df.with_columns(
            (pl.col("close") * self.scale).alias("cfg_one")
        )


class _UnhashableGen(BaseFeatureGenerator):
    feature_cols = ("unh_one",)

    def __init__(self):
        self.handle = object()  # not JSON-serializable

    def generate(self, df):
        return df


class TestGeneratorStateHashing(unittest.TestCase):
    """W1: the frame-cache key must cover generator state, not just class ids."""

    def test_registered_family_version_participates_in_the_hash(self):
        import lab.registry as registry

        spec = FeatureSpec(name="ver", feature_sets=("ver_test",))
        with self.assertRaises(KeyError):
            spec.content_hash()  # unregistered family must be loud

        registry.register_feature("ver_test", columns=("v_one",), version=1)(
            _ExtraGen
        )
        with_version_1 = spec.content_hash()
        self.assertEqual(registry.family_version("ver_test"), 1)

        # Bump the version — the exact act of editing a lookback inside the
        # generator — and the same spec must hash to a NEW cache key.
        registry.register_feature(
            "ver_test", columns=("v_one",), version=2, replace=True
        )(_ExtraGen)
        self.assertNotEqual(with_version_1, spec.content_hash())

        # Restore for other tests in this process.
        registry.register_feature(
            "ver_test", columns=("v_one",), version=1, replace=True
        )(_ExtraGen)
        self.assertEqual(with_version_1, spec.content_hash())

    def test_registration_without_a_version_raises(self):
        import lab.registry as registry

        try:
            registry.register_feature("no_version", columns=("nv_one",))(_ExtraGen)
        except (TypeError, ValueError):
            pass
        else:
            self.fail("registration without a version must raise")
        finally:
            registry._REGISTRY.pop("no_version", None)

    def test_configured_generator_args_move_the_hash(self):
        a = FeatureSpec(
            name="cfg", extra_generators=(_ConfigurableGen(scale=1.0),)
        )
        b = FeatureSpec(
            name="cfg", extra_generators=(_ConfigurableGen(scale=2.0),)
        )
        c = FeatureSpec(
            name="cfg", extra_generators=(_ConfigurableGen(scale=1.0),)
        )
        self.assertNotEqual(a.content_hash(), b.content_hash())
        self.assertEqual(a.content_hash(), c.content_hash())

    def test_same_class_different_args_do_not_collide(self):
        """The core W1 bug: two instances of one class hashed identically."""
        fast = _ConfigurableGen(scale=1.0, lookback=10)
        slow = _ConfigurableGen(scale=1.0, lookback=50)
        a = FeatureSpec(name="pair", extra_generators=(fast,))
        b = FeatureSpec(name="pair", extra_generators=(slow,))
        self.assertNotEqual(a.content_hash(), b.content_hash())

    def test_non_serializable_generator_state_raises(self):
        spec = FeatureSpec(
            name="unh", extra_generators=(_UnhashableGen(),)
        )
        with self.assertRaises(TypeError):
            spec.content_hash()

    def test_dataclass_generator_state_moves_the_hash(self):
        @dataclasses.dataclass
        class DataclassGen(BaseFeatureGenerator):
            feature_cols = ("dc_one",)
            window: int = 20

            def generate(self, df):
                return df

        a = FeatureSpec(name="dc", extra_generators=(DataclassGen(window=20),))
        b = FeatureSpec(name="dc", extra_generators=(DataclassGen(window=50),))
        self.assertNotEqual(a.content_hash(), b.content_hash())

    def test_distinct_dataclass_generator_classes_with_same_fields_do_not_collide(self):
        @dataclasses.dataclass
        class GenOne(BaseFeatureGenerator):
            feature_cols = ("dc_one",)
            window: int = 20

            def generate(self, df):
                return df

        @dataclasses.dataclass
        class GenTwo(BaseFeatureGenerator):
            feature_cols = ("dc_one",)
            window: int = 20

            def generate(self, df):
                return df

        a = FeatureSpec(name="dc", extra_generators=(GenOne(window=20),))
        b = FeatureSpec(name="dc", extra_generators=(GenTwo(window=20),))
        self.assertNotEqual(
            a.content_hash(),
            b.content_hash(),
            "Two distinct dataclass generator classes must not collide even if field values match",
        )

    def test_unhashable_dataclass_generator_state_raises(self):
        @dataclasses.dataclass
        class UnhashableDataclass(BaseFeatureGenerator):
            feature_cols = ("unh_dc",)
            unhashable: object = dataclasses.field(default_factory=object)

            def generate(self, df):
                return df

        spec = FeatureSpec(name="dc_unh", extra_generators=(UnhashableDataclass(),))
        with self.assertRaises(TypeError):
            spec.content_hash()


class _ExtraGenWithoutCols(BaseFeatureGenerator):
    def generate(self, df):
        return df


if __name__ == "__main__":
    unittest.main()
