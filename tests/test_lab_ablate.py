"""
lab ablate: feature-INTERACTION ablation — the polypharmacy question.

The unit of evidence here is edge(full cocktail) − edge(cocktail − X), with
only feature_sets varying. The failure modes pinned in this file, all silent
in production terms:

* a variant that silently changed labels, geometry or data along with the
  family set would measure a different EXPERIMENT, not a family effect;
* a point delta without its Clopper-Pearson interval would read thin-sample
  noise as a finding — the exact +0.179-on-30-trades artifact the
  2026-09-14 work withdrew;
* a single-family spec has no cocktail to subtract against and must refuse
  loudly, not "ablate" into itself.
"""

import sys
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from lab.ablate import (  # noqa: E402
    AblationVariant,
    CP_CONFIDENCE,
    THIN_TRADES,
    delta_with_ci,
    expand_variants,
)
from lab.spec import (  # noqa: E402
    DEFAULT_TRADEABLE_6,
    FeatureSpec,
    GeometrySpec,
    LabelSpec,
)
from ml.core.interfaces import BaseFeatureGenerator  # noqa: E402


class _ThrowawayGen(BaseFeatureGenerator):
    """Registry-mechanics stub; never generates — specs are not built here."""

    feature_cols = ("throwaway_one",)

    def generate(self, df):
        return df


class TestExpandVariants(unittest.TestCase):
    def setUp(self):
        """The variant-name tests need registered families; the content-hash
        lookup (W1) resolves (name, version) through the registry."""
        import lab.registry as registry

        self._registry = registry
        for name in ("third", "fourth", "microstructure"):
            if name not in registry._REGISTRY:
                registry.register_feature(name, columns=(f"{name}_one",), version=1)(
                    _ThrowawayGen
                )
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        for name in ("third", "fourth", "microstructure"):
            # microstructure may be the real registered family in another
            # test's process ordering — only pop what THIS class registered.
            if name in self._registry._REGISTRY and name != "microstructure":
                self._registry._REGISTRY.pop(name, None)

    def test_expands_to_n_plus_1_specs_with_the_right_family_omitted(self):
        spec = FeatureSpec(
            name="ab",
            feature_sets=("v3_base", "microstructure", "third"),
        )
        full, variants = expand_variants(spec)
        self.assertIs(full, spec)
        self.assertEqual(len(variants), 3)
        self.assertEqual(
            [v.feature_sets for v in variants],
            [
                ("microstructure", "third"),
                ("v3_base", "third"),
                ("v3_base", "microstructure"),
            ],
        )
        self.assertEqual(
            [v.name for v in variants],
            ["ab_minus_v3_base", "ab_minus_microstructure", "ab_minus_third"],
        )

    def test_everything_but_feature_sets_is_unchanged(self):
        """Labels, veto, geometry, data, cost table: identical across arms."""
        spec = FeatureSpec(
            name="ab",
            symbols=("GBP_JPY", "EUR_JPY"),
            days_back=365,
            granularity=60,
            feature_sets=("v3_base", "microstructure"),
            geometry=GeometrySpec(1.5, 3.0, 30),
            label=LabelSpec(survival_bars=10, kind="macro"),
            use_spread_table=True,
            extra_generators=(),
        )
        _, variants = expand_variants(spec)
        for v in variants:
            self.assertEqual(v.symbols, spec.symbols)
            self.assertEqual(v.days_back, spec.days_back)
            self.assertEqual(v.granularity, spec.granularity)
            self.assertEqual(v.geometry, spec.geometry)
            self.assertEqual(v.label, spec.label)
            self.assertEqual(v.use_spread_table, spec.use_spread_table)
            self.assertEqual(v.spread_table_path, spec.spread_table_path)
            self.assertEqual(v.extra_generators, spec.extra_generators)

    def test_single_family_spec_refuses(self):
        spec = FeatureSpec(name="solo", feature_sets=("v3_base",))
        with self.assertRaises(ValueError) as ctx:
            expand_variants(spec)
        self.assertIn(">= 2 feature families", str(ctx.exception))
        self.assertIn("`run`", str(ctx.exception))

    def test_extra_generators_do_not_count_as_families(self):
        """A spec whose only registered family is v3_base but which adds an
        extra generator has one family — still no cocktail to subtract."""

        class _Tail(BaseFeatureGenerator):
            feature_cols = ("tail_one",)

            def generate(self, df):
                return df

        spec = FeatureSpec(
            name="solo_tail",
            feature_sets=("v3_base",),
            extra_generators=(_Tail(),),
        )
        with self.assertRaises(ValueError):
            expand_variants(spec)


class TestDeltaWithCI(unittest.TestCase):
    def test_delta_is_full_minus_minus_x(self):
        d = delta_with_ci(20, 30, 15, 30)
        self.assertAlmostEqual(d["delta"], 1 / 6, places=6)

    def test_interval_is_reported_and_spans_the_point(self):
        d = delta_with_ci(20, 30, 15, 30)
        self.assertLessEqual(d["ci_low"], d["delta"])
        self.assertLessEqual(d["delta"], d["ci_high"])

    def test_huge_sample_distinguishable_from_zero(self):
        """30 vs 30 wins in 1000 trades: a 3pp delta is real at this n."""
        d = delta_with_ci(330, 1000, 300, 1000)
        self.assertTrue(d["distinguishable_from_zero"])
        self.assertGreater(d["ci_low"], 0.0)

    def test_thin_trades_delta_consistent_with_zero(self):
        """The v3_base regime: 30-ish trades, a delta spanning zero."""
        d = delta_with_ci(20, 30, 17, 28)
        self.assertFalse(d["distinguishable_from_zero"])
        self.assertLessEqual(d["ci_low"], 0.0)
        self.assertGreaterEqual(d["ci_high"], 0.0)

    def test_identical_arms_give_delta_zero(self):
        d = delta_with_ci(15, 30, 15, 30)
        self.assertAlmostEqual(d["delta"], 0.0, places=12)
        self.assertFalse(d["distinguishable_from_zero"])

    def test_empty_arm_degrades_gracefully(self):
        d = delta_with_ci(0, 0, 5, 10)
        self.assertNotEqual(d["delta"], d["delta"])  # nan point estimate
        self.assertFalse(d["distinguishable_from_zero"])

    def test_perfect_arm_bounds_are_finite(self):
        """CP's edge cases: a perfect arm vs a dead one pins the delta at 1.0,
        so the clipped interval touches (not exceeds) the feasible ceiling."""
        d = delta_with_ci(10, 10, 0, 10)
        self.assertTrue(d["distinguishable_from_zero"])
        self.assertEqual(d["ci_high"], 1.0)
        self.assertGreater(d["ci_low"], 0.0)

    def test_cp_bounds_match_the_gate_conservatism(self):
        """A perfect 3-for-3 arm must NOT clear a wide interval's floor —
        the same exactness the gate's PF bounds use (Wilson would over-claim
        here; CP is the chosen convention)."""
        from scipy.stats import beta as _scipy_beta

        d = delta_with_ci(3, 3, 0, 3)
        expected_full_lo = float(_scipy_beta.ppf(1 - CP_CONFIDENCE, 3, 1))
        # ci_low is delta minus the quadrature width; reconstruct the full
        # arm's own lower bound and compare it to the gate's exact value.
        width = d["ci_high"] - d["ci_low"]
        full_half = (d["ci_high"] - d["delta"]) + (d["delta"] - d["ci_low"])
        # full_half = minus_half = width/sqrt(2) when only one arm is perfect
        # is NOT assumed: verify via the module's own bound helper instead.
        from lab.ablate import _cp_win_rate_bound

        self.assertAlmostEqual(
            _cp_win_rate_bound(3, 3, upper=False), expected_full_lo, places=9
        )
        self.assertAlmostEqual(d["full_wr"], 1.0, places=9)
        self.assertAlmostEqual(d["minus_wr"], 0.0, places=9)


class TestAblationResult(unittest.TestCase):
    def setUp(self):
        import lab.registry as registry

        self._to_pop = []
        for name in ("nothing", "microstructure"):
            if name not in registry._REGISTRY:
                registry.register_feature(name, columns=(f"{name}_one",), version=1)(
                    _ThrowawayGen
                )
                self._to_pop.append(name)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        import lab.registry as registry

        for name in self._to_pop:
            registry._REGISTRY.pop(name, None)

    def _variant(self, family, wins, trades, edge=0.02):
        from types import SimpleNamespace

        report = SimpleNamespace(
            pooled_oos_trades=trades,
            pooled_oos_wins=wins,
            edge_over_random=edge,
            pooled_pf_lower_bound=1.1,
            fold3_pf_lower_bound=0.9,
            gate_passed=False,
        )
        spec = FeatureSpec(
            name=f"ab_minus_{family}", feature_sets=("v3_base", family)
        )
        return AblationVariant(
            family=family,
            spec=spec,
            gate=SimpleNamespace(
                report=report,
                model_family="lightgbm",
                production_threshold=0.5,
            ),
            frame=None,
            frame_from_cache=True,
        )

    def test_summary_carries_delta_with_interval_per_family(self):
        from types import SimpleNamespace

        full = self._variant("nothing", 20, 30)
        var = self._variant("microstructure", 17, 28)
        result = SimpleNamespace(spec=FeatureSpec(name="ab"), full=full, variants=[var])
        from lab.ablate import AblationResult

        result = AblationResult(
            spec=result.spec, full=full, variants=[var],
            deltas={"microstructure": delta_with_ci(20, 30, 17, 28)},
            run_seconds=1.0,
        )
        summary = result.summary()
        self.assertEqual(summary["full"]["pooled_oos_trades"], 30)
        self.assertEqual(len(summary["variants"]), 1)
        row = summary["variants"][0]
        self.assertEqual(row["family"], "microstructure")
        self.assertIn("delta", row)
        self.assertIn("ci_low", row["delta"])
        self.assertIn("ci_high", row["delta"])

    def test_ablate_orchestrator_runs_all_arms_and_computes_deltas(self):
        from unittest import mock
        from types import SimpleNamespace
        from lab.ablate import ablate

        spec = FeatureSpec(name="ab_orch", feature_sets=("nothing", "microstructure"))

        class FakeRunner:
            def prepare_frame(self, sp):
                return SimpleNamespace(df=None, feature_cols=["c1"]), True

        call_records = []

        def fake_run_gate(frame, sp, *, n_folds=None):
            call_records.append((sp.name, n_folds))
            trades = 30 if "minus" not in sp.name else 25
            wins = 15 if "minus" not in sp.name else 10
            rep = SimpleNamespace(
                pooled_oos_trades=trades,
                pooled_oos_wins=wins,
                edge_over_random=0.1,
                pooled_pf_lower_bound=0.8,
                fold3_pf_lower_bound=0.6,
                gate_passed=False,
            )
            return SimpleNamespace(
                report=rep,
                model_family="lightgbm",
                production_threshold=0.5,
            )

        with mock.patch("lab.gate.run_gate", fake_run_gate):
            res = ablate(spec, runner=FakeRunner(), n_folds=3)

        self.assertEqual(len(call_records), 3)  # full + 2 minus-X variants
        self.assertEqual(call_records[0][0], "ab_orch")
        self.assertEqual(call_records[0][1], 3)
        self.assertEqual(res.full.family, "full")
        self.assertEqual(len(res.variants), 2)
        self.assertIn("nothing", res.deltas)
        self.assertIn("microstructure", res.deltas)
        self.assertAlmostEqual(res.deltas["nothing"]["delta"], 15 / 30 - 10 / 25)

    def test_cli_ablate_parser_recognizes_spec_arguments(self):
        import lab.cli as cli_mod
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        ablate_p = sub.add_parser("ablate")
        cli_mod._add_spec_args(ablate_p)
        args = ablate_p.parse_args(["--name", "v3_base_control", "--no-report"])
        self.assertEqual(args.name, "v3_base_control")
        self.assertTrue(args.no_report)


if __name__ == "__main__":
    unittest.main()