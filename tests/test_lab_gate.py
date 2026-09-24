"""
The lab's gate wrapper: it must not invent a metric, and it must not mangle the
retrainer's process-level knobs while calling the real gate.

The real gate itself (validate_candidate + edge_over_random) is exercised
end-to-end by test_base_rate_benchmark.py; this file pins the wrapper's
contract, because the two ways it could silently lie are (a) calling the gate
with different arguments than the spec declares and (b) leaving
RETRAIN_DEVIL_LABEL mutated after the run, which would change the NEXT run's
label without the spec saying so.
"""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from lab.frames import FrameResult  # noqa: E402
from lab.gate import GateResult, require_model_family, run_gate  # noqa: E402
from lab.spec import FeatureSpec, GateConfig, LabelSpec  # noqa: E402


def make_frame() -> FrameResult:
    df = pl.DataFrame(
        {
            "timestamp": pl.datetime_range(
                __import__("datetime").datetime(2026, 1, 1),
                __import__("datetime").datetime(2026, 1, 1, 1),
                interval="15m",
                eager=True,
            ),
            "symbol": ["GBP_JPY"] * 5,
            "close": [100.0] * 5,
            "natr_14": [0.1] * 5,
            "devil_target_macro": [1, 0, 1, 0, 1],
            "feat": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    return FrameResult(
        spec_name="wiring",
        content_hash="deadbeefdeadbeef",
        df=df,
        feature_cols=("feat",),
        chop_veto_rate=0.25,
        alpha_table=None,
    )


def stub_report():
    return SimpleNamespace(
        gate_passed=False,
        edge_over_random=0.045,
        production_angel_threshold=0.3833,
    )


class TestRunGateWiring(unittest.TestCase):
    def _stub(self, captured):
        def fake_validate(df, feature_cols, **kwargs):
            captured["df"] = df
            captured["feature_cols"] = list(feature_cols)
            captured.update(kwargs)
            captured["env_label"] = os.environ.get("RETRAIN_DEVIL_LABEL")
            return (
                stub_report(),
                "angel-model",
                "devil-model",
                ["feat"],
                ["feat", "angel_prob"],
                0.41,
                None,
            )

        return fake_validate

    def test_returns_the_gate_artifacts_and_frozen_thresholds(self):
        captured = {}
        spec = FeatureSpec(name="wiring", feature_sets=("stub",))
        frame = make_frame()
        with mock.patch("core.retrainer._gate.validate_candidate", self._stub(captured)):
            result = run_gate(frame, spec)
        self.assertIsInstance(result, GateResult)
        self.assertEqual(result.angel_model, "angel-model")
        self.assertEqual(result.production_threshold, 0.41)
        self.assertEqual(result.angel_features, ["feat"])
        self.assertEqual(result.edge_over_random, 0.045)

    def test_passes_the_spec_geometry_and_the_frame_veto_rate(self):
        captured = {}
        spec = FeatureSpec(name="wiring", feature_sets=("stub",))
        with mock.patch("core.retrainer._gate.validate_candidate", self._stub(captured)):
            run_gate(make_frame(), spec)
        self.assertEqual(captured["feature_cols"], ["feat"])
        self.assertEqual(captured["sl_mult"], 2.0)
        self.assertEqual(captured["tp_mult"], 4.0)
        self.assertEqual(captured["n_folds"], 3)
        self.assertEqual(captured["chop_veto_rate"], 0.25)
        # The frame the caller built, not one rebuilt from the spec: row
        # alignment with the backtest depends on scoring exactly these rows.
        self.assertEqual(captured["df"].height, 5)
        self.assertEqual(captured["df"].columns, make_frame().df.columns)

    def test_macro_label_kind_is_scoped_to_the_run(self):
        captured = {}
        spec = FeatureSpec(
            name="wiring", feature_sets=("stub",), label=LabelSpec(kind="macro")
        )
        with mock.patch.dict(os.environ, {"RETRAIN_DEVIL_LABEL": "survival"}):
            with mock.patch(
                "core.retrainer._gate.validate_candidate", self._stub(captured)
            ):
                run_gate(make_frame(), spec)
            self.assertEqual(captured["env_label"], "macro")
            self.assertEqual(os.environ["RETRAIN_DEVIL_LABEL"], "survival")

    def test_unset_label_env_is_restored_to_unset(self):
        captured = {}
        spec = FeatureSpec(
            name="wiring", feature_sets=("stub",), label=LabelSpec(kind="macro")
        )
        env = os.environ.copy()
        env.pop("RETRAIN_DEVIL_LABEL", None)
        with mock.patch.dict(os.environ, env, clear=True):
            with mock.patch(
                "core.retrainer._gate.validate_candidate", self._stub(captured)
            ):
                run_gate(make_frame(), spec)
            self.assertNotIn("RETRAIN_DEVIL_LABEL", os.environ)

    def test_model_family_mismatch_fails_loudly(self):
        import core.retrainer._common as common

        original = common.MODEL_FAMILY
        try:
            common.MODEL_FAMILY = "catboost"
            spec = FeatureSpec(
                name="wiring",
                feature_sets=("stub",),
                gate=GateConfig(model_family="lightgbm"),
            )
            with self.assertRaises(RuntimeError) as ctx:
                require_model_family("lightgbm")
            self.assertIn("MODEL_FAMILY", str(ctx.exception))
            self.assertEqual(require_model_family("catboost"), "catboost")
        finally:
            common.MODEL_FAMILY = original

    def test_env_override_is_accepted_and_loud_but_not_spec_contract(self):
        """W4: the estimator A/B launches a lightgbm-pinned seed spec under
        MODEL_FAMILY=catboost. run_gate must run the LOADED family (the
        env-selected arm) and record it, not refuse."""
        import core.retrainer._common as common

        original = common.MODEL_FAMILY
        try:
            common.MODEL_FAMILY = "catboost"
            spec = FeatureSpec(
                name="wiring",
                feature_sets=("stub",),
                gate=GateConfig(model_family="lightgbm"),
            )
            captured = {}
            with mock.patch(
                "core.retrainer._gate.validate_candidate", self._stub(captured)
            ):
                result = run_gate(make_frame(), spec)
            self.assertEqual(result.model_family, "catboost")
            # run_gate's contract (override allowed) accepts the env arm...
            self.assertEqual(require_model_family("lightgbm", allow_env_override=True), "catboost")
            # ...while the bare spec-contract form still refuses.
            with self.assertRaises(RuntimeError):
                require_model_family("lightgbm")
        finally:
            common.MODEL_FAMILY = original


if __name__ == "__main__":
    unittest.main()
