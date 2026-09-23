"""
Served-artifact replay: pinned bars, fit-time schema, and parity with the gate.

Three failure modes are pinned here, all silent in production terms:

* **bars.** A replay that used the gate's calibrated thresholds instead of the
  served ``threshold.json`` would answer a question nobody asked; the live bot
  runs the pinned pair, so the loader's precedence is pinned.
* **schema order.** LightGBM predicts positionally against a numpy array, so a
  replay that passed the frame's column order would score every bar against
  permuted features. The artifact's ``feature_names_in_`` order is load-bearing.
* **divergence.** The artifact path and the gate path share one replay body; if
  they ever produced different trades for the same models, the baseline would
  not be comparable to the seed runs it exists to compare against.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import joblib
import numpy as np
import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from core.thresholds import ANGEL_THRESHOLD  # noqa: E402
from lab.artifact import (  # noqa: E402
    ServedArtifact,
    load_served_artifact,
    predict_probabilities,
    replay_served_artifact,
)
from lab.backtest import run_artifact_backtest, run_model_backtest  # noqa: E402
from lab.frames import FrameResult  # noqa: E402
from lab.report import render_artifact_report, write_artifact_report  # noqa: E402
from lab.spec import FeatureSpec  # noqa: E402


class ConstantModel:
    def __init__(self, prob: float):
        self.prob = float(prob)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.prob), np.full(n, self.prob)])


class FirstColumnModel(ConstantModel):
    """Probability = the first input column, exposing any column scramble."""

    def predict_proba(self, X):
        values = X[:, 0] if not hasattr(X, "columns") else X[:, 0].to_numpy()
        return np.column_stack([1 - values, values])


def replay_frame(n=120, start=datetime(2026, 1, 1)):
    """One up-trending symbol; labels alternate so a base rate exists."""
    ts = pl.datetime_range(
        start, start + timedelta(minutes=15 * (n - 1)), interval="15m", eager=True
    )
    close = 100.0 + np.arange(n) * 0.5
    return pl.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["GBP_JPY"] * n,
            "f": np.full(n, 1.0),
            "g": np.full(n, 0.9),
            "open": close,
            "high": close + 1.2,
            "low": close - 1.2,
            "close": close,
            "natr_14": np.full(n, 1.0),
            "devil_target_macro": np.tile([1, 0], n // 2),
        }
    )


def frame_result(df, feature_cols=("f",)) -> FrameResult:
    return FrameResult(
        spec_name="artifact-bt",
        content_hash="deadbeefdeadbeef",
        df=df,
        feature_cols=feature_cols,
        chop_veto_rate=0.0,
        alpha_table=None,
    )


def artifact_for(
    df, *, angel=None, devil=None, angel_features=("f",), devil_features=("f", "angel_prob"),
    angel_threshold=0.5, devil_threshold=0.5, metadata=None,
) -> ServedArtifact:
    return ServedArtifact(
        model_dir=Path("models/test"),
        angel_model=angel or ConstantModel(1.0),
        devil_model=devil or ConstantModel(1.0),
        angel_threshold=angel_threshold,
        devil_threshold=devil_threshold,
        threshold_source="test",
        angel_feature_cols=tuple(angel_features),
        devil_feature_cols=tuple(devil_features),
        metadata=metadata or {},
    )


def fake_gate():
    return SimpleNamespace(
        angel_model=ConstantModel(1.0),
        devil_model=ConstantModel(1.0),
        production_threshold=0.5,
        report=SimpleNamespace(production_angel_threshold=0.5),
    )


def stub_spec(**kwargs) -> FeatureSpec:
    return FeatureSpec(name="artifact-bt", symbols=("GBP_JPY",), feature_sets=("stub",), **kwargs)


class StubRunner:
    def __init__(self, frame):
        self._frame = frame

    def prepare_frame(self, spec):
        return self._frame, True


class TestLoadServedArtifact(unittest.TestCase):
    def _write_dir(self, tmp, *, threshold=True, metadata=True, angel_schema=True,
                   devil_features=("f", "angel_prob")):
        root = Path(tmp)
        angel = ConstantModel(1.0)
        if angel_schema:
            angel.feature_names_in_ = ["f"]
        devil = ConstantModel(1.0)
        devil.feature_names_in_ = list(devil_features)
        joblib.dump(angel, root / "angel_latest.pkl")
        joblib.dump(devil, root / "devil_latest.pkl")
        if threshold:
            (root / "threshold.json").write_text(
                json.dumps({"angel_threshold": 0.3833, "devil_threshold": 0.44})
            )
        if metadata:
            (root / "metadata.json").write_text(
                json.dumps({"trained_at": "2026-08-30T02:20:16+00:00"})
            )
        return root

    def test_threshold_json_pins_the_live_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = load_served_artifact(str(self._write_dir(tmp)))
        self.assertAlmostEqual(artifact.angel_threshold, 0.3833, places=6)
        self.assertAlmostEqual(artifact.devil_threshold, 0.44, places=6)
        self.assertEqual(artifact.threshold_source, "threshold.json")
        self.assertEqual(artifact.trained_at, "2026-08-30T02:20:16+00:00")
        self.assertEqual(artifact.angel_feature_cols, ("f",))

    def test_missing_threshold_json_falls_back_to_constants(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = load_served_artifact(str(self._write_dir(tmp, threshold=False)))
        self.assertAlmostEqual(artifact.angel_threshold, ANGEL_THRESHOLD, places=6)
        self.assertAlmostEqual(artifact.devil_threshold, 0.50, places=6)
        self.assertEqual(artifact.threshold_source, "constants")

    def test_model_without_a_feature_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_dir(tmp, angel_schema=False)
            with self.assertRaisesRegex(ValueError, "no feature_names_in_"):
                load_served_artifact(str(root))

    def test_catboost_feature_names_spelling_is_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_dir(tmp)
            devil = ConstantModel(1.0)
            devil.feature_names_ = ["f", "angel_prob"]  # no "in_"
            joblib.dump(devil, root / "devil_latest.pkl")
            artifact = load_served_artifact(str(root))
        self.assertEqual(artifact.devil_feature_cols, ("f", "angel_prob"))

    def test_devil_schema_beyond_the_angel_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._write_dir(tmp, devil_features=("f", "angel_prob", "mystery"))
            with self.assertRaisesRegex(ValueError, "mystery"):
                load_served_artifact(str(root))

    def test_incomplete_dir_is_loud(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                load_served_artifact(tmp)


class TestPredictProbabilities(unittest.TestCase):
    def test_columns_are_read_in_the_artifacts_own_order(self):
        df = pl.DataFrame({"f": [0.2, 0.2], "g": [0.9, 0.9]})
        model = FirstColumnModel(0.0)
        model.feature_names_in_ = ["g", "f"]
        artifact = artifact_for(df, angel=model, angel_features=("g", "f"))
        angel, _ = predict_probabilities(df, artifact)
        np.testing.assert_allclose(angel, [0.9, 0.9])


class TestReplayParity(unittest.TestCase):
    """The artifact path and the gate path must agree when given one model pair."""

    def test_same_models_same_trades(self):
        df = replay_frame()
        frame = frame_result(df)
        artifact = artifact_for(df)
        gate = fake_gate()
        with mock.patch.dict(os.environ, {"RISK_CHOP_FILTER_ENABLED": "0"}):
            artifact_report = run_artifact_backtest(frame, stub_spec(), artifact)
            gate_report = run_model_backtest(frame, gate, stub_spec())
        self.assertEqual(artifact_report.total_trades, gate_report.total_trades)
        self.assertAlmostEqual(artifact_report.net_ev_r, gate_report.net_ev_r, places=12)
        self.assertEqual(artifact_report.gate_rejections, gate_report.gate_rejections)


class TestReplayServedArtifact(unittest.TestCase):
    def test_missing_frame_feature_is_loud(self):
        frame = frame_result(replay_frame(), feature_cols=("f",))
        artifact = artifact_for(frame.df, angel_features=("not_in_frame",),
                                devil_features=("not_in_frame", "angel_prob"))
        with self.assertRaisesRegex(ValueError, "not_in_frame"):
            replay_served_artifact(stub_spec(), artifact, runner=StubRunner(frame))

    def test_population_counts_and_window_split(self):
        # 200 bars = 50h starting Dec 31 noon: all three windows have rows.
        start = datetime(2025, 12, 31, 12, tzinfo=timezone.utc)
        df = replay_frame(n=200, start=start)
        frame = frame_result(df, feature_cols=("f", "g"))
        artifact = artifact_for(
            df,
            metadata={
                "holdout": {"start_date": "2026-01-01", "end_date": "2026-01-01"}
            },
        )
        with mock.patch.dict(os.environ, {"RISK_CHOP_FILTER_ENABLED": "0"}):
            result = replay_served_artifact(
                stub_spec(), artifact, runner=StubRunner(frame)
            )
        population = result.population
        self.assertEqual(population["proposed"], df.height)
        self.assertEqual(population["approved"], df.height)
        self.assertAlmostEqual(population["approved_win_rate"], population["base_rate"])
        self.assertAlmostEqual(population["edge_over_random"], 0.0, places=12)
        labels = [window["label"] for window in result.windows]
        self.assertEqual(len(labels), 3)
        self.assertIn("recorded holdout", labels[1])
        self.assertEqual(
            sum(window["rows"] for window in result.windows), df.height
        )
        self.assertGreater(result.windows[0]["rows"], 0)


class TestArtifactReport(unittest.TestCase):
    def _result(self):
        from lab.artifact import ArtifactReplayResult

        df = replay_frame()
        artifact = artifact_for(df, metadata={"trained_at": "2026-08-30T02:20:16+00:00"})
        frame = frame_result(df)
        backtest = SimpleNamespace(
            total_trades=12,
            wins=5,
            win_rate=5 / 12,
            gross_ev_r=0.4,
            net_ev_r=0.31,
            profit_factor_net=1.4,
            max_drawdown_r=3.0,
            gate_rejections={"regime": 4},
            toll_mode="flat",
            per_symbol={},
        )
        return ArtifactReplayResult(
            spec=stub_spec(),
            artifact=artifact,
            frame=frame,
            backtest=backtest,
            population={
                "rows": df.height,
                "base_rate": 0.5,
                "proposed": 20,
                "approved": 12,
                "approved_wins": 5,
                "approved_win_rate": 5 / 12,
                "edge_over_random": -0.0833,
                "profit_factor": 1.0,
            },
            windows=[
                {
                    "label": "2026-01-01 to 2026-01-01 (recorded holdout)",
                    "rows": 60,
                    "base_rate": 0.5,
                    "proposed": 10,
                    "approved": 6,
                    "approved_wins": 3,
                    "approved_win_rate": 0.5,
                    "edge_over_random": 0.0,
                    "profit_factor": 1.0,
                    "trades": 3,
                    "trade_wins": 1,
                    "trade_win_rate": 1 / 3,
                    "net_ev_r": -0.2,
                    "gross_ev_r": -0.1,
                }
            ],
            frame_from_cache=True,
            run_seconds=3.2,
        )

    def test_replay_caveats_are_emitted(self):
        text = render_artifact_report(self._result(), command="cmd")
        self.assertIn("## Caveats", text)
        self.assertIn("Flat toll", text)
        self.assertIn("Replay, not a promotion gate", text)
        self.assertIn("recorded holdout", text)
        self.assertIn("12", text)

    def test_write_uses_the_served_artifact_slug(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = write_artifact_report(self._result(), out_dir=Path(tmp), command="cmd")
            self.assertTrue(out.is_file())
            self.assertIn("lab-served-artifact-artifact-bt", out.name)
            self.assertEqual(list(Path(tmp).glob("*.tmp")), [])
            text = out.read_text()
            self.assertIn("Angel 0.5000 / Devil 0.5000", text)


if __name__ == "__main__":
    unittest.main()
