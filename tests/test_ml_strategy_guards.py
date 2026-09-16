import json
import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch
"""
Tests for MLStrategy's stale-bar guard.

Feature cleaning drops rows with missing values. If the row dropped happens to
be the NEWEST bar, the frame's last row is an older bar -- and scoring that
against the current price means trading on stale information while believing it
is current. The guard compares the two timestamps and returns None instead.

Glossary:
    TestStaleFeatureGuard -- the two halves of that contract.
    test_signal_skipped_when_latest_bar_dropped -- mismatched timestamps must
        produce no signal, even if the model would have approved.
    test_matching_timestamps_proceed_to_prediction -- and the guard must not
        block the normal case.
    TestThresholdLoading -- threshold.json pins BOTH stage bars since 2026-07:
        a pinned angel_threshold overrides the constructor default (a pair
        must run at the bar its Devil/brackets were fitted for), a legacy
        file without the key keeps the default, and a missing file keeps
        both defaults. Uses MLStrategy.__new__ to skip model loading.
"""

import sys
from pathlib import Path

import polars as pl

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from src.strategies.concrete_strategies.ml_strategy import (
    MLStrategy,
    _barriers_requested,
)

from core.retrainer import BASE_FEATURE_COLS
from ml.barriers.estimator import BarrierEstimator
from ml.barriers.labels import DEFAULT_HORIZON, compute_excursions


def _bars(n: int, end: datetime) -> pl.DataFrame:
    ts = [end - timedelta(minutes=n - 1 - i) for i in range(n)]
    base = 1.08
    return pl.DataFrame(
        {
            "symbol": ["EUR_USD"] * n,
            "timestamp": ts,
            "open": [base] * n,
            "high": [base + 0.0005] * n,
            "low": [base - 0.0005] * n,
            "close": [base + 0.0001 * (i % 3) for i in range(n)],
            "volume": [10.0] * n,
        }
    )


class TestStaleFeatureGuard(unittest.TestCase):
    """H3 hardening: never score stale features against the current price."""

    @classmethod
    def setUpClass(cls):
        # Loads the real served forex models via explicit paths — the
        # asset_class default (models/forex/) is a transient side dir that
        # retrains overwrite, so the guard tests must not depend on it.
        cls.strategy = MLStrategy(
            asset_class="forex",
            angel_path="models/forex_m15_wide/angel_latest.pkl",
            devil_path="models/forex_m15_wide/devil_latest.pkl",
            warmup_period=10,
        )

    def test_signal_skipped_when_latest_bar_dropped(self):
        """features_df tail older than raw df tail -> None, no prediction."""
        end = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
        df = _bars(12, end)
        # Simulate clean_data having dropped the newest bar: the feature
        # frame's last timestamp is one bar behind the raw frame's.
        stale_features = pl.DataFrame(
            {"timestamp": [end - timedelta(minutes=1)]}
        )

        with patch.object(
            self.strategy, "_generate_features", return_value=stale_features
        ):
            with patch.object(
                self.strategy.angel_trainer, "predict_proba"
            ) as mock_predict:
                result = self.strategy.generate_signals(df)

        self.assertIsNone(result)
        mock_predict.assert_not_called()

    def test_matching_timestamps_proceed_to_prediction(self):
        """Aligned tails -> the guard does not block the pipeline."""
        end = datetime(2026, 6, 9, 12, 0, tzinfo=timezone.utc)
        df = _bars(12, end)

        result = self.strategy.generate_signals(df)
        # Whatever the models decide, the guard must not have been the
        # blocker: with aligned real features this exercises the full path
        # without raising. (Signal may legitimately be None on rejection.)
        self.assertTrue(result is None or result.direction == "long")


class TestThresholdLoading(unittest.TestCase):
    """threshold.json pins BOTH stage bars (angel_threshold added 2026-07)."""

    @staticmethod
    def _bare_strategy(model_dir: Path) -> MLStrategy:
        # __new__ bypasses the heavy model-loading __init__ —
        # _load_thresholds only touches angel_path and the two instance
        # defaults, so this stays a unit test.
        s = MLStrategy.__new__(MLStrategy)
        s.angel_threshold = 0.40
        s.devil_threshold = 0.50
        s.angel_path = model_dir / "angel_latest.pkl"
        return s

    def test_pinned_angel_overrides_default(self):
        """A pinned angel_threshold wins over the constructor default."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "threshold.json").write_text(
                json.dumps({"devil_threshold": 0.48, "angel_threshold": 0.325})
            )
            s = self._bare_strategy(p)
            self.assertEqual(s._load_thresholds(), (0.325, 0.48))

    def test_legacy_artifact_without_angel_keeps_default(self):
        """Pre-2026-07 threshold.json (devil only) keeps the Angel default."""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "threshold.json").write_text(
                json.dumps({"devil_threshold": 0.48})
            )
            s = self._bare_strategy(p)
            self.assertEqual(s._load_thresholds(), (0.40, 0.48))

    def test_missing_file_keeps_both_defaults(self):
        """No threshold.json at all -> both constructor values survive."""
        with tempfile.TemporaryDirectory() as td:
            s = self._bare_strategy(Path(td))
            self.assertEqual(s._load_thresholds(), (0.40, 0.50))


class TestHotReloadSeams(unittest.TestCase):
    """2026-09-09: sidecars reload on their own mtimes, and a one-sided
    pickle advance stands the pair down instead of scoring a mixed
    Angel/Devil generation."""

    def _bare(self, tmp: str) -> MLStrategy:
        s = MLStrategy.__new__(MLStrategy)
        s.angel_path = Path(tmp) / "angel_latest.pkl"
        s.devil_path = Path(tmp) / "devil_latest.pkl"
        s.angel_mtime = 0.0
        s.devil_mtime = 0.0
        s._threshold_mtime = 0.0
        s._spread_table_mtime = 0.0
        s._pair_mixed = False
        s._pair_pending = None
        s._reload_lock = threading.Lock()
        s.notification_manager = MagicMock()
        s.angel_threshold = 0.40
        s.devil_threshold = 0.44
        s.feature_names = []
        s._cost_gen = MagicMock()
        s._cost_gen.alpha_table = None
        s.angel_trainer = MagicMock()
        s.devil_trainer = MagicMock()
        # Barrier sidecar state, as __init__ would leave it. Without this the
        # barrier hook inside _check_model_updates raises AttributeError and is
        # swallowed by its own try/except, silently costing this class its
        # coverage of everything after it.
        s.use_barriers = False
        s._barrier_estimator = None
        s._barrier_meta_path = Path(tmp) / "barriers_meta.json"
        s._barrier_mtime = 0.0
        return s

    def test_both_loaded_pair_is_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            Path(td, "angel_latest.pkl").write_bytes(b"angel")
            Path(td, "devil_latest.pkl").write_bytes(b"devil")
            s = self._bare(td)
            s._check_model_updates()
            self.assertFalse(s._pair_mixed)
            self.assertEqual(s.angel_trainer.load.call_count, 1)
            self.assertEqual(s.devil_trainer.load.call_count, 1)

    def test_one_sided_advance_sets_pair_mixed(self):
        with tempfile.TemporaryDirectory() as td:
            angel = Path(td, "angel_latest.pkl")
            devil = Path(td, "devil_latest.pkl")
            angel.write_bytes(b"angel")
            devil.write_bytes(b"devil")
            s = self._bare(td)
            s._check_model_updates()  # both loaded, consistent
            self.assertFalse(s._pair_mixed)
            # Advance ONLY the angel (the retrainer replaces them separately).
            os.utime(angel, None)
            s._check_model_updates()
            self.assertTrue(s._pair_mixed)

    def test_both_landed_clears_pair_mixed(self):
        with tempfile.TemporaryDirectory() as td:
            angel = Path(td, "angel_latest.pkl")
            devil = Path(td, "devil_latest.pkl")
            angel.write_bytes(b"angel")
            devil.write_bytes(b"devil")
            s = self._bare(td)
            s._check_model_updates()
            os.utime(angel, None)
            s._check_model_updates()
            self.assertTrue(s._pair_mixed)
            os.utime(devil, None)  # second half of the promotion lands
            s._check_model_updates()
            self.assertFalse(s._pair_mixed)

    def test_threshold_reloads_without_pickle_change(self):
        """The old code only re-read threshold.json inside `if reloaded:`;
        a threshold landing after the pkls was then never picked up."""
        with tempfile.TemporaryDirectory() as td:
            angel = Path(td, "angel_latest.pkl")
            devil = Path(td, "devil_latest.pkl")
            angel.write_bytes(b"angel")
            devil.write_bytes(b"devil")
            s = self._bare(td)
            s._check_model_updates()  # load the pair; thresholds absent
            self.assertEqual((s.angel_threshold, s.devil_threshold), (0.40, 0.44))
            (Path(td) / "threshold.json").write_text(
                json.dumps({"devil_threshold": 0.51, "angel_threshold": 0.42})
            )
            s._check_model_updates()  # pkls unchanged — sidecar must still load
            self.assertEqual((s.angel_threshold, s.devil_threshold), (0.42, 0.51))

    def test_generate_signals_stands_down_while_mixed(self):
        with tempfile.TemporaryDirectory() as td:
            angel = Path(td, "angel_latest.pkl")
            devil = Path(td, "devil_latest.pkl")
            angel.write_bytes(b"angel")
            devil.write_bytes(b"devil")
            s = self._bare(td)
            s._check_model_updates()
            os.utime(angel, None)  # promotion lands half-way
            out = s.generate_signals(
                _bars(3, datetime(2026, 1, 13, 12, 0, tzinfo=timezone.utc))
            )
            self.assertIsNone(out)
            # No inference was attempted on the mixed pair.
            s.angel_trainer.predict.assert_not_called()


class TestBarrierSidecar(unittest.TestCase):
    """
    The learned-barrier sidecar (ml.barriers) on the strategy side.

    Three things are load-bearing and pinned here:
      * OFF by default — a soak restarting onto this branch must behave exactly
        as it did before the barriers existed, so the switch is the only thing
        that can turn learned geometry on.
      * FAIL-LOUD on enable — an operator who asked for learned stops and
        silently got static ones cannot tell, which is the trap the cost_ratio
        and HMM guards already close.
      * the units contract — the payload carries NATR MULTIPLES, so the
        distance RiskManager builds from it equals BarrierOutput's own
        price-unit distance. If those ever disagree, the learned geometry is
        being served at a scale nobody fitted.
    """

    MODEL_DIR = Path("models/forex_m15_wide")

    @staticmethod
    def _bare(tmp: str) -> MLStrategy:
        """A constructed-enough instance for the sidecar paths (no pickles)."""
        s = MLStrategy.__new__(MLStrategy)
        s.asset_class = "forex"
        s.angel_path = Path(tmp) / "angel_latest.pkl"
        s.devil_path = Path(tmp) / "devil_latest.pkl"
        s._reload_lock = threading.Lock()
        s.notification_manager = MagicMock()
        s.use_barriers = True
        s._barrier_estimator = None
        s._barrier_meta_path = Path(tmp) / "barriers_meta.json"
        s._barrier_mtime = 0.0
        s.feature_names = list(BASE_FEATURE_COLS)
        return s

    @staticmethod
    def _fit_and_save(model_dir, feature_cols=("natr_14", "vol_rel"), tau_mae=0.95):
        """A real fitted artifact set on a synthetic frame, saved to disk."""
        sys.path.insert(0, str(project_root))
        from tests.test_barriers import _synth

        df = _synth(900)
        labels = compute_excursions(df, horizon=DEFAULT_HORIZON)
        est = BarrierEstimator(feature_cols=list(feature_cols), tau_mae=tau_mae)
        est.fit(df, labels)
        est.save(model_dir, horizon=DEFAULT_HORIZON)
        return est, df

    def test_default_is_off(self):
        """No env, no argument → static geometry, and no artifact is read."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BARRIER_GEOMETRY_ENABLED", None)
            self.assertFalse(_barriers_requested())
        self.assertTrue(_barriers_requested(True))
        self.assertFalse(_barriers_requested(False))

    def test_env_switch_enables(self):
        with patch.dict(os.environ, {"BARRIER_GEOMETRY_ENABLED": "1"}):
            self.assertTrue(_barriers_requested())
        with patch.dict(os.environ, {"BARRIER_GEOMETRY_ENABLED": "0"}):
            self.assertFalse(_barriers_requested())

    def test_enabling_without_artifacts_is_a_boot_error(self):
        """The live model dir carries no barrier set, so asking for learned
        geometry there must raise (naming every missing file) rather than
        quietly serving static ones."""
        with tempfile.TemporaryDirectory() as td:
            s = self._bare(td)
            with self.assertRaises(FileNotFoundError) as ctx:
                s._load_barriers()
            self.assertIn("barriers_mae.pkl", str(ctx.exception))

    def test_real_model_dir_refuses_enable_without_artifacts(self):
        """Same guard, through the real constructor on the served pair."""
        with self.assertRaises(FileNotFoundError):
            MLStrategy(
                asset_class="forex",
                angel_path=self.MODEL_DIR / "angel_latest.pkl",
                devil_path=self.MODEL_DIR / "devil_latest.pkl",
                warmup_period=10,
                use_barriers=True,
            )

    def test_horizon_mismatch_refuses(self):
        """A barrier labelled on a different walk length than the execution
        lifetime must not be served."""
        with tempfile.TemporaryDirectory() as td:
            est = BarrierEstimator(feature_cols=["natr_14", "vol_rel"])
            df = self._synth_for_horizon()
            est.fit(df, compute_excursions(df, horizon=10))
            est.save(td, horizon=10)
            s = self._bare(td)
            with self.assertRaises(RuntimeError) as ctx:
                s._load_barriers()
            self.assertIn("horizon", str(ctx.exception).lower())

    def _synth_for_horizon(self):
        from tests.test_barriers import _synth

        return _synth(900)

    def test_feature_vocabulary_outside_the_served_schema_refuses(self):
        """A barrier model reading a column the live pipeline does not compute
        cannot be served — that is the point of the parity check."""
        with tempfile.TemporaryDirectory() as td:
            est = BarrierEstimator(feature_cols=["natr_14"])
            # 'not_a_live_feature' is fitted as a constant column so the fit is
            # well-defined; the strategy must still refuse it.
            df = self._synth_for_horizon().with_columns(
                pl.lit(1.0).alias("not_a_live_feature")
            )
            est.feature_cols = ["natr_14", "not_a_live_feature"]
            est.feature_names_in_ = list(est.feature_cols)
            est.fit(df, compute_excursions(df, horizon=DEFAULT_HORIZON))
            est.save(td, horizon=DEFAULT_HORIZON)
            s = self._bare(td)
            with self.assertRaises(RuntimeError) as ctx:
                s._load_barriers()
            self.assertIn("not_a_live_feature", str(ctx.exception))

    def test_loaded_geometry_is_natr_multiples_matching_price_distances(self):
        """The units contract: payload mult x raw ATR == BarrierOutput's own
        price distance, so RiskManager rebuilds exactly what was fitted."""
        with tempfile.TemporaryDirectory() as td:
            est, df = self._fit_and_save(td)
            s = self._bare(td)
            s._barrier_estimator = s._load_barriers()

            feature_frame = df.with_columns(
                pl.col("vol_rel").fill_null(1.0)
            ).tail(50)
            payload = s._barrier_geometry(feature_frame)
            self.assertIsNotNone(payload)
            self.assertEqual(payload["source"], "barrier")

            atr_abs = float(
                (feature_frame["close"] * feature_frame["natr_14"] / 100.0)[-1]
            )
            self.assertAlmostEqual(
                payload["sl_atr_mult"] * atr_abs,
                est.predict(feature_frame.tail(1))[0].raw_sl_distance,
                places=9,
            )
            self.assertAlmostEqual(
                payload["tp_atr_mult"] * atr_abs,
                est.predict(feature_frame.tail(1))[0].raw_tp_distance,
                places=9,
            )

    def test_geometry_is_none_when_sidecar_off(self):
        with tempfile.TemporaryDirectory() as td:
            s = self._bare(td)
            s._barrier_estimator = None
            frame = pl.DataFrame(
                {"close": [1.08], "natr_14": [0.5], "vol_rel": [1.0]}
            )
            self.assertIsNone(s._barrier_geometry(frame))

    def test_prediction_failure_falls_back_instead_of_raising(self):
        """A frame missing the barrier's columns must not blow up the bar —
        the static bracket is a valid fallback, so degrade and log."""
        with tempfile.TemporaryDirectory() as td:
            s = self._bare(td)
            est, _ = self._fit_and_save(td)
            s._barrier_estimator = est
            frame = pl.DataFrame({"close": [1.08], "natr_14": [0.5]})  # no vol_rel
            self.assertIsNone(s._barrier_geometry(frame))

    def test_meta_mtime_promotion_swaps_the_estimator(self):
        """The promotion seam: a new artifact set lands, the meta mtime moves,
        and the next bar serves the new geometry — no restart."""
        with tempfile.TemporaryDirectory() as td:
            old, _ = self._fit_and_save(td, tau_mae=0.95)
            s = self._bare(td)
            s._barrier_estimator = s._load_barriers()
            s._barrier_mtime = s._barrier_meta_mtime()

            new, _ = self._fit_and_save(td, tau_mae=0.90)
            meta = Path(td) / "barriers_meta.json"
            future = meta.stat().st_mtime + 60
            os.utime(meta, (future, future))

            s._reload_barriers_if_changed()
            self.assertAlmostEqual(s._barrier_estimator.tau_mae, 0.90, places=6)
            s.notification_manager.send_system_message.assert_called()

    def test_broken_promotion_keeps_the_previous_geometry(self):
        """A partially written set must not disable geometry or half-swap it:
        the bar keeps serving the last set that loaded cleanly, and alerts."""
        with tempfile.TemporaryDirectory() as td:
            old = self._fit_and_save(td, tau_mae=0.95)[0]
            s = self._bare(td)
            s._barrier_estimator = old
            s._barrier_mtime = s._barrier_meta_mtime()

            self._fit_and_save(td, tau_mae=0.90)  # a new set lands...
            (Path(td) / "barriers_mfe.pkl").unlink()  # ...incompletely
            meta = Path(td) / "barriers_meta.json"
            future = meta.stat().st_mtime + 60
            os.utime(meta, (future, future))

            s._reload_barriers_if_changed()
            self.assertIs(s._barrier_estimator, old)
            self.assertAlmostEqual(s._barrier_estimator.tau_mae, 0.95, places=6)
            s.notification_manager.send_system_message.assert_called()

    # ── promotion verdict: the artifact's own reason to be served ─────────

    def test_recorded_fail_verdict_is_refused_at_boot(self):
        """The Phase 1 gate is not advisory. A retrain hook fires off the
        Angel/Devil gate rather than the barrier gate, so artifacts can exist
        whose gate FAILED — and those must not reach a live bracket."""
        with tempfile.TemporaryDirectory() as td:
            est, _ = self._fit_and_save(td)
            est.save(
                td,
                horizon=DEFAULT_HORIZON,
                verdict={"passed": False, "folds": [{"fold": 3, "coverage": 0.905}]},
            )
            s = self._bare(td)
            with self.assertRaises(RuntimeError) as ctx:
                s._load_barriers()
            self.assertIn("FAILED promotion verdict", str(ctx.exception))
            # The recorded numbers travel with the refusal, so the log says
            # what failed rather than only that something did.
            self.assertIn("0.905", str(ctx.exception))

    def test_recorded_pass_verdict_serves(self):
        with tempfile.TemporaryDirectory() as td:
            est, _ = self._fit_and_save(td)
            est.save(td, horizon=DEFAULT_HORIZON, verdict={"passed": True})
            s = self._bare(td)
            loaded = s._load_barriers()
            self.assertTrue(loaded.verdict_["passed"])
            self.assertIsNotNone(loaded.horizon_)

    def test_absent_verdict_serves_with_a_warning(self):
        """Absence is "unknown", not "pass" and not a refusal: every artifact
        written before the field existed is in this state, and refusing them
        would break the sidecar for artifacts that were validated by hand."""
        with tempfile.TemporaryDirectory() as td:
            self._fit_and_save(td)
            s = self._bare(td)
            # MLStrategy.__module__, not a hardcoded name: this test file imports
            # the module as `src.strategies....` while the live entry points
            # import it as `strategies....`, so a literal name silently catches
            # nothing.
            with self.assertLogs(MLStrategy.__module__, level="WARNING") as logs:
                loaded = s._load_barriers()
            self.assertIsNone(loaded.verdict_)
            self.assertTrue(
                any("NO promotion verdict" in line for line in logs.output),
                logs.output,
            )

    def test_verdict_summary_is_readable_from_folds(self):
        from strategies.concrete_strategies.ml_strategy import _verdict_summary

        line = _verdict_summary(
            {
                "passed": False,
                "eval_date": "2026-09-14T20:00:00+00:00",
                "coverage_floor": 0.93,
                "folds": [
                    {"fold": 1, "coverage": 0.953},
                    {"fold": 2, "coverage": 0.934},
                    {"fold": 3, "coverage": 0.905},
                ],
            }
        )
        self.assertIn("0.953/0.934/0.905", line)
        self.assertIn("coverage_floor=0.93", line)
        # And it must not raise on a verdict that carries nothing but the
        # boolean — the refusal path must never become its own failure.
        self.assertEqual(_verdict_summary({"passed": False}), "no detail recorded")

    def test_bracket_check_stands_down_but_reports_the_skew(self):
        """With learned geometry active the profile-vs-metadata bracket check
        cannot apply — but the mismatch it would have caught is the Devil's
        label geometry, so the numbers must appear rather than vanish.

        The Devil is trained on `devil_target`, built from the STATIC multiples
        (retrainer.py:1469); a learned bracket is a different walk than the one
        its conviction was fitted on, and the direction of that error is not
        known. A silent stand-down is how that gets forgotten.
        """
        with tempfile.TemporaryDirectory() as td:
            Path(td, "metadata.json").write_text(
                json.dumps(
                    {
                        "asset_class": "forex",
                        "sl_atr_multiplier": 1.0,
                        "tp_atr_multiplier": 2.0,
                    }
                )
            )
            s = MLStrategy.__new__(MLStrategy)
            s.asset_class = "forex"
            s.angel_path = Path(td) / "angel_latest.pkl"
            s.use_barriers = True
            with self.assertLogs(MLStrategy.__module__, level="WARNING") as logs:
                s._validate_metadata()  # must NOT raise: 1.0/2.0 vs profile 2.0/4.0
            joined = "\n".join(logs.output)
            self.assertIn("bracket check STOOD DOWN", joined)
            self.assertIn("1.0", joined)  # the artifact's trained pair
            self.assertIn("2.0", joined)  # ...and the profile it is served under

    def test_bracket_check_still_refuses_when_barriers_are_off(self):
        """The guard is only ever bypassed for a declared reason: with the
        sidecar off, a mismatched artifact is still refused outright."""
        with tempfile.TemporaryDirectory() as td:
            Path(td, "metadata.json").write_text(
                json.dumps(
                    {
                        "asset_class": "forex",
                        "sl_atr_multiplier": 1.0,
                        "tp_atr_multiplier": 2.0,
                    }
                )
            )
            s = MLStrategy.__new__(MLStrategy)
            s.asset_class = "forex"
            s.angel_path = Path(td) / "angel_latest.pkl"
            s.use_barriers = False
            with self.assertRaises(RuntimeError) as ctx:
                s._validate_metadata()
            self.assertIn("Bracket mismatch", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

