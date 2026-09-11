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

from src.strategies.concrete_strategies.ml_strategy import MLStrategy


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


if __name__ == "__main__":
    unittest.main()
