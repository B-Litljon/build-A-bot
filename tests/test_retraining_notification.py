import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from src.core.notification_manager import NotificationManager

"""
Tests for the retraining Discord embed -- specifically, that passing the gate
is not reported as going live.

Why this file exists: on 2026-08-17 a side-directory experiment
(RETRAIN_MODEL_DIR=models/_ab_5yr) passed the gate and posted "✅ PROMOTED --
New models passed all validation gates and are now live." Nothing had gone
live; the live model directory was untouched. The alert was indistinguishable
from a real production promotion, which is a false alarm on the only channel
the system uses to reach a human.

Glossary:
    _FakeReport -- minimal stand-in for retrainer.ValidationReport; only the
        fields the embed reads.
    _embed -- sends one report through a patched requests.post and returns the
        Discord embed dict that would have been transmitted.
    test_side_run_* -- a run redirected to a side directory must not be green,
        must not claim "live", and must name where it actually wrote.
    test_production_run_* -- the default path keeps the green PROMOTED verdict
        but still names the directory and warns that the live bot follows
        OANDA_MODEL_DIR.
    test_thresholds_quoted_from_caller -- the embed previously hardcoded a
        Brier bar of 0.25 while the gate had used 0.30 since Phase 5.5; the
        bars are now passed in so they cannot drift.
"""

_THRESHOLDS = {"brier": 0.30, "ev": 0.0005, "profit_factor": 1.2}


class _FakeReport:
    fold_metrics = []
    mean_brier = 0.1892
    mean_ev = 1.386210
    final_profit_factor = 1.8148
    final_win_rate = 0.476
    final_total_trades = 103
    rejection_reasons = ["Profit Factor 1.1538 < 1.2 threshold"]


def _embed(**kwargs) -> dict:
    notifier = NotificationManager(webhook_url="https://example.invalid/hook")
    with patch("src.core.notification_manager.requests.post") as post:
        post.return_value.raise_for_status = lambda: None
        notifier.send_retraining_report(_FakeReport(), **kwargs)
        return post.call_args.kwargs["json"]["embeds"][0]


class TestRetrainingNotification(unittest.TestCase):
    def test_side_run_is_not_announced_as_a_promotion(self):
        e = _embed(promoted=True, model_dir="models/_ab_5yr", is_production_path=False)
        self.assertIn("side candidate", e["title"])
        self.assertNotEqual(e["color"], 0x00FF00, "side runs must not share the green of a real promotion")

    def test_side_run_states_nothing_went_live(self):
        e = _embed(promoted=True, model_dir="models/_ab_5yr", is_production_path=False)
        self.assertIn("nothing has gone live", e["description"])
        self.assertIn("Production weights are untouched", e["description"])

    def test_side_run_names_its_destination(self):
        e = _embed(promoted=True, model_dir="models/_ab_5yr", is_production_path=False)
        self.assertIn("models/_ab_5yr", e["description"])

    def test_side_rejection_says_nothing_was_written(self):
        e = _embed(promoted=False, model_dir="models/_ab_2yr", is_production_path=False)
        self.assertTrue(e["title"].startswith("🚫 REJECTED"))
        self.assertIn("nothing was written", e["description"])

    def test_production_run_keeps_the_promoted_verdict(self):
        e = _embed(promoted=True, model_dir="models/forex", is_production_path=True)
        self.assertTrue(e["title"].startswith("✅ PROMOTED"))
        self.assertEqual(e["color"], 0x00FF00)
        self.assertIn("models/forex", e["description"])

    def test_production_run_warns_that_live_follows_the_env_var(self):
        # The retrainer's default path is models/<asset_class>, but the live
        # soak loads whatever OANDA_MODEL_DIR names -- today a different dir.
        e = _embed(promoted=True, model_dir="models/forex", is_production_path=True)
        self.assertIn("OANDA_MODEL_DIR", e["description"])

    def test_thresholds_quoted_from_caller(self):
        e = _embed(
            promoted=True,
            model_dir="models/forex",
            is_production_path=True,
            gate_thresholds=_THRESHOLDS,
        )
        self.assertIn("threshold ≤ 0.3", e["description"])
        self.assertNotIn("0.25", e["description"])

    def test_thresholds_omitted_rather_than_guessed(self):
        e = _embed(promoted=True, model_dir="models/forex", is_production_path=True)
        self.assertNotIn("threshold", e["description"])

    def test_legacy_two_argument_call_still_works(self):
        e = _embed(promoted=True)
        self.assertTrue(e["title"].startswith("✅ PROMOTED"))


if __name__ == "__main__":
    unittest.main()
