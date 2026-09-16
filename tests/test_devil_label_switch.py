"""
The Devil's label switch: `RETRAIN_DEVIL_LABEL`.

The Devil ships trained on `devil_target` — "did price avoid a 2.0xATR stop for 5
bars". Measured 2026-09-14 on honest OOF probabilities (chronological 5-fold, 227k
rows): that model scores AUC **0.4722** against the MACRO outcome the live path
actually bets on (0.4564 in the held-out final window), while the same model
trained on `devil_target_macro` scores **0.5839** / **0.5806**. The two scores are
anti-correlated (−0.166): the shipping second stage is scoring close to the
opposite of what predicts the bracket.

These tests pin the switch's contract, not the finding: the default preserves
today's behaviour exactly, `macro` selects the validated label, an unrecognised
value warns and falls back rather than crashing a retrain, and the choice is read
per call so an in-process caller (or a test) can flip it without re-importing the
module — the class of bug that silently ignored `RETRAIN_DAYS_BACK` once already.

Glossary:
    devil_label_col -- the accessor under test.
    ENV_DEVIL_LABEL / DEVIL_LABEL_DEFAULT -- the env var and its default.
    TestDevilLabelSwitch -- default preservation, macro selection, fallback,
        per-call reading, and that the engineering step actually produces both
        columns the switch can name.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from core.retrainer import (  # noqa: E402
    DEVIL_LABEL_DEFAULT,
    ENV_DEVIL_LABEL,
    devil_label_col,
)


class TestDevilLabelSwitch(unittest.TestCase):
    def test_default_is_the_shipping_survival_label(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_DEVIL_LABEL, None)
            self.assertEqual(DEVIL_LABEL_DEFAULT, "survival")
            self.assertEqual(devil_label_col(), "devil_target")

    def test_macro_selects_the_validated_label(self):
        for value in ("macro", "MACRO", "devil_target_macro", "bracket"):
            with mock.patch.dict(os.environ, {ENV_DEVIL_LABEL: value}):
                self.assertEqual(devil_label_col(), "devil_target_macro", value)

    def test_survival_spellings_all_agree(self):
        for value in ("survival", "devil_target", "5", ""):
            with mock.patch.dict(os.environ, {ENV_DEVIL_LABEL: value}):
                self.assertEqual(devil_label_col(), "devil_target", value)

    def test_unrecognised_value_warns_and_falls_back(self):
        """A typo must not kill a retrain, and must not silently pick a label."""
        with mock.patch.dict(os.environ, {ENV_DEVIL_LABEL: "maco"}):
            with self.assertLogs("core.retrainer", level="WARNING") as logs:
                self.assertEqual(devil_label_col(), "devil_target")
            self.assertIn("maco", "\n".join(logs.output))

    def test_read_per_call_not_bound_at_import(self):
        """The whole point: an in-process caller can flip it. A module constant
        bound at import is what silently ignored RETRAIN_DAYS_BACK once."""
        with mock.patch.dict(os.environ, {ENV_DEVIL_LABEL: "survival"}):
            self.assertEqual(devil_label_col(), "devil_target")
        with mock.patch.dict(os.environ, {ENV_DEVIL_LABEL: "macro"}):
            self.assertEqual(devil_label_col(), "devil_target_macro")

    def test_both_columns_exist_on_an_engineered_frame(self):
        """The switch may only name columns the pipeline produces."""
        import core.retrainer as R
        from execution.risk_manager import RiskProfile

        n = 400
        ts = pl.datetime_range(
            __import__("datetime").datetime(2026, 1, 1),
            __import__("datetime").datetime(2026, 1, 1)
            + __import__("datetime").timedelta(minutes=n - 1),
            interval="1m",
            eager=True,
        )
        import numpy as np

        rng = np.random.default_rng(3)
        close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
        raw = pl.DataFrame({
            "timestamp": ts,
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.full(n, 10.0),
            "symbol": ["EUR_USD"] * n,
        })
        feats, _, _ = R.engineer_features_and_labels(
            raw, sl_mult=2.0, angel_mult=1.0, tp_mult=4.0, max_hold=30,
            survival_bars=5, htf_timeframe="5m",
            risk_profile=RiskProfile.for_asset_class("forex"), alpha_table=None,
        )
        for col in (devil_label_col(), "devil_target_macro", "devil_target"):
            self.assertIn(col, feats.columns)


if __name__ == "__main__":
    unittest.main()
