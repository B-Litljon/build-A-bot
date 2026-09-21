"""
Tests for the retrainer's opt-in OOS trade ledger.

This capture lives inside `validate_candidate`, which is the promotion gate —
the most consequential offline code in the repo. Two properties matter and are
pinned here:

  1. **It selects the right rows.** `signal_mask` indexes `val_df`, but
     `approved_mask` indexes the *Angel-proposed subset*, not `val_df`. Getting
     that composition wrong would silently attribute one bar's outcome to a
     different bar — and the behavior matrix would then be confidently wrong
     rather than obviously broken.
  2. **It cannot change a promotion decision.** It only reads masks the fold
     already computed, mutates nothing, and is skipped entirely when no ledger
     is passed (every production path).

Glossary:
    _val_df -- a stand-in validation frame with recognisable per-row values so
        mis-selection shows up as a wrong symbol, not a wrong count.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.retrainer import _capture_oos_ledger  # noqa: E402

CARRY = ("timestamp", "symbol", "natr_14", "ppo", "behavior_label")


def _val_df(n=10):
    return pl.DataFrame(
        {
            "timestamp": list(range(n)),
            "symbol": [f"SYM{i}" for i in range(n)],
            "natr_14": [0.01 * i for i in range(n)],
            "ppo": [0.1 * i for i in range(n)],
            "behavior_label": [f"lab{i}" for i in range(n)],
            "unrelated_feature": [999.0] * n,
        }
    )


class TestRowSelection(unittest.TestCase):
    def test_approved_mask_is_relative_to_proposed_not_val_df(self):
        """
        The composition that is easy to get wrong. Angel proposes val rows
        1,3,5,7,9; the Devil then approves the 1st, 3rd and 5th OF THOSE,
        i.e. val rows 1, 5 and 9 — NOT val rows 0, 2, 4.
        """
        val = _val_df(10)
        signal_mask = np.array([False, True, False, True, False, True, False, True, False, True])
        approved_mask = np.array([True, False, True, False, True])

        ledger = []
        _capture_oos_ledger(
            ledger, val, signal_mask, approved_mask,
            macro_targets=np.array([1, 0, 1, 0, 0], dtype=np.int8),
            devil_probs=np.array([0.9, 0.1, 0.8, 0.2, 0.7]),
            angel_probs=np.array([0.5, 0.4, 0.6, 0.4, 0.7]),
            fold_number=3,
            carry_cols=CARRY,
        )

        self.assertEqual(len(ledger), 1)
        got = ledger[0]
        self.assertEqual(got["symbol"].to_list(), ["SYM1", "SYM5", "SYM9"])
        self.assertEqual(got["timestamp"].to_list(), [1, 5, 9])

    def test_outcomes_stay_aligned_with_their_bars(self):
        """macro_win / probs must follow the same composition as the rows."""
        val = _val_df(6)
        signal_mask = np.array([True, False, True, False, True, False])   # rows 0,2,4
        approved_mask = np.array([False, True, True])                     # rows 2,4

        ledger = []
        _capture_oos_ledger(
            ledger, val, signal_mask, approved_mask,
            macro_targets=np.array([0, 1, 0], dtype=np.int8),
            devil_probs=np.array([0.1, 0.77, 0.66]),
            angel_probs=np.array([0.4, 0.55, 0.45]),
            fold_number=2,
            carry_cols=CARRY,
        )

        got = ledger[0]
        self.assertEqual(got["symbol"].to_list(), ["SYM2", "SYM4"])
        self.assertEqual(got["macro_win"].to_list(), [1, 0])
        np.testing.assert_allclose(got["devil_prob"].to_numpy(), [0.77, 0.66])
        np.testing.assert_allclose(got["angel_prob"].to_numpy(), [0.55, 0.45])
        self.assertEqual(got["fold"].to_list(), [2, 2])


class TestNonPerturbation(unittest.TestCase):
    def test_inputs_are_not_mutated(self):
        val = _val_df(6)
        before = val.clone()
        signal_mask = np.array([True, True, False, False, True, True])
        approved_mask = np.array([True, False, True, False])
        macro = np.array([1, 0, 1, 0], dtype=np.int8)
        devil = np.array([0.9, 0.1, 0.8, 0.2])
        angel = np.array([0.5, 0.4, 0.6, 0.4])
        sm, am, mt, dp, ap = (
            signal_mask.copy(), approved_mask.copy(), macro.copy(), devil.copy(), angel.copy()
        )

        _capture_oos_ledger([], val, signal_mask, approved_mask, macro, devil, angel, 1, CARRY)

        self.assertTrue(val.equals(before))
        np.testing.assert_array_equal(signal_mask, sm)
        np.testing.assert_array_equal(approved_mask, am)
        np.testing.assert_array_equal(macro, mt)
        np.testing.assert_array_equal(devil, dp)
        np.testing.assert_array_equal(angel, ap)

    def test_zero_approvals_appends_nothing(self):
        val = _val_df(4)
        ledger = []
        _capture_oos_ledger(
            ledger, val,
            np.array([True, True, False, False]),
            np.array([False, False]),
            np.array([], dtype=np.int8), np.array([]), np.array([]),
            1, CARRY,
        )
        self.assertEqual(ledger, [])

    def test_missing_carry_columns_are_skipped_not_raised(self):
        """A caller may ask for behavior_label before it has been added."""
        val = _val_df(4).drop("behavior_label")
        ledger = []
        _capture_oos_ledger(
            ledger, val,
            np.array([True, False, True, False]),
            np.array([True, True]),
            np.array([1, 0], dtype=np.int8),
            np.array([0.8, 0.7]), np.array([0.5, 0.5]),
            1, CARRY,
        )
        self.assertEqual(len(ledger), 1)
        self.assertNotIn("behavior_label", ledger[0].columns)
        self.assertIn("symbol", ledger[0].columns)

    def test_only_requested_columns_are_carried(self):
        val = _val_df(4)
        ledger = []
        _capture_oos_ledger(
            ledger, val,
            np.array([True, True, False, False]),
            np.array([True, True]),
            np.array([1, 0], dtype=np.int8),
            np.array([0.8, 0.7]), np.array([0.5, 0.5]),
            1, ("symbol",),
        )
        self.assertNotIn("unrelated_feature", ledger[0].columns)
        self.assertNotIn("natr_14", ledger[0].columns)

    def test_folds_accumulate_in_order(self):
        val = _val_df(4)
        ledger = []
        for fold in (1, 2, 3):
            _capture_oos_ledger(
                ledger, val,
                np.array([True, False, True, False]),
                np.array([True, False]),
                np.array([1, 0], dtype=np.int8),
                np.array([0.8, 0.1]), np.array([0.5, 0.4]),
                fold, CARRY,
            )
        self.assertEqual(len(ledger), 3)
        self.assertEqual(pl.concat(ledger)["fold"].to_list(), [1, 2, 3])


class TestCallSiteIsGuarded(unittest.TestCase):
    """
    Source-level guard, in the spirit of tests/test_events.py: prove the
    capture is opt-in and cannot run in a production retrain.
    """

    def test_capture_is_guarded_by_an_explicit_none_check(self):
        src = (Path(__file__).resolve().parents[1] / "src/core/retrainer/_gate.py").read_text()  # validate_candidate lived here since the 2026-09-16 split
        idx = src.index("_capture_oos_ledger(\n                oos_ledger,")
        preceding = src[:idx]
        self.assertTrue(
            preceding.rstrip().endswith("if oos_ledger is not None:"),
            "the ledger capture must sit directly under `if oos_ledger is not None:`",
        )

    def test_ledger_defaults_to_none(self):
        import inspect

        from core.retrainer import validate_candidate

        sig = inspect.signature(validate_candidate)
        self.assertIsNone(sig.parameters["oos_ledger"].default)


if __name__ == "__main__":
    unittest.main()
