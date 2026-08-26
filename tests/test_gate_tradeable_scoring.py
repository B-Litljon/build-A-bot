"""
Tests for the promotion gate's tradeable-instrument scoring.

The gate is a prediction about live results. XAU_USD and XAG_USD are in the
training basket but the account cannot trade them, so their validation trades
must not move a promote/reject decision. Measured 2026-08-23: pooling them
INVERTS the verdict (pooled 1.245 with metals vs 1.500 without, while the same
two configs on the six tradeable crosses read 1.742 vs 1.461), so this is a
correctness fix rather than noise reduction.

Critically, training keeps the full basket — the volatility-first basket is
load-bearing and a shrink was tried and rejected on 2026-07-02.

Glossary:
    _proposed -- builds a validation frame plus a signal mask, so the tests
        exercise the same two-stage indexing the fold does: signal_mask selects
        rows of val_df, approved_mask selects rows of THAT subset.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import retrainer as R  # noqa: E402


def _proposed(symbols):
    """val_df where every row is Angel-proposed, one row per symbol given."""
    val_df = pl.DataFrame(
        {"symbol": list(symbols), "timestamp": list(range(len(symbols)))}
    )
    return val_df, np.ones(len(symbols), dtype=bool)


class TestTradeableScoringMask(unittest.TestCase):
    def test_metals_approvals_are_excluded_from_scoring(self):
        val_df, sig = _proposed(["XAU_USD", "GBP_JPY", "XAG_USD", "EUR_JPY"])
        approved = np.array([True, True, True, True])

        scored, excluded = R._tradeable_scoring_mask(val_df, sig, approved)

        self.assertEqual(excluded, 2)
        self.assertEqual(scored.tolist(), [False, True, False, True])

    def test_unapproved_metals_are_not_double_counted(self):
        """n_excluded counts only trades that were approved to begin with."""
        val_df, sig = _proposed(["XAU_USD", "XAU_USD", "GBP_JPY"])
        approved = np.array([True, False, True])

        scored, excluded = R._tradeable_scoring_mask(val_df, sig, approved)

        self.assertEqual(excluded, 1)
        self.assertEqual(scored.tolist(), [False, False, True])

    def test_mask_aligns_to_proposed_subset_not_val_df(self):
        """
        The indexing trap: signal_mask selects val_df rows, approved_mask
        selects rows of that subset. A mask built against val_df directly would
        silently score the wrong instruments.
        """
        val_df = pl.DataFrame(
            {
                "symbol": ["XAU_USD", "GBP_JPY", "XAG_USD", "EUR_JPY", "XAU_USD"],
                "timestamp": list(range(5)),
            }
        )
        # Angel proposes val rows 1, 2, 4 -> GBP_JPY, XAG_USD, XAU_USD
        sig = np.array([False, True, True, False, True])
        approved = np.array([True, True, True])  # all three proposals approved

        scored, excluded = R._tradeable_scoring_mask(val_df, sig, approved)

        self.assertEqual(scored.tolist(), [True, False, False])
        self.assertEqual(excluded, 2)

    def test_all_tradeable_is_a_no_op(self):
        val_df, sig = _proposed(["GBP_JPY", "EUR_JPY"])
        approved = np.array([True, False])
        scored, excluded = R._tradeable_scoring_mask(val_df, sig, approved)
        self.assertEqual(excluded, 0)
        self.assertEqual(scored.tolist(), approved.tolist())

    def test_symbol_matching_is_case_insensitive(self):
        val_df, sig = _proposed(["xau_usd", "gbp_jpy"])
        scored, excluded = R._tradeable_scoring_mask(
            val_df, sig, np.array([True, True])
        )
        self.assertEqual(excluded, 1)
        self.assertEqual(scored.tolist(), [False, True])

    def test_frame_without_symbol_column_is_untouched(self):
        """Single-symbol callers and older frames must behave as before."""
        val_df = pl.DataFrame({"timestamp": [0, 1]})
        approved = np.array([True, True])
        scored, excluded = R._tradeable_scoring_mask(
            val_df, np.ones(2, dtype=bool), approved
        )
        self.assertEqual(excluded, 0)
        self.assertIs(scored, approved)

    def test_empty_untradeable_set_restores_old_behaviour(self):
        val_df, sig = _proposed(["XAU_USD", "GBP_JPY"])
        approved = np.array([True, True])
        with mock.patch.object(R, "UNTRADEABLE_SYMBOLS", frozenset()):
            scored, excluded = R._tradeable_scoring_mask(val_df, sig, approved)
        self.assertEqual(excluded, 0)
        self.assertIs(scored, approved)

    def test_every_approval_untradeable_yields_empty_mask(self):
        """The fold must be able to detect 'nothing scoreable' and bail."""
        val_df, sig = _proposed(["XAU_USD", "XAG_USD"])
        scored, excluded = R._tradeable_scoring_mask(
            val_df, sig, np.array([True, True])
        )
        self.assertEqual(int(scored.sum()), 0)
        self.assertEqual(excluded, 2)


class TestConfiguration(unittest.TestCase):
    def test_metals_are_untradeable_by_default(self):
        self.assertIn("XAU_USD", R.UNTRADEABLE_SYMBOLS)
        self.assertIn("XAG_USD", R.UNTRADEABLE_SYMBOLS)

    def test_tradeable_crosses_are_not_in_the_exclusion_set(self):
        for sym in ("GBP_JPY", "AUD_JPY", "EUR_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD"):
            self.assertNotIn(sym, R.UNTRADEABLE_SYMBOLS)

    def test_metals_remain_in_the_default_training_basket(self):
        """
        The whole point: excluded from SCORING, kept in TRAINING. A basket
        shrink was tried and rejected 2026-07-02, and metals-in-training
        measured better on the crosses (gross PF 1.742 vs 1.461).
        """
        cfg = R.get_asset_config("oanda")
        tickers = {t.upper() for t in cfg["tickers"]}
        self.assertTrue(
            R.UNTRADEABLE_SYMBOLS <= tickers,
            "untradeable instruments must still be TRAINED on; only scoring "
            f"excludes them (basket={sorted(tickers)})",
        )


class TestBehaviorVetoIsOffByDefault(unittest.TestCase):
    """
    A model trained with a behavior veto REQUIRES a matching live gate. No such
    gate exists yet, so the veto must stay inert unless explicitly requested,
    and any artifact trained with it must declare what it needs.
    """

    def test_veto_is_empty_by_default(self):
        self.assertEqual(R.BEHAVIOR_VETO_LABELS, frozenset())

    def test_env_var_parses_into_labels(self):
        import importlib

        try:
            with mock.patch.dict(
                "os.environ", {"RETRAIN_BEHAVIOR_VETO": "trend_high, range_low"}
            ):
                reloaded = importlib.reload(R)
                self.assertEqual(
                    reloaded.BEHAVIOR_VETO_LABELS,
                    frozenset({"trend_high", "range_low"}),
                )
        finally:
            # Restore OUTSIDE the patch, or the reload re-reads the patched env
            # and leaks a non-empty veto into every later test.
            importlib.reload(R)
        self.assertEqual(R.BEHAVIOR_VETO_LABELS, frozenset())

    def test_metadata_declares_the_required_gate(self):
        """metadata.json must carry behavior_veto so a served model is checkable."""
        src = (Path(__file__).resolve().parents[1] / "src/core/retrainer.py").read_text()
        self.assertIn('"behavior_veto": sorted(BEHAVIOR_VETO_LABELS)', src)


if __name__ == "__main__":
    unittest.main()


class TestHtfTimeframeSymmetry(unittest.TestCase):
    """
    The htf_* features are derived from a higher-timeframe resample. If
    training pairs M15 with 5m while the live bot pairs it with 1h, every
    htf_ column means something different at inference — silent skew, and the
    exact failure that produced an unusable 5yr candidate on 2026-08-24.
    """

    def _cfg(self, tf):
        with mock.patch.dict(
            "os.environ",
            {"DATA_SOURCE": "oanda", "RETRAIN_TIMEFRAME_MINUTES": str(tf)},
        ):
            return R.get_asset_config("oanda")

    def test_m15_pairs_with_one_hour(self):
        self.assertEqual(self._cfg(15)["htf_timeframe"], "1h")

    def test_m1_still_pairs_with_five_minutes(self):
        self.assertEqual(self._cfg(1)["htf_timeframe"], "5m")

    def test_mapping_matches_the_live_bot_exactly(self):
        """run_oanda._GRANULARITY_PROFILES is the authority; mirror or fail."""
        import ast

        src = (Path(__file__).resolve().parents[1] / "run_oanda.py").read_text()
        tree = ast.parse(src)
        live = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "_GRANULARITY_PROFILES":
                live = ast.literal_eval(node.value)
        self.assertIsNotNone(live, "could not read _GRANULARITY_PROFILES")
        for bars, (htf, _warmup) in live.items():
            self.assertEqual(
                R._HTF_FOR_TIMEFRAME.get(bars),
                htf,
                f"retrainer disagrees with the live bot for {bars}m bars",
            )

    def test_env_override_still_wins(self):
        with mock.patch.dict(
            "os.environ",
            {
                "DATA_SOURCE": "oanda",
                "RETRAIN_TIMEFRAME_MINUTES": "15",
                "RETRAIN_HTF_TIMEFRAME": "4h",
            },
        ):
            self.assertEqual(R.get_asset_config("oanda")["htf_timeframe"], "4h")
