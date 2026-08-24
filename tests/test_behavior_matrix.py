"""
Tests for analysis.behavior_matrix — per-behavior candidate scoring.

The matrix's job is to be honest about two things: the spread toll, and thin
evidence. Both have burned this project before (gross PF 1.373 -> net 1.004 on
the shipped model; a 16-trade fold showing PF 3.3). The tests below pin the
arithmetic of the first and the refusal behaviour of the second.

Glossary:
    _ledger -- builds a fake captured OOS ledger with a chosen win/loss mix per
        behavior, so cell arithmetic can be checked against hand computation.
    CAND -- a 2R candidate matching the live forex brackets (2.0x stop /
        4.0x target), used so expected values are easy to verify by hand.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.behavior_matrix import (  # noqa: E402
    DEFAULT_TOLL_R,
    MIN_CELL_TRADES,
    Candidate,
    net_r,
    recommend,
    score_ledger,
    tag_frame,
    to_frame,
)
from ml.regimes.behavior_tagger import LABEL_COLD  # noqa: E402

CAND = Candidate(name="live_2x4x", sl_mult=2.0, tp_mult=4.0, lookback_days=730)


def _ledger(spec):
    """spec: {behavior: (n_wins, n_losses)} -> a ledger frame."""
    labels, wins = [], []
    for behavior, (w, l) in spec.items():
        labels += [behavior] * (w + l)
        wins += [1] * w + [0] * l
    return pl.DataFrame({"behavior_label": labels, "macro_win": np.array(wins, dtype=np.int8)})


class TestNetR(unittest.TestCase):
    def test_payoff_matches_the_retrainer_convention(self):
        """Win = tp/sl R, loss = -1 R, before any toll."""
        r = net_r(np.array([1, 0]), sl_mult=2.0, tp_mult=4.0, toll_r=0.0)
        np.testing.assert_allclose(r, [2.0, -1.0])

    def test_toll_is_charged_on_winners_too(self):
        """The spread is paid on entry, so it reduces wins as well as losses."""
        r = net_r(np.array([1, 0]), sl_mult=2.0, tp_mult=4.0, toll_r=0.33)
        np.testing.assert_allclose(r, [2.0 - 0.33, -1.0 - 0.33])

    def test_toll_default_is_the_worst_admissible_not_a_guess(self):
        """spread_k_base 3.0 admits a trade only if spread <= 1/3 of the stop."""
        self.assertAlmostEqual(DEFAULT_TOLL_R, 1.0 / 3.0, places=2)

    def test_toll_can_flip_a_profitable_cell(self):
        """The headline risk: gross-positive, net-negative."""
        wins = np.array([1] * 34 + [0] * 66)  # 34% win rate at 2R
        gross = net_r(wins, 2.0, 4.0, toll_r=0.0).mean()
        net = net_r(wins, 2.0, 4.0, toll_r=DEFAULT_TOLL_R).mean()
        self.assertGreater(gross, 0.0)
        self.assertLess(net, 0.0)


class TestScoring(unittest.TestCase):
    def test_cell_arithmetic_by_hand(self):
        cells = score_ledger(_ledger({"trend_high": (40, 60)}), CAND, toll_r=0.0)
        self.assertEqual(len(cells), 1)
        c = cells[0]
        self.assertEqual(c.n, 100)
        self.assertAlmostEqual(c.win_rate, 0.40)
        # EV = 0.4*2 - 0.6*1 = 0.2 ; PF = (40*2)/(60*1) = 1.3333
        self.assertAlmostEqual(c.net_ev_r, 0.20, places=6)
        self.assertAlmostEqual(c.profit_factor_gross, 80.0 / 60.0, places=6)

    def test_net_and_gross_are_both_reported(self):
        c = score_ledger(_ledger({"trend_high": (40, 60)}), CAND)[0]
        self.assertGreater(c.gross_ev_r, c.net_ev_r)
        self.assertGreater(c.profit_factor_gross, c.profit_factor_net)

    def test_cold_trades_are_dropped_not_pooled(self):
        led = _ledger({"trend_high": (30, 30), LABEL_COLD: (50, 0)})
        cells = score_ledger(led, CAND)
        self.assertEqual({c.behavior for c in cells}, {"trend_high"})
        self.assertEqual(cells[0].n, 60)

    def test_thin_cell_is_flagged_uninformative(self):
        cells = score_ledger(_ledger({"range_low": (5, 6)}), CAND)
        self.assertFalse(cells[0].informative)
        self.assertEqual(cells[0].n, 11)

    def test_cell_at_the_floor_is_informative(self):
        w = MIN_CELL_TRADES // 2
        cells = score_ledger(_ledger({"range_low": (w, MIN_CELL_TRADES - w)}), CAND)
        self.assertEqual(cells[0].n, MIN_CELL_TRADES)
        self.assertTrue(cells[0].informative)

    def test_ledger_without_labels_is_rejected_loudly(self):
        bad = pl.DataFrame({"macro_win": np.array([1, 0], dtype=np.int8)})
        with self.assertRaises(ValueError) as ctx:
            score_ledger(bad, CAND)
        self.assertIn("behavior_label", str(ctx.exception))

    def test_bootstrap_ci_is_deterministic(self):
        led = _ledger({"trend_high": (40, 60)})
        a = score_ledger(led, CAND, seed=1)[0]
        b = score_ledger(led, CAND, seed=1)[0]
        self.assertEqual((a.ci_low, a.ci_high), (b.ci_low, b.ci_high))

    def test_ci_brackets_the_point_estimate(self):
        c = score_ledger(_ledger({"trend_high": (40, 60)}), CAND)[0]
        self.assertLess(c.ci_low, c.net_ev_r)
        self.assertGreater(c.ci_high, c.net_ev_r)

    def test_significance_tracks_the_interval(self):
        """
        Significant means "distinguishable from zero", in EITHER direction —
        a reliably losing cell is a finding too. The ambiguous case is a cell
        sitting near break-even: net EV = 3*win_rate - 1.33, so ~44% wins at
        2R is indistinguishable from zero on a small sample.
        """
        strong = score_ledger(_ledger({"trend_high": (300, 100)}), CAND)[0]
        self.assertTrue(strong.significant)
        self.assertGreater(strong.net_ev_r, 0)

        reliably_bad = score_ledger(_ledger({"range_low": (60, 240)}), CAND)[0]
        self.assertTrue(reliably_bad.significant)
        self.assertLess(reliably_bad.net_ev_r, 0)

        breakeven = score_ledger(_ledger({"mixed_normal": (22, 28)}), CAND)[0]
        self.assertAlmostEqual(breakeven.net_ev_r, 0.0, delta=0.10)
        self.assertFalse(breakeven.significant)

    def test_all_wins_has_no_infinite_loss_denominator(self):
        c = score_ledger(_ledger({"trend_high": (20, 0)}), CAND, toll_r=0.0)[0]
        self.assertEqual(c.profit_factor_gross, float("inf"))


class TestRecommend(unittest.TestCase):
    def test_picks_the_best_net_candidate(self):
        good = Candidate("good", 2.0, 4.0, 730)
        bad = Candidate("bad", 2.0, 4.0, 730)
        cells = score_ledger(_ledger({"trend_high": (60, 40)}), good) + score_ledger(
            _ledger({"trend_high": (35, 65)}), bad
        )
        self.assertEqual(recommend(cells)["trend_high"], "good")

    def test_refuses_to_name_a_winner_on_thin_evidence(self):
        """A recommender that always recommends is a random number generator."""
        cells = score_ledger(_ledger({"range_low": (8, 3)}), CAND)
        self.assertIsNone(recommend(cells)["range_low"])

    def test_refuses_when_every_candidate_loses_money(self):
        cells = score_ledger(_ledger({"range_low": (20, 80)}), CAND)
        self.assertIsNone(recommend(cells)["range_low"])

    def test_thin_evidence_can_be_allowed_explicitly(self):
        cells = score_ledger(_ledger({"range_low": (8, 3)}), CAND)
        self.assertEqual(
            recommend(cells, require_informative=False)["range_low"], CAND.name
        )


class TestTagFrame(unittest.TestCase):
    def _frame(self, n=400, symbols=("GBP_JPY", "EUR_JPY")):
        rng = np.random.default_rng(5)
        rows = []
        for s in symbols:
            for i in range(n):
                rows.append(
                    {
                        "symbol": s,
                        "timestamp": i,
                        "natr_14": float(abs(rng.normal(0.08, 0.02))),
                        "ppo": float(rng.normal(0, 0.1)),
                    }
                )
        return pl.DataFrame(rows)

    def test_adds_a_label_per_row(self):
        df = self._frame()
        out = tag_frame(df)
        self.assertIn("behavior_label", out.columns)
        self.assertEqual(out.height, df.height)

    def test_tags_are_computed_per_symbol_not_pooled(self):
        """
        A quiet instrument and a violent one must each be judged against their
        OWN range. Pooling would label every bar of the calm pair 'low'.
        """
        rng = np.random.default_rng(3)
        calm = pl.DataFrame(
            {
                "symbol": ["CALM"] * 400,
                "timestamp": list(range(400)),
                "natr_14": np.abs(rng.normal(0.01, 0.002, 400)),
                "ppo": rng.normal(0, 0.01, 400),
            }
        )
        wild = pl.DataFrame(
            {
                "symbol": ["WILD"] * 400,
                "timestamp": list(range(400)),
                "natr_14": np.abs(rng.normal(5.0, 1.0, 400)),
                "ppo": rng.normal(0, 3.0, 400),
            }
        )
        out = tag_frame(pl.concat([calm, wild]))
        for sym in ("CALM", "WILD"):
            labels = set(
                out.filter((pl.col("symbol") == sym) & (pl.col("behavior_label") != LABEL_COLD))[
                    "behavior_label"
                ].to_list()
            )
            self.assertGreater(
                len(labels), 1, f"{sym} collapsed to a single label — pooled?"
            )

    def test_missing_column_is_rejected(self):
        with self.assertRaises(ValueError):
            tag_frame(pl.DataFrame({"symbol": ["A"], "natr_14": [0.1]}))


class TestFrame(unittest.TestCase):
    def test_empty_input_yields_typed_empty_frame(self):
        f = to_frame([])
        self.assertEqual(f.height, 0)
        self.assertIn("net_ev_r", f.columns)

    def test_sorted_best_first(self):
        cells = score_ledger(_ledger({"a": (60, 40), "b": (20, 80)}), CAND)
        f = to_frame(cells)
        self.assertEqual(f["behavior"][0], "a")


if __name__ == "__main__":
    unittest.main()
