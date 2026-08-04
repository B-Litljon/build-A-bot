"""
Tests for the V4 investor benchmark gate (added 2026-08-03).

The gate exists because every other threshold in investor_train_model
measures lift over RANDOM, which cannot tell "better than guessing" from
"better than doing nothing". On 2026-07-03 a model passed all of them
while losing to an equal-weighted basket of the same universe.

Covers: the sector cap the gate simulates, the month-pairing that turns
scores into realised returns, and — most importantly — that an
unmeasurable model fails closed rather than sliding through.

Glossary:
    _mk_prices -- synthetic dates x symbols price matrix; every symbol
        flat except those named in `winners`, so the "right" basket has a
        known, checkable return.
    FakeSectors -- patched SECTORS map, so cap behaviour is tested against
        a fixed layout rather than the live universe's 11 sectors.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts import investor_train_model as itm  # noqa: E402


def _mk_prices(symbols, days=90, winners=(), winner_gain=0.50):
    """Flat prices at 100, except `winners` which rise steadily."""
    idx = pd.date_range("2024-01-01", periods=days, freq="B", tz="UTC")
    data = {}
    for s in symbols:
        if s in winners:
            data[s] = np.linspace(100.0, 100.0 * (1 + winner_gain), days)
        else:
            data[s] = np.full(days, 100.0)
    return pd.DataFrame(data, index=idx)


class TestSectorCappedPick(unittest.TestCase):
    def test_cap_limits_picks_per_sector(self):
        syms = [f"S{i}" for i in range(6)]
        sectors = {s: "tech" for s in syms[:4]}
        sectors.update({s: "energy" for s in syms[4:]})
        # descending scores => S0 best
        scores = np.arange(len(syms))[::-1].astype(float)

        with mock.patch.object(itm, "SECTORS", sectors):
            picked = itm._sector_capped_pick(syms, scores, k=4, cap=2)

        self.assertEqual(len(picked), 4)
        self.assertEqual(picked[:2], ["S0", "S1"])       # best two tech
        self.assertNotIn("S2", picked)                    # tech already full
        self.assertEqual(sorted(picked[2:]), ["S4", "S5"])

    def test_returns_fewer_than_k_when_caps_exhaust_universe(self):
        syms = ["A", "B", "C"]
        with mock.patch.object(itm, "SECTORS", {s: "one" for s in syms}):
            picked = itm._sector_capped_pick(syms, np.array([3.0, 2.0, 1.0]), k=8, cap=2)
        self.assertEqual(picked, ["A", "B"])

    def test_unknown_symbols_share_the_unknown_bucket(self):
        syms = ["X", "Y", "Z"]
        with mock.patch.object(itm, "SECTORS", {}):
            picked = itm._sector_capped_pick(syms, np.array([3.0, 2.0, 1.0]), k=8, cap=2)
        self.assertEqual(picked, ["X", "Y"])


class TestFoldBasketMonths(unittest.TestCase):
    def setUp(self):
        self.syms = [f"S{i}" for i in range(10)]
        self.sectors = {s: f"sec{i}" for i, s in enumerate(self.syms)}

    def test_perfect_picker_beats_equal_weight(self):
        winners = ("S0", "S1")
        px = _mk_prices(self.syms, winners=winners)
        dates = px.index

        test_df = pd.DataFrame(
            [(d, s) for d in dates for s in self.syms], columns=["date", "symbol"]
        )
        # score the winners top
        scores = np.array([10.0 if s in winners else 0.0 for _, s in
                           test_df[["date", "symbol"]].itertuples(index=False)])

        with mock.patch.object(itm, "SECTORS", self.sectors):
            months = itm._fold_basket_months(test_df, scores, px)

        self.assertGreater(len(months), 0)
        for _, basket, bench in months:
            self.assertGreater(basket, bench)

    def test_flat_universe_gives_zero_excess(self):
        px = _mk_prices(self.syms)          # everything flat
        dates = px.index
        test_df = pd.DataFrame(
            [(d, s) for d in dates for s in self.syms], columns=["date", "symbol"]
        )
        scores = np.zeros(len(test_df))

        with mock.patch.object(itm, "SECTORS", self.sectors):
            months = itm._fold_basket_months(test_df, scores, px)

        for _, basket, bench in months:
            self.assertAlmostEqual(basket, bench, places=9)

    def test_symbols_missing_prices_are_excluded(self):
        px = _mk_prices(self.syms, winners=("S0",))
        px["S0"] = np.nan                    # winner has no usable prices
        dates = px.index
        test_df = pd.DataFrame(
            [(d, s) for d in dates for s in self.syms], columns=["date", "symbol"]
        )
        scores = np.array([10.0 if s == "S0" else 1.0 for _, s in
                           test_df[["date", "symbol"]].itertuples(index=False)])

        with mock.patch.object(itm, "SECTORS", self.sectors):
            months = itm._fold_basket_months(test_df, scores, px)

        # S0 must not appear, so the basket cannot inherit its return
        for _, basket, bench in months:
            self.assertTrue(np.isfinite(basket))
            self.assertAlmostEqual(basket, bench, places=9)

    def test_holding_period_may_close_outside_the_test_window(self):
        """
        The last month-end in a fold pairs with the next month-end, which
        lies beyond the window. That is the live behaviour and must not be
        dropped.
        """
        px = _mk_prices(self.syms, days=90)
        dates = px.index
        # test window covers only the first two months
        window = dates[dates < dates[0] + pd.Timedelta(days=62)]
        test_df = pd.DataFrame(
            [(d, s) for d in window for s in self.syms], columns=["date", "symbol"]
        )
        scores = np.zeros(len(test_df))

        with mock.patch.object(itm, "SECTORS", self.sectors):
            months = itm._fold_basket_months(test_df, scores, px)

        self.assertGreaterEqual(len(months), 2)


class TestBenchmarkGateFailsClosed(unittest.TestCase):
    def test_load_close_matrix_returns_none_on_missing_file(self):
        with mock.patch.object(itm, "_RAW_PATH", Path("/nonexistent/nope.parquet")):
            self.assertIsNone(itm._load_close_matrix())

    def test_floor_sits_above_the_measurement_noise_band(self):
        """
        A zero floor gates on noise. Measured 2026-08-03: four reasonable
        implementations of this same quantity spanned ~50 bps, and a
        one-row-per-day label rounding change moved it 23 bps. The floor
        must stay meaningfully above zero.
        """
        self.assertGreaterEqual(itm.GATE_BENCH_MIN_EXCESS_BPS, 20.0)

    def test_stability_is_required_across_multiple_alignments(self):
        """
        The multi-alignment requirement is the part of the gate doing the
        real work — a single measurement passed the very model this gate
        exists to catch.
        """
        self.assertGreaterEqual(len(itm.GATE_BENCH_ALIGNMENTS), 3)
        self.assertEqual(itm.GATE_BENCH_MIN_PASS_SHARE, 1.0)

    def test_deployed_constants_match_the_orchestrator(self):
        """
        The gate must simulate the basket actually traded. If the
        orchestrator's depth or cap changes, this test fails loudly rather
        than the gate quietly measuring the wrong thing.
        """
        from scripts import portfolio_orchestrator as po
        self.assertEqual(itm.TOP_K, po.TOP_K)
        self.assertEqual(itm.SECTOR_CAP, po.SECTOR_CAP)


if __name__ == "__main__":
    unittest.main()
