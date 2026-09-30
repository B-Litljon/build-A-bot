"""
Tests for analysis.decision_grader — grading the live bot's recorded decisions.

The grader's whole value is that it produces MORE honest evidence, so its
failure modes are all forms of dishonesty: silently inventing outcomes for
decisions that have not resolved yet, mis-joining a decision to a different
bar's result, or dropping records because one telemetry line was truncated.
Each of those is pinned below.

Glossary:
    _jsonl -- writes a temporary telemetry file, including the malformed and
        non-"bar" lines the real logs contain, so parsing is tested against
        what the writer actually produces rather than an idealised sample.
    _bars -- a graded-bar frame standing in for the answer key.
    _basket_with_censored_tail -- a dense answer key in the runner's shape
        (last LOOKAHEAD_BARS bars per symbol defaulted to won=0), standing in
        for what _compute_devil_targets_atr hands grade_decisions.
"""

import json
import sys
import tempfile
import unittest
import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.decision_grader import (  # noqa: E402
    LOOKAHEAD_BARS,
    behavior_breakdown,
    calibration_table,
    grade_decisions,
    load_decisions,
    purge_unresolvable_tail,
    threshold_sweep,
)


def _jsonl(records, tmpdir, name="events-2026-08-24.jsonl"):
    p = Path(tmpdir) / name
    with open(p, "w") as fh:
        for r in records:
            fh.write(r if isinstance(r, str) else json.dumps(r))
            fh.write("\n")
    return str(Path(tmpdir) / "events-*.jsonl")


def _bar(sym, ts, angel, devil=None, close=100.0, outcome="angel_reject"):
    return {
        "ts": "2026-08-24T00:00:00+00:00",
        "ev": "bar",
        "sym": sym,
        "bar_ts": ts,
        "close": close,
        "angel": angel,
        "devil": devil,
        "outcome": outcome,
    }


def _bars(rows):
    """rows: (symbol, timestamp, won[, behavior])"""
    return pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "timestamp": [r[1] for r in rows],
            "won": [r[2] for r in rows],
            "behavior_label": [r[3] if len(r) > 3 else "mixed_normal" for r in rows],
        }
    )


class TestLoadDecisions(unittest.TestCase):
    def test_reads_only_bar_events(self):
        with tempfile.TemporaryDirectory() as td:
            pat = _jsonl(
                [
                    _bar("GBP_JPY", "2026-08-20 10:00:00+00:00", 0.31),
                    {"ts": "x", "ev": "heartbeat", "sym": "GBP_JPY", "median": 0.1},
                    {"ts": "x", "ev": "stream", "kind": "disconnect"},
                ],
                td,
            )
            d = load_decisions(pat)
        self.assertEqual(d.height, 1)
        self.assertEqual(d["symbol"][0], "GBP_JPY")

    def test_a_truncated_line_does_not_lose_the_file(self):
        """The telemetry writer is best-effort; a crash mid-write is expected."""
        with tempfile.TemporaryDirectory() as td:
            pat = _jsonl(
                [
                    _bar("GBP_JPY", "2026-08-20 10:00:00+00:00", 0.31),
                    '{"ts": "2026-08-20", "ev": "bar", "sym": "EUR_J',  # truncated
                    _bar("EUR_JPY", "2026-08-20 10:15:00+00:00", 0.12),
                ],
                td,
            )
            d = load_decisions(pat)
        self.assertEqual(d.height, 2)

    def test_records_without_a_timestamp_are_dropped(self):
        with tempfile.TemporaryDirectory() as td:
            bad = _bar("GBP_JPY", None, 0.3)
            pat = _jsonl([bad, _bar("EUR_JPY", "2026-08-20 10:00:00+00:00", 0.2)], td)
            d = load_decisions(pat)
        self.assertEqual(d.height, 1)
        self.assertEqual(d["symbol"][0], "EUR_JPY")

    def test_duplicate_decisions_are_collapsed(self):
        """A reconnect can re-score the same bar; it is still ONE decision."""
        with tempfile.TemporaryDirectory() as td:
            b = _bar("GBP_JPY", "2026-08-20 10:00:00+00:00", 0.31)
            pat = _jsonl([b, dict(b)], td)
            self.assertEqual(load_decisions(pat).height, 1)

    def test_empty_input_returns_typed_frame(self):
        with tempfile.TemporaryDirectory() as td:
            d = load_decisions(str(Path(td) / "nothing-*.jsonl"))
        self.assertEqual(d.height, 0)
        self.assertIn("angel", d.columns)


class TestGrading(unittest.TestCase):
    def _dec(self, rows):
        return pl.DataFrame(
            {
                "symbol": [r[0] for r in rows],
                "bar_ts": [r[1] for r in rows],
                "live_close": [100.0] * len(rows),
                "angel": [r[2] for r in rows],
                "devil": [None] * len(rows),
                "verdict": ["angel_reject"] * len(rows),
            }
        )

    def test_decision_gets_its_own_bars_outcome(self):
        dec = self._dec([("GBP_JPY", "2026-08-20 10:00:00+00:00", 0.31)])
        bars = _bars(
            [
                ("GBP_JPY", "2026-08-20 09:45:00", 0),
                ("GBP_JPY", "2026-08-20 10:00:00", 1),  # the matching bar
                ("EUR_JPY", "2026-08-20 10:00:00", 0),  # right time, wrong symbol
            ]
        )
        g = grade_decisions(dec, bars)
        self.assertEqual(g.height, 1)
        self.assertEqual(g["won"][0], 1)

    def test_unresolved_decisions_are_dropped_not_guessed(self):
        """
        A decision too recent for the 45-bar walk has NO outcome. Inventing one
        is the exact self-deception this module exists to remove.

        The answer key here is long enough to be resolvable, so the only
        unmatched decision is genuinely outside the key entirely — the
        censored-tail case is pinned separately in TestUnresolvableTail, where
        the bar IS in the key but its walk truncated.
        """
        dec = self._dec(
            [
                ("GBP_JPY", "2026-08-20 10:00:00+00:00", 0.31),
                ("GBP_JPY", "2026-08-24 23:45:00+00:00", 0.55),  # no answer key
            ]
        )
        bars = _bars([("GBP_JPY", "2026-08-20 10:00:00", 1)])
        g = grade_decisions(dec, bars)
        self.assertEqual(g.height, 1)
        self.assertEqual(g["angel"][0], 0.31)

    def test_degenerate_answer_key_still_grades(self):
        """
        A key shorter than the walk horizon is kept whole (the retrainer's
        defensive call), so its bars still join — here the one bar resolves.
        """
        dec = self._dec([("GBP_JPY", "2026-08-20 10:00:00+00:00", 0.31)])
        bars = _bars([("GBP_JPY", "2026-08-20 10:00:00", 1)])
        self.assertEqual(grade_decisions(dec, bars).height, 1)

    def test_iso_t_separator_and_space_both_join(self):
        """Telemetry and the bar frame format timestamps differently."""
        dec = self._dec([("GBP_JPY", "2026-08-20T10:00:00+00:00", 0.31)])
        bars = _bars([("GBP_JPY", "2026-08-20 10:00:00", 1)])
        self.assertEqual(grade_decisions(dec, bars).height, 1)

    def test_empty_inputs_are_safe(self):
        dec = self._dec([])
        self.assertEqual(grade_decisions(dec, _bars([])).height, 0)


class TestUnresolvableTail(unittest.TestCase):
    """
    _compute_devil_targets_atr returns a dense int8 with no nulls: bars whose
    forward walk truncated at the frame's right edge default to won=0, exactly
    like a real loss. Decisions on those bars must be DROPPED, not graded 0 —
    the drop path the docstring always promised (2026-09-30 recon, item 3).
    """

    @staticmethod
    def _basket_with_censored_tail(rows_per_symbol: int = 90, symbols=("GBP_JPY",)):
        """
        A dense answer key in the runner's shape (symbol-major blocks, one per
        symbol): the last LOOKAHEAD_BARS rows of each block carry won=0 that
        the walk never earned. Requires rows_per_symbol > LOOKAHEAD_BARS (a
        shorter key has no censored tail).
        """
        n = rows_per_symbol
        if n <= LOOKAHEAD_BARS:
            raise ValueError("use more than LOOKAHEAD_BARS rows for a censored tail")
        start = datetime.datetime(2026, 8, 20)

        def _block(sym: str) -> pl.DataFrame:
            ts = pl.datetime_range(
                start, start + datetime.timedelta(minutes=15 * (n - 1)),
                interval="15m", eager=True,
            )
            return pl.DataFrame(
                {
                    "symbol": pl.Series([sym] * n, dtype=pl.Utf8),
                    "timestamp": ts,
                    "won": pl.Series(
                        [1] * (n - LOOKAHEAD_BARS) + [0] * LOOKAHEAD_BARS,
                        dtype=pl.Int8,
                    ),
                    "behavior_label": ["mixed_normal"] * n,
                }
            )

        return pl.concat([_block(sym) for sym in symbols], how="vertical")

    def test_decision_in_last_lookahead_bars_is_dropped_not_graded_zero(self):
        bars = self._basket_with_censored_tail()
        tail_ts = bars["timestamp"][-1]  # a bar INSIDE the key, censored to 0
        dec = pl.DataFrame(
            {
                "symbol": ["GBP_JPY"],
                "bar_ts": [str(tail_ts)],
                "live_close": [100.0],
                "angel": [0.9],
                "devil": [None],
                "verdict": ["angel_reject"],
            }
        )
        graded = grade_decisions(dec, bars)
        self.assertEqual(
            graded.height, 0,
            "a censored-tail decision was graded instead of dropped",
        )

    def test_resolvable_loss_on_the_same_bar_is_kept(self):
        """A genuine 0 from RESOLVABLE history must survive the purge."""
        bars = self._basket_with_censored_tail()
        resolvable_ts = bars["timestamp"][10]
        bars = bars.with_columns(
            pl.when(pl.col("timestamp") == resolvable_ts)
            .then(pl.lit(0, dtype=pl.Int8))
            .otherwise(pl.col("won"))
            .alias("won")
        )
        dec = pl.DataFrame(
            {
                "symbol": ["GBP_JPY"],
                "bar_ts": [str(resolvable_ts)],
                "live_close": [100.0],
                "angel": [0.3],
                "devil": [None],
                "verdict": ["angel_reject"],
            }
        )
        graded = grade_decisions(dec, bars)
        self.assertEqual(graded.height, 1)
        self.assertEqual(graded["won"][0], 0)

    def test_purge_drops_exactly_lookahead_bars_per_symbol(self):
        bars = self._basket_with_censored_tail(
            rows_per_symbol=90, symbols=("GBP_JPY", "EUR_JPY")
        )
        purged = purge_unresolvable_tail(bars, LOOKAHEAD_BARS)
        self.assertEqual(bars.height - purged.height, LOOKAHEAD_BARS * 2)
        self.assertEqual(purged["won"].min(), 1)  # every censored 0 is gone
        # type contract: won stays Int8-compatible
        self.assertIn(purged["won"].dtype, (pl.Int8, pl.UInt8))

    def test_purge_degenerate_symbol_is_kept_untouched(self):
        """
        A key shorter than the walk horizon is kept whole — the retrainer's
        defensive call (dropping it would empty the key entirely).
        """
        bars = _bars(
            [
                ("GBP_JPY", "2026-08-20 10:00:00", 0),
                ("GBP_JPY", "2026-08-20 10:15:00", 1),
            ]
        )
        self.assertEqual(purge_unresolvable_tail(bars, LOOKAHEAD_BARS).height, 2)


class TestReports(unittest.TestCase):
    def _graded(self, rows):
        return pl.DataFrame(
            {
                "angel": [r[0] for r in rows],
                "won": [r[1] for r in rows],
                "behavior_label": [r[2] if len(r) > 2 else "mixed_normal" for r in rows],
            }
        )

    def test_calibration_reports_the_gap(self):
        # claims ~0.10, wins 50% -> under-confident by +0.40
        g = self._graded([(0.10, 1)] * 5 + [(0.10, 0)] * 5)
        t = calibration_table(g, bands=(0.0, 0.15, 1.01))
        self.assertEqual(t["n"][0], 10)
        self.assertAlmostEqual(t["win_rate"][0], 0.5)
        self.assertAlmostEqual(t["gap"][0], 0.4, places=4)

    def test_threshold_sweep_charges_the_toll(self):
        g = self._graded([(0.5, 1)] * 50 + [(0.5, 0)] * 50)
        free = threshold_sweep(g, 2.0, 4.0, 0.0, thresholds=(0.4,))
        tolled = threshold_sweep(g, 2.0, 4.0, 0.10, thresholds=(0.4,))
        self.assertGreater(free["net_ev_r"][0], tolled["net_ev_r"][0])
        # 50% wins at 2R = +0.5 EV gross, minus the 0.10 toll
        self.assertAlmostEqual(free["net_ev_r"][0], 0.5, places=3)
        self.assertAlmostEqual(tolled["net_ev_r"][0], 0.4, places=3)

    def test_sweep_skips_thresholds_with_no_decisions(self):
        g = self._graded([(0.10, 1), (0.12, 0)])
        self.assertEqual(threshold_sweep(g, 2.0, 4.0, 0.1, thresholds=(0.9,)).height, 0)

    def test_behavior_breakdown_flags_thin_cells(self):
        g = self._graded([(0.2, 1, "trend_high")] * 5 + [(0.2, 0, "range_low")] * 40)
        b = behavior_breakdown(g, min_n=30)
        rows = {r["behavior_label"]: r for r in b.iter_rows(named=True)}
        self.assertFalse(rows["trend_high"]["informative"])
        self.assertTrue(rows["range_low"]["informative"])

    def test_reports_handle_empty_input(self):
        empty = self._graded([])
        self.assertEqual(calibration_table(empty).height, 0)
        self.assertEqual(behavior_breakdown(empty).height, 0)


if __name__ == "__main__":
    unittest.main()
