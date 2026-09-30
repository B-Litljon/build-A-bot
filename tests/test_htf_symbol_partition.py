"""
Pin cross-symbol partitioning in the feature/target generators.

V3HTFFeatures builds htf_bars as a vertical concat of per-symbol blocks, and
V3DirectionalTarget shifts closes forward across the whole frame. Any rolling
or shift op left unpartitioned bleeds the previous symbol's rows into the next
symbol's window start. htf_vol_rel did exactly that (verified 2026-09-30):
symbol B's first ~19 HTF bars divided by a 20-bar mean containing symbol A's
trailing htf_volume, so with wildly different per-symbol volumes the warm-up
rows read absurd values (e.g. 19.6x) instead of 1.0.

Glossary:
    bleed -- a windowed statistic computed across symbol boundaries because
        the expression lacked .over("symbol") on a pooled frame.
    warm-up bars -- the first window_size-1 rows of a symbol's block, where a
        rolling_mean has fewer samples; unpartitioned, these are exactly the
        rows that pick up foreign-symbol volume.

These generators are legacy (the retrainer engineers its own features/labels
in core/retrainer), but ml/feature_pipeline.main() still runs them, so their
train/live behaviour stays pinned here.
"""

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from ml.features.v3_features import V3HTFFeatures  # noqa: E402
from ml.targets.v3_targets import V3DirectionalTarget  # noqa: E402


def _htf_frame(symbols: list[str], n: int, volumes: dict[str, float]) -> pl.DataFrame:
    """A minimal 1m OHLCV frame: n bars per symbol, every bar identical
    within a symbol, so any htf statistic other than the intended one moves
    the value under test."""
    ts = [datetime(2026, 1, 1) + timedelta(minutes=i) for i in range(n)]
    parts = [
        pl.DataFrame(
            {
                "timestamp": ts,
                "symbol": [sym] * n,
                "open": [1.0] * n,
                "high": [1.0] * n,
                "low": [1.0] * n,
                "close": [1.0] * n,
                "volume": [volumes[sym]] * n,
            }
        )
        for sym in symbols
    ]
    return pl.concat(parts).sort(["symbol", "timestamp"])


class TestHTFVolRelSymbolPartition(unittest.TestCase):
    HTF_WINDOW = 20  # htf_vol_rel's rolling_mean window

    def test_second_symbol_warmup_does_not_bleed_first_symbol_volume(self):
        """Two symbols with 3-orders-of-magnitude different volumes: every
        htf_vol_rel in the second symbol's block must be 1.0 (all bars
        identical within the symbol), including the warm-up bars that would
        read absurd values if the rolling mean crossed symbols. Under the
        pre-fix code the second symbol's later HTF bars mix the first
        symbol's volume into their 20-bar window and leave 1.0."""
        n = 120  # 1m bars → 24 HTF bars per symbol: warm-up AND full windows
        df = _htf_frame(
            ["QUIET", "LOUD"], n, volumes={"QUIET": 10.0, "LOUD": 10_000.0}
        )
        out = V3HTFFeatures(timeframe="5m").generate(df)

        self.assertIn("htf_vol_rel", out.columns)
        for sym in ("QUIET", "LOUD"):
            with self.subTest(symbol=sym):
                rel = out.filter(pl.col("symbol") == sym)["htf_vol_rel"]
                # The first 5 1m bars join to nothing (their HTF bar is not
                # yet available — the available_at guard); everything after
                # that must carry the symbol's own relative volume.
                self.assertEqual(rel.null_count(), 5)
                np.testing.assert_allclose(
                    rel.drop_nulls().to_numpy(), 1.0, rtol=1e-12,
                    err_msg=f"{sym}: htf_vol_rel bled across symbols",
                )

    def test_single_symbol_frame_unpartitioned_path_unchanged(self):
        """Frames without a symbol column must take the same unpartitioned
        expression the live path always used — the has_symbol guard is
        a no-op there, and within one symbol's contiguous block a
        .over("symbol") rolling mean is identical to the flat one."""
        n = 120
        df = _htf_frame(["SOLO"], n, volumes={"SOLO": 5.0}).drop("symbol")
        out = V3HTFFeatures(timeframe="5m").generate(df)
        self.assertNotIn("symbol", out.columns)
        self.assertIn("htf_vol_rel", out.columns)
        rel = out["htf_vol_rel"]
        self.assertEqual(rel.null_count(), 5)
        np.testing.assert_allclose(rel.drop_nulls().to_numpy(), 1.0, rtol=1e-12)

    def test_live_single_symbol_value_frame_matches_nosymbol_frame(self):
        """The live path hands MLStrategy a single-symbol frame that still
        CARRIES a symbol column (a literal value per row, ml_strategy.py
        "Option A"). So live takes the has_symbol arm — the .over("symbol")
        one. That arm must produce a frame identical to the historical
        no-symbol path for one symbol, which is what proves the fix is a
        no-op for live."""
        df = _htf_frame(["GBP_JPY"], 120, volumes={"GBP_JPY": 42.0})
        cols = [
            "timestamp", "htf_rsi_14", "htf_vol_rel",
            "htf_bb_pct_b", "htf_trend_agreement",
        ]
        with_sym = V3HTFFeatures(timeframe="5m").generate(df)
        without_sym = V3HTFFeatures(timeframe="5m").generate(df.drop("symbol"))
        self.assertTrue(with_sym.select(cols).equals(without_sym.select(cols)))

    def test_output_schema_unchanged(self):
        """The fix must not add, drop, or retype columns (constraint:
        htf_vol_rel schema identical)."""
        n = 120  # 24 HTF bars per symbol, so full 20-bar windows exist
        df = _htf_frame(["AAA", "BBB"], n, volumes={"AAA": 1.0, "BBB": 2.0})
        out = V3HTFFeatures(timeframe="5m").generate(df)
        self.assertNotIn("htf_volume", out.columns)
        self.assertNotIn("available_at", out.columns)
        self.assertIn("symbol", out.columns)  # symbol frames stay symbol frames
        self.assertEqual(out.schema["htf_vol_rel"], pl.Float64)

        # And no leftover per-symbol intermediate columns.
        self.assertFalse(
            [c for c in out.columns if c.startswith("_htf_")],
            "intermediate _htf_* columns must be dropped",
        )


class TestDirectionalTargetSymbolPartition(unittest.TestCase):
    def test_last_rows_of_nonfinal_symbol_get_null_not_next_symbol_close(self):
        """Unpartitioned shift(-lookahead) copies the NEXT symbol's opening
        closes onto this symbol's final rows, labelling them 1/0 instead of
        null. The final rows of a non-final symbol must be null."""
        n = 25  # lookahead default is 15; rows >= n-15 within a block are null
        df = _htf_frame(["AAA", "ZZZ"], n, volumes={"AAA": 1.0, "ZZZ": 1.0})
        out = V3DirectionalTarget().generate(df)

        tail_a = out.filter(pl.col("symbol") == "AAA")["target"].tail(15).to_list()
        self.assertEqual(
            set(tail_a), {None},
            "AAA's final 15 rows must be null, not labelled from ZZZ's closes",
        )
        tail_z = out.filter(pl.col("symbol") == "ZZZ")["target"].tail(15).to_list()
        self.assertEqual(set(tail_z), {None})
        # Rows with an in-symbol future stay labelled, so the fix isn't nulling
        # everything.
        self.assertEqual(
            out.filter(pl.col("target").is_not_null()).height, 2 * (n - 15)
        )


if __name__ == "__main__":
    unittest.main()