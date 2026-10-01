"""
Tests for the pre-featurization NY-rollover bar exclusion (2026-10-01).

The soak evidence: the ONLY bars that ever cleared the Angel bar live were
illiquid transition bars (the 16:55-17:30 ET rollover, the thin Asian open) —
the model had learned its highest probabilities from those bars' excursions.
Cause: the chop veto drops rollover rows only as trade ENTRIES, after target
generation, so the bars still feed the rolling features of neighboring rows.

The fix under test: _exclude_rollover_bars drops those bars from the RAW frame
BEFORE feature generation, using THE Gate C window definition
(execution.risk_manager.get_blackout_window_et — no third copy of the window),
DST-correct via America/New_York. The post-labeling chop veto must keep
running: the two mechanisms are complementary, not redundant.

Glossary:
    test_summer_and_winter_rollover_bars_excluded -- the DST contract: 16:55-
        17:30 ET is 20:55-21:30 UTC in July (EDT) and 21:55-22:30 UTC in
        January (EST); both exclude exactly the inside-window bars, and 17:30
        NY exactly is KEPT (the live gate's end-exclusive convention).
    test_window_comes_from_gate_c_definition -- RISK_BLACKOUT_ET re-targets the
        exclusion identically to the live gate (one definition, not three).
    test_naive_timestamps_assumed_utc -- the retrainer-wide naive-is-UTC rule.
    test_flag_off_* -- RETRAIN_EXCLUDE_ROLLOVER_BARS=0 restores the pre-fix
        behaviour row for row.
    test_exclusion_on_absent_off_present / test_engineer_parity_with_lab_when_exclusion_disabled --
        (c) the end-to-end acceptance bars are ABSENT from the engineered
        frame when on, present (through feature generation) when off; and the
        lab's frame-parity contract still holds with the exclusion disabled.
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from core.retrainer._pipeline import (  # noqa: E402
    RETRAIN_EXCLUDE_ROLLOVER_BARS,
    _exclude_rollover_bars,
    _rollover_exclusion_mask,
)
from src.execution.risk_manager import (  # noqa: E402
    _DEFAULT_BLACKOUT_ET,
    get_blackout_window_et,
)

_PIPELINE_MOD = sys.modules["core.retrainer._pipeline"]


def _ts_frame(utc_datetimes) -> pl.DataFrame:
    n = len(utc_datetimes)
    return pl.DataFrame(
        {
            "timestamp": utc_datetimes,
            "symbol": ["GBP_JPY"] * n,
            "open": [150.0] * n,
            "high": [150.1] * n,
            "low": [149.9] * n,
            "close": [150.0] * n,
            "volume": [10.0] * n,
        }
    )


SUMMER_TS = [
    datetime(2026, 7, 15, 20, 50, tzinfo=timezone.utc),  # 16:50 NY — before
    datetime(2026, 7, 15, 20, 55, tzinfo=timezone.utc),  # 16:55 NY — start inclusive
    datetime(2026, 7, 15, 21, 0, tzinfo=timezone.utc),   # 17:00 NY — INSIDE
    datetime(2026, 7, 15, 21, 29, tzinfo=timezone.utc),  # 17:29 NY — INSIDE
    datetime(2026, 7, 15, 21, 30, tzinfo=timezone.utc),  # 17:30 NY — end exclusive
    datetime(2026, 7, 15, 21, 31, tzinfo=timezone.utc),  # after
]
SUMMER_EXPECTED = [False, True, True, True, False, False]

WINTER_TS = [
    datetime(2026, 1, 14, 21, 50, tzinfo=timezone.utc),  # 16:50 NY — before
    datetime(2026, 1, 14, 21, 55, tzinfo=timezone.utc),  # 16:55 NY — start inclusive
    datetime(2026, 1, 14, 22, 0, tzinfo=timezone.utc),   # 17:00 NY — INSIDE
    datetime(2026, 1, 14, 22, 30, tzinfo=timezone.utc),  # 17:30 NY — end exclusive
    datetime(2026, 1, 14, 22, 31, tzinfo=timezone.utc),  # after
]
WINTER_EXPECTED = [False, True, True, False, False]


class TestRolloverExclusionMask(unittest.TestCase):
    def test_summer_and_winter_rollover_bars_excluded(self):
        """DST contract on both sides of the boundary (July vs January)."""
        summer = _rollover_exclusion_mask(_ts_frame(SUMMER_TS))
        np.testing.assert_array_equal(summer.to_numpy(), np.array(SUMMER_EXPECTED))
        winter = _rollover_exclusion_mask(_ts_frame(WINTER_TS))
        np.testing.assert_array_equal(winter.to_numpy(), np.array(WINTER_EXPECTED))

    def test_window_comes_from_gate_c_definition(self):
        """The mask must follow THE Gate C window source: the default matches
        _DEFAULT_BLACKOUT_ET, and a RISK_BLACKOUT_ET retune re-targets the
        exclusion identically to the live gate — no third copy of the window."""
        self.assertEqual(get_blackout_window_et()[0].strftime("%H:%M"), "16:55")
        self.assertEqual(get_blackout_window_et()[1].strftime("%H:%M"), "17:30")

        import os

        os.environ["RISK_BLACKOUT_ET"] = "23:00-01:00"
        try:
            shifted = _rollover_exclusion_mask(
                _ts_frame(
                    [
                        datetime(2026, 7, 16, 3, 0, tzinfo=timezone.utc),  # 23:00 NY
                        datetime(2026, 7, 16, 5, 0, tzinfo=timezone.utc),  # 01:00 NY
                    ]
                )
            )
            np.testing.assert_array_equal(shifted.to_numpy(), np.array([True, False]))
        finally:
            del os.environ["RISK_BLACKOUT_ET"]

    def test_naive_timestamps_assumed_utc(self):
        naive = _ts_frame([t.replace(tzinfo=None) for t in SUMMER_TS])
        mask = _rollover_exclusion_mask(naive)
        np.testing.assert_array_equal(mask.to_numpy(), np.array(SUMMER_EXPECTED))

    def test_flag_off_mask_all_false(self):
        with patch.object(_PIPELINE_MOD, "RETRAIN_EXCLUDE_ROLLOVER_BARS", False):
            mask = _rollover_exclusion_mask(_ts_frame(SUMMER_TS))
        np.testing.assert_array_equal(
            mask.to_numpy(), np.zeros(len(SUMMER_TS), dtype=bool)
        )

    def test_missing_timestamp_column_is_a_safe_noop(self):
        df = pl.DataFrame({"close": [1.0, 2.0]})
        mask = _rollover_exclusion_mask(df)
        self.assertEqual(len(mask), 2)
        self.assertFalse(mask.any())

    def test_empty_frame(self):
        mask = _rollover_exclusion_mask(_ts_frame([]))
        self.assertEqual(len(mask), 0)

    def test_unparseable_window_is_a_safe_noop(self):
        import os

        os.environ["RISK_BLACKOUT_ET"] = "not-a-window"
        try:
            mask = _rollover_exclusion_mask(_ts_frame(SUMMER_TS))
            self.assertFalse(mask.any())
        finally:
            del os.environ["RISK_BLACKOUT_ET"]


class TestExcludeRolloverBars(unittest.TestCase):
    def test_drops_only_inside_window(self):
        df = _ts_frame(SUMMER_TS)
        out, n = _exclude_rollover_bars(df)
        self.assertEqual(n, 3)
        self.assertEqual(out.height, 3)
        kept = out["timestamp"].to_list()
        for dropped in (SUMMER_TS[1], SUMMER_TS[2], SUMMER_TS[3]):
            self.assertNotIn(dropped, kept)
        self.assertIn(SUMMER_TS[4], kept)  # 17:30 NY exactly survives

    def test_flag_off_keeps_all_rollover_bars(self):
        df = _ts_frame(SUMMER_TS)
        with patch.object(_PIPELINE_MOD, "RETRAIN_EXCLUDE_ROLLOVER_BARS", False):
            out, n = _exclude_rollover_bars(df)
        self.assertEqual(n, 0)
        self.assertEqual(out.height, df.height)
        self.assertEqual(out["timestamp"].to_list(), df["timestamp"].to_list())

    def test_empty_frame_passthrough(self):
        out, n = _exclude_rollover_bars(_ts_frame([]))
        self.assertEqual((n, out.height), (0, 0))


def _synth_bars(days=3, symbols=("GBP_JPY", "EUR_JPY"), seed=7):
    """M15 bars starting 2026-01-01 (EST: rollover = 21:55-22:30 UTC)."""
    rng = np.random.default_rng(seed)
    n_bars = days * 96
    ts = pl.datetime_range(
        datetime(2026, 1, 1),
        datetime(2026, 1, 1) + timedelta(minutes=15 * (n_bars - 1)),
        interval="15m",
        eager=True,
    )
    bars = {}
    for k, sym in enumerate(symbols):
        close = 150.0 + k + np.cumsum(rng.normal(0, 0.05, n_bars))
        bars[sym] = pl.DataFrame(
            {
                "timestamp": ts,
                "open": close,
                "high": close + 0.05,
                "low": close - 0.05,
                "close": close,
                "volume": np.full(n_bars, 10.0),
            }
        )
    return bars


EXPECTED_ROLLOVER_PER_SYMBOL_PER_DAY = 2  # M15 grid: 17:00 and 17:15 NY land
# inside the 35-minute 16:55-17:30 window (16:55 itself is not grid-aligned).


def _stack(bars):
    """Concat per-symbol frames into the retrainer's stacked layout."""
    parts: list[pl.DataFrame] = []
    for sym, frame in bars.items():
        if frame.is_empty():
            continue
        parts.append(frame.with_columns(pl.lit(sym).alias("symbol")))
    return pl.concat(parts, how="vertical_relaxed").sort(["symbol", "timestamp"])


class TestEngineeredFrameExclusion(unittest.TestCase):
    """(c) End-to-end: bars in the window are absent from the engineered frame
    with the flag on, present (through feature generation) with it off."""

    @classmethod
    def setUpClass(cls):
        cls.bars = _synth_bars()
        cls.stacked = _stack(cls.bars)

    def _engineer(self, stacked):
        from core.retrainer import _purge_boundary_tail, _tail_cutoff_by_symbol
        from core.retrainer._features import engineer_features_and_labels
        from src.execution.risk_manager import RiskProfile

        feats, cols, chop = engineer_features_and_labels(
            stacked,
            sl_mult=2.0,
            angel_mult=1.0,
            tp_mult=4.0,
            max_hold=45,
            survival_bars=5,
            htf_timeframe="1h",
            risk_profile=RiskProfile.for_asset_class("forex"),
            alpha_table=None,
        )
        cutoffs = _tail_cutoff_by_symbol(stacked, 45)
        feats, purged = _purge_boundary_tail(feats, cutoffs)
        return feats, cols, chop, purged

    @staticmethod
    def _roll_count(frame) -> int:
        ny = frame["timestamp"].dt.convert_time_zone("America/New_York")
        secs = (
            ny.dt.hour().cast(pl.Int64) * 3600
            + ny.dt.minute().cast(pl.Int64) * 60
            + ny.dt.second().cast(pl.Int64)
        )
        return int(
            ((secs >= 16 * 3600 + 55 * 60) & (secs < 17 * 3600 + 30 * 60)).sum()
        )

    def test_exclusion_on_absent_off_present(self):
        """ON (module default): zero window bars enter the engineered frame
        AT ALL. OFF: the bars survive FEATURE generation and their excursions
        leak into neighboring rows' features — the observable the fix removes.
        (The post-labeling chop veto still drops the window bars as entries
        either way; this tests the feature-contamination layer.)"""
        self.assertIs(RETRAIN_EXCLUDE_ROLLOVER_BARS, True)
        feats_on, _, _, _ = self._engineer(self.stacked)
        self.assertEqual(self._roll_count(feats_on), 0)

        from core.retrainer._common import (
            V3BaseFeatures,
            V3CostFeatures,
            V3HTFFeatures,
            V3SessionFeatures,
        )

        def generate(stack):
            df = stack
            for gen in (
                V3BaseFeatures(),
                V3HTFFeatures(timeframe="1h"),
                V3SessionFeatures(),
                V3CostFeatures(
                    alpha_table=None, default_alpha=0.15, regime_window=260
                ),
            ):
                df = gen.generate(df)
            return df

        expected = EXPECTED_ROLLOVER_PER_SYMBOL_PER_DAY * 3 * len(self.bars)
        featured_unfiltered = generate(self.stacked)
        self.assertEqual(
            self._roll_count(featured_unfiltered), expected,
            "flag-off path must leave rollover bars in the FEATURED frame",
        )

        # The contamination observable: 17:45 NY sits two bars after the
        # window end; its rolling features must differ once the window bars
        # are excluded. ON arm = exclusion applied; OFF arm = raw stack. Both
        # engineered by the IDENTICAL generator list.
        def feats_at_1745(df):
            ny = df["timestamp"].dt.convert_time_zone("America/New_York")
            ny_min = ny.dt.hour().cast(pl.Int64) * 60 + ny.dt.minute().cast(pl.Int64)
            return (
                df.with_columns(pl.Series("ny_min", ny_min))
                .filter(pl.col("ny_min") == 17 * 60 + 45)
                .select("symbol", "natr_14", "bb_pct_b", "vol_rel")
                .sort("symbol")
            )

        filtered_stack, _ = _exclude_rollover_bars(self.stacked)
        row_on = feats_at_1745(generate(filtered_stack))
        row_off = feats_at_1745(generate(self.stacked))
        self.assertEqual(row_on["symbol"].to_list(), row_off["symbol"].to_list())
        self.assertNotEqual(
            row_on.select("natr_14", "bb_pct_b", "vol_rel").rows(),
            row_off.select("natr_14", "bb_pct_b", "vol_rel").rows(),
            "neighbor features must differ once rollover bars are excluded",
        )

    def test_dropped_count_matches_expected_window_bars(self):
        stacked_on, n_dropped = _exclude_rollover_bars(self.stacked)
        self.assertEqual(
            n_dropped, EXPECTED_ROLLOVER_PER_SYMBOL_PER_DAY * 3 * len(self.bars)
        )
        self.assertEqual(stacked_on.height + n_dropped, self.stacked.height)

    def test_engineer_parity_with_lab_when_exclusion_disabled(self):
        """The lab's frame-parity contract is untouched: with the exclusion
        OFF, production engineering on raw bars == lab build_frame (v3_base)."""
        from lab.frames import build_frame
        from lab.spec import FeatureSpec
        from polars.testing import assert_frame_equal

        spec = FeatureSpec(
            name="rollover-parity",
            symbols=tuple(self.bars.keys()),
            feature_sets=("v3_base",),
            use_spread_table=False,
        )
        with patch.object(_PIPELINE_MOD, "RETRAIN_EXCLUDE_ROLLOVER_BARS", False):
            feats_prod, cols_prod, chop_prod, purged_prod = self._engineer(
                self.stacked
            )
        result = build_frame(spec, self.bars)
        self.assertEqual(tuple(cols_prod), result.feature_cols)
        self.assertAlmostEqual(chop_prod, result.chop_veto_rate, places=12)
        self.assertEqual(purged_prod, result.purged_tail_rows)
        assert_frame_equal(feats_prod, result.df)


if __name__ == "__main__":
    unittest.main()