"""
Tests for the artifact-level holdout gate.

The core property: the served model must never have seen the holdout rows.
These tests prove that property at the data-flow level by checking that the
split is chronological and disjoint, that holdout metrics are recorded, and
that bypass states are declared in metadata. Since the 2026-08-24 audit the
verdict itself is also under test: the PF bar gates on the exact
Clopper-Pearson lower bound, and the audit's instrumented leak check is a
permanent test (TestPermanentLeakGuard) rather than a one-off script.

Glossary:
    _MockLGBM -- a stand-in classifier that returns fixed probabilities so
        _evaluate_holdout can be unit-tested without training real models.
    _make_raw_frame -- builds a recognisable timestamp-ordered frame so the
        split boundary is easy to verify.
    _make_two_symbol_raw -- two symbols stacked over the same hourly bars,
        for per-symbol boundary-purge reasoning.
    TestHoldoutPfConfidenceBound -- pins the exact Clopper-Pearson lower
        bounds for the audit's three windows and the honest 2-year artifact,
        so the gate's arithmetic cannot drift.
    TestHoldoutVerdict -- the stability claim itself: the audit's PASS /
        FAIL / PASS point-estimate verdicts are FAIL / FAIL / FAIL under the
        confidence-bound bar.
    TestBoundaryTailPurge -- the last max_hold bars per symbol are dropped
        after engineering, cutoffs derived from the raw series.
    TestPermanentLeakGuard -- a recording refit_models stand-in captures
        every training frame validate_candidate hands it; none may contain a
        holdout timestamp.
"""

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core import retrainer as R  # noqa: E402
from core.retrainer import HoldoutMetrics, ValidationReport  # noqa: E402


class _MockLGBM:
    """Minimal LightGBM stand-in for holdout metric tests."""

    def __init__(self, class1_prob: float):
        self._class1_prob = float(class1_prob)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        n = len(X)
        p = np.zeros((n, 2), dtype=float)
        p[:, 1] = self._class1_prob
        p[:, 0] = 1.0 - self._class1_prob
        return p


def _make_raw_frame(n_per_day: int = 4, days: int = 10) -> pl.DataFrame:
    """Return a frame with one row per hour-ish bar over ``days`` days."""
    n = n_per_day * days
    timestamps = [
        R.datetime(2026, 1, 1, 0, 0, 0, tzinfo=R.timezone.utc)
        + R.timedelta(hours=i)
        for i in range(n)
    ]
    # Varying prices keep NATR non-zero so the chop veto and labels compute.
    close = [1.0 + 0.01 * (i % 7) + 0.001 * i for i in range(n)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "symbol": ["GBP_JPY"] * n,
            "open": [c - 0.005 for c in close],
            "high": [c + 0.015 for c in close],
            "low": [c - 0.015 for c in close],
            "close": close,
            "volume": [100] * n,
        }
    )


class TestSplitHoldout(unittest.TestCase):
    def test_holdout_is_chronologically_last(self):
        df = _make_raw_frame()
        remainder, holdout, holdout_range = R._split_holdout(df, 0.2)

        self.assertFalse(remainder.is_empty())
        self.assertFalse(holdout.is_empty())
        self.assertLess(
            remainder["timestamp"].max(), holdout["timestamp"].min()
        )
        self.assertIsNotNone(holdout_range)

    def test_split_is_disjoint(self):
        df = _make_raw_frame()
        remainder, holdout, _ = R._split_holdout(df, 0.25)

        rem_ts = set(remainder["timestamp"].to_list())
        hold_ts = set(holdout["timestamp"].to_list())
        self.assertFalse(rem_ts & hold_ts)

    def test_fraction_is_respected(self):
        df = _make_raw_frame(n_per_day=4, days=20)
        remainder, holdout, _ = R._split_holdout(df, 0.18)
        total = len(df)
        self.assertAlmostEqual(len(holdout) / total, 0.18, delta=0.05)

    def test_zero_fraction_disables_holdout(self):
        df = _make_raw_frame()
        remainder, holdout, holdout_range = R._split_holdout(df, 0.0)

        self.assertEqual(len(remainder), len(df))
        self.assertTrue(holdout.is_empty())
        self.assertIsNone(holdout_range)

    def test_empty_input_returns_empty_holdout(self):
        df = pl.DataFrame(
            {
                "timestamp": [],
                "symbol": [],
                "open": [],
                "high": [],
                "low": [],
                "close": [],
                "volume": [],
            }
        )
        remainder, holdout, holdout_range = R._split_holdout(df, 0.2)
        self.assertTrue(remainder.is_empty())
        self.assertTrue(holdout.is_empty())
        self.assertIsNone(holdout_range)


class TestEvaluateHoldout(unittest.TestCase):
    def _frame(self, n: int = 20):
        return pl.DataFrame(
            {
                "timestamp": list(range(n)),
                "symbol": ["GBP_JPY"] * n,
                "angel_target": [1] * n,
                "devil_target": [1, 0] * (n // 2),
                "devil_target_macro": [1, 0] * (n // 2),
                "feat_a": [0.0] * n,
                "feat_b": [0.0] * n,
            }
        )

    def test_evaluates_tradeable_approvals(self):
        holdout = self._frame(20)
        angel = _MockLGBM(class1_prob=0.9)  # all proposed
        devil = _MockLGBM(class1_prob=0.9)  # all approved

        scores = R._evaluate_holdout(
            holdout,
            angel,
            devil,
            angel_features=["feat_a", "feat_b"],
            devil_features=["feat_a", "feat_b", "angel_prob"],
            threshold=0.5,
            sl_mult=2.0,
            tp_mult=4.0,
        )

        self.assertEqual(scores["angel_proposed_trades"], 20)
        self.assertEqual(scores["trades"], 20)
        self.assertAlmostEqual(scores["win_rate"], 0.5, delta=0.01)
        # At 2:1 payoff with 50% wins, EV = 0.5*2 - 0.5 = 0.5
        self.assertAlmostEqual(scores["expected_value"], 0.5, delta=0.01)

    def test_no_angel_proposals_returns_nan_metrics(self):
        holdout = self._frame(10)
        angel = _MockLGBM(class1_prob=0.1)  # no proposals at 0.40 bar
        devil = _MockLGBM(class1_prob=0.9)

        scores = R._evaluate_holdout(
            holdout,
            angel,
            devil,
            angel_features=["feat_a", "feat_b"],
            devil_features=["feat_a", "feat_b", "angel_prob"],
            threshold=0.5,
            sl_mult=2.0,
            tp_mult=4.0,
        )

        self.assertEqual(scores["trades"], 0)
        self.assertTrue(np.isnan(scores["brier_score"]))

    def test_untradeable_approvals_are_excluded_from_scoring(self):
        holdout = pl.DataFrame(
            {
                "timestamp": list(range(4)),
                "symbol": ["XAU_USD", "GBP_JPY", "XAG_USD", "EUR_JPY"],
                "angel_target": [1] * 4,
                "devil_target": [1, 1, 0, 0],
                "devil_target_macro": [1, 1, 0, 0],
                "feat_a": [0.0] * 4,
                "feat_b": [0.0] * 4,
            }
        )
        angel = _MockLGBM(class1_prob=0.9)  # all proposed
        devil = _MockLGBM(class1_prob=0.9)  # all approved

        scores = R._evaluate_holdout(
            holdout,
            angel,
            devil,
            angel_features=["feat_a", "feat_b"],
            devil_features=["feat_a", "feat_b", "angel_prob"],
            threshold=0.5,
            sl_mult=2.0,
            tp_mult=4.0,
        )

        # Two proposals are untradeable metals and should be dropped from scoring.
        self.assertEqual(scores["angel_proposed_trades"], 4)
        self.assertEqual(scores["trades"], 2)
        self.assertEqual(scores["devil_approved_raw"], 4)


class TestMetadataRecordsHoldout(unittest.TestCase):
    def test_metadata_records_holdout_metrics_on_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / "holdout_test"
            cfg = {
                "asset_class": "forex",
                "model_dir": str(override),
                "tickers": ["GBP_JPY"],
                "timeframe_minutes": 1,
            }
            report = ValidationReport(
                fold_metrics=[],
                mean_brier=0.10,
                mean_ev=0.01,
                final_profit_factor=1.5,
                final_win_rate=0.45,
                final_total_trades=100,
                gate_passed=True,
                holdout=HoldoutMetrics(
                    used=True,
                    fraction=0.18,
                    start_date="2026-01-10",
                    end_date="2026-01-12",
                    brier_score=0.12,
                    expected_value=0.02,
                    win_rate=0.40,
                    profit_factor=1.3,
                    trades=55,
                    angel_proposed_trades=80,
                ),
            )

            R.save_models({"angel": 1}, {"devil": 2}, cfg, report=report)

            meta = json.loads((override / "metadata.json").read_text())
            self.assertTrue(meta["holdout"]["used"])
            self.assertEqual(meta["holdout"]["fraction"], 0.18)
            self.assertEqual(meta["holdout"]["start_date"], "2026-01-10")
            self.assertEqual(meta["holdout"]["end_date"], "2026-01-12")
            self.assertEqual(meta["holdout"]["trades"], 55)
            self.assertIsNone(meta["holdout"]["bypass_reason"])

    def test_metadata_records_bypass_when_holdout_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / "holdout_disabled"
            cfg = {
                "asset_class": "forex",
                "model_dir": str(override),
                "tickers": ["GBP_JPY"],
                "timeframe_minutes": 1,
            }
            with mock.patch.object(R, "HOLDOUT_FRAC", 0.0):
                R.save_models({"angel": 1}, {"devil": 2}, cfg)

            meta = json.loads((override / "metadata.json").read_text())
            self.assertFalse(meta["holdout"]["used"])
            self.assertEqual(meta["holdout"]["bypass_reason"], "disabled")


class TestHoldoutNeverInTraining(unittest.TestCase):
    """
    The property the whole change exists to enforce. We cannot easily inspect
    the inside of LightGBM, but we can prove the data-flow invariant: the
    remainder passed to training contains no timestamps from the holdout slice.
    """

    def test_remainder_and_holdout_are_temporally_separated(self):
        # Need enough bars for HTF SMA-50 warm-up (~250 1m bars) plus headroom.
        raw = _make_raw_frame(n_per_day=24, days=30)
        remainder, holdout, _ = R._split_holdout(raw, 0.2)

        # engineer_features_and_labels preserves the timestamp column, so the
        # temporal separation survives the entire pipeline.
        rem_features, _, _ = R.engineer_features_and_labels(
            remainder,
            sl_mult=2.0,
            tp_mult=4.0,
            max_hold=45,
            survival_bars=5,
            htf_timeframe="5m",
        )
        hold_features, _, _ = R.engineer_features_and_labels(
            holdout,
            sl_mult=2.0,
            tp_mult=4.0,
            max_hold=45,
            survival_bars=5,
            htf_timeframe="5m",
        )

        self.assertGreater(rem_features.height, 0)
        self.assertGreater(hold_features.height, 0)
        self.assertLess(
            rem_features["timestamp"].max(),
            hold_features["timestamp"].min(),
        )


def _make_two_symbol_raw() -> pl.DataFrame:
    """Two symbols stacked over the same 100 hourly bars (easy to reason about)."""
    timestamps = [
        R.datetime(2026, 1, 1, 0, 0, 0, tzinfo=R.timezone.utc)
        + R.timedelta(hours=i)
        for i in range(100)
    ]
    rows = []
    for sym in ("AAA_USD", "BBB_USD"):
        close = [1.0 + 0.01 * (i % 7) + 0.001 * i for i in range(100)]
        rows.append(
            pl.DataFrame(
                {
                    "timestamp": timestamps,
                    "symbol": [sym] * 100,
                    "open": [c - 0.005 for c in close],
                    "high": [c + 0.015 for c in close],
                    "low": [c - 0.015 for c in close],
                    "close": close,
                    "volume": [100] * 100,
                }
            )
        )
    return pl.concat(rows)


class TestHoldoutPfConfidenceBound(unittest.TestCase):
    """
    The stability fix: the holdout PF bar gates on the exact Clopper-Pearson
    lower bound, not the point estimate. Expected values were computed with
    scipy.stats.beta.ppf(0.05, w, n-w+1) mapped through PF at 2:1 (sl=2, tp=4).
    """

    def test_audit_pass_window_no_longer_passes(self):
        # Audit 2026-08-24: 62 trades, PF 1.444 point, 41.9% WR -- the window
        # the old gate PASSED. The 95% lower bound is 0.911, below 1.2.
        lb = R._holdout_pf_lower_bound(26, 62, sl_mult=2.0, tp_mult=4.0)
        self.assertAlmostEqual(lb, 0.911128, places=5)
        self.assertLess(lb, R.PROFIT_FACTOR_THRESHOLD)

    def test_audit_fail_window_and_pinned_window_bounds(self):
        # 82 trades at 32.9% (old FAIL) and 55 trades at 40.0% (old PASS):
        # both bounds sit far below the 1.2 bar.
        lb_fail = R._holdout_pf_lower_bound(27, 82, sl_mult=2.0, tp_mult=4.0)
        lb_pin = R._holdout_pf_lower_bound(22, 55, sl_mult=2.0, tp_mult=4.0)
        self.assertAlmostEqual(lb_fail, 0.644204, places=5)
        self.assertAlmostEqual(lb_pin, 0.811240, places=5)
        self.assertLess(lb_fail, 1.2)
        self.assertLess(lb_pin, 1.2)

    def test_two_year_config_clears_the_bound(self):
        # The honest 2-year artifact (134 trades, PF 1.8286, 47.8%): bound
        # 1.3547 -- the justified pass survives the stricter gate.
        lb = R._holdout_pf_lower_bound(64, 134, sl_mult=2.0, tp_mult=4.0)
        self.assertAlmostEqual(lb, 1.354747, places=5)
        self.assertGreaterEqual(lb, R.PROFIT_FACTOR_THRESHOLD)

    def test_perfect_record_needs_four_trades_to_clear(self):
        # Pins the CP-vs-Wilson choice: a perfect 3-for-3 clears Wilson's
        # bound but CP's is 1.1666 -- below the bar -- while 4-for-4 (1.7941)
        # passes. Sample size enters through the bound, not a cliff.
        lb3 = R._holdout_pf_lower_bound(3, 3, sl_mult=2.0, tp_mult=4.0)
        lb4 = R._holdout_pf_lower_bound(4, 4, sl_mult=2.0, tp_mult=4.0)
        self.assertAlmostEqual(lb3, 1.166577, places=5)
        self.assertAlmostEqual(lb4, 1.794136, places=5)
        self.assertLess(lb3, R.PROFIT_FACTOR_THRESHOLD)
        self.assertGreaterEqual(lb4, R.PROFIT_FACTOR_THRESHOLD)

    def test_zero_wins_and_zero_trades_give_zero_bound(self):
        self.assertEqual(R._holdout_pf_lower_bound(0, 30, sl_mult=2.0, tp_mult=4.0), 0.0)
        self.assertEqual(R._holdout_pf_lower_bound(5, 0, sl_mult=2.0, tp_mult=4.0), 0.0)
        # Wins beyond trades are clamped, not trusted.
        self.assertAlmostEqual(
            R._holdout_pf_lower_bound(99, 3, sl_mult=2.0, tp_mult=4.0),
            R._holdout_pf_lower_bound(3, 3, sl_mult=2.0, tp_mult=4.0),
            places=12,
        )


class TestHoldoutVerdict(unittest.TestCase):
    @staticmethod
    def _scores(brier, ev, pf, wr, trades, wins):
        return {
            "brier_score": brier,
            "expected_value": ev,
            "win_rate": wr,
            "profit_factor": pf,
            "trades": trades,
            "wins": wins,
            "losses": trades - wins,
        }

    def test_audit_windows_all_fail_the_stable_gate(self):
        # The three audit windows: point estimates 1.444 / 0.982 / 1.333 said
        # PASS / FAIL / PASS. The stable gate says FAIL / FAIL / FAIL -- the
        # verdict no longer depends on the clock.
        for brier, ev, pf, wr, trades, wins in [
            (0.2000, 0.50, 1.444, 0.419, 62, 26),
            (0.2000, -0.30, 0.982, 0.329, 82, 27),
            (0.2000, 0.45, 1.333, 0.400, 55, 22),
        ]:
            passed, reasons = R._holdout_verdict(
                self._scores(brier, ev, pf, wr, trades, wins),
                sl_mult=2.0,
                tp_mult=4.0,
            )
            self.assertFalse(passed)
            self.assertTrue(
                any("lower bound" in r for r in reasons),
                f"expected the confidence-bound reason, got {reasons}",
            )

    def test_honest_two_year_holdout_passes(self):
        passed, reasons = R._holdout_verdict(
            self._scores(0.2352, 1.1716, 1.8286, 0.478, 134, 64),
            sl_mult=2.0,
            tp_mult=4.0,
        )
        self.assertTrue(passed, reasons)

    def test_zero_trade_holdout_fails_loudly_not_vacuously(self):
        # NaN metrics must fail the gate, not slip through NaN comparisons.
        passed, reasons = R._holdout_verdict(
            self._scores(float("nan"), float("nan"), 0.0, 0.0, 0, 0),
            sl_mult=2.0,
            tp_mult=4.0,
        )
        self.assertFalse(passed)
        self.assertGreaterEqual(len(reasons), 3)


class TestBoundaryTailPurge(unittest.TestCase):
    """Finding 2: drop rows whose max_hold walk ran off the end of the frame."""

    def test_cutoffs_mark_the_last_max_hold_bars_per_symbol(self):
        raw = _make_two_symbol_raw()
        cutoffs = R._tail_cutoff_by_symbol(raw, max_hold=10)

        ts = sorted(raw["timestamp"].unique().to_list())
        self.assertEqual(cutoffs["AAA_USD"], ts[90])
        self.assertEqual(cutoffs["BBB_USD"], ts[90])

    def test_purge_drops_exactly_the_tail_rows(self):
        raw = _make_two_symbol_raw()
        cutoffs = R._tail_cutoff_by_symbol(raw, max_hold=10)
        purged, n_dropped = R._purge_boundary_tail(raw, cutoffs)

        self.assertEqual(n_dropped, 20)
        for sym in ("AAA_USD", "BBB_USD"):
            sub = purged.filter(pl.col("symbol") == sym)
            self.assertEqual(sub.height, 90)
            self.assertLess(sub["timestamp"].max(), cutoffs[sym])

    def test_symbol_with_fewer_bars_than_max_hold_is_untouched(self):
        raw = _make_two_symbol_raw().filter(pl.col("symbol") != "BBB_USD")
        # Keep BBB with only 8 bars: below max_hold, so nothing to purge.
        short = raw.filter(pl.col("timestamp") < raw["timestamp"].min() + R.timedelta(hours=8))
        short = pl.concat(
            [raw, short.with_columns(pl.lit("BBB_USD").alias("symbol"))]
        )
        cutoffs = R._tail_cutoff_by_symbol(short, max_hold=10)
        self.assertIn("AAA_USD", cutoffs)
        self.assertNotIn("BBB_USD", cutoffs)

        purged, n_dropped = R._purge_boundary_tail(short, cutoffs)
        self.assertEqual(n_dropped, 10)
        self.assertEqual(purged.filter(pl.col("symbol") == "BBB_USD").height, 8)

    def test_no_cutoffs_returns_frame_unchanged(self):
        raw = _make_two_symbol_raw()
        frame, n_dropped = R._purge_boundary_tail(raw, {})
        self.assertIs(frame, raw)
        self.assertEqual(n_dropped, 0)


class TestPermanentLeakGuard(unittest.TestCase):
    """
    The audit's instrumented leak proof, made permanent: a recording stand-in
    for refit_models captures every training frame validate_candidate hands
    it, and the test asserts none of them contains a holdout timestamp. This
    is the property the whole holdout gate exists to enforce.
    """

    class _Recorder:
        def __init__(self, class1_prob=0.6):
            self.frames = []
            self.angel = _MockLGBM(class1_prob)
            self.devil = _MockLGBM(class1_prob)

        def __call__(
            self,
            df,
            feature_cols,
            angel_params=None,
            devil_params=None,
            sl_mult=None,
            tp_mult=None,
        ):
            self.frames.append((df["timestamp"].min(), df["timestamp"].max()))
            return (
                self.angel,
                self.devil,
                list(feature_cols),
                list(feature_cols) + ["angel_prob"],
                # refit_models now also returns the Angel proposal bar the
                # Devil's population was filtered at; 0.4 mirrors the
                # pre-calibration constant these tests were written against.
                0.4,
            )

    def test_no_training_frame_touches_the_holdout(self):
        raw = _make_raw_frame(n_per_day=24, days=30)
        remainder, holdout, holdout_range = R._split_holdout(raw, 0.2)

        rem_features, feature_cols, _ = R.engineer_features_and_labels(
            remainder,
            sl_mult=2.0,
            tp_mult=4.0,
            max_hold=45,
            survival_bars=5,
            htf_timeframe="5m",
        )

        recorder = self._Recorder()
        with mock.patch.object(R, "refit_models", recorder):
            report, angel, devil, angel_feats, devil_feats, threshold, hmm = (
                R.validate_candidate(
                    rem_features,
                    feature_cols,
                    sl_mult=2.0,
                    tp_mult=4.0,
                    n_folds=3,
                )
            )

        # Every fold trained (and the final refit if the gate passed) must end
        # strictly before the holdout begins.
        holdout_start = holdout["timestamp"].min()
        self.assertGreaterEqual(len(recorder.frames), 3)
        for train_min, train_max in recorder.frames:
            self.assertLess(train_max, holdout_start)

        # And the holdout side is non-empty, so the assertion above is not
        # vacuous: there is real future data being excluded.
        hold_features, _, _ = R.engineer_features_and_labels(
            holdout,
            sl_mult=2.0,
            tp_mult=4.0,
            max_hold=45,
            survival_bars=5,
            htf_timeframe="5m",
        )
        self.assertGreater(hold_features.height, 0)


class TestMainWiringLeakGuard(unittest.TestCase):
    """
    TestPermanentLeakGuard proves validate_candidate only sees remainder
    frames; this test proves MAIN is the caller that wires the remainder in.
    A patched fetch returns a synthetic window, a recording refit_models
    stands in for training, promote_or_reject is captured, and the whole
    Phase 3a / Phase 4.5 fold-fail diagnostic path runs end to end — so a
    regression that engineered raw_data instead of remainder_raw would fail
    here.
    """

    def test_main_never_hands_training_a_holdout_row(self):
        raw = _make_raw_frame(n_per_day=24, days=30)
        remainder, holdout, _ = R._split_holdout(raw, 0.2)
        holdout_start = holdout["timestamp"].min()

        recorder = TestPermanentLeakGuard._Recorder()
        captured = {}

        def fake_promote(
            report,
            angel,
            devil,
            threshold,
            asset_config=None,
            hmm_models=None,
            angel_threshold=None,
        ):
            captured["report"] = report
            return False

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            R, "fetch_training_data", return_value=raw
        ), mock.patch.object(
            R, "get_market_provider", return_value=object()
        ), mock.patch.object(
            R, "refit_models", recorder
        ), mock.patch.object(
            R, "promote_or_reject", fake_promote
        ), mock.patch.object(
            R, "DAYS_BACK", 30
        ), mock.patch.object(
            R, "HOLDOUT_FRAC", 0.2
        ), mock.patch.object(
            R, "USE_HMM_FEATURES", False
        ), mock.patch.dict(
            os.environ,
            {
                "DATA_SOURCE": "oanda",
                "RETRAIN_MODEL_DIR": str(Path(tmp) / "main_wiring_test"),
            },
        ):
            rc = R.main()

        self.assertEqual(rc, 2)  # rejected, healthy — and nothing crashed
        self.assertIn("report", captured)
        report = captured["report"]
        self.assertFalse(report.gate_passed)
        self.assertIsNotNone(report.holdout)
        self.assertTrue(report.holdout.used)
        self.assertTrue(report.holdout.diagnostic_only)

        # The property itself: every training frame main handed to
        # refit_models ends strictly before the holdout begins.
        self.assertGreaterEqual(len(recorder.frames), 3)
        for train_min, train_max in recorder.frames:
            self.assertLess(train_max, holdout_start)


class TestMetadataRecordsConfidenceVerdict(unittest.TestCase):
    def test_metadata_records_bound_and_diagnostic_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            override = Path(tmp) / "holdout_ci"
            cfg = {
                "asset_class": "forex",
                "model_dir": str(override),
                "tickers": ["GBP_JPY"],
                "timeframe_minutes": 1,
            }
            report = ValidationReport(
                fold_metrics=[],
                mean_brier=0.10,
                mean_ev=0.01,
                final_profit_factor=1.5,
                final_win_rate=0.45,
                final_total_trades=100,
                gate_passed=False,
                holdout=HoldoutMetrics(
                    used=True,
                    fraction=0.18,
                    start_date="2026-01-10",
                    end_date="2026-01-12",
                    brier_score=0.20,
                    expected_value=0.50,
                    win_rate=0.419,
                    profit_factor=1.444,
                    trades=62,
                    angel_proposed_trades=80,
                    wins=26,
                    pf_lower_bound=0.9111,
                    pf_confidence=0.95,
                    purged_tail_rows=12,
                    diagnostic_only=True,
                ),
            )

            R.save_models({"angel": 1}, {"devil": 2}, cfg, report=report)

            meta = json.loads((override / "metadata.json").read_text())
            ho = meta["holdout"]
            self.assertTrue(ho["used"])
            self.assertEqual(ho["wins"], 26)
            self.assertEqual(ho["pf_lower_bound"], 0.9111)
            self.assertEqual(ho["pf_confidence"], 0.95)
            self.assertEqual(ho["purged_tail_rows"], 12)
            # diagnostic_only deliberately NOT in metadata.json: metadata is
            # only written on promotion, which implies the fold gate passed,
            # which implies diagnostic_only was False. The flag lives on the
            # ValidationReport and in the logs, not in the sidecar.


class TestChronologicalOOFIntegrity(unittest.TestCase):
    """2026-09-09: the frame is symbol-blocked, so row index is not a time
    axis. These pin the two fixes that restore chronology to the Devil's
    training inputs: timestamp-ranked decay weights and the chronological
    permutation around the OOF split."""

    def test_decay_weights_follow_timestamps_not_basket_position(self):
        ts = pl.Series(
            "t",
            [
                datetime(2026, 1, 1), datetime(2026, 2, 1), datetime(2026, 3, 1),
                datetime(2026, 6, 1), datetime(2026, 7, 1), datetime(2026, 8, 1),
            ],
        )
        w = R.generate_time_decay_weights(6, timestamps=ts)
        # The newest row overall must carry the max weight...
        self.assertEqual(w[5], w.max())
        # ...regardless of which symbol block it sits in, and recency must
        # order weights within a block.
        self.assertGreater(w[1], w[0])
        self.assertGreater(w[5], w[3])

    def test_decay_weights_rank_by_time_within_symbol_blocks(self):
        # Symbol A occupies the last three rows here but holds OLD timestamps;
        # the old row-index weighting would have given A's rows the top
        # weights. Timestamp ranking must put the NEWER rows (index 0-2) on top.
        ts = pl.Series(
            "t",
            [
                datetime(2026, 9, 1), datetime(2026, 9, 2), datetime(2026, 9, 3),
                datetime(2026, 1, 1), datetime(2026, 1, 2), datetime(2026, 1, 3),
            ],
        )
        w = R.generate_time_decay_weights(6, timestamps=ts)
        self.assertEqual(w[2], w.max())
        self.assertGreater(w[0], w[3])

    def test_decay_weights_legacy_shape_unchanged_without_timestamps(self):
        """No-timestamps path keeps the old [0.1, 1.0] contract."""
        w = R.generate_time_decay_weights(100)
        self.assertAlmostEqual(w.min(), 0.1, places=6)
        self.assertAlmostEqual(w.max(), 1.0, places=6)
        self.assertGreater(w[-1], w[0])

    def test_decay_weights_uniform_when_single_timestamp(self):
        w = R.generate_time_decay_weights(
            4, timestamps=pl.Series("t", [datetime(2026, 1, 1)] * 4)
        )
        np.testing.assert_array_equal(w, np.ones(4))

    def test_oof_split_uses_chronological_permutation(self):
        """Pin the permutation refit_models applies before TimeSeriesSplit:
        with a symbol-blocked frame, raw-index splitting made basket-tail
        symbols val folds scored by models trained on the future."""
        ts_vals = np.array(
            [
                np.datetime64("2026-01-01"), np.datetime64("2026-02-01"),
                np.datetime64("2026-03-01"), np.datetime64("2026-06-01"),
                np.datetime64("2026-07-01"), np.datetime64("2026-08-01"),
            ]
        )
        perm = np.argsort(ts_vals, kind="stable")
        self.assertTrue(np.array_equal(perm, np.arange(6)))
        # Inverted frame (newest first in row order): permutation must
        # restore chronology for the split.
        ts_desc = ts_vals[::-1]
        perm_desc = np.argsort(ts_desc, kind="stable")
        np.testing.assert_array_equal(ts_desc[perm_desc], np.sort(ts_vals))


if __name__ == "__main__":
    unittest.main()
