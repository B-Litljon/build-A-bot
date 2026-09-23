"""
The lab's model-aware backtest: signal correctness, row alignment, live gating.

Two failure modes are pinned here, both silent in production terms:

* **raw_sl_distance units.** The RiskManager multiplies raw ATR itself, so a
  strategy that pre-multiplies doubles the bracket. MLStrategy emits
  ``close * natr_14 / 100``; so must this one.
* **row alignment.** The strategy's probabilities are indexed against the frame
  the backtest walks. A window ending at bar i must read bar i's score, not
  row 0's or a shifted row's — otherwise every trade is placed on another
  bar's conviction while looking perfectly healthy.

Plus the live-gate funnel: with the chop filter on, vetoed proposals must be
counted (gate_rejections), not silently dropped; with it off, trades happen.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from lab.backtest import LabModelStrategy, run_model_backtest  # noqa: E402
from lab.frames import FrameResult  # noqa: E402
from lab.spec import FeatureSpec  # noqa: E402


class ConstantModel:
    """predict_proba returns a constant column, row count preserved."""

    def __init__(self, prob: float):
        self.prob = float(prob)
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.prob), np.full(n, self.prob)])


class FeatureEchoModel:
    """Angel prob equals the row's feature value — exposes any row shift."""

    def __init__(self, feature: str):
        self.feature = feature
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        values = X[self.feature].to_numpy() if hasattr(X, "columns") else np.asarray(X)[:, 0]
        return np.column_stack([1 - values, values])


def tiny_frame(probs_source=("feat",), row_probs=(0.1, 0.9, 0.5, 0.7)):
    ts = pl.datetime_range(
        datetime(2026, 1, 1),
        datetime(2026, 1, 1) + timedelta(minutes=15 * (len(row_probs) - 1)),
        interval="15m",
        eager=True,
    )
    return pl.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["GBP_JPY"] * len(row_probs),
            "f": list(row_probs),
            "close": [100.0, 101.0, 102.0, 103.0][: len(row_probs)],
            "open": [100.0, 101.0, 102.0, 103.0][: len(row_probs)],
            "high": [101.0, 102.0, 103.0, 104.0][: len(row_probs)],
            "low": [99.0, 100.0, 101.0, 102.0][: len(row_probs)],
            "natr_14": [1.0] * len(row_probs),
        }
    )


class TestLabModelStrategy(unittest.TestCase):
    def _strategy(self, frame, angel, devil, angel_thr=0.0, devil_thr=0.0):
        return LabModelStrategy(frame, ("f",), angel, devil, angel_thr, devil_thr)

    def test_emits_raw_atr_not_a_multiplied_stop(self):
        frame = tiny_frame()
        strategy = self._strategy(frame, ConstantModel(1.0), ConstantModel(1.0))
        signal = strategy.generate_signals(frame.slice(0, 1))
        self.assertIsNotNone(signal)
        expected = 100.0 * 1.0 / 100.0  # close * natr / 100
        self.assertAlmostEqual(signal.raw_sl_distance, expected, places=12)
        self.assertAlmostEqual(signal.entry_price, 100.0, places=12)
        self.assertEqual(signal.direction, "long")

    def test_window_ending_at_bar_i_reads_bar_i(self):
        frame = tiny_frame()
        strategy = self._strategy(
            frame, FeatureEchoModel("f"), ConstantModel(1.0)
        )
        # A window ending at the second row must carry the second row's score.
        signal = strategy.generate_signals(frame.slice(0, 2))
        self.assertAlmostEqual(signal.metadata["angel_prob"], 0.9, places=12)
        signal = strategy.generate_signals(frame.slice(2, 1))
        self.assertAlmostEqual(signal.metadata["angel_prob"], 0.5, places=12)

    def test_angel_and_devil_thresholds_both_gate(self):
        frame = tiny_frame()
        strategy = self._strategy(
            frame, ConstantModel(0.4), ConstantModel(0.4),
            angel_thr=0.5, devil_thr=0.0,
        )
        self.assertIsNone(strategy.generate_signals(frame.slice(1, 1)))
        strategy = self._strategy(
            frame, ConstantModel(0.9), ConstantModel(0.4),
            angel_thr=0.0, devil_thr=0.5,
        )
        self.assertIsNone(strategy.generate_signals(frame.slice(1, 1)))
        strategy = self._strategy(
            frame, ConstantModel(0.9), ConstantModel(0.6),
            angel_thr=0.5, devil_thr=0.5,
        )
        self.assertIsNotNone(strategy.generate_signals(frame.slice(1, 1)))


def backtest_frame(n=120):
    """A one-symbol frame whose bars trend up hard enough to hit a 4xATR target."""
    ts = pl.datetime_range(
        datetime(2026, 1, 1),
        datetime(2026, 1, 1) + timedelta(minutes=15 * (n - 1)),
        interval="15m",
        eager=True,
    )
    close = 100.0 + np.arange(n) * 0.5
    return pl.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["GBP_JPY"] * n,
            "f": np.full(n, 1.0),
            "open": close,
            "high": close + 1.2,
            "low": close - 1.2,
            "close": close,
            "natr_14": np.full(n, 1.0),
        }
    )


class TestRunModelBacktest(unittest.TestCase):
    def _frame_result(self, df, alpha_table=None):
        return FrameResult(
            spec_name="bt",
            content_hash="deadbeefdeadbeef",
            df=df,
            feature_cols=("f",),
            chop_veto_rate=0.0,
            alpha_table=alpha_table,
        )

    def _gate(self):
        return SimpleNamespace(
            angel_model=ConstantModel(1.0),
            devil_model=ConstantModel(1.0),
            production_threshold=0.5,
            report=SimpleNamespace(production_angel_threshold=0.5),
        )

    def test_trades_are_taken_with_gates_off(self):
        spec = FeatureSpec(name="bt", symbols=("GBP_JPY",), feature_sets=("stub",))
        frame = self._frame_result(backtest_frame())
        with mock.patch.dict(os.environ, {"RISK_CHOP_FILTER_ENABLED": "0"}):
            report = run_model_backtest(frame, self._gate(), spec)
        self.assertGreater(report.total_trades, 0)
        self.assertEqual(report.toll_mode, "flat")
        self.assertEqual(report.symbol_count, 1)
        self.assertGreater(report.gross_ev_r, 0.0)

    def test_gate_funnel_counts_vetoed_proposals(self):
        """With a forced Gate A veto, proposals must be counted, not traded."""
        spec = FeatureSpec(name="bt", symbols=("GBP_JPY",), feature_sets=("stub",))
        frame = self._frame_result(backtest_frame())
        with mock.patch.dict(
            os.environ,
            {
                "RISK_CHOP_FILTER_ENABLED": "1",
                "RISK_SPREAD_K": "1000000.0",
                "RISK_REGIME_PCTILE": "0.0",
            },
        ):
            report = run_model_backtest(frame, self._gate(), spec)
        self.assertEqual(report.total_trades, 0)
        self.assertTrue(report.gate_rejections, "vetoes were not counted")
        # Gate A dominates; the 'time' keys are the two bars that fall inside
        # the 16:55-17:30 ET blackout over this 30-hour synthetic window.
        self.assertIn("spread", report.gate_rejections)
        self.assertGreater(report.gate_rejections["spread"], 0)

    def test_spread_table_prices_per_symbol(self):
        spec = FeatureSpec(name="bt", symbols=("GBP_JPY",), feature_sets=("stub",))
        frame = self._frame_result(backtest_frame(), alpha_table={"GBP_JPY": 0.9})
        with mock.patch.dict(os.environ, {"RISK_CHOP_FILTER_ENABLED": "0"}):
            with_table = run_model_backtest(frame, self._gate(), spec)
        flat_frame = self._frame_result(backtest_frame(), alpha_table=None)
        with mock.patch.dict(os.environ, {"RISK_CHOP_FILTER_ENABLED": "0"}):
            flat = run_model_backtest(flat_frame, self._gate(), spec)
        self.assertEqual(with_table.toll_mode, "spread_table")
        self.assertEqual(flat.toll_mode, "flat")
        # The measured alpha (0.9) is far above the flat default toll, so the
        # cost-priced run must book a strictly worse net EV.
        self.assertLess(with_table.net_ev_r, flat.net_ev_r)


if __name__ == "__main__":
    unittest.main()
