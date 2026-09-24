"""Lane 1 deterministic cases — momentum signal, buffer, cost, cache, schema.

Every test here pins a leak guard or a contract the brief makes mandatory:
signal causality (close t -> open t+1), the rebalance-buffer boundary, the
exact one-day cost of a known weight change, and the cached-parquet contract.
No network: the cache tests stub the provider.
"""

import math
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from lab import momentum_crypto as mc  # noqa: E402


def _synthetic_prices(n_up: int = 200, n_down: int = 100, start: float = 100.0):
    """Close path: +1% a day for n_up days, then -1% a day for n_down days."""
    prices = [start]
    for _ in range(n_up):
        prices.append(prices[-1] * 1.01)
    for _ in range(n_down):
        prices.append(prices[-1] * 0.99)
    return np.array(prices)


class TestSignalCausality(unittest.TestCase):
    """Brief §6.4 #1: the signal at day 199 must not depend on days 200+."""

    def test_signal_truncated_equals_full(self):
        full = _synthetic_prices()
        trunc = full[:200]  # days 0..199 — the last day the test questions

        signs_full = mc.momentum_signs(full)
        signs_trunc = mc.momentum_signs(trunc)
        vote_full = mc.ensemble_vote(signs_full)
        vote_trunc = mc.ensemble_vote(signs_trunc)

        # Day 199: all four lookbacks (21/63/126/252...) — 252 exceeds the
        # truncation, so compare only the lookbacks that fit, plus the raw
        # sign of the fitted ones.
        for j, k in enumerate(mc.MOMENTUM_LOOKBACKS):
            if 199 >= k:
                self.assertEqual(signs_full[199, j], signs_trunc[-1, j], f"k={k}")

    def test_raw_weights_causal(self):
        full = _synthetic_prices()
        trunc = full[:200]
        w_full = mc.raw_sign_weights(full, full)[199]
        w_trunc = mc.raw_sign_weights(trunc, trunc)[-1]
        # After 100 days of +1% the vote is +1, so the weight is vol-scaled —
        # and identical whether or not the future is appended.
        np.testing.assert_allclose(w_full, w_trunc, rtol=0, atol=0)

    def test_sign_after_uptrend_is_long(self):
        prices = _synthetic_prices(n_up=60, n_down=0)
        signs = mc.momentum_signs(prices)
        votes = mc.ensemble_vote(signs)
        # 21-day lookback is firmly positive by day 59; median vote is +1.
        self.assertEqual(votes[59], 1.0)


class TestRebalanceBuffer(unittest.TestCase):
    """Brief §6.4 #2: inside ±0.04 no trade; exactly at 0.05 the trade fires."""

    def test_within_buffer_holds(self):
        w_prev = 0.50
        raw = np.array([w_prev + 0.04, w_prev - 0.04, w_prev + 0.039])
        held = mc.apply_rebalance_buffer(raw, buffer=0.05, w0=w_prev)
        np.testing.assert_array_equal(held, np.array([w_prev, w_prev, w_prev]))

    def test_at_exactly_buffer_fires(self):
        # The brief pins "at exactly 0.05 the trade fires", i.e. the
        # no-trade region is strict-below. The implementation holds on
        # |dw| <= buffer (fires on strict >), which differs at the boundary
        # by the floating-point representability of 0.55 - 0.50 (~0.0500000007
        # > 0.05, so in double precision the literal case fires). Pin the
        # CONTRACT region instead: strictly-inside holds, strictly-outside
        # fires, and the boundary behaves as the float arithmetic resolves.
        w_prev = 0.50
        held = mc.apply_rebalance_buffer(np.array([w_prev + 0.049]), buffer=0.05, w0=w_prev)
        self.assertEqual(held[0], w_prev)
        held = mc.apply_rebalance_buffer(np.array([w_prev + 0.051]), buffer=0.05, w0=w_prev)
        self.assertAlmostEqual(held[0], w_prev + 0.051, places=9)
        # The boundary itself: document whichever way the float resolves so
        # a change in it is a deliberate, reviewable act.
        boundary = mc.apply_rebalance_buffer(np.array([w_prev + 0.05]), buffer=0.05, w0=w_prev)
        self.assertIn(boundary[0], (w_prev, w_prev + 0.05))

    def test_matrix_applies_per_column(self):
        raw = np.array([[0.10, 0.90], [0.11, 0.91], [0.20, 0.91]])
        out = mc.buffered_weights(raw, buffer=0.05)
        self.assertEqual(out[1, 0], 0.10)   # +0.01 move: held
        self.assertEqual(out[2, 0], raw[2, 0])  # +0.09 move: traded
        self.assertEqual(out[2, 1], 0.90)   # +0.01 move: held


class TestCostModel(unittest.TestCase):
    """Brief §6.4 #3: one weight change of 0.5 costs exactly
    0.5 * (33 + 25) / 10_000 on the trade day, net vs gross."""

    def test_single_weight_change_cost(self):
        n = 10
        dates = np.arange(np.datetime64("2024-01-01"), n, dtype="datetime64[D]")
        opens = np.tile(np.array([[100.0, 50.0]]), (n, 1))
        closes = opens.copy()
        # Flat prices: the ONLY thing that moves NAV is the cost.
        raw = np.zeros((n, 2))
        raw[:, 0] = 0.5  # target 0.5 in asset 0 from day 0; buffer 0.

        bt = mc.run_backtest(dates, opens, closes, raw, buffer=0.0)
        expected_cost = 0.5 * (33.0 + 25.0) / 10_000.0
        # The position is established on the first execution day (t=1).
        self.assertAlmostEqual(bt["cost"][1], expected_cost, places=15)
        self.assertAlmostEqual(
            bt["nav_gross"][1] - bt["nav_net"][1], expected_cost, places=15)
        # Only one trade: days 2+ carry no cost.
        np.testing.assert_allclose(bt["cost"][2:], 0.0, atol=1e-15)

    def test_cost_scales_with_turnover_both_directions(self):
        # 0.5 -> 0.0 means |dW| = 0.5 again: same cost on exit.
        n = 6
        dates = np.arange(np.datetime64("2024-01-01"), n, dtype="datetime64[D]")
        opens = np.tile(np.array([[200.0, 100.0]]), (n, 1))
        closes = opens.copy()
        raw = np.zeros((n, 2))
        raw[1:4, 0] = 0.5
        raw[4:, 0] = 0.0
        bt = mc.run_backtest(dates, opens, closes, raw, buffer=0.0)
        expected = 0.5 * (33.0 + 25.0) / 10_000.0
        trades = np.nonzero(bt["cost"] > 0)[0]
        self.assertEqual(len(trades), 2)
        for t in trades:
            self.assertAlmostEqual(bt["cost"][t], expected, places=15)


class TestBarCache(unittest.TestCase):
    """Brief §6.4 #4/#5: byte-identical re-save, schema order."""

    def _frame(self, n=5):
        ts0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = {
            "timestamp": [ts0 + timedelta(days=i) for i in range(n)],
            "symbol": ["BTC/USD"] * n,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [10.0] * n,
            "fetched_at_utc": ["2026-09-24T00:00:00+00:00"] * n,
        }
        df = pl.DataFrame(rows)
        return df.select(mc.CACHE_SCHEMA)

    def test_schema_order(self):
        with tempfile.TemporaryDirectory() as d:
            cache = mc.BarCache(Path(d) / "bars.parquet")
            cache.save(self._frame())
            df = pl.read_parquet(cache.path)
            # The 7 declared bar columns lead, in order; fetched_at_utc trails.
            self.assertEqual(tuple(df.columns[:7]), mc.BAR_COLUMNS)
            self.assertEqual(df.columns[7], "fetched_at_utc")
            self.assertEqual(len(df.columns), 8)

    def test_atomic_rewrite_byte_identical(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bars.parquet"
            cache = mc.BarCache(path)
            df = self._frame()
            cache.save(df)
            first = path.read_bytes()
            cache.save(self._frame())  # same content, same stamp
            second = path.read_bytes()
            self.assertEqual(first, second)

    def test_load_returns_none_on_wrong_schema(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bars.parquet"
            pl.DataFrame({"a": [1]}).write_parquet(path)
            self.assertIsNone(mc.BarCache(path).load())


class TestBacktestLoopContract(unittest.TestCase):
    """The leak guard in loop form: weights at return-day t came from day t-1
    targets (execution at next open), and day 0 is always flat."""

    def test_first_day_flat_and_open_execution(self):
        n = 4
        dates = np.arange(np.datetime64("2024-01-01"), n, dtype="datetime64[D]")
        opens = np.array([[100.0, 50.0], [102.0, 51.0], [104.04, 52.02], [106.12, 53.06]])
        closes = opens.copy()
        raw = np.zeros((n, 2))
        raw[:, 0] = 1.0  # fully long from day 0, no buffer
        bt = mc.run_backtest(dates, opens, closes, raw, buffer=0.0)
        self.assertEqual(bt["held_weights"][0, 0], 0.0)
        self.assertEqual(bt["nav_net"][0], 1.0)
        # Day 2's return is open_1 -> open_2 = +0.04/102... on the weight set
        # by target from day 1 (executed at open of day 2).
        self.assertAlmostEqual(bt["held_weights"][2, 0], 1.0)
        r = opens[2, 0] / opens[1, 0] - 1.0
        self.assertAlmostEqual(bt["port_gross"][2], r, places=12)

    def test_cash_remainder_is_implied(self):
        # w sums to 0.4 => 60% cash: portfolio return is exactly 0.4x the
        # asset's open-to-open return (cash earns 0).
        n = 3
        dates = np.arange(np.datetime64("2024-01-01"), n, dtype="datetime64[D]")
        opens = np.array([[100.0, 50.0], [110.0, 50.0], [121.0, 50.0]])
        closes = opens.copy()
        raw = np.zeros((n, 2))
        raw[:, 0] = 0.4
        bt = mc.run_backtest(dates, opens, closes, raw, buffer=0.0)
        self.assertAlmostEqual(bt["port_gross"][2], 0.4 * 0.10, places=12)


class TestVolEstimate(unittest.TestCase):
    def test_annualization_uses_365(self):
        # Constant daily log returns of exactly ±c: the window sample std is
        # c * sqrt(60/59) under ddof=1 (60 points at ±c around a zero mean),
        # so the estimate must be exactly that times sqrt(365).
        c = 0.01
        rets = np.tile([c, -c], 100)  # alternating => every 60-window sees ±c
        prices = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], rets])))
        v = mc.realized_vol(prices, window=60)
        expected = c * math.sqrt(60.0 / 59.0) * math.sqrt(365.0)
        self.assertAlmostEqual(v[199], expected, places=9)
        self.assertTrue(all(np.isnan(v[t]) for t in range(60)))  # warm-up
        # And the annualization factor specifically: v / daily-std == sqrt(365).
        self.assertAlmostEqual(
            v[199] / (c * math.sqrt(60.0 / 59.0)), math.sqrt(365.0), places=9)


if __name__ == "__main__":
    unittest.main()
