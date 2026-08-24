"""
Tests for ml.regimes.behavior_tagger -- the causal market-behavior labels.

The tagger's only real job is to be honest about time. Every number the
behavior matrix will report is conditioned on these labels, so a tag that
peeks even one bar into the future turns the whole downstream analysis into
confident fiction. Two tests carry that weight:
``test_no_lookahead_prefix_invariance`` and
``test_rank_agrees_with_live_gate_b_decision``.

Glossary:
    _series -- a deterministic pseudo-random volatility/momentum pair, seeded
        so a failure reproduces exactly.
    TestCausality -- proves a tag never depends on a later bar.
    TestLiveSymmetry -- proves the offline estimator is the one the live Gate B
        actually runs, so a tag means the same thing in a backtest and in the
        bot.
    TestBands -- the rank -> band mapping, including both cut boundaries.
    TestColdStart -- what happens before the window is warm; must be an
        explicit "cold", never a guessed band.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from execution.risk_manager import (  # noqa: E402
    GATE_NONE,
    GATE_REGIME,
    RiskManager,
    RiskProfile,
)
from ml.regimes.behavior_tagger import (  # noqa: E402
    DEFAULT_MIN_SAMPLES,
    DEFAULT_WINDOW,
    HIGH_CUT,
    LABEL_COLD,
    LOW_CUT,
    TREND_MIXED,
    TREND_RANGING,
    TREND_TRENDING,
    VOL_HIGH,
    VOL_LOW,
    VOL_NORMAL,
    BehaviorTag,
    tag_bar,
    tag_series,
    trailing_pctile_rank,
    trend_strength_from_ppo,
)


def _series(n=900, seed=7):
    rng = np.random.default_rng(seed)
    # Volatility is positive and autocorrelated, like real NATR.
    natr = np.abs(rng.normal(0.05, 0.02, n).cumsum() * 0.01 + 0.08)
    ppo = rng.normal(0.0, 0.15, n)
    return natr, np.abs(ppo)


class TestCausality(unittest.TestCase):
    def test_no_lookahead_prefix_invariance(self):
        """
        THE causality test. Tagging the first k bars must give byte-identical
        results to the first k tags of the full history. If any tag consulted a
        later bar, truncating the future would change it.
        """
        natr, trend = _series()
        full = tag_series(natr, trend)

        for k in (61, 100, 259, 260, 261, 500, 899):
            prefix = tag_series(natr[:k], trend[:k])
            self.assertEqual(
                prefix, full[:k], f"tags changed when the future was removed at k={k}"
            )

    def test_appending_a_bar_never_rewrites_history(self):
        """The incremental case: one more bar must not relabel earlier bars."""
        natr, trend = _series(n=400)
        before = tag_series(natr[:300], trend[:300])
        after = tag_series(natr[:301], trend[:301])
        self.assertEqual(before, after[:300])

    def test_window_slides_not_expands_forever(self):
        """
        Beyond DEFAULT_WINDOW a bar must forget the distant past, exactly as
        the live bounded deque does. A huge early spike must stop influencing
        the rank once it has rolled out of the window.
        """
        n = DEFAULT_WINDOW * 2
        natr = np.full(n, 0.10)
        natr[0] = 99.0  # enormous early outlier
        trend = np.full(n, 0.05)

        tags = tag_series(natr, trend)
        # Once the spike has rolled out, every remaining value is identical, so
        # the rank is 1.0 (all values <= current).
        self.assertEqual(tags[-1].vol_rank, 1.0)

    def test_determinism(self):
        natr, trend = _series()
        self.assertEqual(tag_series(natr, trend), tag_series(natr, trend))


class TestLiveSymmetry(unittest.TestCase):
    """The offline tagger must use the live gate's estimator, not its own."""

    def test_constants_match_the_forex_risk_profile(self):
        """
        Drift guard. If someone retunes the live regime window, this fails and
        forces the tagger to follow rather than silently disagreeing with the
        bot it is supposed to model.
        """
        profile = RiskProfile.for_asset_class("forex")
        self.assertEqual(DEFAULT_WINDOW, profile.regime_window)
        self.assertEqual(DEFAULT_MIN_SAMPLES, profile.regime_min_samples)

    def test_rank_agrees_with_live_gate_b_decision(self):
        """
        Drive the REAL Gate B and check our rank predicts its verdict on every
        window. Gate B vetoes exactly when pctile_rank < regime_pctile/100, so
        agreement here means our rank is its rank.
        """
        profile = RiskProfile.for_asset_class("forex")
        # Isolate Gate B: silence the cost gate and the time blackout.
        profile = RiskProfile(
            **{
                **profile.__dict__,
                "spread_k_base": 0.0,
                "regime_pctile": 20.0,
            }
        )
        rm = RiskManager(profile)
        cutoff = profile.regime_pctile / 100.0

        rng = np.random.default_rng(11)
        checked = 0
        for _ in range(300):
            window = np.abs(rng.normal(0.1, 0.05, rng.integers(60, 300)))
            gate = rm._evaluate_dynamic_gates(
                entry_price=150.0,
                sl_dist=10.0,          # huge, so Gate A cannot bind
                symbol="GBP_JPY",
                spread=0.0001,
                spread_fresh=True,
                regime_series=window,
                timestamp=None,
            )
            rank = trailing_pctile_rank(window)
            self.assertIsNotNone(rank)
            expected_veto = rank < cutoff
            self.assertEqual(
                gate == GATE_REGIME,
                expected_veto,
                f"disagreed with live Gate B at rank={rank:.4f}",
            )
            self.assertIn(gate, (GATE_NONE, GATE_REGIME))
            checked += 1
        self.assertEqual(checked, 300)

    def test_tag_series_equals_repeated_tag_bar(self):
        """
        tag_series is *defined* as repeated tag_bar over trailing windows.
        Pinning it means the offline path and the live-shaped path cannot drift.
        """
        natr, trend = _series(n=500)
        series_tags = tag_series(natr, trend)

        for i in range(len(natr)):
            lo = max(0, i - DEFAULT_WINDOW + 1)
            manual = tag_bar(natr[lo : i + 1], trend[lo : i + 1])
            self.assertEqual(manual, series_tags[i], f"divergence at bar {i}")


class TestRank(unittest.TestCase):
    def test_rank_is_fraction_at_or_below_current(self):
        self.assertEqual(trailing_pctile_rank([1.0, 2.0, 3.0, 4.0]), 1.0)
        self.assertEqual(trailing_pctile_rank([4.0, 3.0, 2.0, 1.0]), 0.25)
        self.assertEqual(trailing_pctile_rank([1.0, 2.0, 3.0, 2.0]), 0.75)

    def test_trailing_nan_ranks_the_last_real_value(self):
        """Matches Gate B: filter non-finite FIRST, then take the last."""
        self.assertEqual(
            trailing_pctile_rank([4.0, 3.0, 2.0, 1.0, np.nan]),
            trailing_pctile_rank([4.0, 3.0, 2.0, 1.0]),
        )

    def test_all_nan_window_has_no_rank(self):
        self.assertIsNone(trailing_pctile_rank([np.nan, np.inf, -np.inf]))

    def test_empty_window_has_no_rank(self):
        self.assertIsNone(trailing_pctile_rank([]))


class TestBands(unittest.TestCase):
    @staticmethod
    def _window_with_rank(k, n=300):
        """
        A window of n values whose LAST value (the bar being tagged) has
        exactly k of the n values at or below it, i.e. rank == k/n.

        Layout: k-1 values below, n-k values above, then the current bar.
        """
        return np.concatenate([np.zeros(k - 1), np.full(n - k, 2.0), [1.0]])

    def test_window_helper_produces_the_intended_rank(self):
        """Guard the fixture itself — a wrong fixture silently weakens Bands."""
        self.assertAlmostEqual(trailing_pctile_rank(self._window_with_rank(100)), 100 / 300)
        self.assertAlmostEqual(trailing_pctile_rank(self._window_with_rank(1)), 1 / 300)

    def test_low_and_high_bands(self):
        n = 300
        # Current bar is the minimum of its window -> rank 1/n -> low.
        low = np.concatenate([np.ones(n - 1), [0.0]])
        tag = tag_bar(low, low)
        self.assertEqual(tag.vol_band, VOL_LOW)
        self.assertEqual(tag.trend_state, TREND_RANGING)

        # Current bar is the maximum -> rank 1.0 -> high.
        high = np.concatenate([np.zeros(n - 1), [1.0]])
        tag = tag_bar(high, high)
        self.assertEqual(tag.vol_band, VOL_HIGH)
        self.assertEqual(tag.trend_state, TREND_TRENDING)

    def test_cut_points_are_low_exclusive_high_inclusive(self):
        """
        Documented semantics: rank < LOW_CUT is low, rank >= HIGH_CUT is high,
        everything between is normal. Both boundaries are pinned, because an
        off-by-one here quietly reassigns whole swathes of bars between cells.
        """
        # Just below LOW_CUT -> low.
        self.assertEqual(tag_bar(*[self._window_with_rank(99)] * 2).vol_band, VOL_LOW)
        # Exactly LOW_CUT (100/300 == 1/3) -> NOT low; it is normal.
        self.assertEqual(tag_bar(*[self._window_with_rank(100)] * 2).vol_band, VOL_NORMAL)
        # Just below HIGH_CUT -> still normal.
        self.assertEqual(tag_bar(*[self._window_with_rank(199)] * 2).vol_band, VOL_NORMAL)
        # Exactly HIGH_CUT (200/300 == 2/3) -> high.
        self.assertEqual(tag_bar(*[self._window_with_rank(200)] * 2).vol_band, VOL_HIGH)

    def test_label_is_trend_then_vol(self):
        n = 300
        vol_high = np.concatenate([np.zeros(n - 1), [1.0]])   # rank 1.0
        trend_low = np.concatenate([np.ones(n - 1), [0.0]])   # rank 1/n
        tag = tag_bar(vol_high, trend_low)
        self.assertEqual(tag.label, f"{TREND_RANGING}_{VOL_HIGH}")
        self.assertEqual(tag.label, "range_high")

    def test_label_vocabulary_is_closed(self):
        """Only the 9 composites plus 'cold' may ever appear."""
        natr, trend = _series(n=2000, seed=3)
        labels = {t.label for t in tag_series(natr, trend)}
        allowed = {
            f"{t}_{v}"
            for t in (TREND_RANGING, TREND_MIXED, TREND_TRENDING)
            for v in (VOL_LOW, VOL_NORMAL, VOL_HIGH)
        } | {LABEL_COLD}
        self.assertTrue(labels <= allowed, f"unexpected labels: {labels - allowed}")


class TestColdStart(unittest.TestCase):
    def test_below_min_samples_is_cold_not_guessed(self):
        window = np.abs(np.random.default_rng(1).normal(0.1, 0.02, DEFAULT_MIN_SAMPLES - 1))
        tag = tag_bar(window, window)
        self.assertEqual(tag.label, LABEL_COLD)
        self.assertEqual(tag.vol_band, LABEL_COLD)
        self.assertFalse(tag.warm)

    def test_exactly_min_samples_is_warm(self):
        window = np.abs(np.random.default_rng(1).normal(0.1, 0.02, DEFAULT_MIN_SAMPLES))
        self.assertTrue(tag_bar(window, window).warm)

    def test_series_is_cold_then_warm(self):
        natr, trend = _series(n=200)
        tags = tag_series(natr, trend)
        self.assertTrue(all(t.label == LABEL_COLD for t in tags[: DEFAULT_MIN_SAMPLES - 1]))
        self.assertTrue(all(t.warm for t in tags[DEFAULT_MIN_SAMPLES:]))

    def test_warmth_counts_only_finite_bars(self):
        """A window padded with NaN is not warm just because it is long."""
        window = np.full(DEFAULT_WINDOW, np.nan)
        window[:10] = 0.1
        self.assertFalse(tag_bar(window, window).warm)

    def test_missing_trend_data_degrades_to_mixed_not_cold(self):
        """The volatility axis governs warmth; a blind trend axis is 'mixed'."""
        natr = np.abs(np.random.default_rng(2).normal(0.1, 0.02, 300))
        trend = np.full(300, np.nan)
        tag = tag_bar(natr, trend)
        self.assertTrue(tag.warm)
        self.assertEqual(tag.trend_state, TREND_MIXED)
        self.assertNotEqual(tag.label, LABEL_COLD)


class TestMisc(unittest.TestCase):
    def test_trend_strength_discards_direction(self):
        up = trend_strength_from_ppo([0.3, 0.2])
        down = trend_strength_from_ppo([-0.3, -0.2])
        np.testing.assert_array_equal(up, down)

    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            tag_series([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_tag_is_immutable(self):
        tag = tag_bar(np.ones(100), np.ones(100))
        self.assertIsInstance(tag, BehaviorTag)
        with self.assertRaises(Exception):
            tag.label = "tampered"  # type: ignore[misc]

    def test_ranks_are_retained_for_rebucketing(self):
        tag = tag_bar(np.linspace(0.05, 0.2, 300), np.linspace(0.0, 0.4, 300))
        self.assertIsNotNone(tag.vol_rank)
        self.assertIsNotNone(tag.trend_rank)
        self.assertGreaterEqual(tag.vol_rank, 0.0)
        self.assertLessEqual(tag.vol_rank, 1.0)


if __name__ == "__main__":
    unittest.main()
