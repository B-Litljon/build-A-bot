"""
Tests for the per-instrument spread-cost feature (V3CostFeatures) and the
per-instrument chop-veto alphas (2026-07-07 cost-awareness experiment).

Covers the five verification points from the implementation plan:
  1. Generator baseline parity with the training veto's expanding/rolling median.
  2. alpha_table=None → generate() is a bit-identical no-op.
  3. Veto mask prices instruments asymmetrically from the table.

Glossary:
    parity (point 1) -- the feature and the training-side veto must compute
        baseline volatility the SAME way. If they diverge, the model sees a
        different cost than the gate enforces.
    bit-identical no-op (point 2) -- the safety guarantee: with no cost table,
        output must be unchanged from before this feature existed, so models
        predating the experiment are unaffected.
    asymmetric pricing (point 3) -- a cheap instrument and an expensive one must
        be judged by their own measured costs, not one shared assumption. This
        is the entire point of the experiment.
    cost_ratio -- see GLOSSARY.md; higher means the toll is large relative to
        the move being targeted.
  4. clean_data interplay: the cost column drops no extra rows.
  5. Train/live golden parity: pooled multi-symbol path == single-symbol path.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.feature_pipeline import FeaturePipeline
from ml.features.v3_features import V3BaseFeatures, V3CostFeatures

REGIME_W = 10  # small window keeps fixtures readable


def _frame(symbols: list[str], n: int, seed: int = 7) -> pl.DataFrame:
    """Synthetic OHLCV frame with a natr_14 column carrying leading NaNs
    (mimics TA-Lib output)."""
    rng = np.random.default_rng(seed)
    parts = []
    for k, sym in enumerate(symbols):
        natr = rng.uniform(0.05, 0.5, n)
        natr[:13] = np.nan  # TA-Lib NATR warmup emits float NaN
        parts.append(
            pl.DataFrame(
                {
                    "timestamp": pl.datetime_range(
                        pl.datetime(2026, 1, 1),
                        pl.datetime(2026, 1, 1) + pl.duration(minutes=15 * (n - 1)),
                        interval="15m",
                        eager=True,
                    ),
                    "symbol": [sym] * n,
                    "close": rng.uniform(1.0, 2.0, n),
                    "natr_14": natr,
                }
            )
        )
    return pl.concat(parts).sort(["symbol", "timestamp"])


def _veto_style_baseline(natr: np.ndarray, w: int) -> np.ndarray:
    """The training veto's baseline, verbatim semantics from
    retrainer._compute_chop_veto_mask: expanding median below w-1, strict
    rolling median after."""
    from numpy.lib.stride_tricks import sliding_window_view

    m = len(natr)
    baseline = np.full(m, np.nan)
    if m >= w:
        sw = sliding_window_view(natr, w)
        baseline[w - 1 :] = np.median(sw, axis=1)
    for i in range(0, min(w - 1, m)):
        baseline[i] = float(np.median(natr[: i + 1]))
    return baseline


class TestBaselineParity:
    def test_matches_veto_median_where_finite(self):
        df = _frame(["XAU_USD"], 60)
        gen = V3CostFeatures(
            alpha_table={"XAU_USD": 1.0}, regime_window=REGIME_W
        )  # alpha=1 → cost_ratio = baseline/natr
        out = gen.generate(df)

        natr = df["natr_14"].to_numpy()
        expected_baseline = _veto_style_baseline(natr, REGIME_W)
        got_baseline = (out["cost_ratio"] * out["natr_14"]).to_numpy()

        both = np.isfinite(expected_baseline) & np.isfinite(got_baseline)
        assert both.sum() > 30, "fixture should have a large comparable region"
        np.testing.assert_allclose(
            got_baseline[both], expected_baseline[both], rtol=1e-12
        )
        # Divergence (polars skips NaN, numpy poisons the window) is confined
        # to the NATR warmup + one regime window.
        divergent = np.where(np.isfinite(got_baseline) != np.isfinite(expected_baseline))[0]
        assert all(i < REGIME_W + 13 for i in divergent)


class TestNoOp:
    def test_none_table_returns_df_unchanged(self):
        df = _frame(["XAU_USD", "GBP_NZD"], 40)
        out = V3CostFeatures(alpha_table=None).generate(df)
        assert out.columns == df.columns
        assert out.equals(df)

    def test_missing_natr_raises(self):
        df = _frame(["XAU_USD"], 20).drop("natr_14")
        with pytest.raises(ValueError, match="natr_14"):
            V3CostFeatures(alpha_table={"XAU_USD": 0.1}).generate(df)


class TestAlphaLookup:
    def test_per_symbol_alpha_and_default_fallback(self):
        # Identical natr path per symbol → cost_ratio must scale exactly
        # with each symbol's alpha.
        parts = [
            _frame([s], 30, seed=42).with_columns(pl.lit(s).alias("symbol"))
            for s in ["CHEAP", "PRICY", "UNLISTED"]
        ]
        df = pl.concat(parts).sort(["symbol", "timestamp"])
        gen = V3CostFeatures(
            alpha_table={"CHEAP": 0.07, "PRICY": 0.90},
            default_alpha=0.15,
            regime_window=REGIME_W,
        )
        out = gen.generate(df)
        by_sym = {
            s: out.filter(pl.col("symbol") == s)["cost_ratio"].to_numpy()
            for s in ["CHEAP", "PRICY", "UNLISTED"]
        }
        finite = np.isfinite(by_sym["CHEAP"])
        np.testing.assert_allclose(
            by_sym["PRICY"][finite] / by_sym["CHEAP"][finite], 0.90 / 0.07
        )
        np.testing.assert_allclose(
            by_sym["UNLISTED"][finite] / by_sym["CHEAP"][finite], 0.15 / 0.07
        )


class TestVetoMaskAsymmetry:
    def test_expensive_symbol_vetoes_more(self):
        from src.core.retrainer import _compute_chop_veto_mask
        from src.execution.risk_manager import RiskProfile

        profile = RiskProfile.for_asset_class("forex")
        df = _frame(["CHEAP", "PRICY"], 400, seed=11)
        table = {"CHEAP": 0.05, "PRICY": 0.95}
        veto = _compute_chop_veto_mask(df, profile, sl_mult=1.0, alpha_table=table)

        symbols = df["symbol"].to_numpy()
        cheap_rate = veto[symbols == "CHEAP"].mean()
        pricy_rate = veto[symbols == "PRICY"].mean()
        assert pricy_rate > cheap_rate, (
            f"expensive instrument should veto more: {pricy_rate=} {cheap_rate=}"
        )

    def test_no_table_matches_flat_alpha(self):
        from src.core.retrainer import _compute_chop_veto_mask
        from src.execution.risk_manager import RiskProfile

        profile = RiskProfile.for_asset_class("forex")
        df = _frame(["XAU_USD"], 300, seed=3)
        flat = _compute_chop_veto_mask(df, profile, sl_mult=1.0)
        tabled_at_flat = _compute_chop_veto_mask(
            df, profile, sl_mult=1.0,
            alpha_table={"XAU_USD": profile.spread_atr_alpha},
        )
        np.testing.assert_array_equal(flat, tabled_at_flat)


class TestCleanDataInterplay:
    def test_cost_column_drops_no_extra_rows(self):
        base_cols = ["natr_14"]
        df = _frame(["XAU_USD", "GBP_NZD"], 50)
        with_cost = V3CostFeatures(
            alpha_table={"XAU_USD": 0.07, "GBP_NZD": 0.90},
            regime_window=REGIME_W,
        ).generate(df)

        cleaned_without = FeaturePipeline.clean_data(df, feature_cols=base_cols)
        cleaned_with = FeaturePipeline.clean_data(
            with_cost, feature_cols=base_cols + ["cost_ratio"]
        )
        assert len(cleaned_with) == len(cleaned_without)


class TestTrainLiveParity:
    def test_pooled_equals_single_symbol_tail(self):
        """Retrainer path (pooled, full history) vs live path (single-symbol
        buffer): identical cost_ratio on the newest row."""
        table = {"XAU_USD": 0.0718, "GBP_NZD": 0.9032}
        n = 300
        pooled = _frame(["GBP_NZD", "XAU_USD"], n, seed=5)
        gen = V3CostFeatures(alpha_table=table, regime_window=260)
        pooled_out = gen.generate(pooled)

        for sym in ["XAU_USD", "GBP_NZD"]:
            # Live: the orchestrator hands the strategy a single-symbol tail
            # buffer (279 bars at M15). Same generator instance, live-style.
            single = pooled.filter(pl.col("symbol") == sym).tail(279)
            single_out = gen.generate(single)
            live_val = single_out["cost_ratio"].tail(1)[0]

            pooled_val = (
                pooled_out.filter(pl.col("symbol") == sym)["cost_ratio"].tail(1)[0]
            )
            # 279-bar tail spans ≥ regime window + NATR warmup at these sizes;
            # min_samples=1 keeps the newest row defined in both paths.
            assert live_val is not None and pooled_val is not None
            assert abs(live_val - pooled_val) < 1e-12

    def test_exactly_warmup_history_yields_finite_newest_row(self):
        """The trap test: at exactly warmup-length history the newest row's
        cost_ratio must be finite (min_samples=1), or live would silently
        never signal."""
        df = _frame(["XAU_USD"], 260, seed=9)
        out = V3CostFeatures(
            alpha_table={"XAU_USD": 0.0718}, regime_window=260
        ).generate(df)
        val = out["cost_ratio"].tail(1)[0]
        assert val is not None and np.isfinite(val)
