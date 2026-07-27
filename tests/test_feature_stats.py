"""
Tests for the feature-stats sidecar + PSI (src/ml/feature_stats.py).

Covers both the artifact's shape and the PSI maths -- including the property
that motivated the whole design: identical distributions must score near zero,
and the null calibration must sit well above zero on ordinary autocorrelated
data. Without that second fact the textbook thresholds would false-alarm
constantly.

Glossary:
    pooled vs per_symbol -- both populations are written; drift diagnosis needs
        per-symbol because instruments sit at different baseline levels.
    continuous vs categorical binning -- features with few distinct values
        (session flags, hour_of_day) get one bin per value instead of deciles.
    null_psi -- the calibration quantiles; see GLOSSARY.md (PSI).
    round-trip -- stats are saved and reloaded to confirm the JSON artifact
        survives serialisation unchanged.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from ml.feature_stats import (
    classify_psi,
    classify_psi_calibrated,
    compute_feature_stats,
    drift_flags,
    load_feature_stats,
    psi,
    psi_report,
    save_feature_stats,
)


def _train_frame(n=5000, seed=1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "symbol": ["XAU_USD"] * n + ["GBP_NZD"] * n,
            "rsi_14": np.concatenate(
                [rng.normal(50, 10, n), rng.normal(55, 12, n)]
            ),
            "session_ny": rng.integers(0, 2, 2 * n).astype(float),
            "natr_14": np.concatenate(
                [rng.lognormal(-1.5, 0.4, n), rng.lognormal(-3.0, 0.4, n)]
            ),
        }
    )


FEATS = ["rsi_14", "session_ny", "natr_14"]


class TestComputeStats:
    def test_shapes_and_kinds(self):
        stats = compute_feature_stats(_train_frame(), FEATS)
        assert set(stats["pooled"]) == set(FEATS)
        assert set(stats["per_symbol"]) == {"XAU_USD", "GBP_NZD"}
        assert stats["pooled"]["rsi_14"]["kind"] == "continuous"
        assert len(stats["pooled"]["rsi_14"]["decile_edges"]) == 9
        assert stats["pooled"]["session_ny"]["kind"] == "categorical"

    def test_roundtrip(self, tmp_path):
        stats = compute_feature_stats(_train_frame(), FEATS)
        save_feature_stats(stats, tmp_path)
        loaded = load_feature_stats(tmp_path)
        assert loaded is not None
        assert loaded["pooled"]["rsi_14"]["decile_edges"] == pytest.approx(
            stats["pooled"]["rsi_14"]["decile_edges"]
        )

    def test_missing_dir_returns_none(self, tmp_path):
        assert load_feature_stats(tmp_path / "nope") is None


class TestPSI:
    def test_same_distribution_is_stable(self):
        stats = compute_feature_stats(_train_frame(seed=1), FEATS)
        live = _train_frame(n=500, seed=99)  # same params, fresh draws
        for sym in ["XAU_USD", "GBP_NZD"]:
            rep = psi_report(
                live.filter(pl.col("symbol") == sym), stats, symbol=sym
            )
            for feat, v in rep.items():
                assert v < 0.10, f"{sym}/{feat} unexpectedly drifted: {v}"

    def test_shifted_distribution_flags_severe(self):
        stats = compute_feature_stats(_train_frame(), FEATS)
        rng = np.random.default_rng(5)
        # Volatility collapses to a fraction of training levels (the
        # "holiday chop" scenario).
        live = pl.DataFrame(
            {"natr_14": rng.lognormal(-1.5, 0.4, 500) * 0.25}
        )
        v = psi(
            live["natr_14"].to_numpy(),
            stats["per_symbol"]["XAU_USD"]["natr_14"],
        )
        assert v is not None and v > 0.25
        assert classify_psi(v) == "SEVERE"

    def test_categorical_unseen_value_flags(self):
        stats = compute_feature_stats(_train_frame(), FEATS)
        live = pl.DataFrame({"session_ny": [2.0] * 100})  # never in training
        v = psi(live["session_ny"].to_numpy(), stats["pooled"]["session_ny"])
        assert v is not None and v > 0.25

    def test_pooled_vs_per_symbol_reference_matters(self):
        """natr differs by instrument: gold's live natr should be stable vs
        gold's own stats but drifted vs the pooled reference — the reason we
        store per-symbol stats at all."""
        stats = compute_feature_stats(_train_frame(), FEATS)
        rng = np.random.default_rng(11)
        gold_live = pl.DataFrame({"natr_14": rng.lognormal(-1.5, 0.4, 500)})
        v_own = psi(gold_live["natr_14"].to_numpy(),
                    stats["per_symbol"]["XAU_USD"]["natr_14"])
        v_pooled = psi(gold_live["natr_14"].to_numpy(),
                       stats["pooled"]["natr_14"])
        assert v_own < 0.10 < v_pooled

    def test_tiny_sample_returns_none(self):
        stats = compute_feature_stats(_train_frame(), FEATS)
        assert psi(np.array([50.0] * 5), stats["pooled"]["rsi_14"]) is None


def _autocorr_frame(n=8000, seed=3, shift=0.0) -> pl.DataFrame:
    """Slowly-varying (autocorrelated) series — the realistic market case
    where raw PSI thresholds false-alarm on short windows."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = 0.98 * x[i - 1] + rng.normal(0, 0.2)
    return pl.DataFrame({"symbol": ["XAU_USD"] * n, "slow_feat": x + shift})


class TestCalibratedDrift:
    def test_null_quantiles_present(self):
        stats = compute_feature_stats(_autocorr_frame(), ["slow_feat"])
        null = stats["per_symbol"]["XAU_USD"]["slow_feat"]["null_psi"]
        assert null["window_bars"] == 100
        assert null["p50"] < null["p95"] < null["p99"]

    def test_normal_window_not_flagged_despite_high_raw_psi(self):
        """The exact failure mode observed live: an ordinary contiguous
        window of an autocorrelated series has raw PSI >> 0.25 (textbook
        'SEVERE') but is NOT drift — calibration must say stable."""
        train = _autocorr_frame()
        stats = compute_feature_stats(train, ["slow_feat"])
        ref = stats["per_symbol"]["XAU_USD"]["slow_feat"]
        # a held-back contiguous window from the same process
        window = train["slow_feat"].to_numpy()[4200:4300]
        raw = psi(window, ref)
        assert raw is not None and raw > 0.25  # textbook would scream
        assert classify_psi(raw) == "SEVERE"   # ...and does
        assert classify_psi_calibrated(raw, ref) in ("stable", "moderate")

    def test_genuinely_foreign_regime_still_flagged(self):
        """A level the training series never visited must beat the null."""
        train = _autocorr_frame()
        stats = compute_feature_stats(train, ["slow_feat"])
        ref = stats["per_symbol"]["XAU_USD"]["slow_feat"]
        foreign = _autocorr_frame(n=200, seed=9, shift=30.0)  # way off-range
        v = psi(foreign["slow_feat"].to_numpy()[:100], ref)
        assert classify_psi_calibrated(v, ref) == "SEVERE"

    def test_drift_flags_uses_per_symbol_null(self):
        train = _autocorr_frame()
        stats = compute_feature_stats(train, ["slow_feat"])
        window = train["slow_feat"].to_numpy()[4200:4300]
        raw = psi(window, stats["per_symbol"]["XAU_USD"]["slow_feat"])
        flags = drift_flags({"slow_feat": raw}, stats, symbol="XAU_USD")
        assert flags["slow_feat"] in ("stable", "moderate")
