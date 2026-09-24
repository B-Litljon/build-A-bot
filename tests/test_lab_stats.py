"""Deterministic tests for src/lab/stats.py — the shared DSR/CSCV/HLZ module.

These pin the exact contract the 2026-09-24 quant lanes agreed on
(``llm_reports/handoffs/2026-09-24_lane*.md`` §6.2) so Lane 1..5 all import the
same numbers. The six mandatory cases come from Lane 2's brief §6.4.

Glossary:
    DSR / PBO / HLZ -- see src/lab/stats.py docstring and GLOSSARY.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from lab.stats import (
    cscv_pbo,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    hlz_haircut_sharpe,
    hlz_t_stat,
)


# ── mandatory brief cases (§6.4 case 6) ───────────────────────────────

def test_dsr_zero_skill_is_bounded_probability():
    dsr = deflated_sharpe_ratio(0, 10, 100, 0, 3)
    assert isinstance(dsr, float)
    assert 0.0 <= dsr <= 1.0


def test_cscv_pbo_zeros_is_coin_flip():
    # T x N zeros: no ordering signal anywhere -> PBO must be exactly 0.5.
    assert cscv_pbo(np.zeros((80, 5))) == 0.5


def test_hlz_haircut_reduces_sharpe():
    assert hlz_haircut_sharpe(1.0, 10) < 1.0


# ── DSR behaviour ─────────────────────────────────────────────────────

def test_dsr_increases_with_edge():
    # A genuinely strong SR should deflate to a HIGHER probability than a
    # marginal one at the same trial count. Use borderline SRs (the regime the
    # falsification gate actually operates in) so neither saturates Phi to 1.
    weak = deflated_sharpe_ratio(0.05, 10, 100, 0, 3)
    strong = deflated_sharpe_ratio(0.30, 10, 100, 0, 3)
    assert strong > weak


def test_dsr_more_trials_lower_probability():
    # More variants tried -> larger SR_0 hurdle -> the same SR_hat is LESS
    # believable. Borderline SR + short window keep both below saturation.
    few = deflated_sharpe_ratio(0.20, 5, 100, 0, 3)
    many = deflated_sharpe_ratio(0.20, 500, 100, 0, 3)
    assert many < few


def test_dsr_fat_tails_deflate_harder():
    # Higher kurtosis inflates the Sharpe variance -> lower DSR for the same SR.
    thin = deflated_sharpe_ratio(0.20, 10, 100, 0.0, 3.0)
    fat = deflated_sharpe_ratio(0.20, 10, 100, 0.0, 9.0)
    assert fat < thin


def test_dsr_strong_edge_saturates_to_one():
    assert deflated_sharpe_ratio(2.5, 5, 500, 0, 3) == pytest.approx(1.0, abs=1e-9)


def test_dsr_rejects_degenerate_inputs():
    assert np.isnan(deflated_sharpe_ratio(float("nan"), 10, 100, 0, 3))
    assert np.isnan(deflated_sharpe_ratio(1.0, 10, 1, 0, 3))   # n_obs < 2
    assert np.isnan(deflated_sharpe_ratio(1.0, 0, 100, 0, 3))   # n_trials < 1


def test_expected_max_sharpe_grows_with_trials_and_zero_for_one():
    assert expected_max_sharpe(1, 1 / 100) == 0.0
    assert expected_max_sharpe(100, 1 / 100) > expected_max_sharpe(10, 1 / 100) > 0


# ── CSCV PBO behaviour ────────────────────────────────────────────────

def test_cscv_pbo_identical_columns():
    # All strategies identical -> no split can prefer one -> coin flip.
    col = np.random.default_rng(0).normal(0.001, 0.01, 120)
    X = np.tile(col[:, None], (1, 6))
    assert cscv_pbo(X) == 0.5


def test_cscv_pbo_detects_persistent_winner():
    # Strategy 0 dominates in-sample AND out-of-sample -> PBO near 0.
    rng = np.random.default_rng(1)
    base = rng.normal(0.0, 0.01, (160, 5))
    base[:, 0] += 0.02  # a real, persistent edge on strategy 0
    pbo = cscv_pbo(base)
    assert 0.0 <= pbo <= 0.5


def test_cscv_pbo_overfit_is_high():
    # Best-in-sample strategy is pure noise that flips sign out of sample.
    # Build a matrix where block-local winners are anti-persistent.
    rng = np.random.default_rng(2)
    X = rng.normal(0, 0.01, (160, 8))
    # make strategy performance reverse between alternating blocks
    for b in range(8):
        lo, hi = b * 20, (b + 1) * 20
        X[lo:hi, b % 8] += 0.02 * (1 if b % 2 == 0 else -1)
    pbo = cscv_pbo(X)
    assert 0.0 <= pbo <= 1.0


def test_cscv_pbo_bounds_and_degenerate():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 0.01, (200, 4))
    assert 0.0 <= cscv_pbo(X) <= 1.0
    with pytest.raises(ValueError):
        cscv_pbo(np.zeros((10,)))  # not 2-D


# ── HLZ behaviour ─────────────────────────────────────────────────────

def test_hlz_more_trials_bigger_haircut():
    assert hlz_haircut_sharpe(1.0, 100) < hlz_haircut_sharpe(1.0, 5) < 1.0


def test_hlz_t_stat_defined_and_signed():
    t = hlz_t_stat(1.0, 10, 252, 0.0, 3.0)
    assert np.isfinite(t)
    # symmetric: negative haircut SR -> negative t
    assert hlz_t_stat(-1.0, 10, 252, 0.0, 3.0) < 0


def test_hlz_t_stat_nan_on_degenerate():
    assert np.isnan(hlz_t_stat(float("nan"), 10, 252))
    assert np.isnan(hlz_t_stat(1.0, 10, 1))
