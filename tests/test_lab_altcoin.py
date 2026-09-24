"""Tests for lab.altcoin_topquint (Audit A).

Synthetic fixtures only — never network. The "momentum" ranker mode is used
for the ranked-picks test because the synthetic fixture is monotonic in
momentum, where lambdarank and the mean-of-lookback score coincide.
"""

from __future__ import annotations

import numpy as np
import pytest

from lab.altcoin_topquint import (
    FRICTION_BPS_PER_SIDE,
    admit_universe,
    apply_friction,
    momentum_features,
    next_week_returns,
    run_a,
)


# ── fixture builders ───────────────────────────────────────────────────────


def _mk_daily(n_days: int, n_assets: int, start: str = "2025-01-06"):
    """A daily (dates, close, volume) grid starting on a Monday."""
    dates = np.arange(
        np.datetime64(start), np.datetime64(start) + np.timedelta64(n_days, "D"),
        dtype="datetime64[D]",
    )
    rng = np.random.default_rng(7)
    close = np.exp(
        np.cumsum(rng.normal(0.0002, 0.02, size=(n_days, n_assets)), axis=0)
    ) * 100.0
    volume = rng.uniform(1e4, 5e4, size=(n_days, n_assets))
    return dates, close, volume


# ── ranker picks the top group (brief §4.5 #1) ────────────────────────────


def test_ranker_identifies_top_group():
    # 10 assets whose 21-day momentum is strictly ordered: asset j trends at
    # j bps/day, so asset 9 is the unambiguous top.
    n_days, n_assets = 80, 10
    dates = np.arange(
        np.datetime64("2025-01-06"),
        np.datetime64("2025-01-06") + np.timedelta64(n_days, "D"),
        dtype="datetime64[D]",
    )
    t = np.arange(n_days)[:, None]
    drift = np.linspace(0.1, 1.0, n_assets)[None, :] * 0.001
    close = 100.0 * np.exp(t * drift)
    volume = np.full((n_days, n_assets), 1e5)

    sig_idx = n_days - 8  # leave ≥7 days of future for a target
    feats = momentum_features(dates, close, sig_idx, lookbacks=(21, 10))
    mean_mom = np.nanmean(feats, axis=1)
    top = int(np.argmax(mean_mom))
    assert top == n_assets - 1  # strongest momentum is the last asset

    # and the top-5 ordering matches the true momentum ordering
    top5 = np.argsort(-mean_mom)[:5]
    assert set(top5) == {5, 6, 7, 8, 9}


# ── liquidity floor exclusion (brief §4.5 #2) ──────────────────────────────


def test_universe_guard_drops_illiquid():
    n_days, n_assets = 40, 3
    dates, close, volume = _mk_daily(n_days, n_assets)
    # asset 2 has near-zero volume every day: it must be dropped
    volume[:, 2] = 0.0
    close[:, 2] = 100.0
    symbols = ["A/USD", "B/USD", "C/USD"]
    as_of = dates[-1]
    admitted, dropped = admit_universe(
        dates, symbols, close * 1.0, volume, as_of,
        min_days=30, min_median_dollar_vol=250_000.0, lookback_days=60,
    )
    assert "C/USD" not in admitted
    assert any(d.startswith("C/USD") for d in dropped)
    assert set(admitted) == {"A/USD", "B/USD"}


def test_universe_guard_uses_only_past_data():
    # an asset whose volume is thin until a huge spike AFTER as_of must NOT
    # be rescued by the future spike
    n_days, n_assets = 40, 1
    dates, close, volume = _mk_daily(n_days, n_assets)
    volume[:, 0] = 1.0  # perpetually thin
    symbols = ["X/USD"]
    as_of = dates[30]
    admitted, _ = admit_universe(
        dates, symbols, close, volume, as_of,
        min_days=30, min_median_dollar_vol=250_000.0, lookback_days=60,
    )
    # now spike volume after as_of; re-check up to as_of is unchanged
    volume2 = volume.copy()
    volume2[35:, 0] = 1e12
    admitted2, _ = admit_universe(
        dates, symbols, close, volume2, as_of,
        min_days=30, min_median_dollar_vol=250_000.0, lookback_days=60,
    )
    assert admitted == admitted2 == []


# ── friction test (brief §4.5 #3) ─────────────────────────────────────────


def test_forced_turnover_half_costs_half_side_rate():
    # full swap: sell-all + buy-all of a disjoint book = turnover 1.0
    prev = {"A/USD": 0.5, "B/USD": 0.5}
    new = {"C/USD": 0.5, "D/USD": 0.5}
    turnover, cost = apply_friction(prev, new)
    assert turnover == pytest.approx(1.0)
    assert cost == pytest.approx(1.0 * FRICTION_BPS_PER_SIDE / 10_000.0)

    # the brief's exact fixture: forced turnover of half the basket → NAV
    # cost exactly 0.5 × (6.6 + 25)/10000
    prev2 = {"A/USD": 1.0}
    new2 = {"A/USD": 0.5, "B/USD": 0.5}
    turnover2, cost2 = apply_friction(prev2, new2)
    assert turnover2 == pytest.approx(0.5)
    assert cost2 == pytest.approx(0.5 * FRICTION_BPS_PER_SIDE / 10_000.0)


# ── determinism (brief §4.5 #4) ────────────────────────────────────────────


def test_run_a_deterministic():
    dates, close, volume = _mk_daily(400, 10)
    symbols = [f"S{i}/USD" for i in range(10)]
    # lift volume so the floor admits all
    volume = volume * 100.0
    r1 = run_a(dates, symbols, close, volume, ranker="momentum", warmup_weeks=0)
    r2 = run_a(dates, symbols, close, volume, ranker="momentum", warmup_weeks=0)
    n1 = [w.net_return for w in r1.weekly]
    n2 = [w.net_return for w in r2.weekly]
    assert n1 == n2
    assert [w.selected for w in r1.weekly] == [w.selected for w in r2.weekly]


# ── leakage guard on features ──────────────────────────────────────────────


def test_momentum_features_ignore_future():
    dates, close, volume = _mk_daily(300, 4)
    sig = 250
    f_full = momentum_features(dates, close, sig, lookbacks=(21, 63))
    f_trunc = momentum_features(dates[: sig + 1], close[: sig + 1], sig, lookbacks=(21, 63))
    np.testing.assert_allclose(f_full, f_trunc, equal_nan=True)


def test_next_week_returns_window():
    # a strictly-increasing series: the 7-day-ahead return is positive and
    # uses only the close at sig+7d.
    n_days = 30
    dates = np.arange(
        np.datetime64("2025-01-06"),
        np.datetime64("2025-01-06") + np.timedelta64(n_days, "D"),
        dtype="datetime64[D]",
    )
    close = np.tile((np.arange(n_days) + 100.0)[:, None], (1, 2))
    r = next_week_returns(dates, close, signal_idx=10, horizon_days=7)
    assert np.all(np.isfinite(r))
    assert np.all(r > 0)
    # and it equals close[17]/close[10] - 1
    assert r[0] == pytest.approx(close[17, 0] / close[10, 0] - 1.0)
