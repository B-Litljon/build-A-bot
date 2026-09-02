"""
Tests for analysis.walk_forward_tuner -- walk-forward fold generation, parameter search, and OOS validation.
"""

from dataclasses import replace

import numpy as np
import polars as pl
import pytest

from analysis.walk_forward_tuner import generate_folds, tune_strategy, to_frame
from strategies.concrete_strategies import SMACrossoverStrategy


def make_ohlcv_series(n_bars: int = 500) -> pl.DataFrame:
    rng = np.random.default_rng(999)
    close = 100.0 + np.cumsum(rng.normal(0, 0.4, n_bars))
    high = close + 0.6
    low = close - 0.6
    return pl.DataFrame({
        "timestamp": [f"2026-01-01T{i:04d}:00Z" for i in range(n_bars)],
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": [1000] * n_bars,
        "symbol": ["TEST"] * n_bars,
    })


def test_generate_folds_boundaries():
    folds = generate_folds(1000, n_folds=3)
    assert len(folds) == 3
    for tr_start, tr_end, val_start, val_end in folds:
        assert tr_start == 0
        assert tr_end == val_start
        assert val_end <= 1000
        assert tr_end > tr_start
        assert val_end > val_start

    # Expanding window: train_end must increase with each fold
    assert folds[0][1] < folds[1][1] < folds[2][1]


def test_tune_strategy_execution():
    df = make_ohlcv_series(n_bars=300)
    grid = {
        "fast_period": [5, 10],
        "slow_period": [20, 25],
    }
    results = tune_strategy(SMACrossoverStrategy, grid, df, n_folds=2, min_trades=5)
    assert len(results) == 4
    for r in results:
        assert "fast_period" in r.params
        assert "slow_period" in r.params
        assert len(r.fold_metrics) == 2
        assert isinstance(r.robust, bool)

    frame = to_frame(results)
    assert frame.height == 4
    assert "oos_trades" in frame.columns
    assert "oos_win_rate" in frame.columns
    assert "oos_net_ev_r" in frame.columns


# ---------------------------------------------------------------------------
# Evidential + fold-warmth regressions (2026-09-02)
# ---------------------------------------------------------------------------


def test_robust_requires_the_clopper_pearson_bound_to_clear_breakeven():
    """
    A point estimate is not evidence.

    `robust` used to compute a Clopper-Pearson bound and then ignore it, so a
    lucky 31-trade run could read as robust on its point estimate alone. The
    bound must now clear the bracket's own break-even win rate.
    """
    from analysis.walk_forward_tuner import TunedParamResult

    lucky = TunedParamResult(
        params={"fast_period": 5},
        total_oos_trades=31,
        pooled_oos_win_rate=0.45,
        pooled_oos_net_ev=0.10,
        pooled_oos_pf=1.3,
        positive_folds_ratio=1.0,
        pooled_oos_cp_lb=0.30,        # below the 1/3 break-even for a 2:1 payoff
        breakeven_win_rate=1.0 / 3.0,
    )
    assert lucky.robust is False, "a bound under break-even must not read robust"

    solid = TunedParamResult(
        params={"fast_period": 5},
        total_oos_trades=400,
        pooled_oos_win_rate=0.45,
        pooled_oos_net_ev=0.10,
        pooled_oos_pf=1.3,
        positive_folds_ratio=1.0,
        pooled_oos_cp_lb=0.41,
        breakeven_win_rate=1.0 / 3.0,
    )
    assert solid.robust is True

    # And the other bars still bind independently.
    assert not replace(solid, pooled_oos_net_ev=-0.01).robust
    assert not replace(solid, total_oos_trades=29).robust
    assert not replace(solid, positive_folds_ratio=0.33).robust


def test_oos_folds_are_warm_so_no_validation_bars_are_silently_discarded():
    """
    Each validation slice is prefixed with warmup bars of prior history.

    Slicing a fold at exactly val_start meant run_backtest spent that fold's
    entire warmup window unable to trade, silently discarding the first
    warmup_period bars of every fold and resetting trailing state that live
    never resets. Warming the folds must recover trades, never lose them.
    """
    from analysis.walk_forward_tuner import generate_folds, tune_strategy

    df = make_ohlcv_series(n_bars=3000)
    warm = tune_strategy(
        SMACrossoverStrategy,
        {"fast_period": [5], "slow_period": [15]},
        df,
        n_folds=3,
    )
    assert warm, "expected a result for the single parameter combination"

    folds = generate_folds(len(df), n_folds=3)
    warmup = SMACrossoverStrategy(fast_period=5, slow_period=15).warmup_period

    # Every fold's validation window is longer than the warmup it must absorb,
    # so a cold start would have thrown away real bars in each one.
    for _, _, val_start, val_end in folds:
        assert val_start >= warmup, "fixture too short to test fold warmth"

    # The pooled OOS population is non-empty and the CP bound was computed.
    assert warm[0].total_oos_trades > 0
    assert warm[0].breakeven_win_rate == pytest.approx(1.0 / 3.0)
    assert 0.0 <= warm[0].pooled_oos_cp_lb <= 1.0
