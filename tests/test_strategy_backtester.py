"""
Tests for analysis.strategy_backtester -- execution realism, no-lookahead, and ledger capture.
"""

import numpy as np
import polars as pl
import pytest

from analysis.strategy_backtester import run_backtest, BacktestReport
from strategies.base import BaseStrategy, Signal
from strategies.concrete_strategies import SMACrossoverStrategy


def make_trending_ohlcv(n_bars: int = 300, trend: float = 30.0) -> pl.DataFrame:
    rng = np.random.default_rng(123)
    # Sine wave + drift guarantees multiple moving average crosses
    t = np.linspace(0, 8 * np.pi, n_bars)
    wave = 10.0 * np.sin(t)
    drift = np.linspace(0, trend, n_bars)
    close = 100.0 + drift + wave + rng.normal(0, 0.2, n_bars)
    high = close + 1.0
    low = close - 1.0
    open_p = close
    return pl.DataFrame({
        "timestamp": [f"2026-01-01T{i:04d}:00Z" for i in range(n_bars)],
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": [1000] * n_bars,
        "symbol": ["EUR_USD"] * n_bars,
        "behavior_label": ["trend_high"] * n_bars,
    })


def test_backtest_empty_or_short_data():
    strat = SMACrossoverStrategy(fast_period=5, slow_period=15)
    short_df = make_trending_ohlcv(n_bars=20)
    rep = run_backtest(strat, short_df)
    assert rep.total_trades == 0
    assert rep.win_rate == 0.0
    assert rep.to_ledger().is_empty()


def test_backtest_trade_execution_and_ledger():
    strat = SMACrossoverStrategy(fast_period=5, slow_period=15, atr_period=5)
    df = make_trending_ohlcv(n_bars=250, trend=40.0)

    rep = run_backtest(strat, df, sl_mult=2.0, tp_mult=4.0, toll_r=0.33)
    assert isinstance(rep, BacktestReport)
    assert rep.total_trades > 0
    assert rep.wins + rep.losses == rep.total_trades
    assert 0.0 <= rep.win_rate <= 1.0

    ledger = rep.to_ledger()
    assert not ledger.is_empty()
    assert "macro_win" in ledger.columns
    assert "behavior_label" in ledger.columns
    assert "net_r" in ledger.columns
    assert "gross_r" in ledger.columns

    # Verify spread toll was deducted: net_r == gross_r - toll_r
    for row in ledger.iter_rows(named=True):
        assert pytest.approx(row["net_r"], abs=1e-5) == row["gross_r"] - 0.33
        assert row["macro_win"] in (0, 1)


# ---------------------------------------------------------------------------
# Execution-realism regressions (2026-09-02)
#
# Three defects found in review, each of which inflated results silently:
#   1. a timeout was booked at the bracket's full nominal payoff on sign alone
#   2. a bar that GAPPED past a level was filled AT the level anyway
#   3. the live chop-filter gates were never applied
# ---------------------------------------------------------------------------


class _AlwaysLong(BaseStrategy):
    """Fires every bar. Isolates the exit machinery from any entry logic."""

    warmup_period = 5

    def generate_signals(self, df: pl.DataFrame):
        c = float(df["close"][-1])
        return Signal(direction="long", entry_price=c,
                      raw_sl_distance=c * 0.01, raw_tp_distance=c * 0.01)


def _flat_frame(n: int = 600, drift_per_bar: float = 0.00001) -> pl.DataFrame:
    """A market that crawls upward and never reaches either bracket level."""
    close = 100.0 * (1.0 + np.arange(n) * drift_per_bar)
    return pl.DataFrame({
        "open": close,
        "high": close * 1.0001,
        "low": close * 0.9999,
        "close": close,
        "volume": np.ones(n),
        "symbol": ["EUR_JPY"] * n,
    })


def test_timeout_is_not_a_win_and_pays_only_the_realised_move():
    """
    A trade that never touched target or stop did not earn the bracket payoff.

    Regression: this frame drifts ~0.6% in total and touches neither level, so
    every trade times out. The old code booked each one +2R on sign alone and
    reported a 100% win rate.
    """
    rep = run_backtest(_AlwaysLong(), _flat_frame(),
                       sl_mult=2.0, tp_mult=4.0, max_hold_bars=45, toll_r=0.0)

    assert rep.total_trades > 0
    assert {t.exit_reason for t in rep.trades} == {"timeout"}

    # macro_win is bracket resolution: a timeout never resolved the target.
    assert rep.wins == 0
    assert rep.win_rate == 0.0

    # ...and R is the realised move, which on this frame is tiny but positive,
    # nowhere near the +2.0R the nominal payoff would have paid.
    assert 0.0 < rep.gross_ev_r < 0.25
    for t in rep.trades:
        assert -1.0 < t.gross_r < 2.0


def test_gap_through_stop_fills_at_the_open_not_the_stop_price():
    """A price that gapped past the stop never traded at the stop."""
    n = 60
    px = np.full(n, 100.0)
    px[10:] = 90.0                     # entry ~100, stop at 98.0, opens at 90.0
    df = pl.DataFrame({"open": px, "high": px, "low": px, "close": px,
                       "volume": np.ones(n), "symbol": ["EUR_JPY"] * n})

    rep = run_backtest(_AlwaysLong(), df, sl_mult=2.0, tp_mult=4.0, toll_r=0.0)
    first = rep.trades[0]

    assert first.exit_reason == "sl_gap"
    assert first.exit_price == pytest.approx(90.0)
    # A 10-point adverse move against a 2-point stop is a 5R loss, not 1R.
    assert first.gross_r == pytest.approx(-5.0)
    assert first.macro_win == 0


def test_gap_through_target_fills_at_the_open_and_counts_as_a_win():
    n = 60
    px = np.full(n, 100.0)
    px[10:] = 120.0                    # target at 104.0, opens far beyond it
    df = pl.DataFrame({"open": px, "high": px, "low": px, "close": px,
                       "volume": np.ones(n), "symbol": ["EUR_JPY"] * n})

    rep = run_backtest(_AlwaysLong(), df, sl_mult=2.0, tp_mult=4.0, toll_r=0.0)
    first = rep.trades[0]

    assert first.exit_reason == "tp_gap"
    assert first.exit_price == pytest.approx(120.0)
    assert first.macro_win == 1
    assert first.gross_r == pytest.approx(10.0)


def test_clean_level_hits_still_book_exactly_the_nominal_payoff():
    """
    Realised-R booking must not disturb the ordinary case.

    Entry at 100 with raw_sl 1.0 gives a 2.0 stop distance and a 4.0 target
    distance, so a clean fill at the 104.0 target is exactly +2R.
    """
    n = 60
    o = np.full(n, 100.0)
    h = np.full(n, 100.5)
    l = np.full(n, 99.5)
    c = np.full(n, 100.0)
    h[10] = 104.5          # trades THROUGH the 104.0 target intrabar...
    o[10] = 100.0          # ...but opens below it, so this is a fill, not a gap

    df = pl.DataFrame({"open": o, "high": h, "low": l, "close": c,
                       "volume": np.ones(n), "symbol": ["EUR_JPY"] * n})

    rep = run_backtest(_AlwaysLong(), df, sl_mult=2.0, tp_mult=4.0, toll_r=0.0)
    hits = [t for t in rep.trades if t.exit_reason == "tp"]
    assert hits, "expected at least one clean target hit"
    for t in hits:
        assert t.exit_price == pytest.approx(104.0)
        assert t.gross_r == pytest.approx(2.0)   # tp_mult / sl_mult
        assert t.macro_win == 1


def test_gates_veto_signals_and_the_funnel_is_recorded():
    """
    With a RiskManager supplied, vetoed signals are counted, not traded.

    Gate B vetoes the bottom 20% of the trailing volatility window, which is
    ~60% of the tagger's *_low band -- so a backtest without gates measures a
    population the live bot would never take.
    """
    from execution.risk_manager import RiskManager, RiskProfile

    n = 900
    close = 100.0 * (1.0 + np.arange(n) * 0.00001)
    # Gate B ranks the CURRENT bar inside its own trailing window, so a flat
    # low-volatility stretch ranks high, not low. A monotonically DECAYING
    # series is what keeps the newest bar at the bottom of its own window and
    # therefore continuously vetoed.
    natr = np.concatenate([
        np.full(300, 0.30),
        np.linspace(0.30, 0.01, n - 300),
    ])
    df = pl.DataFrame({
        "open": close, "high": close * 1.0001, "low": close * 0.9999,
        "close": close, "volume": np.ones(n),
        "symbol": ["EUR_JPY"] * n, "natr_14": natr,
    })

    rm = RiskManager(RiskProfile.for_asset_class("forex"))
    gated = run_backtest(_AlwaysLong(), df, risk_manager=rm, toll_r=0.0)
    ungated = run_backtest(_AlwaysLong(), df, toll_r=0.0)

    assert sum(gated.gate_rejections.values()) > 0, "gates never fired"
    assert "regime" in gated.gate_rejections, "expected Gate B to bite"
    assert gated.total_trades < ungated.total_trades
    assert ungated.gate_rejections == {}


def test_spread_alphas_produce_a_per_instrument_toll():
    """The shipped alphas span 12.6x; one flat constant cannot represent them."""
    n = 600
    close = 100.0 * (1.0 + np.arange(n) * 0.00001)
    df = pl.DataFrame({
        "open": close, "high": close * 1.0001, "low": close * 0.9999,
        "close": close, "volume": np.ones(n),
        "symbol": ["GBP_NZD"] * n, "natr_14": np.full(n, 0.10),
    })

    cheap = run_backtest(_AlwaysLong(), df, spread_alphas={"GBP_NZD": 0.072})
    dear = run_backtest(_AlwaysLong(), df, spread_alphas={"GBP_NZD": 0.903})

    assert cheap.total_trades == dear.total_trades
    assert cheap.net_ev_r > dear.net_ev_r, "a wider spread must cost more"
    assert cheap.gross_ev_r == pytest.approx(dear.gross_ev_r), "gross is cost-free"
