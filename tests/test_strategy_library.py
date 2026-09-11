"""
Tests for the non-ML strategy library (SMACrossover, RSIMeanReversion, BollingerBreakout, DonchianBreakout, Momentum).
"""

import numpy as np
import polars as pl
import pytest
import os
from pathlib import Path
from strategies.base import BaseStrategy

from strategies.concrete_strategies import (
    BollingerBreakoutStrategy,
    DonchianBreakoutStrategy,
    MomentumStrategy,
    RSIMeanReversionStrategy,
    SMACrossoverStrategy,
    STRATEGIES,
)


def make_synthetic_ohlcv(n_bars: int = 150, trend: float = 0.0) -> pl.DataFrame:
    """Generate synthetic OHLCV data for testing."""
    rng = np.random.default_rng(42)
    base = 100.0
    drift = np.linspace(0, trend, n_bars)
    noise = rng.normal(0, 0.5, n_bars)
    close = base + drift + noise
    high = close + np.abs(rng.normal(0.5, 0.2, n_bars))
    low = close - np.abs(rng.normal(0.5, 0.2, n_bars))
    open_p = (high + low) / 2.0
    vol = rng.integers(100, 1000, n_bars)

    return pl.DataFrame({
        "timestamp": [f"2026-01-01T{i:04d}:00Z" for i in range(n_bars)],
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
        "symbol": ["TEST"] * n_bars,
    })


def test_registry_contains_all_strategies():
    expected = [
        "ml_strategy",
        "sma_crossover",
        "rsi_mean_reversion",
        "bollinger_breakout",
        "donchian_breakout",
        "momentum",
        "regime_router",
    ]
    for name in expected:
        assert name in STRATEGIES, f"Strategy {name} missing from STRATEGIES registry"


@pytest.mark.parametrize(
    "strat_cls,kwargs",
    [
        (SMACrossoverStrategy, {"fast_period": 10, "slow_period": 30}),
        (RSIMeanReversionStrategy, {"rsi_period": 14, "oversold": 30.0, "overbought": 70.0}),
        (BollingerBreakoutStrategy, {"bb_period": 20, "bb_std": 2.0}),
        (DonchianBreakoutStrategy, {"channel_period": 20}),
        (MomentumStrategy, {"fast_period": 12, "slow_period": 26}),
    ],
)
def test_input_validation(strat_cls, kwargs):
    strat = strat_cls(**kwargs)
    with pytest.raises(ValueError, match="Expected polars.DataFrame"):
        strat.generate_signals("not_a_df")  # type: ignore

    with pytest.raises(ValueError, match="Input DataFrame is empty"):
        strat.generate_signals(pl.DataFrame())


@pytest.mark.parametrize(
    "strat_cls,kwargs",
    [
        (SMACrossoverStrategy, {"fast_period": 10, "slow_period": 30}),
        (RSIMeanReversionStrategy, {"rsi_period": 14}),
        (BollingerBreakoutStrategy, {"bb_period": 20}),
        (DonchianBreakoutStrategy, {"channel_period": 20}),
        (MomentumStrategy, {}),
    ],
)
def test_warmup_guard(strat_cls, kwargs):
    strat = strat_cls(**kwargs)
    # Less data than warmup_period must return None safely
    short_df = make_synthetic_ohlcv(n_bars=strat.warmup_period - 5)
    sig = strat.generate_signals(short_df)
    assert sig is None


def test_sma_crossover_signal():
    strat = SMACrossoverStrategy(fast_period=5, slow_period=15, atr_period=5)
    assert strat.warmup_period == 15 + 5 + 2

    # Bullish scenario: sharp upward move
    df_bull = make_synthetic_ohlcv(n_bars=80, trend=25.0)
    sig = strat.generate_signals(df_bull)
    if sig is not None:
        assert sig.direction in ("long", "short")
        assert sig.entry_price > 0
        assert sig.raw_sl_distance > 0
        assert "fast_sma" in sig.metadata
        assert "slow_sma" in sig.metadata


def test_rsi_mean_reversion_validation():
    with pytest.raises(ValueError, match="must be strictly less than"):
        RSIMeanReversionStrategy(oversold=70, overbought=30)


def test_bollinger_breakout_signal():
    strat = BollingerBreakoutStrategy(bb_period=10, bb_std=1.5, atr_period=5)
    # Huge spike at end to trigger upper breakout
    df = make_synthetic_ohlcv(n_bars=60, trend=0.0)
    # Create sudden jump
    df = df.with_columns(
        pl.when(pl.int_range(0, pl.len()) == 59)
        .then(pl.col("close") + 50.0)
        .otherwise(pl.col("close"))
        .alias("close"),
        pl.when(pl.int_range(0, pl.len()) == 59)
        .then(pl.col("high") + 50.0)
        .otherwise(pl.col("high"))
        .alias("high"),
    )
    sig = strat.generate_signals(df)
    assert sig is not None
    assert sig.direction == "long"
    assert sig.raw_sl_distance > 0


def test_donchian_breakout_lookahead_free():
    strat = DonchianBreakoutStrategy(channel_period=10, atr_period=5)
    df = make_synthetic_ohlcv(n_bars=50, trend=0.0)
    # Create a breakout above previous channel
    df = df.with_columns(
        pl.when(pl.int_range(0, pl.len()) == 49)
        .then(pl.col("close") + 20.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    sig = strat.generate_signals(df)
    assert sig is not None
    assert sig.direction == "long"
    assert "channel_high" in sig.metadata


# ---------------------------------------------------------------------------
# Live-path import hygiene (2026-09-02)
# ---------------------------------------------------------------------------


def test_registry_keys_are_enumerable_without_importing_anything():
    """
    `run_oanda.py` builds its --strategy choices from STRATEGIES.keys().

    Listing the options must not load the strategies, or the live bot's import
    graph silently contains every research module in the library.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import sys
        from strategies.concrete_strategies import STRATEGIES
        names = sorted(STRATEGIES.keys())
        leaked = [m for m in sys.modules
                  if m.startswith("strategies.concrete_strategies.")
                  and not m.endswith(("ml_strategy", "base"))]
        print("NAMES=" + ",".join(names))
        print("LEAKED=" + ";".join(sorted(leaked)))
    """)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
        env={**os.environ, "PYTHONPATH": "src:."},
    )
    assert out.returncode == 0, out.stderr
    parsed = dict(
        line.split("=", 1) for line in out.stdout.splitlines() if "=" in line
    )
    names, leaked = parsed["NAMES"], parsed["LEAKED"]

    assert "momentum" in names and "regime_router" in names, names
    assert leaked == "", f"listing strategy names imported: {leaked}"


def test_selecting_one_strategy_does_not_import_the_others():
    """A typo in an unused research strategy must not be able to stop the bot."""
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import sys
        from strategies.concrete_strategies import STRATEGIES
        STRATEGIES["momentum"]()
        loaded = sorted(m.rsplit(".", 1)[-1] for m in sys.modules
                        if m.startswith("strategies.concrete_strategies."))
        print(";".join(loaded))
    """)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
        env={**os.environ, "PYTHONPATH": "src:."},
    )
    assert out.returncode == 0, out.stderr
    loaded = out.stdout.strip().split(";")

    assert "momentum" in loaded
    for other in ("donchian_breakout", "bollinger_breakout",
                  "rsi_mean_reversion", "sma_crossover", "regime_router"):
        assert other not in loaded, f"selecting momentum also imported {other}"


def test_build_strategy_round_trips_every_registered_name():
    from strategies.concrete_strategies import build_strategy

    for name in sorted(STRATEGIES.keys()):
        if name == "ml_strategy":
            continue  # needs model artifacts on disk
        strat = build_strategy(name)
        assert isinstance(strat, BaseStrategy)
        assert strat.warmup_period > 0
