"""
Tests for strategies.concrete_strategies.regime_router -- regime-routed dispatching and stand-down rules.
"""

import json
from pathlib import Path
import numpy as np
import polars as pl
import pytest

from ml.regimes.behavior_tagger import LABEL_COLD
from strategies.base import BaseStrategy, Signal
from strategies.concrete_strategies.regime_router import RegimeRouterStrategy


class MockSignalStrategy(BaseStrategy):
    """Mock sub-strategy that always emits a known signal."""

    def __init__(self, direction: str = "long", **kwargs):
        super().__init__(**kwargs)
        self.direction = direction
        self.warmup_period = 10

    def generate_signals(self, df: pl.DataFrame) -> Signal:
        return Signal(
            direction=self.direction,
            entry_price=100.0,
            raw_sl_distance=1.5,
            raw_tp_distance=1.5,
            metadata={"mock": True},
        )


def make_large_ohlcv(n_bars: int = 350) -> pl.DataFrame:
    rng = np.random.default_rng(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n_bars))
    high = close + 0.5
    low = close - 0.5
    return pl.DataFrame({
        "timestamp": [f"2026-01-01T{i:04d}:00Z" for i in range(n_bars)],
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": [1000] * n_bars,
        "symbol": ["EUR_USD"] * n_bars,
    })


def test_regime_router_cold_window_stands_down():
    # Warmup guard or cold window must decline to trade
    router = RegimeRouterStrategy(regime_window=260, min_samples=60)
    short_df = make_large_ohlcv(n_bars=50)
    sig = router.generate_signals(short_df)
    assert sig is None


def test_regime_router_explicit_stand_down():
    # Routing table with all regimes mapped to None (stand down)
    routing_table = {
        "trend_high": None,
        "trend_normal": None,
        "trend_low": None,
        "range_high": None,
        "range_normal": None,
        "range_low": None,
        "mixed_high": None,
        "mixed_normal": None,
        "mixed_low": None,
    }
    router = RegimeRouterStrategy(
        routing_table=routing_table,
        regime_window=50,
        min_samples=20,
    )
    df = make_large_ohlcv(n_bars=150)
    sig = router.generate_signals(df)
    # Must decline even though bars are plenty
    assert sig is None


def test_regime_router_delegates_to_sub_strategy():
    mock_strat = MockSignalStrategy(direction="short")
    routing_table = {
        "trend_high": "mock_strat",
        "trend_normal": "mock_strat",
        "trend_low": "mock_strat",
        "range_high": "mock_strat",
        "range_normal": "mock_strat",
        "range_low": "mock_strat",
        "mixed_high": "mock_strat",
        "mixed_normal": "mock_strat",
        "mixed_low": "mock_strat",
    }
    router = RegimeRouterStrategy(
        routing_table=routing_table,
        sub_strategies={"mock_strat": mock_strat},
        regime_window=50,
        min_samples=20,
    )
    df = make_large_ohlcv(n_bars=150)
    sig = router.generate_signals(df)

    assert sig is not None
    assert sig.direction == "short"
    assert sig.entry_price == 100.0
    assert "regime_label" in sig.metadata
    assert sig.metadata["routed_strategy"] == "mock_strat"


def test_regime_router_loads_config_file(tmp_path: Path):
    cfg_file = tmp_path / "test_routing.json"
    cfg = {"range_normal": "sma_crossover", "trend_high": None}
    with open(cfg_file, "w") as f:
        json.dump(cfg, f)

    router = RegimeRouterStrategy(routing_config_path=cfg_file)
    assert router.routing_table["range_normal"] == "sma_crossover"
    assert router.routing_table["trend_high"] is None


# ---------------------------------------------------------------------------
# Registry hygiene (2026-09-02)
# ---------------------------------------------------------------------------


def test_sub_strategies_registered_once_under_a_canonical_name():
    """
    One strategy, one key.

    The library used to be registered twice -- once as snake_case, once as
    ClassName -- so a routing table could name the same strategy two ways and a
    rename would half-break in silence.
    """
    router = RegimeRouterStrategy()
    keys = list(router.sub_strategies)

    assert len(keys) == len(set(keys)), "duplicate keys in the sub-strategy registry"
    assert all(k == k.lower() for k in keys), f"non-canonical keys present: {keys}"
    # Every registered instance is distinct -- no key aliases the same object.
    assert len({id(v) for v in router.sub_strategies.values()}) == len(keys)


def test_classname_spelling_still_resolves_to_the_canonical_key():
    """Older tables spelled routes as ClassName; they must keep working."""
    router = RegimeRouterStrategy()
    assert router._resolve_name("momentum") == "momentum"
    assert router._resolve_name("MomentumStrategy") == "momentum"
    assert router._resolve_name("NoSuchStrategy") is None


def test_documentation_keys_are_not_treated_as_routes(tmp_path: Path):
    """
    The shipped templates carry a "_WARNING" key. It is documentation, not a
    regime, and must never be mistaken for one.
    """
    cfg = tmp_path / "routing.json"
    cfg.write_text(json.dumps({
        "_WARNING": "template only, not measured",
        "_source": "hand-authored",
        "trend_high": None,
        "range_normal": "momentum",
    }))

    router = RegimeRouterStrategy(routing_config_path=cfg)

    assert "_WARNING" not in router.routing_table
    assert "_source" not in router.routing_table
    assert router.routing_table == {"trend_high": None, "range_normal": "momentum"}
