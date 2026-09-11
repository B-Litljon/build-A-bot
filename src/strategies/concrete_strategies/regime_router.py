"""
Regime Router strategy -- dynamically routes market data to regime-specialized algorithms.

Glossary:
    routing_table -- maps BehaviorTag.label ('trend_high', 'range_low', ...) to a
        strategy name, or None to explicitly stand down ('trade nothing'). Keys
        beginning with "_" are documentation and are dropped on load.
    sub_strategies -- name -> BaseStrategy, keyed by ONE canonical snake_case
        name each. Not aliased; see _resolve_name.
    _resolve_name -- maps a routing-table value to a canonical key, accepting a
        ClassName spelling for backward compatibility. Returns None for unknown
        names so a typo stands down instead of trading something arbitrary.
    stand down -- returning None: no trade, deliberately. Three causes, all
        intended -- cold window, a None route, or an unresolvable strategy name.
    default_strategy -- what an unlisted behavior label falls back to. Leave it
        None so an unmeasured regime declines rather than guesses.
    causal regime tagging -- uses ml.regimes.behavior_tagger.tag_bar over
        trailing windows, strictly maintaining the live symmetry contract.
    warmup_period -- max(regime_window + 14, every sub-strategy's warmup), so
        the router never delegates to a strategy that is not itself warm.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence
import numpy as np
import polars as pl
import talib

from strategies.base import BaseStrategy, Signal
from ml.regimes.behavior_tagger import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_WINDOW,
    LABEL_COLD,
    BehaviorTag,
    tag_bar,
    trend_strength_from_ppo,
)

logger = logging.getLogger(__name__)


class RegimeRouterStrategy(BaseStrategy):
    """
    Master regime router strategy.
    
    Inspects trailing market behavior (volatility & trend strength ranks),
    resolves the corresponding strategy from a validated routing table,
    and delegates signal generation -- or stands down (returns None) if
    the regime provides no statistical edge.
    """

    def __init__(
        self,
        routing_table: Optional[Dict[str, Optional[str]]] = None,
        routing_config_path: Optional[str | Path] = None,
        sub_strategies: Optional[Dict[str, BaseStrategy]] = None,
        regime_window: int = DEFAULT_WINDOW,
        min_samples: int = DEFAULT_MIN_SAMPLES,
        default_strategy: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            regime_window=regime_window,
            min_samples=min_samples,
            default_strategy=default_strategy,
            **kwargs,
        )
        self.regime_window = regime_window
        self.min_samples = min_samples
        self.default_strategy = default_strategy

        # Load routing table from file if provided
        self.routing_table: Dict[str, Optional[str]] = {}
        if routing_config_path is not None:
            cfg_path = Path(routing_config_path)
            if cfg_path.exists():
                with open(cfg_path) as f:
                    raw = json.load(f)
                # Keys beginning with "_" are documentation (e.g. "_WARNING" on
                # the shipped templates), not regime labels. Drop them so they
                # can never be mistaken for a route.
                self.routing_table = {
                    k: v for k, v in raw.items() if not k.startswith("_")
                }
                routed = sorted(k for k, v in self.routing_table.items() if v is not None)
                logger.info(
                    "RegimeRouter loaded routing table from %s -- %d/%d regimes routed (%s), "
                    "rest stand down",
                    cfg_path,
                    len(routed),
                    len(self.routing_table),
                    ", ".join(routed) or "none",
                )
                if ".example." in cfg_path.name:
                    logger.warning(
                        "Routing table %s is a TEMPLATE with no empirical basis. "
                        "Its assignments are textbook priors, not measurements. "
                        "Regenerate with build_strategy_matrix.py before trusting it.",
                        cfg_path.name,
                    )
            else:
                logger.warning("Routing config file %s not found", cfg_path)

        # In-memory routing_table overrides or supplements config
        if routing_table is not None:
            self.routing_table.update(routing_table)

        # Initialize sub-strategies if not supplied
        if sub_strategies is not None:
            self.sub_strategies = dict(sub_strategies)
        else:
            self.sub_strategies = self._default_strategy_library()

        # Warmup period needs to cover regime_window and the sub-strategies
        sub_warmups = [s.warmup_period for s in self.sub_strategies.values() if hasattr(s, "warmup_period")]
        self.warmup_period = max([self.regime_window + 14] + sub_warmups)

    def _default_strategy_library(self) -> Dict[str, BaseStrategy]:
        """
        Instantiate the standard suite of non-ML sub-strategies.

        Keyed by snake_case ONLY -- one name per strategy. An earlier version
        registered every strategy twice, under both its snake_case key and its
        ClassName, so a routing table could name the same strategy two ways and
        a rename would half-break silently. ClassName spellings are still
        accepted, but they are RESOLVED to the canonical key by
        ``_resolve_name`` rather than being a second registry entry.
        """
        from strategies.concrete_strategies.sma_crossover import SMACrossoverStrategy
        from strategies.concrete_strategies.rsi_mean_reversion import RSIMeanReversionStrategy
        from strategies.concrete_strategies.bollinger_breakout import BollingerBreakoutStrategy
        from strategies.concrete_strategies.donchian_breakout import DonchianBreakoutStrategy
        from strategies.concrete_strategies.momentum import MomentumStrategy

        return {
            "sma_crossover": SMACrossoverStrategy(),
            "rsi_mean_reversion": RSIMeanReversionStrategy(),
            "bollinger_breakout": BollingerBreakoutStrategy(),
            "donchian_breakout": DonchianBreakoutStrategy(),
            "momentum": MomentumStrategy(),
        }

    def _resolve_name(self, name: str) -> Optional[str]:
        """
        Map a routing-table value to a canonical sub-strategy key.

        Accepts the canonical snake_case key, or a ClassName spelling for
        backward compatibility with tables written before the registry was
        de-duplicated. Returns None when the name matches nothing.
        """
        if name in self.sub_strategies:
            return name
        for key, strat in self.sub_strategies.items():
            if strat.__class__.__name__ == name:
                logger.debug(
                    "Routing table used ClassName '%s'; canonical key is '%s'",
                    name,
                    key,
                )
                return key
        return None

    def determine_regime(self, df: pl.DataFrame) -> BehaviorTag:
        """Calculate causal BehaviorTag for the latest bar in df."""
        close = df["close"].to_numpy().astype(float)
        high = df["high"].to_numpy().astype(float)
        low = df["low"].to_numpy().astype(float)

        natr = talib.NATR(high, low, close, timeperiod=14)
        ppo = talib.PPO(close, fastperiod=12, slowperiod=26, matype=talib.MA_Type.SMA)
        trend = trend_strength_from_ppo(ppo)

        # Slice trailing window ending at the latest bar
        natr_window = natr[-self.regime_window :]
        trend_window = trend[-self.regime_window :]

        return tag_bar(
            natr_window,
            trend_window,
            min_samples=self.min_samples,
        )

    def generate_signals(self, df: pl.DataFrame) -> Optional[Signal]:
        self.validate_input(df)

        if len(df) < self.warmup_period:
            return None

        # 1. Determine causal regime for the latest bar
        tag = self.determine_regime(df)

        symbol = str(df["symbol"].tail(1)[0]) if "symbol" in df.columns else "unknown"

        # 2. Cold window guard: absence of evidence -> stand down
        if not tag.warm or tag.label == LABEL_COLD:
            logger.debug("[%s] Regime cold (insufficient samples) -> standing down", symbol)
            return None

        # 3. Route to target strategy
        target_name = self.routing_table.get(tag.label, self.default_strategy)

        # 4. Explicit stand-down (trade nothing)
        if target_name is None:
            logger.debug("[%s] Regime '%s' routed to None -> standing down", symbol, tag.label)
            return None

        resolved = self._resolve_name(target_name)
        if resolved is None:
            logger.warning(
                "[%s] Target strategy '%s' for regime '%s' not found in sub_strategies "
                "(known: %s) -- standing down",
                symbol,
                target_name,
                tag.label,
                sorted(self.sub_strategies),
            )
            return None

        # 5. Delegate signal generation
        target_strat = self.sub_strategies[resolved]
        signal = target_strat.generate_signals(df)

        if signal is None:
            return None

        # 6. Enrich signal metadata with regime diagnostics
        if signal.metadata is None:
            signal.metadata = {}
        signal.metadata.update({
            "regime_label": tag.label,
            "regime_vol_band": tag.vol_band,
            "regime_trend_state": tag.trend_state,
            "regime_vol_rank": tag.vol_rank,
            "regime_trend_rank": tag.trend_rank,
            "routed_strategy": resolved,
        })

        return signal
