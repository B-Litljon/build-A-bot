"""
Abstract base class for all trading strategies, and the Signal they emit.

This is now the ONLY Signal class: the Alpaca path's `core.signal.Signal`
(bracket levels in a metadata dict) was deleted 2026-09-16 along with that
whole dormant lane. Direction here is a plain string, not an enum.

Glossary:
    Signal -- what a strategy returns when it wants to trade. Returning None
        instead means "no trade", which is the overwhelmingly common case.
    direction -- "long" (profit if price rises) or "short" (profit if it
        falls). A plain string, not an enum, unlike the Alpaca path's
        SignalType.
    entry_price -- the price the decision was made at, i.e. the latest sealed
        bar's close. Not the fill price.
    raw_sl_distance -- how far from entry the stop belongs, as a PRICE
        DISTANCE, not a percentage or a level. The RiskManager turns it into an
        actual stop level and applies its own multipliers.
    raw_tp_distance -- ⚠️ WRITTEN BUT NEVER READ. MLStrategy sets it to the
        same value as raw_sl_distance, and no execution path consumes it: the
        RiskManager owns target sizing via its own multipliers, deliberately,
        so that live brackets always match the ones the model was trained
        against. Treat this field as vestigial.
    metadata -- free-form dict; the ML strategy puts angel_prob, devil_prob and
        diagnostics here for logging and Discord alerts. One key is a real
        contract rather than a diagnostic — see BARRIER_GEOMETRY_KEY.
    BARRIER_GEOMETRY_KEY -- the metadata key carrying LEARNED bracket geometry
        ("barrier_geometry"): the per-bar NATR-multiple stop/target pair from
        ml.barriers.BarrierEstimator, which RiskManager substitutes for its
        static profile multipliers. Defined here, beside the Signal it travels
        on, because execution and analysis both read it and neither may import
        the strategy implementation to get it.
    BaseStrategy -- the contract: implement generate_signals(df) -> Signal|None.
    params -- whatever kwargs the strategy was constructed with, kept for
        logging and reproducibility.
    name -- the concrete class's own name, used in logs.
    validate_input -- shared sanity check (is it a DataFrame, is it non-empty).
        Subclasses call it before doing real work.

NOTE (glossary pass, 2026-07-27): generate_signals' docstring below says
"standard 18-feature microstructure input". That count is STALE -- the live
feature set is 22 columns, or 23 with cost_ratio. MLStrategy no longer hardcodes
a count at all; it reads the schema off the trained model
(``feature_names_in_``). Flagged, not changed.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import polars as pl

# Signal.metadata key carrying learned bracket geometry. Lives here, next to the
# Signal that carries it, because three layers need the same string and only one
# of them may own it: the producer (strategies.concrete_strategies.ml_strategy),
# the consumer (execution.risk_manager, which must stay import-light) and the
# offline replay (analysis.strategy_backtester, which keeps execution out of
# module scope). An import of strategies.base is the cheapest of those edges and
# pulls nothing but polars.
BARRIER_GEOMETRY_KEY = "barrier_geometry"


@dataclass
class Signal:
    """Normalized signal output container."""

    direction: str  # 'long' or 'short'
    entry_price: float
    raw_sl_distance: float
    raw_tp_distance: float
    metadata: Optional[Dict[str, Any]] = None


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must inherit from this class and implement
    the generate_signals method to produce normalized signal outputs.
    """

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize strategy with custom parameters.

        Args:
            **kwargs: Strategy-specific configuration parameters
        """
        self.params = kwargs
        self.name = self.__class__.__name__

    @abstractmethod
    def generate_signals(self, df: pl.DataFrame) -> Signal:
        """
        Generate trading signals from microstructure data.

        Args:
            df: Polars DataFrame containing standard 18-feature microstructure input

        Returns:
            Signal object containing direction, entry_price, raw_sl_distance,
            and raw_tp_distance

        Raises:
            ValueError: If input DataFrame is invalid or missing required features
        """
        pass

    def validate_input(self, df: pl.DataFrame) -> None:
        """
        Validate that input DataFrame meets requirements.

        Args:
            df: Input DataFrame to validate

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(df, pl.DataFrame):
            raise ValueError(f"Expected polars.DataFrame, got {type(df)}")

        if df.is_empty():
            raise ValueError("Input DataFrame is empty")

    def __repr__(self) -> str:
        return f"{self.name}(params={self.params})"
