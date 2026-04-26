"""Abstract base class for all trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional
import polars as pl


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
