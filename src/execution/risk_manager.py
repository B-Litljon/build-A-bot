"""Centralized execution safety and risk management."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class TimeInForce(Enum):
    """Time-in-force order options."""

    GTC = "GTC"  # Good Till Canceled
    DAY = "DAY"  # Day order


class AssetClass(Enum):
    """Asset class classifications."""

    CRYPTO = "crypto"
    EQUITY = "equity"


@dataclass
class RiskValidatedSignal:
    """Signal after risk management processing."""

    direction: str
    entry_price: float
    actual_sl_distance: float
    actual_tp_distance: float
    time_in_force: TimeInForce
    is_valid: bool
    rejection_reason: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class RiskManager:
    """
    Centralizes proprietary execution safety nets.

    Enforces minimum SL floors, slippage guards, and dynamic TIF routing
    based on asset class.
    """

    MIN_SL_PCT = 0.0015  # 0.15% minimum stop-loss percentage
    DEFAULT_ATR_MULTIPLIER = 0.5

    def __init__(self, atr_multiplier: float = DEFAULT_ATR_MULTIPLIER) -> None:
        """
        Initialize risk manager with dynamic risk parameters.

        Args:
            atr_multiplier: Risk multiplier for ATR-based calculations (default: 0.5)
        """
        self.atr_multiplier = atr_multiplier

    def validate_signal(
        self,
        direction: str,
        entry_price: float,
        raw_sl_distance: float,
        raw_tp_distance: float,
        asset_class: AssetClass,
        **kwargs: Any,
    ) -> RiskValidatedSignal:
        """
        Validate and process signal through all safety nets.

        Args:
            direction: 'long' or 'short'
            entry_price: Expected entry price
            raw_sl_distance: Raw stop-loss distance from strategy
            raw_tp_distance: Raw take-profit distance from strategy
            asset_class: Asset classification for TIF routing
            **kwargs: Additional validation parameters

        Returns:
            RiskValidatedSignal with processed values and validation status
        """
        # Calculate actual SL with floor enforcement
        min_sl_distance = entry_price * self.MIN_SL_PCT
        actual_sl_distance = max(raw_sl_distance, min_sl_distance)

        # Apply ATR multiplier to distances
        actual_sl_distance *= self.atr_multiplier
        actual_tp_distance = raw_tp_distance * self.atr_multiplier

        # Calculate absolute price levels
        if direction == "long":
            expected_sl = entry_price - actual_sl_distance
            expected_tp = entry_price + actual_tp_distance
        elif direction == "short":
            expected_sl = entry_price + actual_sl_distance
            expected_tp = entry_price - actual_tp_distance
        else:
            return RiskValidatedSignal(
                direction=direction,
                entry_price=entry_price,
                actual_sl_distance=actual_sl_distance,
                actual_tp_distance=actual_tp_distance,
                time_in_force=self._route_tif(asset_class),
                is_valid=False,
                rejection_reason=f"Invalid direction: {direction}",
            )

        # Literal slippage guard - check for mathematical inversion
        rejection_reason = None
        if direction == "long":
            if expected_tp <= entry_price:
                rejection_reason = (
                    f"Long TP inversion: TP {expected_tp} <= entry {entry_price}"
                )
            elif expected_sl >= entry_price:
                rejection_reason = (
                    f"Long SL inversion: SL {expected_sl} >= entry {entry_price}"
                )
        elif direction == "short":
            if expected_tp >= entry_price:
                rejection_reason = (
                    f"Short TP inversion: TP {expected_tp} >= entry {entry_price}"
                )
            elif expected_sl <= entry_price:
                rejection_reason = (
                    f"Short SL inversion: SL {expected_sl} <= entry {entry_price}"
                )

        # Dynamic TIF routing
        tif = self._route_tif(asset_class)

        return RiskValidatedSignal(
            direction=direction,
            entry_price=entry_price,
            actual_sl_distance=actual_sl_distance,
            actual_tp_distance=actual_tp_distance,
            time_in_force=tif,
            is_valid=rejection_reason is None,
            rejection_reason=rejection_reason,
            metadata=kwargs,
        )

    def _route_tif(self, asset_class: AssetClass) -> TimeInForce:
        """
        Route to appropriate time-in-force based on asset class.

        Args:
            asset_class: Asset classification

        Returns:
            Appropriate TimeInForce for the asset class
        """
        if asset_class == AssetClass.CRYPTO:
            return TimeInForce.GTC
        elif asset_class == AssetClass.EQUITY:
            return TimeInForce.DAY
        else:
            return TimeInForce.GTC
