import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class RiskProfile:
    sl_atr_multiplier: float = 0.5
    tp_atr_multiplier: float = 3.0
    min_sl_pct: float = 0.0015  # 0.15% absolute floor
    risk_per_trade: float = 0.02 # 2% of account
    max_notional_cap: float = 100000.0

class RiskManager:
    """
    The Shield: Enforces institutional-grade safety nets and dynamic sizing.
    """
    def __init__(self, profile: RiskProfile = RiskProfile()):
        self.profile = profile

    def calculate_bracket(self, entry_price: float, atr: float) -> tuple[float, float]:
        """
        Calculates SL and TP with an absolute floor for SL distance.
        """
        # Dynamic 0.5x ATR sizing (multiplier from profile)
        raw_sl_dist = atr * self.profile.sl_atr_multiplier

        # 0.15% absolute Stop Loss floor
        min_sl_dist = entry_price * self.profile.min_sl_pct

        actual_sl_dist = max(raw_sl_dist, min_sl_dist)

        sl_price = round(entry_price - actual_sl_dist, 4)
        tp_price = round(entry_price + (atr * self.profile.tp_atr_multiplier), 4)

        return sl_price, tp_price

    def calculate_quantity(self, equity: float, buying_power: float, entry_price: float, sl_price: float) -> float:
        """
        Calculates fractional position size based on risk-per-trade.
        """
        risk_dollars = equity * self.profile.risk_per_trade
        risk_per_share = entry_price - sl_price

        if risk_per_share <= 0:
            return 0.0

        risk_qty = risk_dollars / risk_per_share
        notional_qty = self.profile.max_notional_cap / entry_price
        bp_qty = (buying_power * 0.95) / entry_price

        final_qty = min(risk_qty, notional_qty, bp_qty)

        if final_qty < risk_qty:
            logger.warning(
                f"Quantity scaled down from {risk_qty:.4f} to {final_qty:.4f} to meet notional/bp limits."
            )

        return max(round(final_qty, 4), 0.0001)
