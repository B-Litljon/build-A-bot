import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class RiskProfile:
    sl_atr_multiplier: float = 0.5
    tp_atr_multiplier: float = 3.0
    min_sl_pct: float = 0.0015  # 0.15% absolute floor
    min_sl_pips: float = 2.0     # Default Forex pip floor (2.0 pips)
    risk_per_trade: float = 0.02 # 2% of account
    max_notional_cap: float = 100000.0
    round_precision: int = 4

    @classmethod
    def for_asset_class(cls, asset_class: str) -> "RiskProfile":
        if asset_class == "forex":
            return cls(
                sl_atr_multiplier=1.0,
                tp_atr_multiplier=2.0,
                min_sl_pips=2.0,
                round_precision=5,
            )
        return cls()

class RiskManager:
    """
    The Shield: Enforces institutional-grade safety nets and dynamic sizing.
    """
    def __init__(self, profile: RiskProfile = RiskProfile()):
        self.profile = profile

    def calculate_bracket(
        self, entry_price: float, raw_atr: float, symbol: Optional[str] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Applies multipliers and A3 chop filter to raw ATR volatility.

        Returns (sl_distance, tp_distance) or None if A3 filter rejects the trade
        (volatility too low — 0.5x ATR stop would sit below the floor).
        """
        sl_dist = raw_atr * self.profile.sl_atr_multiplier
        tp_dist = raw_atr * self.profile.tp_atr_multiplier

        # Determine the stop-loss floor
        if symbol and self._is_forex_symbol(symbol):
            pip_size = self._get_forex_pip_size(symbol)
            floor = self.profile.min_sl_pips * pip_size
        else:
            floor = entry_price * self.profile.min_sl_pct

        # A3 chop filter: if the stop floor would override the ATR stop, skip entirely
        if sl_dist < floor:
            return None

        return round(sl_dist, self.profile.round_precision), round(tp_dist, self.profile.round_precision)

    def _is_forex_symbol(self, symbol: str) -> bool:
        clean = symbol.replace("_", "").replace("/", "").upper()
        return len(clean) == 6 and clean.isalpha()

    def _get_forex_pip_size(self, symbol: str) -> float:
        clean = symbol.replace("_", "").replace("/", "").upper()
        quote = clean[-3:]
        if quote == "JPY":
            return 0.01
        return 0.0001

    def calculate_quantity(
        self,
        equity: float,
        buying_power: float,
        entry_price: float,
        sl_price: float,
        cash: float = 0.0,
        is_crypto: bool = False,
    ) -> float:
        """
        Calculates fractional position size based on risk-per-trade.

        For crypto, uses cash * 0.95 as the buying-power cap (Alpaca reports
        crypto available funds in the cash field, not buying_power).
        Returns 0.0 if the resulting notional is below the $50 zombie-trade floor.
        """
        # 05192026: shouldn't apply to forex trades
        risk_dollars = equity * self.profile.risk_per_trade
        risk_per_share = entry_price - sl_price

        if risk_per_share <= 0:
            return 0.0

        risk_qty = risk_dollars / risk_per_share
        notional_qty = self.profile.max_notional_cap / entry_price
        bp_source = cash if is_crypto else buying_power
        bp_qty = (bp_source * 0.95) / entry_price

        final_qty = min(risk_qty, notional_qty, bp_qty)

        if final_qty < risk_qty:
            logger.warning(
                f"Quantity scaled down from {risk_qty:.4f} to {final_qty:.4f} to meet notional/bp limits."
            )

        # $50 minimum notional — prevents zombie fractional-share trades
        if final_qty * entry_price < 50.0:
            return 0.0

        return max(round(final_qty, 4), 0.0001)
