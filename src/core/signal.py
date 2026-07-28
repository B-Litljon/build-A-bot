"""
The Signal shape used by the Alpaca (equities / crypto) execution path.

IMPORTANT -- there are two different Signal classes in this repo and they are
not interchangeable:
  * ``core.signal.Signal`` (this one) is what ``LiveOrchestrator`` emits and
    what ``NotificationManager.send_trade_alert`` formats. Bracket levels ride
    inside ``metadata``.
  * ``strategies.base.Signal`` is what the OANDA / forex path uses, and it
    carries explicit ``raw_sl_distance`` / ``raw_tp_distance`` fields.
See GLOSSARY.md ("Signal -- two of them") before assuming which one you have.

Glossary:
    SignalType -- BUY / SELL / HOLD. Only BUY and SELL reach execution; HOLD
        means the strategy evaluated the bar and declined to trade.
    Signal.symbol -- broker ticker the signal refers to (e.g. "BTC/USD").
    Signal.type -- the SignalType above.
    Signal.price -- the price the decision was made at (the sealed bar's
        close). Not the fill price, which the broker decides later.
    Signal.confidence -- 0.0-1.0 single summary score. The two-stage model path
        does not really use it; it carries angel_prob / devil_prob in metadata
        instead and leaves this as a display fallback.
    Signal.timestamp -- the sealed bar's timestamp, i.e. when the decision was
        made rather than when the order was sent.
    Signal.metadata -- free-form dict. In practice the live path stores
        angel_prob, devil_prob, sl_price, tp_price and expected_pct_growth
        here, and NotificationManager reads exactly those keys.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    symbol: str
    type: SignalType
    price: float
    confidence: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
