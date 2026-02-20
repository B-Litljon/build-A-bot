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
