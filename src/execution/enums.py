"""Broker-agnostic order enums for the Build-A-Bot Factory SDK.

Adapters in ``src/execution/<broker>_adapter.py`` are responsible for
translating these into broker-specific types. Do not import broker SDK
types into this file under any circumstance.
"""

from enum import Enum


class OrderSide(str, Enum):
    """Direction of an order."""

    BUY = "buy"
    SELL = "sell"


class TimeInForce(str, Enum):
    """Duration for which an order remains active."""

    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"


class OrderType(str, Enum):
    """Type of order to submit."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
