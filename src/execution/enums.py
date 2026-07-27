"""Broker-agnostic order enums for the Build-A-Bot Factory SDK.

Adapters in ``src/execution/<broker>_adapter.py`` are responsible for
translating these into broker-specific types. Do not import broker SDK
types into this file under any circumstance.

Glossary:
    OrderSide -- BUY or SELL.
    OrderType -- MARKET (take whatever price is available now), LIMIT (only at
        my price or better), STOP (become a market order once price reaches a
        level), STOP_LIMIT (both). The live paths use MARKET only.
    TimeInForce -- how long an unfilled order stays alive: DAY (until the
        session ends), GTC (good till cancelled), IOC (fill what you can right
        now, cancel the rest), FOK (all of it immediately or none).
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
