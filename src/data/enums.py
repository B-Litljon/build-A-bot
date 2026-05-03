"""Broker-agnostic data-layer enums for the Build-A-Bot Factory SDK.

Adapters in ``src/data/<broker>_provider.py`` are responsible for
translating these into broker-specific types. Do not import broker SDK
types into this file under any circumstance.
"""

from enum import Enum


class AssetClass(str, Enum):
    """Asset class for discovery and trading filters."""

    US_EQUITY = "us_equity"
    CRYPTO = "crypto"


class AssetStatus(str, Enum):
    """Tradability status of a listed asset."""

    ACTIVE = "active"
    INACTIVE = "inactive"


class DataFeed(str, Enum):
    """Market-data feed routing for historical/realtime queries."""

    IEX = "iex"
    SIP = "sip"
