"""Broker-agnostic data-layer enums for the Build-A-Bot Factory SDK.

Adapters in ``src/data/<broker>_provider.py`` are responsible for
translating these into broker-specific types. Do not import broker SDK
types into this file under any circumstance.

Glossary:
    AssetClass -- what kind of instrument: US_EQUITY or CRYPTO. Used to filter
        discovery queries. Note forex is absent -- the OANDA path predates and
        bypasses these enums.
    AssetStatus -- ACTIVE or INACTIVE, i.e. whether the venue still lists the
        asset for trading.
    DataFeed -- which market-data feed to route a query through. IEX is the
        free tier and sees only a fraction of national volume, which is why
        volume-derived numbers carry accuracy warnings throughout the repo;
        SIP is the paid consolidated feed covering all exchanges.
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
