"""
Abstract base class for unified market data providers.

A `MarketDataProvider` is responsible for the full data lifecycle of one
broker / data vendor: symbol discovery, historical REST queries, and
real-time streaming. This unified shape (vs. split historical/streaming
ABCs) reflects that credentials, client lifecycles, and rate limits
are typically shared across both modes for a given vendor.

Concrete implementations:
  - AlpacaProvider      (src/data/alpaca_provider.py)
  - PolygonDataProvider (src/data/polygon_provider.py)
  - YahooDataProvider   (src/data/yahoo_provider.py)

Adapters for new vendors should subclass MarketDataProvider and
implement all four abstract methods.

This file must not import any vendor SDKs.

Glossary:
    MarketDataProvider -- the contract every vendor adapter implements.
        Discovery, historical bars and live streaming live in one class
        because a vendor's credentials, client lifecycle and rate limits are
        shared across all three.
    _BAR_SCHEMA -- the canonical six-column bar shape every provider must
        return: timestamp (microsecond UTC), open/high/low/close/volume as
        Float64. Everything downstream assumes exactly this.
    _empty_bars() -- a correctly-typed empty DataFrame. Providers return this
        on failure rather than raising, so one bad symbol cannot abort a fetch.
    get_active_symbols -- up to *limit* currently-tradable tickers, ranked by
        activity where the vendor supports it.
    get_historical_bars -- OHLCV for one symbol over a datetime range. Must
        never raise; returns empty on failure or no data.
    subscribe -- register a callback for live bars. Non-blocking; it only sets
        up the vendor's stream client.
    run_stream -- the blocking loop that actually delivers bars. Always called
        after subscribe().
    timeframe_minutes -- bar size in minutes. Daily and weekly bars are
        deliberately out of scope for this contract.
"""

import abc
from datetime import datetime
from typing import Callable, List

import polars as pl


class MarketDataProvider(abc.ABC):
    """Unified historical + streaming + discovery contract."""

    _BAR_SCHEMA = {
        "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
        "open": pl.Float64,
        "high": pl.Float64,
        "low": pl.Float64,
        "close": pl.Float64,
        "volume": pl.Float64,
    }

    @classmethod
    def _empty_bars(cls) -> pl.DataFrame:
        return pl.DataFrame({col: [] for col in cls._BAR_SCHEMA}, schema=cls._BAR_SCHEMA)

    @abc.abstractmethod
    def get_active_symbols(self, limit: int = 10) -> List[str]:
        """
        Return up to *limit* currently-tradable symbols for this vendor.

        Implementations should prefer volume-ranked or activity-ranked
        results where the vendor supports it, otherwise return any
        deterministic subset (and document the caveat).
        """
        ...

    @abc.abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch historical OHLCV bars for a single symbol.

        Parameters
        ----------
        symbol:
            Ticker (e.g. "AAPL", "BTC/USD"). Crypto pairs use a slash.
        timeframe_minutes:
            Bar granularity in minutes. Daily/weekly bars are out of
            scope for this contract.
        start, end:
            Timezone-aware datetimes (UTC strongly preferred).

        Returns
        -------
        polars.DataFrame
            Columns at minimum: timestamp, open, high, low, close, volume.
            Returns an empty DataFrame on failure or no data — never raises.
            The `timestamp` column should be timezone-aware (UTC).
        """
        ...

    @abc.abstractmethod
    def subscribe(self, symbols: List[str], callback: Callable) -> None:
        """
        Register *callback* for real-time bar updates.

        This method is non-blocking: it only registers the callback and
        prepares vendor-specific stream clients. Call run_stream() to
        actually start receiving bars.

        The callback will receive a dict with keys:
        symbol, timestamp, open, high, low, close, volume.

        Implementations may be sync (using internal asyncio bridges) or
        async; the public signature is sync to keep the SDK consumer
        contract simple.
        """
        ...

    @abc.abstractmethod
    def run_stream(self) -> None:
        """
        Start the blocking stream event loop.

        Must be called after subscribe(). Runs until the process is
        interrupted (Ctrl+C) or the underlying stream errors out.
        """
        ...
