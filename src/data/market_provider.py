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
"""

import abc
from datetime import datetime
from typing import Callable, List

import polars as pl


class MarketDataProvider(abc.ABC):
    """Unified historical + streaming + discovery contract."""

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
