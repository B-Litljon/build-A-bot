"""
Abstract base class defining the contract for market data providers.

Any data source (Alpaca, Polygon, IBKR, CSV replay, etc.) must implement
this interface so the TradingBot remains decoupled from vendor specifics.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, List

import polars as pl


class MarketDataProvider(ABC):
    """
    Contract for all market-data adapters.

    Implementations must supply:
        - symbol discovery  (get_active_symbols)
        - historical OHLCV  (get_historical_bars)
        - real-time stream   (subscribe + run_stream)
    """

    @abstractmethod
    def get_active_symbols(self, limit: int = 10) -> List[str]:
        """
        Return the top *limit* most-active ticker symbols.

        The definition of "most active" is provider-specific (e.g. volume,
        trade count).  The returned list is ordered by activity descending.
        """
        ...

    @abstractmethod
    def get_historical_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch historical OHLCV bars and return a Polars DataFrame.

        Required schema:
            timestamp : pl.Datetime(time_unit="us", time_zone="UTC")
            open      : pl.Float64
            high      : pl.Float64
            low       : pl.Float64
            close     : pl.Float64
            volume    : pl.Float64
        """
        ...

    @abstractmethod
    def subscribe(self, symbols: List[str], callback: Callable) -> None:
        """
        Register *callback* for real-time bar updates on *symbols*.

        This method performs **non-blocking** setup only.  The callback
        will receive a dict with keys:
            symbol, timestamp, open, high, low, close, volume

        The actual event loop is started by :meth:`run_stream`.
        """
        ...

    @abstractmethod
    def run_stream(self) -> None:
        """
        Start the **blocking** data-stream event loop.

        This should be the last call in ``main()``; it runs until the
        process is interrupted.
        """
        ...
