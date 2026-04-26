"""Abstract base class for historical and snapshot market data providers.

Concrete implementations live alongside this file (e.g.
``alpaca_provider.py``). Do not add streaming / live-feed methods here —
those belong in ``LiveDataFeed`` (see ``feed.py``).
"""

import abc
from datetime import datetime
from typing import List, Optional

import polars as pl

from data.timeframe import TimeFrame


class MarketDataProvider(abc.ABC):
    """Historical and snapshot market data provider contract."""

    @abc.abstractmethod
    def get_historical_bars(
        self,
        symbols: List[str],
        timeframe: TimeFrame,
        start: datetime,
        end: Optional[datetime] = None,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV bars.

        Parameters
        ----------
        symbols:
            List of ticker symbols (e.g. ``["AAPL", "BTC/USD"]``).
        timeframe:
            Bar granularity — see :class:`data.timeframe.TimeFrame`.
        start:
            Inclusive start datetime (**timezone-aware UTC**).
        end:
            Optional inclusive end datetime (**timezone-aware UTC**).

        Returns
        -------
        polars.DataFrame
            Columns **at minimum**:
            ``symbol``, ``timestamp``, ``open``, ``high``, ``low``,
            ``close``, ``volume``.  ``timestamp`` must be timezone-aware UTC.
        """
        ...

    @abc.abstractmethod
    def get_latest_bar(
        self,
        symbol: str,
        timeframe: TimeFrame,
    ) -> Optional[pl.DataFrame]:
        """Return the most recent completed bar as a single-row DataFrame.

        Returns ``None`` if no bar is available.
        """
        ...

    @abc.abstractmethod
    def is_market_open(self, symbol: str) -> bool:
        """Return ``True`` if the asset's market is currently open.

        Crypto is conventionally always open; equities depend on exchange
        hours.
        """
        ...
