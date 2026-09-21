"""
Concrete MarketDataProvider backed by Alpaca (US equities + crypto).

Wraps Alpaca's REST and WebSocket SDKs behind the generic
:class:`MarketDataProvider` interface. Selected with DATA_SOURCE=alpaca (the
default). Equities and crypto need different Alpaca clients and different
stream classes, so this adapter holds both and routes per symbol.

Glossary:
    AlpacaProvider -- the adapter itself.
    api_key / secret_key -- Alpaca credentials, passed in by the factory.
    paper -- True routes trading calls to the paper-money account.
    _timeframe_for -- maps a bar size in minutes onto the Alpaca TimeFrame unit that
        can express it (1440 -> 1Day, 240 -> 4Hour, 15 -> 15Minute); raises for sizes
        no single unit covers (e.g. 90). Minute amounts above 59 are rejected by
        Alpaca's API, which is why H4/D1 must not be built as 240/1440 minutes.
    stock_client / crypto_client -- separate historical REST clients; Alpaca
        splits the two asset types across different endpoints.
    trading_client -- the account/asset endpoint, used for symbol discovery
        rather than for placing orders.
    Symbol convention -- crypto pairs carry a slash ("BTC/USD") while equities
        do not ("AAPL"); the slash is how this adapter decides which client to
        use, and Alpaca is inconsistent about it in responses (see the
        matching fix-up in src/data/feed.py).
    DataFeed.IEX -- the free tier's data feed, which sees only part of total
        market volume. Any volume-derived feature computed from it is an
        approximation.
"""

import asyncio
import logging
from datetime import datetime
from typing import Callable, List, Optional

import pandas as pd
import polars as pl
from alpaca.data.enums import DataFeed
from alpaca.data.historical import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.live import CryptoDataStream, StockDataStream
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, AssetStatus
from alpaca.trading.requests import GetAssetsRequest

from data.market_provider import MarketDataProvider

logger = logging.getLogger(__name__)


MINUTES_PER_HOUR = 60
MINUTES_PER_DAY = 1440


def _timeframe_for(timeframe_minutes: int) -> TimeFrame:
    """
    Build an Alpaca ``TimeFrame`` for a bar size expressed in minutes.

    Why this exists: ``TimeFrame(n, TimeFrameUnit.Minute)`` is rejected by Alpaca for
    n > 59 ("Second or Minute units can only be used with amounts between 1-59"), so
    the inline construction this replaces could not request **H4 (240) or D1 (1440) at
    all** — the two sizes the crypto-expansion plan depends on (see
    llm_reports/recons/2026-08-11_crypto-expansion-feasibility.md, which recommends
    daily crypto precisely because the round-trip fee stops dominating there). Requests
    built that way failed at the API, were swallowed by the caller's broad ``except``,
    and surfaced as "no data" rather than as an error.

    Two independent API restrictions shape the mapping, both confirmed against
    alpaca-py's own ``TimeFrame.validate_timeframe``:
      * minute amounts are limited to 1-59, so hours/days must change unit;
      * **Day and Week units accept amount 1 only**, so a multi-day bar size (2880
        minutes = 2 days) cannot be expressed at all.

    Sizes no single unit covers (90 minutes, 2880 minutes, ...) raise ``ValueError``
    here rather than failing remotely inside a broad ``except``.
    """
    n = int(timeframe_minutes)
    if n <= 0:
        raise ValueError(f"timeframe_minutes must be positive, got {timeframe_minutes}")
    if n == MINUTES_PER_DAY:
        return TimeFrame(1, TimeFrameUnit.Day)
    if n % MINUTES_PER_HOUR == 0 and n // MINUTES_PER_HOUR <= 24:
        return TimeFrame(n // MINUTES_PER_HOUR, TimeFrameUnit.Hour)
    if n > 59:
        raise ValueError(
            f"{n} minutes is not expressible in Alpaca's timeframe units "
            f"(minute amounts are 1-59; 60-1440 must divide by 60; day amounts must be 1)"
        )
    return TimeFrame(n, TimeFrameUnit.Minute)


class AlpacaProvider(MarketDataProvider):
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.stock_client = StockHistoricalDataClient(api_key, secret_key)
        self.crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)

        # Streaming state — populated by subscribe()
        self._callback: Optional[Callable] = None
        self._symbols: List[str] = []
        self._crypto_stream: Optional[CryptoDataStream] = None
        self._stock_stream: Optional[StockDataStream] = None

        logger.info("AlpacaProvider initialized (Universal Stock/Crypto).")

    # ── MarketDataProvider interface ──────────────────────────────────

    def get_active_symbols(self, limit: int = 10) -> List[str]:
        """
        Return up to *limit* tradable US-equity symbols.

        Note: Alpaca does not expose a volume-ranked "most active" endpoint
        cheaply. This returns the first *limit* tradable, active US equities
        in the order Alpaca's asset listing returns them. For a true
        most-active list, use PolygonDataProvider.
        """
        try:
            req = GetAssetsRequest(
                asset_class=AssetClass.US_EQUITY,
                status=AssetStatus.ACTIVE,
            )
            assets = self.trading_client.get_all_assets(req)
            tradable = [a.symbol for a in assets if a.tradable]
            return tradable[:limit]
        except Exception as e:
            logger.error("AlpacaProvider.get_active_symbols failed: %s", e)
            return []

    def get_historical_bars(
        self, symbol: str, timeframe_minutes: int, start: datetime, end: datetime
    ) -> pl.DataFrame:
        """Fetches bars for either Stocks or Crypto based on symbol format."""
        try:
            # Detect if it's a Crypto symbol (contains '/')
            is_crypto = "/" in symbol

            if is_crypto:
                req = CryptoBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=_timeframe_for(timeframe_minutes),
                    start=start,
                    end=end,
                )
                bars = self.crypto_client.get_crypto_bars(req)
            else:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=_timeframe_for(timeframe_minutes),
                    start=start,
                    end=end,
                    feed=DataFeed.IEX,
                )
                bars = self.stock_client.get_stock_bars(req)

            if not bars.data or symbol not in bars.data:
                return self._empty_bars()

            # Data Washing: Strip Alpaca metadata for Polars compatibility
            df_pandas = bars.df.loc[symbol].reset_index()
            df_pandas.columns = [col.lower() for col in df_pandas.columns]

            df_numpy_backed = pd.DataFrame(
                {
                    col: df_pandas[col].to_numpy(dtype=None, copy=True)
                    for col in df_pandas.columns
                }
            )

            return pl.from_pandas(df_numpy_backed)

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return self._empty_bars()

    def subscribe(self, symbols: List[str], callback: Callable) -> None:
        """
        Register *callback* for real-time bar updates.

        Routes symbols to the appropriate Alpaca stream based on whether
        they contain '/' (crypto) or not (equity). Both streams are
        initialized but not started — call run_stream() to begin.

        The callback receives a dict with keys:
        symbol, timestamp, open, high, low, close, volume.
        """
        self._symbols = symbols
        self._callback = callback

        crypto_symbols = [s for s in symbols if "/" in s]
        stock_symbols = [s for s in symbols if "/" not in s]

        async def _bar_handler(bar):
            await self._callback(
                {
                    "symbol": bar.symbol,
                    "timestamp": bar.timestamp,
                    "open": bar.open,
                    "high": bar.high,
                    "low": bar.low,
                    "close": bar.close,
                    "volume": bar.volume,
                }
            )

        if crypto_symbols:
            self._crypto_stream = CryptoDataStream(self.api_key, self.secret_key)
            self._crypto_stream.subscribe_bars(_bar_handler, *crypto_symbols)
            logger.info(
                "AlpacaProvider: subscribed to crypto bars for %s", crypto_symbols
            )

        if stock_symbols:
            self._stock_stream = StockDataStream(self.api_key, self.secret_key)
            self._stock_stream.subscribe_bars(_bar_handler, *stock_symbols)
            logger.info(
                "AlpacaProvider: subscribed to stock bars for %s", stock_symbols
            )

    def run_stream(self) -> None:
        """
        Start the blocking stream event loop(s).

        If both crypto and stock subscriptions exist, both streams run
        concurrently via asyncio.gather. Blocks until interrupted.

        Must be called after subscribe().
        """
        if self._callback is None:
            raise RuntimeError("Call subscribe() before run_stream().")

        async def _run_all():
            tasks = []
            if self._crypto_stream is not None:
                tasks.append(self._crypto_stream._run_forever())
            if self._stock_stream is not None:
                tasks.append(self._stock_stream._run_forever())
            if not tasks:
                raise RuntimeError("No symbols subscribed — nothing to stream.")
            await asyncio.gather(*tasks)

        asyncio.run(_run_all())
