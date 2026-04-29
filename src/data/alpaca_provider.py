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
                    timeframe=TimeFrame(timeframe_minutes, TimeFrameUnit.Minute),
                    start=start,
                    end=end,
                )
                bars = self.crypto_client.get_crypto_bars(req)
            else:
                req = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame(timeframe_minutes, TimeFrameUnit.Minute),
                    start=start,
                    end=end,
                    feed=DataFeed.IEX,
                )
                bars = self.stock_client.get_stock_bars(req)

            if not bars.data or symbol not in bars.data:
                return pl.DataFrame()

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
            return pl.DataFrame()

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
