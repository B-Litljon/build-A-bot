"""
Concrete MarketDataProvider backed by Alpaca.

Wraps alpaca-py REST + WebSocket clients behind the generic
:class:`MarketDataProvider` interface so the rest of the application
never imports ``alpaca`` directly.
"""

import logging
from datetime import datetime, timezone
from typing import Callable, List

import polars as pl
from alpaca.data import (
    StockBarsRequest,
    StockHistoricalDataClient,
    TimeFrame,
    TimeFrameUnit,
)
from alpaca.data.enums import MostActivesBy
from alpaca.data.historical.screener import ScreenerClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import MostActivesRequest
from alpaca.trading.client import TradingClient

from data.market_provider import MarketDataProvider

logger = logging.getLogger(__name__)

# ── canonical Polars schema every method must return ──────────────────
_BAR_SCHEMA = {
    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


class AlpacaProvider(MarketDataProvider):
    """
    Alpaca adapter for market data.

    Parameters
    ----------
    api_key : str
        Alpaca API key.
    secret_key : str
        Alpaca API secret.
    paper : bool, default True
        If ``True`` use the paper-trading endpoint.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self._api_key = api_key
        self._secret_key = secret_key

        # Trading client – exposed so the bot can pass it to OrderManager
        self.trading_client = TradingClient(api_key, secret_key, paper=paper)

        # Historical / screener REST clients (internal)
        self._stock_client = StockHistoricalDataClient(api_key, secret_key)
        self._screener_client = ScreenerClient(api_key, secret_key)

        # WebSocket stream (internal)
        self._data_stream = StockDataStream(api_key, secret_key)

    # ── MarketDataProvider interface ──────────────────────────────────

    def get_active_symbols(self, limit: int = 10) -> List[str]:
        """
        Return the *limit* most-active tickers ranked by volume.

        Alpaca's screener returns symbols prefixed with a space character
        (e.g. `` "AAPL"``).  We strip the leading character to normalise.
        """
        request = MostActivesRequest(by=MostActivesBy.VOLUME, top=limit)
        response = self._screener_client.get_most_actives(request)

        watchlist = pl.DataFrame(response.most_actives)
        # Strip leading whitespace / padding character from symbol column
        symbols: List[str] = watchlist["symbol"].str.strip_chars().to_list()
        return symbols

    def get_historical_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch OHLCV bars from Alpaca and return a Polars DataFrame.

        The returned DataFrame conforms to the canonical schema:
            timestamp (Datetime UTC), open, high, low, close, volume
        """
        try:
            tf = TimeFrame(timeframe_minutes, TimeFrameUnit.Minute)
            request_params = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start,
                end=end,
            )
            bars = self._stock_client.get_stock_bars(request_params)

            # Alpaca returns a multi-index pandas DF (symbol, timestamp)
            pandas_df = bars.df.reset_index()

            try:
                df = pl.from_pandas(pandas_df)
            except ImportError:
                df = pl.DataFrame(pandas_df.to_dict(orient="list"))

            # Normalise to canonical schema
            df = self._normalise_bars_df(df)
            return df

        except Exception as e:
            logger.error("Alpaca get_historical_bars failed for %s: %s", symbol, e)
            return pl.DataFrame(
                {col: [] for col in _BAR_SCHEMA}, schema=_BAR_SCHEMA
            )

    def subscribe(self, symbols: List[str], callback: Callable) -> None:
        """
        Register *callback* for live 1-minute bar updates (non-blocking).

        The raw Alpaca ``Bar`` is converted to a plain dict before being
        forwarded to *callback*, so consumers never depend on alpaca-py
        types.
        """

        async def _wrapper(raw_bar):  # noqa: ANN001
            generic = self._convert_bar(raw_bar)
            await callback(generic)

        self._data_stream.subscribe_bars(_wrapper, *symbols)
        logger.info("Subscribed to bar updates for %s", symbols)

    def run_stream(self) -> None:
        """Start the blocking Alpaca WebSocket event loop."""
        self._data_stream.run()

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _convert_bar(raw_bar) -> dict:  # noqa: ANN001
        """
        Convert an Alpaca ``Bar`` object to a provider-agnostic dict.

        Ensures the timestamp is always a UTC-aware ``datetime``.
        """
        ts = raw_bar.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        return {
            "symbol": raw_bar.symbol,
            "timestamp": ts,
            "open": float(raw_bar.open),
            "high": float(raw_bar.high),
            "low": float(raw_bar.low),
            "close": float(raw_bar.close),
            "volume": float(raw_bar.volume),
        }

    @staticmethod
    def _normalise_bars_df(df: pl.DataFrame) -> pl.DataFrame:
        """
        Select and cast columns to the canonical bar schema.

        Handles the common Alpaca quirks:
        - ``symbol`` column present from ``reset_index()`` – drop it.
        - Timestamp may arrive as a naive datetime – force UTC.
        """
        # Alpaca sometimes names columns differently; pick what we need
        col_map = {
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        df = df.select([pl.col(src).alias(dst) for src, dst in col_map.items()])

        # Ensure timestamp is Datetime with UTC tz
        ts_dtype = df["timestamp"].dtype
        if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is None:
            df = df.with_columns(
                pl.col("timestamp").dt.replace_time_zone("UTC")
            )
        elif not isinstance(ts_dtype, pl.Datetime):
            df = df.with_columns(
                pl.col("timestamp")
                .cast(pl.Datetime(time_unit="us"))
                .dt.replace_time_zone("UTC")
            )

        # Cast numeric columns to Float64
        for col in ("open", "high", "low", "close", "volume"):
            df = df.with_columns(pl.col(col).cast(pl.Float64))

        return df
