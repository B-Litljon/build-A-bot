"""
Concrete MarketDataProvider backed by Polygon.io.

Wraps the ``polygon-api-client`` REST + WebSocket clients behind the
generic :class:`MarketDataProvider` interface.

Requires the ``POLYGON_API_KEY`` environment variable.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Callable, List

import polars as pl
from polygon import RESTClient
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage

from data.market_provider import MarketDataProvider

logger = logging.getLogger(__name__)

# ── canonical Polars schema (mirrors alpaca_provider._BAR_SCHEMA) ─────
_BAR_SCHEMA = {
    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}


class PolygonDataProvider(MarketDataProvider):
    """
    Polygon.io adapter for market data.

    Best suited for **speed-sensitive** and **ML** workloads thanks to
    Polygon's low-latency feeds and deep history.

    Parameters
    ----------
    api_key : str, optional
        Polygon API key.  Falls back to the ``POLYGON_API_KEY`` env var.
    """

    def __init__(self, api_key: str | None = None):
        self._api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Polygon API key is required. Set POLYGON_API_KEY in your "
                "environment or pass api_key= directly."
            )

        self._rest = RESTClient(api_key=self._api_key)

        # WebSocket state — initialised lazily in subscribe()
        self._ws: WebSocketClient | None = None
        self._callback: Callable | None = None
        self._symbols: List[str] = []

    # ── MarketDataProvider interface ──────────────────────────────────

    def get_active_symbols(self, limit: int = 10) -> List[str]:
        """
        Return the *limit* most-active tickers by volume from the
        previous trading day's snapshot.

        Uses ``/v2/snapshot/locale/us/markets/stocks/tickers`` which
        requires a Polygon subscription that includes snapshots.
        Falls back to a ``/v3/reference/tickers`` query on error.
        """
        try:
            snapshots = list(self._rest.get_snapshot_all("stocks"))

            # Sort by day volume descending and take top N
            snapshots.sort(
                key=lambda s: getattr(s.day, "volume", 0) or 0,
                reverse=True,
            )
            symbols = [s.ticker for s in snapshots[:limit]]
            if symbols:
                return symbols
        except Exception as e:
            logger.warning(
                "Polygon snapshot endpoint failed (%s). "
                "Falling back to reference tickers.",
                e,
            )

        # Fallback: grab tickers sorted by market cap (no volume ranking)
        try:
            tickers = list(
                self._rest.list_tickers(
                    market="stocks",
                    active=True,
                    limit=limit,
                    order="desc",
                    sort="market_cap",
                )
            )
            return [t.ticker for t in tickers[:limit]]
        except Exception as e:
            logger.error("Polygon fallback ticker lookup failed: %s", e)
            return []

    def get_historical_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch OHLCV bars via ``RESTClient.list_aggs`` and return a
        Polars DataFrame conforming to the canonical schema.

        Polygon returns timestamps as **milliseconds since epoch**.
        These are converted to UTC-aware ``datetime`` objects.
        """
        try:
            # Convert aware datetimes to millisecond timestamps for the API
            start_ms = int(start.timestamp() * 1_000)
            end_ms = int(end.timestamp() * 1_000)

            aggs = list(
                self._rest.list_aggs(
                    ticker=symbol,
                    multiplier=timeframe_minutes,
                    timespan="minute",
                    from_=start_ms,
                    to=end_ms,
                    limit=50_000,
                )
            )

            if not aggs:
                logger.warning("Polygon returned 0 bars for %s.", symbol)
                return pl.DataFrame(
                    {col: [] for col in _BAR_SCHEMA}, schema=_BAR_SCHEMA
                )

            rows = []
            for a in aggs:
                ts = datetime.fromtimestamp(a.timestamp / 1_000, tz=timezone.utc)
                rows.append(
                    {
                        "timestamp": ts,
                        "open": float(a.open),
                        "high": float(a.high),
                        "low": float(a.low),
                        "close": float(a.close),
                        "volume": float(a.volume),
                    }
                )

            df = pl.DataFrame(rows, schema=_BAR_SCHEMA)
            return df

        except Exception as e:
            logger.error("Polygon get_historical_bars failed for %s: %s", symbol, e)
            return pl.DataFrame({col: [] for col in _BAR_SCHEMA}, schema=_BAR_SCHEMA)

    def subscribe(self, symbols: List[str], callback: Callable) -> None:
        """
        Register *callback* for real-time aggregate (1-min bar) updates.

        Uses Polygon's WebSocket ``A.*`` (per-minute aggregates) channel.
        This method is **non-blocking** — the event loop starts in
        :meth:`run_stream`.
        """
        self._symbols = symbols
        self._callback = callback

        self._ws = WebSocketClient(
            api_key=self._api_key,
            subscriptions=[f"A.{s}" for s in symbols],
        )
        logger.info("Polygon: subscribed to aggregate updates for %s", symbols)

    def run_stream(self) -> None:
        """
        Start the **blocking** Polygon WebSocket event loop.

        Must be called after :meth:`subscribe`.  Runs until the process
        is interrupted.
        """
        if self._ws is None or self._callback is None:
            raise RuntimeError("Call subscribe() before run_stream().")

        import asyncio

        async def _dispatch(msgs: List[WebSocketMessage]) -> None:
            for msg in msgs:
                bar = self._convert_agg(msg)
                if bar is not None:
                    await self._callback(bar)

        # Polygon's WebSocketClient.run() expects a sync handler;
        # bridge to our async callback with asyncio.run().
        def _sync_handler(msgs: List[WebSocketMessage]) -> None:
            for msg in msgs:
                bar = self._convert_agg(msg)
                if bar is not None:
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._callback(bar))
                    except RuntimeError:
                        asyncio.run(self._callback(bar))

        self._ws.run(handle_msg=_sync_handler)

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _convert_agg(msg) -> dict | None:
        """
        Convert a Polygon WebSocket aggregate message to the
        provider-agnostic bar dict.

        Polygon aggregate messages have the event type ``"A"`` and carry:
            sym, s (start ms), e (end ms), o, h, l, c, v, ...

        Returns ``None`` for non-aggregate messages.
        """
        # The polygon-api-client >= 1.x delivers typed objects.
        # Aggregate messages expose .symbol, .open, .close, etc.
        try:
            symbol = getattr(msg, "symbol", None) or getattr(msg, "sym", None)
            if symbol is None:
                return None

            # Polygon timestamps: start_timestamp is ms since epoch
            ts_ms = getattr(msg, "start_timestamp", None) or getattr(msg, "s", None)
            if ts_ms is None:
                return None

            ts = datetime.fromtimestamp(ts_ms / 1_000, tz=timezone.utc)

            return {
                "symbol": symbol,
                "timestamp": ts,
                "open": float(getattr(msg, "open", 0) or getattr(msg, "o", 0)),
                "high": float(getattr(msg, "high", 0) or getattr(msg, "h", 0)),
                "low": float(getattr(msg, "low", 0) or getattr(msg, "l", 0)),
                "close": float(getattr(msg, "close", 0) or getattr(msg, "c", 0)),
                "volume": float(getattr(msg, "volume", 0) or getattr(msg, "v", 0)),
            }
        except Exception as e:
            logger.warning("Failed to convert Polygon aggregate message: %s", e)
            return None
