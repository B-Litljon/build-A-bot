"""
Concrete MarketDataProvider backed by the OANDA v20 REST + Streaming API.

Wraps ``oandapyV20`` behind the generic :class:`MarketDataProvider`
interface. Designed for CFTC/NFA-compliant forex scalping (FIFO, no
hedging, 50:1 leverage caps) and the V5 Angel/Devil meta-labeling
workflow.

Required environment variables:
    OANDA_API_KEY       - Bearer token from hub.oanda.com
    OANDA_ACCOUNT_ID    - Account ID (numeric string)
"""

import asyncio
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional

import oandapyV20
import oandapyV20.endpoints.accounts as v20_accounts
import oandapyV20.endpoints.instruments as v20_instruments
import oandapyV20.endpoints.pricing as v20_pricing
import polars as pl

from data.market_provider import MarketDataProvider

logger = logging.getLogger(__name__)

# Maps timeframe_minutes → OANDA granularity string
_GRANULARITY: Dict[int, str] = {
    1: "M1",
    2: "M2",
    4: "M4",
    5: "M5",
    10: "M10",
    15: "M15",
    30: "M30",
    60: "H1",
    120: "H2",
    180: "H3",
    240: "H4",
    360: "H6",
    480: "H8",
    720: "H12",
    1440: "D",
}

_MAX_CANDLES = 5000


def _to_oanda_symbol(symbol: str) -> str:
    """Normalise 'EUR/USD', 'EURUSD', or 'EUR_USD' → 'EUR_USD'."""
    return symbol.replace("/", "_").upper()


def _parse_iso(ts: str) -> datetime:
    """
    Parse OANDA's ISO-8601 timestamp to a UTC-aware datetime.

    OANDA returns nanosecond precision (e.g. "2024-01-01T00:00:00.000000000Z"),
    which fromisoformat rejects — truncate to seconds before parsing.
    """
    clean = ts.rstrip("Z")
    if "." in clean:
        clean = clean[: clean.index(".")]
    return datetime.fromisoformat(clean).replace(tzinfo=timezone.utc)


class OandaMarketProvider(MarketDataProvider):
    """
    OANDA v20 adapter for forex market data.

    Parameters
    ----------
    environment : str
        ``'practice'`` for paper trading, ``'live'`` for real money.
    api_key : str, optional
        Falls back to the ``OANDA_API_KEY`` environment variable.
    account_id : str, optional
        Falls back to the ``OANDA_ACCOUNT_ID`` environment variable.
    stream_granularity_minutes : int
        Bar size used when aggregating real-time price ticks (default 1).
    """

    def __init__(
        self,
        environment: str = "practice",
        api_key: Optional[str] = None,
        account_id: Optional[str] = None,
        stream_granularity_minutes: int = 1,
    ):
        self._api_key = api_key or os.getenv("OANDA_API_KEY")
        self._account_id = account_id or os.getenv("OANDA_ACCOUNT_ID")
        if not self._api_key:
            raise ValueError(
                "OANDA API key required. Set OANDA_API_KEY or pass api_key=."
            )
        if not self._account_id:
            raise ValueError(
                "OANDA account ID required. Set OANDA_ACCOUNT_ID or pass account_id=."
            )

        self._environment = environment
        self._stream_gran = stream_granularity_minutes
        self._client = oandapyV20.API(
            access_token=self._api_key,
            environment=environment,
        )

        # Streaming state — populated by subscribe()
        self._callback: Optional[Callable] = None
        self._symbols: List[str] = []
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Per-symbol tick accumulator: symbol → bar state dict
        self._tick_bars: Dict[str, dict] = {}

        logger.info("OandaMarketProvider initialized (environment=%s).", environment)

    # ── internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _granularity(timeframe_minutes: int) -> str:
        gran = _GRANULARITY.get(timeframe_minutes)
        if gran is None:
            raise ValueError(
                f"Unsupported timeframe_minutes={timeframe_minutes}. "
                f"Valid values: {sorted(_GRANULARITY)}"
            )
        return gran

    def _handle_tick(self, msg: dict) -> None:
        """
        Aggregate a single PRICE tick into the current bar for its instrument.

        Mid price is (best bid + best ask) / 2. Flushes the completed bar
        and fires the callback when the bar period rolls over.
        """
        try:
            instrument = msg["instrument"]
            ts = _parse_iso(msg["time"])

            bids = msg.get("bids") or []
            asks = msg.get("asks") or []
            if not bids or not asks:
                return

            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            mid = (bid + ask) / 2.0

            bar_epoch = int(ts.timestamp()) // (self._stream_gran * 60)
            bar_start = datetime.fromtimestamp(
                bar_epoch * self._stream_gran * 60, tz=timezone.utc
            )
            state = self._tick_bars.get(instrument)

            if state is None or state["epoch"] != bar_epoch:
                if state is not None:
                    self._flush_bar(instrument, state)
                self._tick_bars[instrument] = {
                    "epoch": bar_epoch,
                    "bar_start": bar_start,
                    "open": mid,
                    "high": mid,
                    "low": mid,
                    "close": mid,
                    "volume": 0,
                }
            else:
                state["high"] = max(state["high"], mid)
                state["low"] = min(state["low"], mid)
                state["close"] = mid
                state["volume"] += 1  # tick count used as proxy volume

        except Exception as e:
            logger.warning("OandaMarketProvider tick parse error: %s", e)

    def _flush_bar(self, instrument: str, state: dict) -> None:
        """Fire the callback with the completed bar dict."""
        bar = {
            "symbol": instrument,
            "timestamp": state["bar_start"],
            "open": state["open"],
            "high": state["high"],
            "low": state["low"],
            "close": state["close"],
            "volume": float(state["volume"]),
        }
        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._callback(bar), self._loop)
        else:
            asyncio.run(self._callback(bar))

    # ── MarketDataProvider interface ──────────────────────────────────

    def get_active_symbols(self, limit: int = 10) -> List[str]:
        """
        Return up to *limit* tradable instruments on this OANDA account.

        OANDA does not rank by volume; instruments are returned in the
        order the API lists them for the account.
        """
        try:
            req = v20_accounts.AccountInstruments(accountID=self._account_id)
            self._client.request(req)
            return [i["name"] for i in req.response.get("instruments", [])[:limit]]
        except Exception as e:
            logger.error("OandaMarketProvider.get_active_symbols failed: %s", e)
            return []

    def get_historical_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch historical mid-price OHLCV candles from OANDA.

        Requests up to 5 000 candles per call using the ``from`` + ``count``
        parameter group (startOnly) and pages forward until the full
        ``start``–``end`` range is covered. In-progress (incomplete)
        candles are excluded.
        """
        oanda_symbol = _to_oanda_symbol(symbol)
        gran = self._granularity(timeframe_minutes)
        rows: list = []
        chunk_start = start

        try:
            while chunk_start < end:
                params = {
                    "granularity": gran,
                    "from": chunk_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "count": _MAX_CANDLES,
                    "price": "M",
                }
                req = v20_instruments.InstrumentsCandles(
                    instrument=oanda_symbol, params=params
                )
                self._client.request(req)
                candles = req.response.get("candles", [])

                if not candles:
                    break

                for c in candles:
                    ts = _parse_iso(c["time"])
                    if ts > end:
                        break
                    if not c.get("complete", True):
                        continue
                    mid = c["mid"]
                    rows.append(
                        {
                            "timestamp": ts,
                            "open": float(mid["o"]),
                            "high": float(mid["h"]),
                            "low": float(mid["l"]),
                            "close": float(mid["c"]),
                            "volume": float(c.get("volume", 0)),
                        }
                    )

                last_ts = _parse_iso(candles[-1]["time"])
                if last_ts <= chunk_start or len(candles) < _MAX_CANDLES:
                    break
                chunk_start = last_ts + timedelta(minutes=timeframe_minutes)

        except Exception as e:
            logger.error(
                "OandaMarketProvider.get_historical_bars failed for %s: %s",
                oanda_symbol,
                e,
            )
            return self._empty_bars()

        if not rows:
            return self._empty_bars()

        return pl.DataFrame(rows, schema=self._BAR_SCHEMA)

    def subscribe(self, symbols: List[str], callback: Callable) -> None:
        """
        Register *callback* for real-time bar updates.

        Ticks are aggregated into fixed-duration bars (``stream_granularity_minutes``).
        Non-blocking — call :meth:`run_stream` to start receiving data.
        """
        self._symbols = [_to_oanda_symbol(s) for s in symbols]
        self._callback = callback
        self._tick_bars = {}
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()
        logger.info("OandaMarketProvider: subscribed to %s", self._symbols)

    def run_stream(self) -> None:
        """
        Start the blocking OANDA pricing stream.

        Aggregates bid/ask mid-price ticks into bars and invokes the
        registered callback when each bar closes. Blocks until
        interrupted (KeyboardInterrupt) or :meth:`stop_stream` is called.

        Must be called after :meth:`subscribe`.
        """
        if self._callback is None:
            raise RuntimeError("Call subscribe() before run_stream().")

        self._stop_event.clear()
        params = {"instruments": ",".join(self._symbols)}
        req = v20_pricing.PricingStream(accountID=self._account_id, params=params)

        try:
            for msg in self._client.request(req):
                if self._stop_event.is_set():
                    break
                if msg.get("type") == "PRICE":
                    self._handle_tick(msg)
        except KeyboardInterrupt:
            logger.info("OandaMarketProvider stream stopped by user.")
        except Exception as e:
            logger.error("OandaMarketProvider stream error: %s", e)

    def stop_stream(self) -> None:
        """Signal the stream loop to exit and flush all in-flight bars."""
        self._stop_event.set()
        for instrument, state in list(self._tick_bars.items()):
            if state:
                self._flush_bar(instrument, state)
        self._tick_bars.clear()
