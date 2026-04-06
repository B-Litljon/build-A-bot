"""
Concrete MarketDataProvider backed by Yahoo Finance (yfinance).

**For Testing / Paper Trading Only.**

Yahoo Finance does not provide a real-time WebSocket feed.  The
:meth:`run_stream` implementation polls for the latest 1-minute candle
every 60 seconds, which introduces up to 60 s of latency.  This is
acceptable for strategy validation and paper trading but **must not**
be used for latency-sensitive live execution.

No API key is required.
"""

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, List

import polars as pl
import yfinance as yf

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

# Yahoo Finance screener: pre-defined "most active" tickers as fallback.
# yf.download can be slow for screener-style queries; the Ticker module
# exposes no ranked "most active" endpoint.  We use the yf screener when
# available, otherwise fall back to a well-known liquid list.
_FALLBACK_ACTIVE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "AMZN",
    "META",
    "GOOG",
    "AMD",
    "SPY",
    "QQQ",
    "INTC",
    "BAC",
    "F",
    "PLTR",
    "SOFI",
]


class YahooDataProvider(MarketDataProvider):
    """
    Yahoo Finance adapter for market data.

    .. warning::

        **For Testing / Paper Trading Only.**

        Yahoo Finance is rate-limited, has no official WebSocket feed,
        and its data may be delayed by 15+ minutes for some exchanges.
        Do **not** rely on this provider for live capital deployment.

    Parameters
    ----------
    poll_interval : int, default 60
        Seconds between polls in :meth:`run_stream`.
    """

    def __init__(self, poll_interval: int = 60):
        self._poll_interval = poll_interval

        # Stream state — populated by subscribe()
        self._callback: Callable | None = None
        self._symbols: List[str] = []
        self._last_seen_ts: dict[str, datetime] = {}

    # ── MarketDataProvider interface ──────────────────────────────────

    def get_active_symbols(self, limit: int = 10) -> List[str]:
        """
        Return the *limit* most-active tickers.

        Attempts to use the ``yfinance`` screener.  If that fails (API
        changes, rate limiting), returns a hard-coded list of high-volume
        US equities.
        """
        try:
            screener = yf.Screener()
            screener.set_default_body({"query": "most_actives", "count": limit})
            response = screener.response
            quotes = response.get("quotes", [])
            if quotes:
                symbols = [q["symbol"] for q in quotes[:limit]]
                if symbols:
                    return symbols
        except Exception as e:
            logger.warning("yfinance screener failed (%s). Using fallback list.", e)

        return _FALLBACK_ACTIVE[:limit]

    def get_historical_bars(
        self,
        symbol: str,
        timeframe_minutes: int,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame:
        """
        Fetch OHLCV bars via ``yf.download`` and return a Polars
        DataFrame conforming to the canonical schema.

        Yahoo returns a pandas DataFrame with a (possibly MultiIndex)
        column header. This method flattens and normalises it.
        """
        try:
            interval = self._yf_interval(timeframe_minutes)

            pdf = yf.download(
                tickers=symbol,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d %H:%M:%S"),
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if pdf is None or pdf.empty:
                logger.warning("Yahoo returned 0 bars for %s.", symbol)
                return pl.DataFrame(
                    {col: [] for col in _BAR_SCHEMA}, schema=_BAR_SCHEMA
                )

            # yf.download can return MultiIndex columns when called with a
            # single ticker string — flatten to simple column names.
            if hasattr(pdf.columns, "nlevels") and pdf.columns.nlevels > 1:
                pdf.columns = [
                    col[0] if isinstance(col, tuple) else col for col in pdf.columns
                ]

            # Ensure index (Datetime) becomes a column
            pdf = pdf.reset_index()

            # Rename the index column — Yahoo names it "Datetime" for
            # intraday intervals and "Date" for daily+.
            ts_col = "Datetime" if "Datetime" in pdf.columns else "Date"

            pdf = pdf.rename(columns={ts_col: "timestamp"})

            # Normalise column names to lowercase
            pdf.columns = [c.lower() for c in pdf.columns]

            # Select only the columns we need (drop 'adj close', etc.)
            keep = ["timestamp", "open", "high", "low", "close", "volume"]
            pdf = pdf[[c for c in keep if c in pdf.columns]]

            # Convert to Polars
            df = pl.from_pandas(pdf)
            df = self._normalise_bars_df(df)
            return df

        except Exception as e:
            logger.error("Yahoo get_historical_bars failed for %s: %s", symbol, e)
            return pl.DataFrame({col: [] for col in _BAR_SCHEMA}, schema=_BAR_SCHEMA)

    def subscribe(self, symbols: List[str], callback: Callable) -> None:
        """
        Register *callback* for polled bar updates (non-blocking setup).

        Because Yahoo has no WebSocket, the actual polling happens in
        :meth:`run_stream`.
        """
        self._symbols = symbols
        self._callback = callback
        self._last_seen_ts = {
            s: datetime.min.replace(tzinfo=timezone.utc) for s in symbols
        }
        logger.info("Yahoo: registered poller for %s", symbols)

    def run_stream(self) -> None:
        """
        Start the **blocking** poll loop.

        Every ``poll_interval`` seconds, fetches the most recent 1-minute
        candle for each subscribed symbol.  If the candle's timestamp is
        newer than the last one seen, fires the callback.

        Runs until the process is interrupted.
        """
        if self._callback is None:
            raise RuntimeError("Call subscribe() before run_stream().")

        import asyncio

        logger.info(
            "Yahoo poller started (interval=%ds). Ctrl+C to stop.",
            self._poll_interval,
        )

        while True:
            for symbol in self._symbols:
                try:
                    bar = self._fetch_latest_bar(symbol)
                    if bar is None:
                        continue

                    last = self._last_seen_ts.get(
                        symbol, datetime.min.replace(tzinfo=timezone.utc)
                    )
                    if bar["timestamp"] > last:
                        self._last_seen_ts[symbol] = bar["timestamp"]
                        try:
                            loop = asyncio.get_running_loop()
                            loop.create_task(self._callback(bar))
                        except RuntimeError:
                            asyncio.run(self._callback(bar))

                except Exception as e:
                    logger.warning("Yahoo poll error for %s: %s", symbol, e)

            time.sleep(self._poll_interval)

    # ── internal helpers ──────────────────────────────────────────────

    def _fetch_latest_bar(self, symbol: str) -> dict | None:
        """
        Download the most recent 1-minute candle for *symbol*.

        Returns a provider-agnostic bar dict or ``None`` on failure.
        """
        try:
            pdf = yf.download(
                tickers=symbol,
                period="1d",
                interval="1m",
                progress=False,
                auto_adjust=True,
            )

            if pdf is None or pdf.empty:
                return None

            # Flatten MultiIndex columns if present
            if hasattr(pdf.columns, "nlevels") and pdf.columns.nlevels > 1:
                pdf.columns = [
                    col[0] if isinstance(col, tuple) else col for col in pdf.columns
                ]

            last = pdf.iloc[-1]

            # The index is the timestamp
            ts = pdf.index[-1].to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)

            return {
                "symbol": symbol,
                "timestamp": ts,
                "open": float(last["Open"] if "Open" in last.index else last["open"]),
                "high": float(last["High"] if "High" in last.index else last["high"]),
                "low": float(last["Low"] if "Low" in last.index else last["low"]),
                "close": float(
                    last["Close"] if "Close" in last.index else last["close"]
                ),
                "volume": float(
                    last["Volume"] if "Volume" in last.index else last["volume"]
                ),
            }
        except Exception as e:
            logger.warning("Failed to fetch latest bar for %s: %s", symbol, e)
            return None

    @staticmethod
    def _yf_interval(minutes: int) -> str:
        """
        Map a timeframe in minutes to a yfinance interval string.

        Yahoo supports: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, ...
        """
        mapping = {
            1: "1m",
            2: "2m",
            5: "5m",
            15: "15m",
            30: "30m",
            60: "60m",
            90: "90m",
        }
        if minutes in mapping:
            return mapping[minutes]

        # For unlisted intervals, round down to the nearest supported one
        # and let the bar aggregator handle the rest.
        for threshold in sorted(mapping.keys(), reverse=True):
            if minutes >= threshold:
                logger.warning(
                    "Yahoo has no native %dm interval; fetching %dm bars instead.",
                    minutes,
                    threshold,
                )
                return mapping[threshold]

        return "1m"

    @staticmethod
    def _normalise_bars_df(df: pl.DataFrame) -> pl.DataFrame:
        """
        Select and cast columns to the canonical bar schema.

        Mirrors ``AlpacaProvider._normalise_bars_df``.
        """
        df = df.select(
            [
                pl.col("timestamp"),
                pl.col("open"),
                pl.col("high"),
                pl.col("low"),
                pl.col("close"),
                pl.col("volume"),
            ]
        )

        # Ensure timestamp is Datetime with UTC tz
        ts_dtype = df["timestamp"].dtype
        if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is None:
            df = df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC"))
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
