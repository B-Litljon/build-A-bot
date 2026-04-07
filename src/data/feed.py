import abc
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Callable

import pandas as pd
import polars as pl
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.live import CryptoDataStream

logger = logging.getLogger(__name__)


class MarketDataFeed(abc.ABC):
    @abc.abstractmethod
    async def warmup_history(
        self, symbols: List[str], lookback_minutes: int
    ) -> Dict[str, pl.DataFrame]:
        pass

    @abc.abstractmethod
    async def subscribe(self, symbols: List[str], on_tick: Callable):
        pass

    @abc.abstractmethod
    async def stop(self):
        pass


class AlpacaCryptoFeed(MarketDataFeed):
    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self._stream: CryptoDataStream | None = None

    async def warmup_history(
        self, symbols: List[str], lookback_minutes: int
    ) -> Dict[str, pl.DataFrame]:
        """Fetches historical bars from Alpaca REST and injects them into a per-symbol dict."""
        client = CryptoHistoricalDataClient(self.api_key, self.secret_key)
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=lookback_minutes)

        req = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start,
            end=end,
        )

        bars = await asyncio.to_thread(client.get_crypto_bars, req)
        raw_df = bars.df  # pandas MultiIndex DataFrame: (symbol, timestamp)

        # --- DEBUG DIAGNOSTIC (HOTFIX 4) ---
        print(f"[DEBUG] Raw history dataframe type: {type(raw_df)}")
        if raw_df is not None and not raw_df.empty:
            print(f"[DEBUG] Raw history index levels: {raw_df.index.names}")
            print(
                f"[DEBUG] Raw history unique symbols in index: {raw_df.index.get_level_values(0).unique().tolist()}"
            )
        else:
            print("[DEBUG] Raw history dataframe is EMPTY directly from Alpaca.")
        # --- END DEBUG ---

        result: Dict[str, pl.DataFrame] = {}

        if raw_df is None or raw_df.empty:
            logger.warning(
                "Alpaca returned an empty DataFrame for warmup. No bars injected."
            )
            return {s: pl.DataFrame() for s in symbols}

        # Determine what symbol keys are actually present in the index
        actual_symbols_in_index = raw_df.index.get_level_values(0).unique().tolist()
        print(f"[DEBUG] Requested symbols: {symbols}")
        print(f"[DEBUG] Symbols present in index: {actual_symbols_in_index}")

        for symbol in symbols:
            # Build a lookup map: handle BTC/USD <-> BTCUSD mismatch gracefully
            index_key = self._resolve_index_key(symbol, actual_symbols_in_index)

            if index_key is None:
                logger.warning(
                    f"[WARMUP] Symbol '{symbol}' not found in Alpaca response index. Skipping."
                )
                result[symbol] = pl.DataFrame()
                continue

            try:
                slice_df = raw_df.loc[index_key].reset_index()
                slice_df.columns = [col.lower() for col in slice_df.columns]

                # Convert to numpy-backed pandas then to Polars (strips Alpaca metadata)
                df_numpy = pd.DataFrame(
                    {
                        col: slice_df[col].to_numpy(dtype=None, copy=True)
                        for col in slice_df.columns
                    }
                )
                pl_df = pl.from_pandas(df_numpy)
                logger.info(
                    f"[WARMUP] {symbol}: {len(pl_df)} bars fetched (index key: '{index_key}')"
                )
                result[symbol] = pl_df
            except Exception as exc:
                logger.error(
                    f"[WARMUP] Failed to slice bars for '{symbol}' (key='{index_key}'): {exc}"
                )
                result[symbol] = pl.DataFrame()

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_index_key(requested: str, available: List[str]) -> str | None:
        """
        Maps a requested symbol (e.g. 'BTC/USD') to whatever key Alpaca actually
        stored in the MultiIndex (e.g. 'BTCUSD' or 'BTC/USD').
        Falls back to a slash-stripped comparison.
        """
        if requested in available:
            return requested

        # Try stripping '/' (BTC/USD -> BTCUSD)
        stripped = requested.replace("/", "")
        if stripped in available:
            logger.debug(f"Symbol mismatch fixed: '{requested}' -> '{stripped}'")
            return stripped

        # Try inserting '/' before USD/USDT/EUR (BTCUSD -> BTC/USD)
        for avail in available:
            if avail.replace("/", "") == stripped:
                logger.debug(f"Symbol mismatch fixed: '{requested}' -> '{avail}'")
                return avail

        return None

    async def subscribe(self, symbols: List[str], on_tick: Callable):
        """Subscribes to the Alpaca crypto WebSocket stream."""
        self._stream = CryptoDataStream(self.api_key, self.secret_key)

        async def _bar_handler(bar):
            await on_tick(
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

        self._stream.subscribe_bars(_bar_handler, *symbols)
        await self._stream._run_forever()

    async def stop(self):
        if self._stream:
            await self._stream.stop()
