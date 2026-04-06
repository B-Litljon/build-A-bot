import logging
import pandas as pd
import polars as pl
from datetime import datetime
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

logger = logging.getLogger(__name__)


class AlpacaProvider:
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.stock_client = StockHistoricalDataClient(api_key, secret_key)
        self.crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
        logger.info("AlpacaProvider initialized (Universal Stock/Crypto).")

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
