"""
Out-of-Sample (OOS) Data Harvester for Universal Scalper v3.0.

Fetches historical 1-minute bars from Alpaca API for OOS testing.
Usage:
    python -m src.data.harvester

Environment Variables:
    ALPACA_API_KEY: Alpaca API key
    ALPACA_SECRET_KEY: Alpaca API secret
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import polars as pl
import pyarrow.parquet as pq
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = Path("data/oos_bars.parquet")
TICKERS: List[str] = ["TSLA", "NVDA", "MARA", "COIN", "SMCI"]
TIMEFRAME = TimeFrame(1, TimeFrameUnit.Minute)
DATA_FEED = DataFeed.IEX
LOOKBACK_DAYS = 7


def get_alpaca_client() -> StockHistoricalDataClient:
    """Initialize Alpaca client from environment variables."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables must be set"
        )

    return StockHistoricalDataClient(api_key, secret_key)


def fetch_ticker_data(
    client: StockHistoricalDataClient,
    symbol: str,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """Fetch 1-minute bars for a single ticker."""
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TIMEFRAME,
            start=start,
            end=end,
            feed=DATA_FEED,
        )

        bars = client.get_stock_bars(request)

        if not bars.data or symbol not in bars.data:
            logger.warning(f"No data returned for {symbol}")
            return pl.DataFrame()

        # Reset Alpaca MultiIndex and convert to Polars
        df_pandas = bars.df.reset_index()

        # Standardize column names (lowercase)
        df_pandas.columns = [col.lower() for col in df_pandas.columns]

        # Convert to Polars DataFrame
        df = pl.from_pandas(df_pandas)

        # Ensure symbol column exists
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(symbol).alias("symbol"))

        logger.info(f"Fetched {len(df):,} bars for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return pl.DataFrame()


def harvest_oos_data() -> None:
    """Main harvester function - fetches OOS data for all tickers."""
    logger.info("=" * 70)
    logger.info("OOS DATA HARVESTER v3.0")
    logger.info("=" * 70)

    # Initialize client
    client = get_alpaca_client()
    logger.info("Alpaca client initialized")

    # Calculate date range (last 7 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    logger.info(f"Fetching data from {start_date.date()} to {end_date.date()}")
    logger.info(f"Tickers: {', '.join(TICKERS)}")
    logger.info(f"Timeframe: 1-minute bars")
    logger.info(f"Data Feed: {DATA_FEED.value}")

    # Fetch data for all tickers
    all_frames: List[pl.DataFrame] = []

    for ticker in TICKERS:
        df = fetch_ticker_data(client, ticker, start_date, end_date)
        if len(df) > 0:
            all_frames.append(df)

    if not all_frames:
        logger.error("No data fetched for any ticker. Exiting.")
        return

    # Combine all ticker data
    combined = pl.concat(all_frames, how="vertical_relaxed")

    # Sort by timestamp and symbol for consistency
    combined = combined.sort(["timestamp", "symbol"])

    logger.info(f"Combined dataset: {len(combined):,} total rows")
    logger.info(f"Symbols: {combined['symbol'].unique().to_list()}")
    logger.info(
        f"Date range: {combined['timestamp'].min()} to {combined['timestamp'].max()}"
    )

    # Ensure output directory exists
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Write to Parquet with Snappy compression
    combined.write_parquet(
        DATA_PATH,
        compression="snappy",
        use_pyarrow=True,
    )

    # Verify with pyarrow
    file_size = DATA_PATH.stat().st_size / (1024 * 1024)  # MB
    logger.info(f"Data saved to {DATA_PATH}")
    logger.info(f"File size: {file_size:.2f} MB")
    logger.info("Harvest complete!")


if __name__ == "__main__":
    harvest_oos_data()
