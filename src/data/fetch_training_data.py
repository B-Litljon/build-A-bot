#!/usr/bin/env python3
"""
Data fetcher for Universal Scalper training.
Fetches 1-minute historical data for specified tickers and stores as Parquet.
"""

import os
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import polars as pl
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

_RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def fetch_ticker_data(
    client: StockHistoricalDataClient,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
) -> pl.DataFrame:
    """Fetch 1-minute bars for a single ticker."""
    logger.info(f"Fetching {symbol} from {start_date.date()} to {end_date.date()}...")

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date,
    )

    bars = client.get_stock_bars(request)

    if symbol not in bars.data or not bars.data[symbol]:
        logger.warning(f"No data returned for {symbol}")
        return pl.DataFrame()

    # Convert to Polars DataFrame
    data = []
    for bar in bars.data[symbol]:
        data.append(
            {
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
            }
        )

    df = pl.DataFrame(data)
    logger.info(f"  Retrieved {len(df)} rows for {symbol}")
    return df


def main():
    # Configuration
    symbols = ["SPY", "TSLA", "NVDA", "COIN"]
    days_back = 60  # Fetch last 60 days of data

    # Initialize client - support both standard and legacy env var names
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("alpaca_key")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("alpaca_secret")

    if not api_key or not secret_key:
        logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in environment")
        return

    client = StockHistoricalDataClient(api_key, secret_key)

    # Create output directory
    _RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    # Fetch data for each symbol
    for symbol in symbols:
        df = fetch_ticker_data(client, symbol, start_date, end_date)

        if len(df) == 0:
            logger.error(f"No data fetched for {symbol}, skipping...")
            continue

        # Save to Parquet
        output_path = _RAW_DIR / f"{symbol}_1min.parquet"
        df.write_parquet(output_path)
        logger.info(f"  Saved {len(df)} rows to {output_path}")

    logger.info("Data fetch complete!")


if __name__ == "__main__":
    main()
