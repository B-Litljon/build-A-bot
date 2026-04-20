"""
src/day_trading/harvester_5m.py
Day Trade Dataset Harvester — Universal Scalper V4.0

Fetches two complementary datasets for the day trading model:

  1. 5-minute OHLCV bars  (base inference timeframe)
     Output: data/raw/dt_{SYMBOL}_5min.parquet
     Window: DAYS_BACK trading days (~1 year)

  2. Daily OHLCV bars     (for Daily ATR, gap_pct, first_30m_vol_rel)
     Output: data/raw/dt_{SYMBOL}_daily.parquet
     Window: DAYS_BACK + DAILY_WARMUP_DAYS (extra history so NATR-14
             on daily bars is fully warm from the first 5m training bar)

Both use Adjustment.SPLIT so split-adjusted prices are continuous
across the full training window (critical for NVDA/TSLA histories).

Usage:
    python -m src.day_trading.harvester_5m

Environment Variables:
    ALPACA_API_KEY    — Alpaca API key
    ALPACA_SECRET_KEY — Alpaca API secret
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd
import polars as pl
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# UNIVERSE & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# High-liquidity, high-beta instruments suited to intraday trend following.
# SPY/QQQ anchor macro regime context; TSLA/NVDA/AAPL/AMD/MSFT carry
# sufficient intraday range and institutional flow to produce clean trends.
DAY_TRADE_UNIVERSE: List[str] = [
    "SPY",  # S&P 500 ETF — macro regime reference
    "QQQ",  # Nasdaq-100 ETF — tech regime reference
    "TSLA",  # High-beta, strong intraday trend character
    "NVDA",  # AI proxy — volatile, momentum-driven
    "AAPL",  # Large-cap liquid — trend + mean-reversion mix
    "AMD",  # High-beta semiconductor
    "MSFT",  # Large-cap tech — trend anchor
]

# ~1 trading year of 5-minute data.  At ~78 RTH bars/day this produces
# ~78 × 252 ≈ 19,656 bars per symbol before feature-engineering dropna.
DAYS_BACK: int = 365

# Additional calendar days fetched for the daily bars only.
# NATR(14) on daily bars needs 14 days to warm up, so 30 gives comfortable
# headroom even when some days are holidays or early closes.
DAILY_WARMUP_DAYS: int = 30

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"

# Prefix distinguishes day-trade raw files from V3.4 scalper raw files
# (which are stored as {SYMBOL}_1min.parquet / {SYMBOL}_5min.parquet).
_FILE_PREFIX = "dt_"

TIMEFRAME_5MIN = TimeFrame(5, TimeFrameUnit.Minute)
TIMEFRAME_DAILY = TimeFrame(1, TimeFrameUnit.Day)
DATA_FEED = DataFeed.IEX


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENT
# ═══════════════════════════════════════════════════════════════════════════════


def _build_client() -> StockHistoricalDataClient:
    """Initialise Alpaca REST client from environment variables."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise EnvironmentError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment "
            "or in a .env file loaded before calling this script."
        )
    return StockHistoricalDataClient(api_key, secret_key)


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH HELPER
# ═══════════════════════════════════════════════════════════════════════════════


def _fetch_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    timeframe: TimeFrame,
    start: datetime,
    end: datetime,
) -> Optional[pl.DataFrame]:
    """
    Fetch OHLCV bars for one symbol and return a clean, symbol-tagged Polars
    DataFrame.

    Handling the Alpaca MultiIndex
    ──────────────────────────────
    `client.get_stock_bars()` returns a `BarSet` whose `.df` property is a
    Pandas DataFrame with a (symbol, timestamp) MultiIndex.  We:
      1. Call `.loc[symbol]` to select only this symbol's rows (drops the
         outer symbol level, leaving a DatetimeIndex named "timestamp").
      2. Call `.reset_index()` to promote "timestamp" to a plain column.
      3. Force every column through `.to_numpy(copy=True)` before handing it
         to `pl.from_pandas`.  This strips any Alpaca-specific ExtensionArray
         metadata (nullable Int64, Alpaca enums) that cause silent conversion
         failures in some versions of polars/pyarrow.

    Returns None (not an empty DataFrame) on any failure so the caller can
    log and skip cleanly without masking shape errors.
    """
    try:
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DATA_FEED,
            adjustment=Adjustment.SPLIT,  # split-adjusted prices — continuous history
        )
        bar_set = client.get_stock_bars(request)

        if not bar_set.data or symbol not in bar_set.data:
            logger.warning("%-6s  no data returned from Alpaca.", symbol)
            return None

        # Strip MultiIndex → flat Pandas DataFrame
        df_pd = bar_set.df.loc[symbol].reset_index()
        df_pd.columns = [c.lower() for c in df_pd.columns]

        # Force numpy-backed copy to neutralise Alpaca ExtensionArrays
        df_clean = pd.DataFrame(
            {col: df_pd[col].to_numpy(dtype=None, copy=True) for col in df_pd.columns}
        )

        df = pl.from_pandas(df_clean)
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

        return df

    except Exception as exc:
        logger.error("%-6s  fetch error: %s", symbol, exc)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HARVEST
# ═══════════════════════════════════════════════════════════════════════════════


def harvest(
    symbols: List[str] = DAY_TRADE_UNIVERSE,
    days_back: int = DAYS_BACK,
    daily_warmup: int = DAILY_WARMUP_DAYS,
) -> None:
    """
    Fetch and persist 5-minute and daily bars for every symbol in the universe.

    Output layout
    ─────────────
    data/raw/dt_{SYMBOL}_5min.parquet   — 5-minute OHLCV, `days_back` calendar days
    data/raw/dt_{SYMBOL}_daily.parquet  — Daily OHLCV,    `days_back + daily_warmup` days

    The daily file intentionally starts `daily_warmup` days earlier than the
    5-minute file.  This guarantees that when `DayTradeDailyJoin` computes
    NATR(14) on the daily bars, it has ≥14 warm-up days before the first
    5-minute training bar.  Without this buffer, the first 14 rows of every
    symbol would receive NaN for `daily_atr_abs` and be dropped by the
    pipeline's `clean_data()` step, wasting the earliest training data.

    Args:
        symbols:      List of ticker symbols to harvest.
        days_back:    Calendar days of 5-minute history to fetch.
        daily_warmup: Extra calendar days fetched for daily bars only.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = _build_client()

    end_dt = datetime.utcnow()
    start_5min = end_dt - timedelta(days=days_back)
    start_daily = end_dt - timedelta(days=days_back + daily_warmup)

    logger.info("=" * 70)
    logger.info("DAY TRADE HARVESTER  —  Universal Scalper V4.0")
    logger.info("=" * 70)
    logger.info("Universe      : %s", ", ".join(symbols))
    logger.info("5m  window    : %s  →  %s", start_5min.date(), end_dt.date())
    logger.info("Daily window  : %s  →  %s", start_daily.date(), end_dt.date())
    logger.info("Adjustment    : SPLIT")
    logger.info("Feed          : IEX")
    logger.info("-" * 70)

    failed: List[str] = []

    for symbol in symbols:
        # ── 5-minute bars ──────────────────────────────────────────────────────
        df_5min = _fetch_bars(client, symbol, TIMEFRAME_5MIN, start_5min, end_dt)

        if df_5min is None:
            logger.error("%-6s  SKIPPED — no 5-minute data.", symbol)
            failed.append(symbol)
            continue

        out_5min = _RAW_DIR / f"{_FILE_PREFIX}{symbol}_5min.parquet"
        df_5min.write_parquet(out_5min, compression="snappy")
        logger.info(
            "%-6s  5min   %7d bars  →  %s",
            symbol,
            len(df_5min),
            out_5min.name,
        )

        # ── Daily bars ─────────────────────────────────────────────────────────
        df_daily = _fetch_bars(client, symbol, TIMEFRAME_DAILY, start_daily, end_dt)

        if df_daily is None:
            logger.warning(
                "%-6s  daily bars unavailable — daily_atr_abs will be NaN for this symbol.",
                symbol,
            )
            continue

        out_daily = _RAW_DIR / f"{_FILE_PREFIX}{symbol}_daily.parquet"
        df_daily.write_parquet(out_daily, compression="snappy")
        logger.info(
            "%-6s  daily  %7d bars  →  %s",
            symbol,
            len(df_daily),
            out_daily.name,
        )

    logger.info("=" * 70)
    if failed:
        logger.warning("Failed symbols: %s", ", ".join(failed))
        sys.exit(1)
    logger.info("Harvest complete — all symbols written to %s", _RAW_DIR)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    harvest()
