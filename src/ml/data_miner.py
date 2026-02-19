"""
Historical data mining pipeline for ML training datasets.

Uses the :func:`data.factory.get_market_provider` factory to fetch
multi-year, 1-minute OHLCV bars in monthly chunks and persist them as
Parquet files under ``data/raw/{SYMBOL}_1min.parquet``.

Usage (from project root)::

    # Uses whatever DATA_SOURCE is set in .env (default: alpaca)
    python -m ml.data_miner

    # Override to use Polygon for speed
    DATA_SOURCE=polygon python -m ml.data_miner

The resulting Parquet files use the canonical schema:

    timestamp : Datetime(us, UTC)
    open      : Float64
    high      : Float64
    low       : Float64
    close     : Float64
    volume    : Float64
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import polars as pl

# ── path setup (mirrors main.py) ─────────────────────────────────────
# When run as `python -m ml.data_miner` from the project root, `src/`
# is already on sys.path.  When run directly as a script, we need to
# add it explicitly so `from data.factory import ...` resolves.
_SRC_DIR = Path(__file__).resolve().parent.parent  # src/
_PROJECT_ROOT = _SRC_DIR.parent  # build-A-bot/
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from data.factory import get_market_provider  # noqa: E402
from data.market_provider import MarketDataProvider  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── canonical schema (single source of truth for empty-frame creation) ─
_BAR_SCHEMA = {
    "timestamp": pl.Datetime(time_unit="us", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}

# Defaults
_DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "data" / "raw"
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 5  # seconds; doubles each attempt
_INTER_CHUNK_DELAY = 1.0  # polite pause between API calls (seconds)


# ─────────────────────────────────────────────────────────────────────
# DataMiner
# ─────────────────────────────────────────────────────────────────────
class DataMiner:
    """
    Bulk historical data fetcher.

    Iterates a date range in calendar-month chunks, fetches 1-minute
    bars from the configured :class:`MarketDataProvider`, concatenates
    them, and writes one Parquet file per symbol.

    Parameters
    ----------
    provider : MarketDataProvider
        The data source (obtained from ``get_market_provider()``).
    output_dir : Path | str, optional
        Directory for Parquet output.  Created automatically if absent.
        Defaults to ``<project_root>/data/raw/``.
    max_retries : int, default 3
        Per-chunk retry limit before the chunk is skipped.
    """

    def __init__(
        self,
        provider: MarketDataProvider,
        output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
        max_retries: int = _MAX_RETRIES,
    ):
        self._provider = provider
        self._output_dir = Path(output_dir)
        self._max_retries = max_retries

        # Guarantee the output directory exists
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Output directory: %s", self._output_dir)

    # ── public API ────────────────────────────────────────────────────

    def mine_history(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Fetch 1-minute bars for every *symbol* from *start_date* to
        *end_date* and write ``{symbol}_1min.parquet`` files.

        Parameters
        ----------
        symbols : list[str]
            Ticker symbols to download.
        start_date : datetime
            Inclusive start (UTC-aware recommended).
        end_date : datetime
            Exclusive end   (UTC-aware recommended).
        """
        # Normalise to UTC if naive
        start_date = self._ensure_utc(start_date)
        end_date = self._ensure_utc(end_date)

        logger.info(
            "Mining %d symbol(s) from %s to %s",
            len(symbols),
            start_date.date(),
            end_date.date(),
        )

        for symbol in symbols:
            self._mine_symbol(symbol, start_date, end_date)

    # ── per-symbol pipeline ───────────────────────────────────────────

    def _mine_symbol(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> None:
        """Download, concatenate, deduplicate, and save one symbol."""
        logger.info("=== %s  %s -> %s ===", symbol, start.date(), end.date())

        chunks: list[pl.DataFrame] = []
        months = list(self._month_ranges(start, end))
        total = len(months)

        for idx, (m_start, m_end) in enumerate(months, 1):
            label = m_start.strftime("%Y-%m")
            logger.info("  [%d/%d] %s  %s ...", idx, total, symbol, label)

            chunk = self._fetch_chunk_with_retry(symbol, m_start, m_end)

            if chunk is not None and not chunk.is_empty():
                chunks.append(chunk)
                logger.info(
                    "  [%d/%d] %s  %s  => %d bars",
                    idx,
                    total,
                    symbol,
                    label,
                    len(chunk),
                )
            else:
                logger.warning(
                    "  [%d/%d] %s  %s  => 0 bars (skipped)",
                    idx,
                    total,
                    symbol,
                    label,
                )

            # Polite inter-request pause
            time.sleep(_INTER_CHUNK_DELAY)

        if not chunks:
            logger.warning("%s: no data collected. Nothing to write.", symbol)
            return

        # Concatenate all monthly frames
        df = pl.concat(chunks, how="vertical_relaxed")

        # Deduplicate & sort (month boundaries may overlap by a bar)
        df = df.unique(subset=["timestamp"]).sort("timestamp")

        # Write
        out_path = self._output_dir / f"{symbol}_1min.parquet"
        df.write_parquet(out_path)
        logger.info(
            "%s: wrote %d bars to %s (%.2f MB)",
            symbol,
            len(df),
            out_path,
            out_path.stat().st_size / (1024 * 1024),
        )

    # ── fetch with retry ──────────────────────────────────────────────

    def _fetch_chunk_with_retry(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> pl.DataFrame | None:
        """
        Call ``provider.get_historical_bars`` with exponential-backoff
        retry on failure.

        Returns the DataFrame on success, or ``None`` after exhausting
        retries.
        """
        for attempt in range(1, self._max_retries + 1):
            try:
                df = self._provider.get_historical_bars(
                    symbol=symbol,
                    timeframe_minutes=1,
                    start=start,
                    end=end,
                )
                # Providers return an empty schema-typed DF on error;
                # treat that as success (just no data for this period).
                return df

            except Exception as exc:
                backoff = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "  Attempt %d/%d for %s failed: %s  (retrying in %ds)",
                    attempt,
                    self._max_retries,
                    symbol,
                    exc,
                    backoff,
                )
                time.sleep(backoff)

        logger.error(
            "  All %d attempts failed for %s [%s -> %s]. Skipping chunk.",
            self._max_retries,
            symbol,
            start.date(),
            end.date(),
        )
        return None

    # ── date helpers ──────────────────────────────────────────────────

    @staticmethod
    def _month_ranges(start: datetime, end: datetime):
        """
        Yield ``(month_start, month_end)`` pairs covering *start* to
        *end* in calendar-month increments.  Both bounds are UTC-aware.
        """
        cursor = start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        while cursor < end:
            # First day of next month
            if cursor.month == 12:
                next_month = cursor.replace(year=cursor.year + 1, month=1)
            else:
                next_month = cursor.replace(month=cursor.month + 1)

            # Clamp to the user-requested window
            m_start = max(cursor, start)
            m_end = min(next_month, end)

            yield m_start, m_end
            cursor = next_month

    @staticmethod
    def _ensure_utc(dt: datetime) -> datetime:
        """Attach UTC if the datetime is naive."""
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env")

    symbols = ["SPY", "QQQ", "IWM", "NVDA", "AMD", "MSFT", "AAPL"]

    start_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    end_date = datetime.now(timezone.utc)

    logger.info("DATA_SOURCE = %s", os.getenv("DATA_SOURCE", "alpaca"))

    provider = get_market_provider()
    miner = DataMiner(provider=provider)
    miner.mine_history(symbols, start_date, end_date)


if __name__ == "__main__":
    main()
