"""fetch_training_data (bars + spread samples from the provider) and _split_holdout
(chronological tail carve, carved BEFORE any feature engineering).
Split out of core/retrainer.py on 2026-09-16.
"""
from __future__ import annotations

from ._common import (
    DAYS_BACK,
    List,
    MarketDataProvider,
    Optional,
    Tuple,
    datetime,
    logger,
    os,
    pl,
    time,
    timedelta,
    timezone,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ALPACA DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════


def fetch_training_data(
    provider: MarketDataProvider,
    symbols: List[str],
    days_back: int = DAYS_BACK,
    timeframe_minutes: int = 1,
) -> pl.DataFrame:
    """
    Fetch historical 1-minute bars for training using the unified provider.

    Args:
        provider: MarketDataProvider instance
        symbols: List of symbols to fetch
        days_back: Number of days to fetch (default: 60)

    Returns:
        Polars DataFrame with OHLCV data for all symbols
    """
    logger.info("=" * 70)
    logger.info("FETCHING TRAINING DATA")
    logger.info("=" * 70)

    # Use timezone-aware UTC datetime. RETRAIN_END_DATE (YYYY-MM-DD) lets
    # us shift the window backward for soak-readiness reruns — the audit
    # report flagged single-window results as a deployment risk.
    end_override = os.getenv("RETRAIN_END_DATE", "").strip()
    if end_override:
        end_date = datetime.strptime(end_override, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
        logger.info(f"RETRAIN_END_DATE override active: {end_date.date()}")
    else:
        end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=days_back)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Timeframe: {timeframe_minutes}-minute bars")

    all_frames: List[pl.DataFrame] = []
    failed_symbols: List[str] = []

    # OANDA intermittently rejects burst requests with a 401 ("Insufficient
    # authorization") that succeeds seconds later; the provider returns an
    # empty frame on any error, so retry-on-empty covers both cases.
    fetch_retries = int(os.getenv("RETRAIN_FETCH_RETRIES", "3"))

    for ticker in symbols:
        df = None
        for attempt in range(1, fetch_retries + 1):
            try:
                df = provider.get_historical_bars(
                    symbol=ticker,
                    timeframe_minutes=timeframe_minutes,
                    start=start_date,
                    end=end_date,
                )
            except Exception as e:
                df = None
                logger.error(f"Error fetching {ticker} (attempt {attempt}/{fetch_retries}): {e}")
            if df is not None and not df.is_empty():
                break
            logger.warning(f"No data returned for {ticker} (attempt {attempt}/{fetch_retries})")
            if attempt < fetch_retries:
                time.sleep(5 * attempt)

        if df is None or df.is_empty():
            failed_symbols.append(ticker)
            continue

        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]

        # Add symbol column if not present
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(ticker).alias("symbol"))

        all_frames.append(df)
        logger.info(f"Fetched {len(df):,} bars for {ticker}")

    if failed_symbols:
        # Training on a silently shrunk basket corrupts the experiment AND the
        # metadata sidecar (trained_on_symbols would claim the full list).
        raise ValueError(
            f"Fetch failed for {failed_symbols} after {fetch_retries} attempts — "
            "refusing to train on a partial basket"
        )

    if not all_frames:
        raise ValueError("No data fetched for any symbol")

    # Combine all symbols
    combined = pl.concat(all_frames, how="vertical_relaxed")
    combined = combined.sort(["symbol", "timestamp"])

    logger.info(f"Combined dataset: {len(combined):,} total rows")
    return combined


def _split_holdout(
    df: pl.DataFrame, frac: float
) -> Tuple[pl.DataFrame, pl.DataFrame, Optional[Tuple[datetime, datetime]]]:
    """
    Carve a chronologically last holdout slice from raw fetched data.

    The split is by timestamp BEFORE feature engineering, so no indicator,
    label, veto, or model parameter can see across the boundary. Returns
    ``(remainder, holdout, (holdout_start, holdout_end))``. When ``frac`` is 0
    or the holdout would be empty, the holdout frame is empty and the date
    range is None.

    Args:
        df: Raw stacked bars with a ``timestamp`` column.
        frac: Fraction of the chronological span to reserve (0.0–1.0).

    Returns:
        Tuple of (remainder DataFrame, holdout DataFrame, optional date range).
    """
    if frac <= 0.0 or df.is_empty():
        return df, pl.DataFrame(), None

    min_ts = df["timestamp"].min()
    max_ts = df["timestamp"].max()
    span_seconds = (max_ts - min_ts).total_seconds()
    if span_seconds <= 0:
        return df, pl.DataFrame(), None

    holdout_seconds = span_seconds * frac
    holdout_start = max_ts - timedelta(seconds=holdout_seconds)
    # Keep the boundary clean: remainder is strictly before holdout_start,
    # holdout is from holdout_start onward. A bar exactly on the boundary
    # belongs to the holdout.
    remainder = df.filter(pl.col("timestamp") < holdout_start)
    holdout = df.filter(pl.col("timestamp") >= holdout_start)

    if holdout.is_empty():
        return remainder, holdout, None

    return remainder, holdout, (holdout_start, max_ts)
