"""
V4 Investor Data Miner — daily OHLCV + macro + fundamental alignment.

Fetches 5 years of daily adjusted price data, quarterly income-statement
fundamentals (with a 45-day look-ahead bias prevention shift to simulate
SEC reporting lag), and daily macro series (VIX, 10Y Treasury yield).
Merges all three layers into a single point-in-time-safe, daily-indexed
pandas DataFrame and saves as Parquet.

Usage:
    pipenv run python scripts/investor_data_miner.py

Output:
    data/raw/v4_investor_data.parquet

Data layers:
    1. Daily OHLCV  — yf.download() via direct yfinance call
                      (YahooDataProvider targets intraday; daily bars
                       are fetched directly here at the orchestration layer)
    2. Macro        — YFinanceMacroProvider  (src/data/providers/yf_macro.py)
    3. Fundamentals — YFinanceFundamentalProvider  (src/data/providers/)
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

# ── path setup (mirrors run_paper_live.py) ────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))

load_dotenv(_PROJECT_ROOT / ".env")

from data.providers.yf_fundamentals import YFinanceFundamentalProvider  # noqa: E402
from data.providers.yf_macro import YFinanceMacroProvider  # noqa: E402

# ── logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("yfinance").setLevel(logging.WARNING)

# ── V4 universe configuration ─────────────────────────────────────────
UNIVERSE: list[str] = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "WMT", "JNJ"]
MACRO_INDICATORS: list[str] = ["VIX", "10Y_YIELD"]

_END_DATE = datetime.now(timezone.utc)
_START_DATE = _END_DATE.replace(year=_END_DATE.year - 5)
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "raw" / "v4_investor_data.parquet"

# SEC 10-Q/10-K filing window: companies have up to 40–45 days after
# a quarter closes to file.  We use 45 days as a conservative margin so
# that no data from a quarterly report leaks into the model before it
# was publicly available.
_FUNDAMENTAL_LAG_DAYS: int = 45


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _fetch_daily_ohlcv(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetch daily adjusted OHLCV bars for one symbol via yf.download().

    Returns a DataFrame with a UTC-aware DatetimeIndex named "date" and
    columns: open, high, low, close, volume, symbol.
    Returns an empty DataFrame on failure.
    """
    raw = yf.download(
        tickers=symbol,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=True,
    )

    if raw is None or raw.empty:
        logger.warning("  No OHLCV data for %s.", symbol)
        return pd.DataFrame()

    # yfinance 1.x may return a MultiIndex (Price, Ticker) — flatten to
    # simple column labels regardless of version
    if hasattr(raw.columns, "nlevels") and raw.columns.nlevels > 1:
        raw.columns = [
            col[0] if isinstance(col, tuple) else col for col in raw.columns
        ]

    df = raw.rename(columns=str.lower).copy()
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()

    # Ensure UTC-aware DatetimeIndex
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "date"
    df["symbol"] = symbol
    return df


def _align_macro(price_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex macro DataFrame onto the daily price index, then forward-fill.

    Uses reindex + ffill rather than merge_asof because macro series are
    already daily and only need alignment (no irregular spacing).
    """
    if macro_df.empty:
        return price_df

    macro_aligned = macro_df.reindex(price_df.index, method="ffill")
    return price_df.join(macro_aligned, how="left")


def _align_fundamentals(
    price_df: pd.DataFrame,
    fundamentals_df: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    """
    Merge lagged quarterly fundamentals onto the daily price index.

    The ``fundamentals_df`` index is already shifted forward by
    _FUNDAMENTAL_LAG_DAYS days by the caller (the 45-day safety margin
    is applied before this function is called — not inside it).

    Uses merge_asof(direction="backward") so each trading day receives
    the most recent quarterly report whose lagged date is <= that day.
    This is the correct point-in-time join: the model only sees data
    that was publicly available on the day being evaluated.

    Both indexes must be UTC-aware before calling this function.
    """
    if fundamentals_df.empty:
        logger.warning("  No fundamental data for %s — skipping join.", symbol)
        return price_df

    # merge_asof requires both sides sorted ascending on the key
    left = price_df.reset_index().sort_values("date")
    right = (
        fundamentals_df
        .reset_index()
        .rename(columns={"period_end": "date"})
        .sort_values("date")
    )

    # pandas 3.x requires identical datetime64 units on the merge key.
    # yfinance / tz_localize may produce different resolutions (s vs us).
    # Cast right to match left's exact dtype to avoid MergeError.
    right["date"] = right["date"].astype(left["date"].dtype)

    merged = pd.merge_asof(left, right, on="date", direction="backward")
    return merged.set_index("date")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 70)
    logger.info("V4 Investor Data Miner")
    logger.info("Universe : %s", UNIVERSE)
    logger.info("Window   : %s → %s", _START_DATE.date(), _END_DATE.date())
    logger.info("Lag      : %d days (SEC reporting safety margin)", _FUNDAMENTAL_LAG_DAYS)
    logger.info("=" * 70)

    fundamental_provider = YFinanceFundamentalProvider()
    macro_provider = YFinanceMacroProvider()

    # ── Stage 1 — Macro series ────────────────────────────────────────
    logger.info("\n[Stage 1/3] Fetching macro series ...")
    macro_series: dict[str, pd.Series] = {}

    for indicator in MACRO_INDICATORS:
        series = macro_provider.get_macro_series(
            indicator_name=indicator,
            start_date=_START_DATE.strftime("%Y-%m-%d"),
        )
        if series.empty:
            logger.warning("  '%s' returned empty — skipped.", indicator)
            continue

        # ⚠️  Yahoo Finance reports the 10-Year Treasury yield as its
        # raw index value (e.g. 44.5).  Divide by 10 to get the actual
        # percentage yield (4.45%).
        if indicator == "10Y_YIELD":
            raw_sample = float(series.iloc[0])
            series = series / 10.0
            logger.info(
                "  10Y_YIELD: applied ÷10 correction  "
                "(raw sample %.3f → corrected %.4f%%)",
                raw_sample,
                float(series.iloc[0]),
            )

        macro_series[indicator] = series
        logger.info("  %s: %d observations.", indicator, len(series))

    macro_df = pd.DataFrame(macro_series) if macro_series else pd.DataFrame()

    # ── Stage 2 — Per-symbol: OHLCV + fundamentals + merge ───────────
    logger.info("\n[Stage 2/3] Fetching per-symbol data ...")
    symbol_frames: list[pd.DataFrame] = []

    for symbol in UNIVERSE:
        logger.info("\n  ─── %s ───", symbol)

        # --- Daily OHLCV ---
        logger.info("  Fetching daily OHLCV ...")
        price_df = _fetch_daily_ohlcv(symbol, _START_DATE, _END_DATE)
        if price_df.empty:
            logger.warning("  OHLCV empty for %s — symbol skipped.", symbol)
            continue
        logger.info("  OHLCV: %d daily bars.", len(price_df))

        # --- Macro join (same for all symbols; reindex onto each price grid) ---
        if not macro_df.empty:
            price_df = _align_macro(price_df, macro_df)
            logger.info(
                "  Macro joined: %s", [c for c in macro_df.columns]
            )

        # --- Quarterly fundamentals with 45-day look-ahead bias prevention ---
        logger.info("  Fetching quarterly fundamentals ...")
        fundamentals_df = fundamental_provider.get_quarterly_financials(symbol)

        if not fundamentals_df.empty:
            n_quarters = len(fundamentals_df)

            # POINT-IN-TIME SAFETY
            # ─────────────────────────────────────────────────────────
            # Quarterly reports (10-Q / 10-K) are not publicly available
            # on their period-end date.  The SEC allows up to 45 days for
            # large accelerated filers to file a 10-Q.  We shift every
            # fundamental index date forward by exactly 45 days.
            #
            # Example:
            #   Q1 2024 period end  : 2024-03-31
            #   Earliest safe use   : 2024-03-31 + 45d = 2024-05-15
            #
            # merge_asof(direction="backward") then ensures a daily row
            # dated 2024-05-14 sees no Q1 2024 data — only Q4 2023.
            # ─────────────────────────────────────────────────────────
            fundamentals_df.index = (
                fundamentals_df.index + pd.Timedelta(days=_FUNDAMENTAL_LAG_DAYS)
            )

            # Ensure UTC-aware so merge_asof doesn't raise a tz mismatch
            if fundamentals_df.index.tz is None:
                fundamentals_df.index = fundamentals_df.index.tz_localize("UTC")
            else:
                fundamentals_df.index = fundamentals_df.index.tz_convert("UTC")

            # Rename to "period_end" so the merge key is self-documenting
            fundamentals_df.index.name = "period_end"

            logger.info(
                "  Fundamentals: %d quarters | "
                "earliest safe date after 45d lag: %s",
                n_quarters,
                fundamentals_df.index.min().date(),
            )

            price_df = _align_fundamentals(price_df, fundamentals_df, symbol)

        symbol_frames.append(price_df)

    if not symbol_frames:
        logger.error("No symbol data collected — aborting.")
        return

    # ── Stage 3 — Concatenate and save ───────────────────────────────
    logger.info("\n[Stage 3/3] Concatenating and saving ...")

    combined = (
        pd.concat(symbol_frames, axis=0)
        .reset_index()                              # date → column for sort
        .sort_values(["symbol", "date"])
        .set_index("date")
    )

    n_rows, n_cols = combined.shape
    n_symbols = combined["symbol"].nunique()
    logger.info(
        "Final dataset: %d rows × %d columns | %d symbols",
        n_rows, n_cols, n_symbols,
    )
    logger.info("Columns (%d): %s", n_cols, list(combined.columns))

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(_OUTPUT_PATH, index=True)

    size_mb = _OUTPUT_PATH.stat().st_size / (1024 * 1024)
    logger.info("\nSaved → %s  (%.2f MB)", _OUTPUT_PATH, size_mb)
    logger.info("V4 data miner complete.")


if __name__ == "__main__":
    main()
