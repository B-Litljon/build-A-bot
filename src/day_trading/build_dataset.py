"""
src/day_trading/build_dataset.py
Master Training Dataset Compiler — Universal Scalper V4.0

Loads raw 5-minute and daily Parquet files produced by harvester_5m.py,
runs the full feature-engineering and target-labeling pipeline for each
symbol, concatenates the results, and writes the final training dataset.

Output
------
data/processed/dt_training_data.parquet

  Rows    : all clean 5-minute bars across the 7-symbol universe
            (~125 k rows after TA-Lib warmup + EOD null drop)
  Columns : OHLCV metadata + 22 features + 3 target columns

Usage
-----
    python -m src.day_trading.build_dataset

    # or from the project root:
    PYTHONPATH=src python src/day_trading/build_dataset.py

Why per-symbol pipeline execution?
-----------------------------------
TA-Lib functions (RSI, PPO, NATR, …) operate on a 1-D numpy array.
If every symbol's bars were concatenated into one array, the last bar of
TSLA and the first bar of NVDA would be treated as consecutive bars in a
single time-series, producing meaningless indicator values at every symbol
boundary.  Processing one symbol at a time eliminates this contamination.

DayTradeDailyJoin is constructed ONCE from the combined daily DataFrame
for all symbols.  Its _precompute() method uses `.over("symbol")` for
gap_pct shift arithmetic, so it correctly handles the full multi-symbol
table.  Each per-symbol call to `.generate()` then joins only the daily
rows relevant to that symbol via `by="symbol"` in join_asof.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

# ── Project paths ──────────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent  # src/
_PROJECT_ROOT = _SRC_DIR.parent  # build-A-bot/
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Internal imports (requires PYTHONPATH=src or python -m execution) ─────────
from ml.feature_pipeline import FeaturePipeline  # noqa: E402
from day_trading.features import (  # noqa: E402
    DAY_TRADE_FEATURE_COLS,
    DayTradeBaseFeatures,
    DayTradeDailyJoin,
    DayTradeIntradayFeatures,
)
from day_trading.targets import DayTradeTargets  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

UNIVERSE: List[str] = [
    "SPY",
    "QQQ",
    "TSLA",
    "NVDA",
    "AAPL",
    "AMD",
    "MSFT",
]

OUTPUT_PATH = _PROCESSED_DIR / "dt_training_data.parquet"

# Columns to drop from the final dataset — they are pipeline intermediates or
# metadata that the model should not receive as features.  They remain in the
# file as human-readable metadata but are NOT part of DAY_TRADE_FEATURE_COLS.
_METADATA_COLS = [
    "open",
    "high",
    "low",  # OHLCV raw (close/volume kept for audit)
    "day_open",  # intermediate for trend_vs_open
    "vwap",  # intermediate for vwap_dist
    "intraday_range",  # intermediate for range_exhaustion
    "bb_upper",
    "bb_middle",  # should have been dropped by generator
    "bb_lower",
    "sma_50",  # (defensive cleanup)
]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _load_raw(symbol: str, suffix: str) -> Optional[pl.DataFrame]:
    """
    Load data/raw/dt_{symbol}_{suffix}.parquet.
    Returns None if the file does not exist or cannot be parsed.
    """
    path = _RAW_DIR / f"dt_{symbol}_{suffix}.parquet"
    if not path.exists():
        logger.error("Missing raw file: %s", path)
        return None
    try:
        df = pl.read_parquet(path)
        if "symbol" not in df.columns:
            df = df.with_columns(pl.lit(symbol).alias("symbol"))
        return df
    except Exception as exc:
        logger.error("Failed to read %s: %s", path, exc)
        return None


def _target_balance(df: pl.DataFrame, col: str) -> str:
    """
    Return a compact balance string for a binary Int8 target column.
    Example: 'pos=12,345 (41.2%)  neg=17,655 (58.8%)'
    """
    if col not in df.columns:
        return "column not found"
    counts = df[col].value_counts().sort(col)
    total = len(df)
    parts = []
    for row in counts.iter_rows(named=True):
        label = "pos" if row[col] == 1 else "neg"
        n = row["count"]
        pct = 100.0 * n / total if total > 0 else 0.0
        parts.append(f"{label}={n:,} ({pct:.1f}%)")
    return "  ".join(parts)


def _drop_if_present(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    """Drop columns that exist in df, silently ignore missing ones."""
    to_drop = [c for c in cols if c in df.columns]
    return df.drop(to_drop) if to_drop else df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def build(universe: List[str] = UNIVERSE) -> pl.DataFrame:
    """
    Full pipeline: load → engineer features → label targets → clean → save.

    Returns the final processed DataFrame (also written to OUTPUT_PATH).
    Raises SystemExit(1) if any critical step fails.
    """
    t0 = time.perf_counter()

    logger.info("=" * 70)
    logger.info("DAY TRADE DATASET COMPILER  —  Universal Scalper V4.0")
    logger.info("=" * 70)
    logger.info("Universe : %s", ", ".join(universe))
    logger.info("Output   : %s", OUTPUT_PATH)
    logger.info("-" * 70)

    # ── Step 1: Load ALL daily bars (needed by DayTradeDailyJoin constructor) ──
    #
    # DayTradeDailyJoin._precompute() runs NATR(14) and gap_pct over the full
    # multi-symbol daily DataFrame with `.over("symbol")` partitioning.
    # All symbols must be present at construction time; the join is symbol-aware.
    logger.info("[1/4] Loading daily bars for all symbols …")

    daily_frames: List[pl.DataFrame] = []
    for sym in universe:
        df_daily = _load_raw(sym, "daily")
        if df_daily is None:
            logger.error("Cannot build dataset without daily bars for %s. Abort.", sym)
            sys.exit(1)
        daily_frames.append(df_daily)

    daily_df = pl.concat(daily_frames, how="vertical_relaxed").sort(
        ["symbol", "timestamp"]
    )
    logger.info(
        "    Combined daily DataFrame: %d rows across %d symbols",
        len(daily_df),
        daily_df["symbol"].n_unique(),
    )

    # ── Step 2: Build the pipeline (constructed once, reused per symbol) ────────
    #
    # DayTradeDailyJoin is initialised here with the full daily_df.  Its
    # _precompute() runs once and stores the join table; subsequent calls
    # to .generate() are cheap join operations.
    logger.info("[2/4] Building feature pipeline …")

    pipeline = FeaturePipeline(
        feature_generators=[
            DayTradeBaseFeatures(),
            DayTradeDailyJoin(daily_df),  # pre-computes daily scalars here
            DayTradeIntradayFeatures(),
        ],
        target_generator=DayTradeTargets(),
    )
    logger.info("    Pipeline ready: 3 feature generators + DayTradeTargets")

    # ── Step 3: Process each symbol ─────────────────────────────────────────────
    #
    # Each symbol is passed through the pipeline independently.
    # Reason: DayTradeBaseFeatures calls talib.RSI/PPO/NATR on numpy arrays.
    # Concatenating all symbols into one array would treat the last bar of
    # symbol A and the first bar of symbol B as consecutive — corrupting the
    # indicator warmup window at every symbol boundary.
    logger.info("[3/4] Processing symbols …")
    logger.info("")

    processed_frames: List[pl.DataFrame] = []
    per_symbol_stats: Dict[str, Dict] = {}

    for sym in universe:
        sym_t0 = time.perf_counter()

        df_5min = _load_raw(sym, "5min")
        if df_5min is None:
            logger.warning("Skipping %s — missing 5-minute file.", sym)
            continue

        raw_rows = len(df_5min)

        try:
            df_processed = pipeline.run(df_5min)
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", sym, exc, exc_info=True)
            continue

        # Defensive: strip any intermediate columns the generators left behind
        df_processed = _drop_if_present(df_processed, _METADATA_COLS)

        clean_rows = len(df_processed)
        elapsed_ms = (time.perf_counter() - sym_t0) * 1000

        per_symbol_stats[sym] = {
            "raw_rows": raw_rows,
            "clean_rows": clean_rows,
            "dropped": raw_rows - clean_rows,
        }

        logger.info(
            "    %-6s  raw=%7d  clean=%7d  dropped=%5d  (%.0f ms)",
            sym,
            raw_rows,
            clean_rows,
            raw_rows - clean_rows,
            elapsed_ms,
        )

        processed_frames.append(df_processed)

    if not processed_frames:
        logger.error("No symbols produced any data. Aborting.")
        sys.exit(1)

    # ── Step 4: Concatenate, sort, and save ──────────────────────────────────
    logger.info("")
    logger.info("[4/4] Concatenating, sorting, writing …")

    combined = pl.concat(processed_frames, how="vertical_relaxed").sort(
        ["symbol", "timestamp"]
    )

    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(OUTPUT_PATH, compression="snappy")
    file_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)

    elapsed_total = time.perf_counter() - t0

    # ── Diagnostics ──────────────────────────────────────────────────────────
    _print_summary(combined, per_symbol_stats, file_mb, elapsed_total)

    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY PRINTER
# ═══════════════════════════════════════════════════════════════════════════════


def _print_summary(
    df: pl.DataFrame,
    sym_stats: Dict[str, Dict],
    file_mb: float,
    elapsed_s: float,
) -> None:
    """Print a structured post-build diagnostics block."""
    bar = "=" * 70
    thin = "-" * 70

    logger.info("")
    logger.info(bar)
    logger.info("BUILD COMPLETE")
    logger.info(bar)

    # ── Dataset shape ─────────────────────────────────────────────────────────
    n_rows, n_cols = df.shape
    n_feat = len([c for c in DAY_TRADE_FEATURE_COLS if c in df.columns])
    n_found = len(DAY_TRADE_FEATURE_COLS)
    logger.info("Shape      : %d rows × %d columns", n_rows, n_cols)
    logger.info(
        "Features   : %d / %d expected feature columns present",
        n_feat,
        n_found,
    )

    # Warn loudly if any expected feature columns are missing
    missing_feats = [c for c in DAY_TRADE_FEATURE_COLS if c not in df.columns]
    if missing_feats:
        logger.warning("MISSING FEATURE COLUMNS: %s", missing_feats)

    # ── Date range ────────────────────────────────────────────────────────────
    ts_min = df["timestamp"].min()
    ts_max = df["timestamp"].max()
    logger.info("Date range : %s  →  %s", ts_min, ts_max)

    # ── Per-symbol row counts ─────────────────────────────────────────────────
    logger.info(thin)
    logger.info("Per-symbol row counts (after clean):")
    sym_counts = df.group_by("symbol").len().sort("symbol")
    for row in sym_counts.iter_rows(named=True):
        raw = sym_stats.get(row["symbol"], {}).get("raw_rows", 0)
        clean = sym_stats.get(row["symbol"], {}).get("clean_rows", 0)
        pct = 100.0 * clean / raw if raw > 0 else 0.0
        logger.info(
            "    %-6s : %7d rows  (%.1f%% of %d raw)",
            row["symbol"],
            row["len"],
            pct,
            raw,
        )

    # ── Target class balance ──────────────────────────────────────────────────
    logger.info(thin)
    logger.info("Target class balance:")
    for col in ("angel_target", "devil_target", "devil_target_macro"):
        bal = _target_balance(df, col)
        logger.info("    %-22s : %s", col, bal)

    # ── Angel / Devil joint approval rate ────────────────────────────────────
    if "angel_target" in df.columns and "devil_target" in df.columns:
        joint = ((df["angel_target"] == 1) & (df["devil_target"] == 1)).sum()
        logger.info(
            "    %-22s : %d (%.1f%%) — dual-approval trades",
            "angel AND devil",
            joint,
            100.0 * joint / n_rows,
        )

    # ── Schema snapshot (feature columns only) ───────────────────────────────
    logger.info(thin)
    logger.info("Feature column dtypes:")
    for col in DAY_TRADE_FEATURE_COLS:
        dtype = df.schema.get(col, "MISSING")
        marker = "" if str(dtype) != "MISSING" else "  ← MISSING"
        logger.info("    %-25s %s%s", col, dtype, marker)

    # ── File info ─────────────────────────────────────────────────────────────
    logger.info(thin)
    logger.info("Output     : %s", OUTPUT_PATH)
    logger.info("File size  : %.2f MB", file_mb)
    logger.info("Wall time  : %.1f s", elapsed_s)
    logger.info(bar)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    build()
