"""
V4 Investor Feature Pipeline — momentum, macro, fundamental, and target.

Ingests the aligned daily Parquet produced by investor_data_miner.py and
engineers three feature families plus the cross-sectional ranking target:

  1. Momentum         — 3-month (63d), 6-month (126d), 12-month (252d) trailing returns
  2. Macro trends     — VIX and 10Y Yield: 20-day SMA and rate-of-change
  3. Fundamental      — Derived margin ratios from quarterly income-statement data
  4. Target           — Binary 1 if asset is in the top quintile (Q5) of
                        60-day cross-sectional forward returns, 0 otherwise

Usage:
    pipenv run python scripts/investor_feature_pipeline.py

Input:
    data/raw/v4_investor_data.parquet

Output:
    data/processed/v4_training_features.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))

_INPUT_PATH = _PROJECT_ROOT / "data" / "raw" / "v4_investor_data.parquet"
_OUTPUT_PATH = _PROJECT_ROOT / "data" / "processed" / "v4_training_features.parquet"

# ── logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────
# Momentum lookback windows in trading days
MOM_WINDOWS: dict[str, int] = {
    "mom_3m": 63,
    "mom_6m": 126,
    "mom_12m": 252,
}

# Macro rolling window (trading days)
MACRO_WINDOW: int = 20

# Forward return horizon (trading days)
FORWARD_DAYS: int = 60

# Fundamental line items needed for margin ratios.
# These are sourced from the quarterly_financials join in the miner.
# Rows outside the fundamentals window will remain NaN —
# LightGBM handles missing values natively.
_NUMERATOR_COLS: dict[str, str] = {
    "gross_margin": "Gross Profit",
    "operating_margin": "Operating Income",
    "net_margin": "Net Income",
    "ebitda_margin": "EBITDA",
}
_REVENUE_COL: str = "Total Revenue"


# ─────────────────────────────────────────────────────────────────────
# Target helper
# ─────────────────────────────────────────────────────────────────────

def _top_quintile_label(series: pd.Series) -> pd.Series:
    """
    Cross-sectional top-quintile classifier for a single date group.

    Given a Series of forward returns (one per symbol on a given date),
    returns 1 for symbols in the highest return quintile (Q5) and 0 for
    all others.

    Edge-case handling
    ------------------
    * Fewer than 2 non-null values   → all NaN (date is unusable)
    * qcut produces < 5 bins (ties)  → use the observed maximum bin as
                                       the "top" so at least one symbol
                                       always receives label 1
    * qcut raises for any reason     → fall back to rank-percentile ≥ 80%
    """
    result = pd.Series(float("nan"), index=series.index)
    valid_mask = series.notna()
    valid = series[valid_mask]

    if len(valid) < 2:
        return result

    try:
        quintiles = pd.qcut(valid, q=5, labels=False, duplicates="drop")
        top_bin = int(quintiles.max())
        result[valid_mask] = (quintiles == top_bin).astype(float)
    except Exception:
        # Fallback: percentile rank — top 20% gets label 1
        ranks = valid.rank(pct=True, ascending=True)
        result[valid_mask] = (ranks >= 0.8).astype(float)

    return result


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 70)
    logger.info("V4 Investor Feature Pipeline")
    logger.info("Input  : %s", _INPUT_PATH)
    logger.info("Output : %s", _OUTPUT_PATH)
    logger.info("=" * 70)

    # ── Load & normalise ─────────────────────────────────────────────
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{_INPUT_PATH} not found. Run scripts/investor_data_miner.py first."
        )

    df = pd.read_parquet(_INPUT_PATH)
    logger.info("Loaded raw data: %d rows × %d columns", *df.shape)

    # Ensure 'date' is a regular column for groupby operations
    if df.index.name == "date":
        df = df.reset_index()

    # Canonical sort: symbol primary, date secondary (required for correct
    # pct_change / shift calculations within each symbol's time series)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    logger.info("Symbols: %s", sorted(df["symbol"].unique().tolist()))

    # ── Stage 1 — Momentum features ─────────────────────────────────
    logger.info("\n[Stage 1/4] Computing momentum features ...")
    for col_name, window in MOM_WINDOWS.items():
        df[col_name] = df.groupby("symbol")["close"].pct_change(periods=window)
        n_valid = df[col_name].notna().sum()
        logger.info(
            "  %-10s  window=%3dd  non-null rows: %d / %d",
            col_name, window, n_valid, len(df),
        )

    # ── Stage 2 — Macro trend features ──────────────────────────────
    logger.info("\n[Stage 2/4] Computing macro trend features ...")

    for raw_col, sma_col, roc_col in [
        ("VIX", "vix_sma_20", "vix_roc_20"),
        ("10Y_YIELD", "yield_sma_20", "yield_roc_20"),
    ]:
        df[sma_col] = df.groupby("symbol")[raw_col].transform(
            lambda x: x.rolling(window=MACRO_WINDOW, min_periods=MACRO_WINDOW).mean()
        )
        df[roc_col] = df.groupby("symbol")[raw_col].transform(
            lambda x: x.pct_change(periods=MACRO_WINDOW)
        )
        logger.info(
            "  %-15s  SMA non-null: %d | ROC non-null: %d",
            raw_col,
            df[sma_col].notna().sum(),
            df[roc_col].notna().sum(),
        )

    # ── Stage 3 — Fundamental ratio features ────────────────────────
    logger.info("\n[Stage 3/4] Computing fundamental ratio features ...")

    if _REVENUE_COL in df.columns:
        for ratio_col, numerator_col in _NUMERATOR_COLS.items():
            if numerator_col in df.columns:
                # Avoid division by zero; result is NaN where either input is NaN/zero
                df[ratio_col] = df[numerator_col] / df[_REVENUE_COL].replace(0, float("nan"))
                n_valid = df[ratio_col].notna().sum()
                logger.info("  %-20s  non-null rows: %d / %d", ratio_col, n_valid, len(df))
            else:
                logger.warning("  '%s' not found — skipping %s.", numerator_col, ratio_col)
    else:
        logger.warning(
            "  '%s' column missing — skipping all margin ratios.", _REVENUE_COL
        )

    # ── Stage 4 — Forward return and cross-sectional target ──────────
    logger.info("\n[Stage 4/4] Computing forward return and cross-sectional target ...")

    # 60-trading-day forward return: close(t+60) / close(t) - 1
    df["forward_return_60d"] = df.groupby("symbol")["close"].transform(
        lambda x: x.shift(-FORWARD_DAYS) / x - 1
    )
    n_fwd = df["forward_return_60d"].notna().sum()
    logger.info(
        "  forward_return_60d: %d non-null (embargo: %d rows with NaN)",
        n_fwd, len(df) - n_fwd,
    )

    # Cross-sectional target: 1 if top quintile on that date, 0 otherwise.
    # Groups by date — each daily slice contains one row per symbol.
    # _top_quintile_label handles edge cases (ties, small groups).
    df["target_top_quintile"] = (
        df.groupby("date")["forward_return_60d"]
        .transform(_top_quintile_label)
    )

    n_target = df["target_top_quintile"].notna().sum()
    n_positive = (df["target_top_quintile"] == 1).sum()
    logger.info(
        "  target_top_quintile: %d labelled rows | positive rate: %.1f%%",
        n_target,
        n_positive / n_target * 100 if n_target > 0 else 0,
    )

    # ── Clean-up: drop the embargo period ────────────────────────────
    # Rows where target_top_quintile is NaN are the final ~60 trading
    # days where we cannot compute the forward return.  These must be
    # excluded from training.  Feature NaNs (momentum warm-up, sparse
    # fundamentals) are intentionally retained — LightGBM handles them.
    pre_drop = len(df)
    df = df.dropna(subset=["target_top_quintile"])
    logger.info(
        "\nDropped %d embargo rows (NaN target). Remaining: %d rows.",
        pre_drop - len(df), len(df),
    )

    # ── Final label distribution audit ───────────────────────────────
    total = len(df)
    pos = int((df["target_top_quintile"] == 1).sum())
    neg = int((df["target_top_quintile"] == 0).sum())
    logger.info(
        "Target distribution — positive (Q5): %d (%.1f%%) | "
        "negative: %d (%.1f%%)",
        pos, pos / total * 100,
        neg, neg / total * 100,
    )

    # Per-symbol target rate
    logger.info("Per-symbol positive rate:")
    for sym, grp in df.groupby("symbol"):
        rate = (grp["target_top_quintile"] == 1).mean() * 100
        logger.info("  %-6s  %.1f%%", sym, rate)

    # ── Save ─────────────────────────────────────────────────────────
    df = df.set_index("date")
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUTPUT_PATH, index=True)

    size_mb = _OUTPUT_PATH.stat().st_size / (1024 * 1024)
    logger.info(
        "\nSaved → %s  (%d rows × %d cols, %.2f MB)",
        _OUTPUT_PATH, *df.shape, size_mb,
    )
    logger.info("V4 feature pipeline complete.")


if __name__ == "__main__":
    main()
