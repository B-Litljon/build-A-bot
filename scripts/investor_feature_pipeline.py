"""
V4 Investor Feature Pipeline — momentum, macro, fundamental, and target.

Ingests the aligned daily Parquet produced by investor_data_miner.py and
engineers three feature families plus the cross-sectional ranking target:

  1. Momentum         — 3-month (63d), 6-month (126d), 12-month (252d) trailing returns
  2. Macro trends     — VIX and 10Y Yield: 20-day SMA and rate-of-change
  3. Fundamental      — Derived margin ratios from quarterly income-statement data
  4. Target           — Binary 1 if asset is in the top quintile (Q5) of
                        60-day cross-sectional forward returns, 0 otherwise

Modes
-----
Training mode (default)
    Drops the embargo window (rows where forward_return_60d / target are
    NaN — the most recent ~60 trading days).  Output:
        data/processed/v4_training_features.parquet

Inference mode (``--inference``)
    Retains the embargo window so today's row survives — required by the
    monthly portfolio orchestrator.  Output:
        data/processed/v4_inference_features.parquet

Usage:
    pipenv run python scripts/investor_feature_pipeline.py
    pipenv run python scripts/investor_feature_pipeline.py --inference

Input:
    data/raw/v4_investor_data.parquet

Step 2 of the V4 Investor. Turns the merged daily table into model inputs plus
the ranking target.

Glossary:
    MOM_WINDOWS -- the trailing-return lookbacks: 63, 126 and 252 trading days
        (roughly 3, 6 and 12 months). "Momentum" here just means how much the
        stock has already gone up over each span.
    MACRO_WINDOW -- 20 days, the smoothing window applied to VIX and yields so
        the model sees a trend rather than one noisy day.
    FORWARD_DAYS -- 60. The target looks 60 trading days ahead.
    target_top_quintile -- 1 if the stock lands in the best-performing FIFTH of
        the universe over those 60 days, else 0. Note this is a RELATIVE
        (cross-sectional) question -- "did it beat its peers", not "did it go
        up" -- which is what makes the model a ranker rather than a predictor.
    _NUMERATOR_COLS / _REVENUE_COL -- income-statement lines turned into margin
        ratios, so a large company and a small one are comparable.
    --inference flag -- switches the output file and RETAINS the most recent
        rows that training deliberately discards. Training must drop them
        (their 60-day future has not happened yet, so they have no label);
        inference needs exactly those rows, because today is the day being
        predicted.
    _OUTPUT_PATH_TRAINING / _OUTPUT_PATH_INFERENCE --
        data/processed/v4_training_features.parquet and
        v4_inference_features.parquet respectively.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"
sys.path.insert(0, str(_SRC_DIR))

_INPUT_PATH = _PROJECT_ROOT / "data" / "raw" / "v4_investor_data.parquet"
_OUTPUT_PATH_TRAINING = _PROJECT_ROOT / "data" / "processed" / "v4_training_features.parquet"
_OUTPUT_PATH_INFERENCE = _PROJECT_ROOT / "data" / "processed" / "v4_inference_features.parquet"

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

def _top_k_label(series: pd.Series) -> pd.Series:
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

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Engineer V4 investor features and (optionally) the cross-sectional target."
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        help=(
            "Inference mode: retain the 60-day embargo window so today's "
            "row survives (required by the monthly orchestrator). Writes "
            "to v4_inference_features.parquet instead of v4_training_features.parquet."
        ),
    )
    args = parser.parse_args(argv)

    inference_mode: bool = args.inference
    output_path = _OUTPUT_PATH_INFERENCE if inference_mode else _OUTPUT_PATH_TRAINING

    logger.info("=" * 70)
    logger.info("V4 Investor Feature Pipeline")
    logger.info("Mode   : %s", "INFERENCE (embargo retained)" if inference_mode else "TRAINING (embargo dropped)")
    logger.info("Input  : %s", _INPUT_PATH)
    logger.info("Output : %s", output_path)
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

    # ── Stage 1 — Momentum & Volatility features ──────────────────────
    logger.info("\n[Stage 1/4] Computing momentum & volatility features ...")
    for col_name, window in MOM_WINDOWS.items():
        df[col_name] = df.groupby("symbol")["close"].pct_change(periods=window)
        n_valid = df[col_name].notna().sum()
        logger.info(
            "  %-10s  window=%3dd  non-null rows: %d / %d",
            col_name, window, n_valid, len(df),
        )

    # Trailing standard deviations of daily returns
    daily_ret = df.groupby("symbol")["close"].pct_change()
    df["vol_60d"] = daily_ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(60, min_periods=60).std()
    )
    df["vol_120d"] = daily_ret.groupby(df["symbol"]).transform(
        lambda x: x.rolling(120, min_periods=120).std()
    )
    logger.info("  vol_60d     non-null rows: %d / %d", df["vol_60d"].notna().sum(), len(df))
    logger.info("  vol_120d    non-null rows: %d / %d", df["vol_120d"].notna().sum(), len(df))

    # Classic momentum (skipping recent month) & short-term reversal
    df["mom_12_1"] = df.groupby("symbol")["close"].transform(
        lambda x: x.shift(21) / x.shift(252) - 1
    )
    df["reversal_1m"] = df.groupby("symbol")["close"].transform(
        lambda x: x / x.shift(21) - 1
    )
    logger.info("  mom_12_1    non-null rows: %d / %d", df["mom_12_1"].notna().sum(), len(df))
    logger.info("  reversal_1m non-null rows: %d / %d", df["reversal_1m"].notna().sum(), len(df))

    # ── Stage 2 — Fundamental & Quality ratio features ──────────────
    logger.info("\n[Stage 2/4] Computing fundamental & quality ratio features ...")

    # Margin ratios
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

    # Quality ratios
    if "Net Income" in df.columns and "Total Assets" in df.columns:
        df["roa"] = df["Net Income"] / df["Total Assets"].replace(0, float("nan"))
        logger.info("  roa                   non-null rows: %d / %d", df["roa"].notna().sum(), len(df))
    else:
        logger.warning("  Missing 'Net Income' or 'Total Assets' — skipping roa.")
        df["roa"] = float("nan")

    if "Total Liabilities" in df.columns and "Total Equity" in df.columns:
        df["debt_to_equity"] = df["Total Liabilities"] / df["Total Equity"].replace(0, float("nan"))
        logger.info("  debt_to_equity        non-null rows: %d / %d", df["debt_to_equity"].notna().sum(), len(df))
    else:
        logger.warning("  Missing 'Total Liabilities' or 'Total Equity' — skipping debt_to_equity.")
        df["debt_to_equity"] = float("nan")

    if "Gross Profit" in df.columns and "Total Assets" in df.columns:
        df["gross_profitability"] = df["Gross Profit"] / df["Total Assets"].replace(0, float("nan"))
        logger.info("  gross_profitability   non-null rows: %d / %d", df["gross_profitability"].notna().sum(), len(df))
    else:
        logger.warning("  Missing 'Gross Profit' or 'Total Assets' — skipping gross_profitability.")
        df["gross_profitability"] = float("nan")

    # ── Stage 3 — Cross-sectional rank-normalization ──────────────────
    logger.info("\n[Stage 3/4] Performing cross-sectional rank-normalization ...")

    FACTOR_COLS = [
        "mom_3m", "mom_6m", "mom_12m", "mom_12_1", "reversal_1m",
        "vol_60d", "vol_120d",
        "roa", "debt_to_equity", "gross_profitability",
        "gross_margin", "operating_margin", "net_margin", "ebitda_margin",
    ]

    for f in FACTOR_COLS:
        if f in df.columns:
            # Rank within each date to convert to percentile ranks [0, 1]
            df[f] = df.groupby("date")[f].rank(pct=True)
            logger.info("  Rank-normalised: %s (non-null: %d)", f, df[f].notna().sum())
        else:
            logger.warning("  Factor '%s' not found — cannot rank-normalise.", f)

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
    # _top_k_label handles edge cases (ties, small groups).
    df["target_top_quintile"] = (
        df.groupby("date")["forward_return_60d"]
        .transform(_top_k_label)
    )

    n_target = df["target_top_quintile"].notna().sum()
    n_positive = (df["target_top_quintile"] == 1).sum()
    logger.info(
        "  target_top_quintile: %d labelled rows | positive rate: %.1f%%",
        n_target,
        n_positive / n_target * 100 if n_target > 0 else 0,
    )

    # ── Embargo handling ─────────────────────────────────────────────
    # Rows where target_top_quintile is NaN are the final ~60 trading
    # days where we cannot compute the forward return.
    #   Training mode  : drop them (no usable label).
    #   Inference mode : KEEP them — today's row lives here and is the
    #                    row the orchestrator will predict on.  The
    #                    forward_return_60d / target_top_quintile columns
    #                    will simply be NaN for those rows; downstream
    #                    inference excludes them as features anyway.
    if inference_mode:
        n_embargo = int(df["target_top_quintile"].isna().sum())
        logger.info(
            "\nInference mode — retaining %d embargo rows (NaN target). "
            "Total rows: %d.",
            n_embargo, len(df),
        )
    else:
        pre_drop = len(df)
        df = df.dropna(subset=["target_top_quintile"])
        logger.info(
            "\nDropped %d embargo rows (NaN target). Remaining: %d rows.",
            pre_drop - len(df), len(df),
        )

    # ── Label distribution audit (training only — meaningless on NaN) ─
    if not inference_mode:
        total = len(df)
        pos = int((df["target_top_quintile"] == 1).sum())
        neg = int((df["target_top_quintile"] == 0).sum())
        logger.info(
            "Target distribution — positive (Q5): %d (%.1f%%) | "
            "negative: %d (%.1f%%)",
            pos, pos / total * 100,
            neg, neg / total * 100,
        )

        logger.info("Per-symbol positive rate:")
        for sym, grp in df.groupby("symbol"):
            rate = (grp["target_top_quintile"] == 1).mean() * 100
            logger.info("  %-6s  %.1f%%", sym, rate)
    else:
        # In inference mode, log the most recent observation date so
        # operators can confirm today's row is present.
        logger.info(
            "Inference snapshot — most recent date in frame: %s",
            df["date"].max().date().isoformat(),
        )

    # ── Select Only Allow-Listed Columns to Persist ──────────────────
    keep_cols = ["date", "symbol", "forward_return_60d", "target_top_quintile"] + [
        f for f in FACTOR_COLS if f in df.columns
    ]
    df = df[keep_cols]

    # ── Save ─────────────────────────────────────────────────────────
    df = df.set_index("date")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=True)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        "\nSaved → %s  (%d rows × %d cols, %.2f MB)",
        output_path, *df.shape, size_mb,
    )
    logger.info("V4 feature pipeline complete (mode=%s).",
                "inference" if inference_mode else "training")


if __name__ == "__main__":
    main()
