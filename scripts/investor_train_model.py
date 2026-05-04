"""
V4 Investor Walk-Forward LightGBM Ranker Training Pipeline.

Implements a strict expanding-window walk-forward cross-validation with
a 60-trading-day embargo between train and test splits to prevent data
leakage from the 60-day forward-return target.

Architecture
------------
  Objective  : LambdaRank (pairwise learning-to-rank)
  Target     : target_top_quintile (1 = top-quintile forward return, 0 = not)
  Group unit : One trading date = one LightGBM query group
  Features   : Momentum (3m/6m/12m) + Macro trends + Fundamental margin ratios
               + raw quarterly income-statement line items (NaN-safe via LGBM)

Walk-forward structure (expanding training window)
---------------------------------------------------
  Fold k  →  Train: dates[0 : 504 + k*60]
             Embargo: next 60 trading days  ← gap equal to forward-return horizon
             Test:  next 60 trading days

  With 1195 unique dates this yields ~10 out-of-sample folds.

Usage:
    pipenv run python scripts/investor_train_model.py

Output:
    models/v4_investor_lgbm.txt   (LightGBM native text format)
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

_INPUT_PATH  = _PROJECT_ROOT / "data" / "processed" / "v4_training_features.parquet"
_MODEL_PATH  = _PROJECT_ROOT / "models" / "v4_investor_lgbm.txt"

# ── logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── walk-forward parameters ───────────────────────────────────────────
TRAIN_DAYS   = 504   # minimum expanding train window (~2 calendar years)
EMBARGO_DAYS = 60    # embargo = forward-return horizon prevents leakage
TEST_DAYS    = 60    # fold width; also the roll-forward step size

# ── columns excluded from the feature matrix ─────────────────────────
# OHLCV: raw price data leaks forward returns if included as features.
# Metadata + targets: must never appear in X.
# 'date': time index — must be excluded to prevent the model from
#         memorizing calendar patterns instead of learning true signal.
_EXCLUDE_COLS = frozenset({
    "date",
    "symbol",
    "forward_return_60d",
    "target_top_quintile",
    "open", "high", "low", "close", "volume",
})

# ── LGBMRanker hyperparameters ────────────────────────────────────────
LGBM_PARAMS: dict = dict(
    objective="lambdarank",
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=5,
    importance_type="gain",
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _sanitize(name: str) -> str:
    """
    Replace characters that LightGBM dislikes in feature names
    (spaces, parentheses, slashes) with underscores.
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", name).strip("_")


def _build_groups(df: pd.DataFrame, date_col: str = "date") -> np.ndarray:
    """
    Build the group-size array required by LGBMRanker.

    Each trading date is one query group; its size is the number of
    symbols present on that date.  ``df`` MUST be sorted by ``date_col``
    before calling — LightGBM assumes rows within a group are contiguous.

    Returns
    -------
    np.ndarray of shape (n_unique_dates,) with dtype int32.
    """
    sizes = df.groupby(date_col, sort=False).size().to_numpy(dtype=np.int32)
    assert sizes.sum() == len(df), (
        f"Group array sum ({sizes.sum()}) ≠ DataFrame length ({len(df)}). "
        "Ensure df is sorted by date before calling _build_groups()."
    )
    return sizes


def _precision_at_k(
    test_df: pd.DataFrame,
    scores: np.ndarray,
    k: int,
    date_col: str = "date",
    target_col: str = "target_top_quintile",
) -> float:
    """
    Cross-sectional Precision@K averaged over all dates in *test_df*.

    For each date, select the K symbols with the highest predicted score
    and compute the fraction whose ground-truth label is 1.

    Parameters
    ----------
    test_df : DataFrame with columns *date_col* and *target_col*.
    scores  : Predicted relevance scores, aligned with test_df rows.
    k       : Number of top symbols to consider per date.
    """
    df = test_df[[date_col, target_col]].copy()
    df["_score"] = scores

    daily = []
    for _, grp in df.groupby(date_col, sort=True):
        top_k = grp.nlargest(k, "_score")
        daily.append(float(top_k[target_col].mean()))

    return float(np.mean(daily)) if daily else float("nan")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("=" * 70)
    logger.info("V4 Walk-Forward LightGBM Ranker Training")
    logger.info("Train window : %d trading days (expanding)", TRAIN_DAYS)
    logger.info("Embargo      : %d trading days", EMBARGO_DAYS)
    logger.info("Test window  : %d trading days per fold", TEST_DAYS)
    logger.info("=" * 70)

    # ── Load & sort ──────────────────────────────────────────────────
    if not _INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{_INPUT_PATH} not found. "
            "Run scripts/investor_feature_pipeline.py first."
        )

    df = pd.read_parquet(_INPUT_PATH)

    # Ensure 'date' is a regular column for groupby and mask operations
    if df.index.name == "date":
        df = df.reset_index()

    # ── Strict sort by date (required by LGBMRanker + group logic) ──
    df = df.sort_values("date").reset_index(drop=True)
    logger.info(
        "Loaded: %d rows × %d cols | dates %s → %s | %d symbols",
        *df.shape,
        df["date"].min().date(), df["date"].max().date(),
        df["symbol"].nunique(),
    )

    # ── Build feature matrix ─────────────────────────────────────────
    raw_feat_cols = [c for c in df.columns if c not in _EXCLUDE_COLS]

    # Sanitize column names: LightGBM text format chokes on spaces /
    # special characters when saving/loading the model.
    col_rename = {c: _sanitize(c) for c in raw_feat_cols}
    # Guard against any name collisions after sanitization
    seen: dict[str, int] = {}
    for orig, sanitized in col_rename.items():
        if sanitized in seen.values():
            idx = sum(1 for v in seen.values() if v == sanitized)
            col_rename[orig] = f"{sanitized}_{idx}"
        seen[orig] = col_rename[orig]

    # Coerce fundamentals to numeric (some quarterly line items can be
    # object-typed when a symbol has no data in a quarter)
    X_df = (
        df[raw_feat_cols]
        .apply(pd.to_numeric, errors="coerce")
        .rename(columns=col_rename)
    )
    feature_cols: list[str] = list(X_df.columns)
    y_all = df["target_top_quintile"].to_numpy(dtype=np.int32)

    logger.info(
        "Feature matrix: %d cols | target positive rate: %.1f%%",
        len(feature_cols),
        y_all.mean() * 100,
    )

    # ── Walk-forward splits ──────────────────────────────────────────
    unique_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    n_dates = len(unique_dates)
    logger.info("Unique trading dates: %d", n_dates)

    fold_results: list[dict] = []
    fold_num = 0

    logger.info("\n%s", "─" * 70)

    while True:
        # Expanding window: train grows by TEST_DAYS each fold
        train_end_idx   = TRAIN_DAYS + fold_num * TEST_DAYS
        embargo_end_idx = train_end_idx + EMBARGO_DAYS
        test_end_idx    = embargo_end_idx + TEST_DAYS

        if test_end_idx > n_dates:
            break   # not enough future dates for a full test window

        # Date boundary values (inclusive cutoffs)
        train_cutoff   = unique_dates[train_end_idx - 1]
        embargo_cutoff = unique_dates[embargo_end_idx - 1]
        test_cutoff    = unique_dates[test_end_idx - 1]

        # Boolean masks on the sorted-by-date DataFrame
        train_mask = df["date"] <= train_cutoff
        test_mask  = (df["date"] > embargo_cutoff) & (df["date"] <= test_cutoff)

        train_df = df[train_mask].copy()
        test_df  = df[test_mask].copy()

        X_train = X_df[train_mask]
        y_train = y_all[train_mask.values]
        # ── GROUP ARRAY: number of symbols per trading date ──────────
        # LGBMRanker treats each date as one query. group_train[i] is
        # the count of symbols on the i-th unique date in the training
        # set.  Rows MUST be sorted by date (guaranteed above).
        group_train = _build_groups(train_df)

        X_test  = X_df[test_mask]
        y_test  = y_all[test_mask.values]
        group_test = _build_groups(test_df)

        # ── Train fold ───────────────────────────────────────────────
        evals_result: dict = {}
        model = lgb.LGBMRanker(**LGBM_PARAMS)
        model.fit(
            X_train, y_train,
            group=group_train,
            eval_set=[(X_test, y_test)],
            eval_group=[group_test],
            eval_metric="ndcg",
            callbacks=[
                lgb.record_evaluation(evals_result),
                lgb.log_evaluation(period=-1),    # suppress per-tree stdout
            ],
        )

        # ── Metrics ──────────────────────────────────────────────────
        valid_0 = evals_result.get("valid_0", {})
        ndcg_key = next((k for k in valid_0 if "ndcg" in k.lower()), None)
        final_ndcg = float(valid_0[ndcg_key][-1]) if ndcg_key else float("nan")

        scores = model.predict(X_test)
        p_at_1 = _precision_at_k(test_df, scores, k=1)
        p_at_2 = _precision_at_k(test_df, scores, k=2)

        logger.info(
            "Fold %2d │ Train → %s (%4d dates, %5d rows) │ "
            "Embargo → %s │ Test %s→%s │ "
            "%s=%.4f │ P@1=%.3f │ P@2=%.3f",
            fold_num + 1,
            train_cutoff.date(), train_end_idx, len(train_df),
            embargo_cutoff.date(),
            unique_dates[embargo_end_idx].date(), test_cutoff.date(),
            ndcg_key or "NDCG", final_ndcg,
            p_at_1, p_at_2,
        )

        fold_results.append({
            "fold":          fold_num + 1,
            "train_dates":   train_end_idx,
            "train_rows":    len(train_df),
            ndcg_key or "ndcg": final_ndcg,
            "precision_at_1": p_at_1,
            "precision_at_2": p_at_2,
        })

        fold_num += 1

    # ── Walk-forward summary ─────────────────────────────────────────
    logger.info("\n%s", "─" * 70)
    results_df = pd.DataFrame(fold_results)
    logger.info("Walk-forward summary (%d folds):\n%s", fold_num, results_df.to_string(index=False))

    ndcg_col = [c for c in results_df.columns if "ndcg" in c.lower()]
    if ndcg_col:
        mean_ndcg = results_df[ndcg_col[0]].mean()
        mean_p1   = results_df["precision_at_1"].mean()
        mean_p2   = results_df["precision_at_2"].mean()
        logger.info(
            "\nMean across folds │ %s=%.4f │ P@1=%.3f │ P@2=%.3f",
            ndcg_col[0], mean_ndcg, mean_p1, mean_p2,
        )

    # ── Final model — retrain on ALL available labelled data ─────────
    logger.info("\n%s", "─" * 70)
    logger.info("Training final model on full dataset (%d rows) ...", len(df))

    group_all = _build_groups(df)   # sorted by date → group array for all data

    final_model = lgb.LGBMRanker(**LGBM_PARAMS)
    final_model.fit(
        X_df, y_all,
        group=group_all,
        callbacks=[lgb.log_evaluation(period=-1)],
    )

    # ── Save ─────────────────────────────────────────────────────────
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_model.booster_.save_model(str(_MODEL_PATH))
    size_kb = _MODEL_PATH.stat().st_size / 1024
    logger.info("Model saved → %s  (%.1f KB)", _MODEL_PATH, size_kb)

    # ── Feature importance ───────────────────────────────────────────
    importances = (
        pd.Series(final_model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )
    logger.info("\nTop 15 features by gain importance:")
    for feat, gain in importances.head(15).items():
        logger.info("  %-45s  %.1f", feat, gain)

    logger.info("\nV4 walk-forward training complete.")


if __name__ == "__main__":
    main()
