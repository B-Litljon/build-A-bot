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

Step 3 of the V4 Investor, with its own promotion gate -- structurally the same
idea as the forex bot's retrainer, but scored on RANKING quality rather than
profit factor.

Glossary:
    LambdaRank -- a learning-to-rank objective. It optimises the ORDER of the
        list rather than each stock's individual score, which matches the job:
        only the relative ordering decides what gets bought.
    Group unit -- one trading date is one ranking problem ("rank today's 96
        candidates"), which is why dates form the query groups.
    TRAIN_DAYS -- 504, the minimum expanding training window (~2 years).
    TEST_DAYS -- 60, the width of each out-of-sample fold and the roll-forward
        step.
    EMBARGO_DAYS -- 60, AND THIS IS THE SUBTLE ONE. The target looks 60 days
        ahead, so the last 60 days of any training window overlap the future of
        the test window. Skipping a 60-day gap between them prevents the model
        being scored on outcomes it partly saw during training.
    P@K (precision at K) -- of the top K names the model picked, what fraction
        really landed in the top quintile.
    lift -- P@K divided by the base rate. Since the target is a top QUINTILE,
        random guessing scores 0.20; a lift of 1.3 means 30% better than chance.
        Lift, not raw precision, is what the gate thresholds on.
    GATE_P1_MIN_LIFT / GATE_P2_MIN_LIFT -- 1.3 / 1.2 for the top 1 and 2 picks.
    GATE_P8_MIN_LIFT -- 1.1, and the one that actually matters: the orchestrator
        deploys TOP_K=8, so this measures the gate at the depth really traded.
        Added when the strategy moved from 2 concentrated picks to 8.
    GATE_NDCG_MIN -- 0.0, i.e. informational only until calibrated. NDCG scores
        the whole ordering, rewarding good names placed near the top.
    FORCE_SAVE -- INVESTOR_GATE_FORCE=1, an escape hatch to save a model that
        failed the gate. Deliberately awkward to trigger.
    _EXCLUDE_COLS -- columns never fed to the model (identifiers, raw prices,
        the target itself).

    ── benchmark gate (added 2026-08-03) ──
    Every gate above measures lift over RANDOM. That cannot distinguish
    "better than guessing" from "better than doing nothing", and on
    2026-08-02 a model passed all of them while failing to beat an
    equal-weighted basket of the whole universe out of sample. These
    measure lift over DOING NOTHING.
    GATE_BENCH_MIN_EXCESS_BPS -- INVESTOR_GATE_BENCH_BPS, default 25.0. The
        mean monthly return of the deployed basket minus the mean monthly
        return of equal-weighting all 96, in basis points, averaged over
        every out-of-sample month. A sanity floor, not the real bar --
        see GATE_BENCH_MIN_T.
    GATE_BENCH_MIN_T -- INVESTOR_GATE_BENCH_T, default 2.0, AND THIS IS THE
        ONE THAT BINDS. The mean monthly excess divided by its standard
        error. That standard error is about 60 bps on ~30 months, so t >= 2
        means roughly +120 bps/month. Deliberately stringent: on this much
        data, nothing smaller can be distinguished from luck. Every target
        tried so far lands within one or two standard errors of zero.
    GATE_BENCH_ALIGNMENTS -- fold-start offsets in trading days
        (default 0,7,14,21,28). The benchmark is re-measured at each to
        catch results that hang on one lucky set of fold boundaries.
    GATE_BENCH_MIN_PASS_SHARE -- 1.0, i.e. every alignment must clear the
        floor. Guards against a lucky slice -- but DO NOT read it as five
        confirmations: the alignments share nearly all their rows and their
        excess series correlate 0.71, so five of them are worth about 2.2
        independent samples (measured 2026-08-04). Statistical power comes
        from GATE_BENCH_MIN_T, not from this.
    _benchmark_at_alignment -- re-runs the whole walk-forward with the start
        slid by N trading days and returns that slice's mean excess.
    alignment_results -- offset -> mean excess in bps, the spread the gate
        actually decides on.
    TOP_K / SECTOR_CAP -- 8 and 2, mirroring portfolio_orchestrator so the
        gate simulates the basket actually traded, not the raw ranking.
    _load_close_matrix -- dates x symbols closing prices from the raw
        parquet, needed because the training frame carries no prices.
        If it cannot be loaded the benchmark gate FAILS CLOSED: an
        unmeasurable model must not be promoted.
    _sector_capped_pick -- the orchestrator's greedy top-K-with-sector-cap
        selection, duplicated here rather than imported to keep the trainer
        independent of the broker-facing module.
    bench_months -- one row per held month across all folds:
        (month, basket return, equal-weight return).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import os
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from investor_universe import SECTORS  # noqa: E402

_INPUT_PATH  = _PROJECT_ROOT / "data" / "processed" / "v4_training_features.parquet"
_RAW_PATH    = _PROJECT_ROOT / "data" / "raw" / "v4_investor_data.parquet"
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

# ── Self-approval gate thresholds (mirror src/core/retrainer.py:214-219) ──
# Ranker gate is LIFT-OVER-RANDOM, not absolute: a random picker scores
# Precision@K ≈ the positive base rate (top-quintile target ≈ 0.20). The
# gate requires the model to clear the base rate by a margin.
GATE_P1_MIN_LIFT = float(os.getenv("INVESTOR_GATE_P1_LIFT", "1.3"))  # P@1 ≥ 1.3× base rate
GATE_P2_MIN_LIFT = float(os.getenv("INVESTOR_GATE_P2_LIFT", "1.2"))  # P@2 ≥ 1.2× base rate
GATE_P8_MIN_LIFT = float(os.getenv("INVESTOR_GATE_P8_LIFT", "1.1"))  # P@8 ≥ 1.1× base — the depth we deploy (orchestrator TOP_K=8)
GATE_NDCG_MIN    = float(os.getenv("INVESTOR_GATE_NDCG_MIN", "0.0")) # absolute NDCG floor (0 = informational until calibrated)
FORCE_SAVE       = os.getenv("INVESTOR_GATE_FORCE", "0").strip() == "1"  # escape hatch

# ── Benchmark gate: lift over DOING NOTHING, not over random ─────────
# Added 2026-08-03. Every threshold above compares the model to a coin
# flip; none of them notices a model that ranks better than chance while
# still losing to an equal-weighted basket of the same universe — which is
# exactly what the 2026-07-03 promotion turned out to be.
#
# Sizing the bar (revised 2026-08-04): the mean monthly excess over ~30
# months has a standard error of roughly 60 bps. Everything measured so far
# — the shipped model, and every alternative target tried — lands inside
# one or two standard errors of zero. A gate on the point estimate alone is
# therefore weak, so the t-statistic below is the binding constraint and
# the bps floor is a secondary sanity check.
GATE_BENCH_MIN_EXCESS_BPS = float(os.getenv("INVESTOR_GATE_BENCH_BPS", "25.0"))
# Mean excess divided by its standard error. At ~30 months and SE ~60 bps,
# t >= 2 means roughly +120 bps/month — deliberately stringent, because on
# this much data nothing smaller can be told apart from luck. If nothing
# ever clears it, that is the honest finding, not a broken gate: the
# fallback is equal-weighting, which the evidence supports as well as any
# model produced so far.
GATE_BENCH_MIN_T = float(os.getenv("INVESTOR_GATE_BENCH_T", "2.0"))
TOP_K: int = 8        # mirrors portfolio_orchestrator.TOP_K
SECTOR_CAP: int = 2   # mirrors portfolio_orchestrator.SECTOR_CAP

# Fold-start offsets, in trading days, at which the benchmark is measured.
# Re-slicing catches results that depend on one lucky set of fold
# boundaries — for instance, rounding the top-quintile label from 19 names
# to 20 (one row per day out of 96) moves the measured excess by 23 bps.
#
# BUT DO NOT READ "passed at all 5" AS FIVE CONFIRMATIONS. Measured
# 2026-08-04: the alignments' monthly excess series correlate 0.71 on
# average, because they share almost all the same rows — five alignments
# are worth about 2.2 independent samples. This is a guard against a lucky
# slice, not a substitute for statistical power. That is what
# GATE_BENCH_MIN_T is for.
GATE_BENCH_ALIGNMENTS: list[int] = [
    int(x) for x in os.getenv("INVESTOR_GATE_BENCH_ALIGNMENTS", "0,7,14,21,28").split(",")
    if x.strip()
]
# Share of alignments that must clear the floor. 1.0 = all of them.
GATE_BENCH_MIN_PASS_SHARE = float(os.getenv("INVESTOR_GATE_BENCH_SHARE", "1.0"))

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

FACTOR_COLS = [
    "mom_3m", "mom_6m", "mom_12m", "mom_12_1", "reversal_1m",
    "vol_60d", "vol_120d",
    "roa", "debt_to_equity", "gross_profitability",
    "gross_margin", "operating_margin", "net_margin", "ebitda_margin",
]

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


def _load_close_matrix() -> pd.DataFrame | None:
    """
    Closing prices as a dates x symbols matrix.

    The training frame deliberately carries no prices (they leak the
    forward return), so the benchmark gate reads them from the raw
    parquet.  Returns None if unavailable — the caller must then FAIL the
    gate rather than skip it.
    """
    try:
        raw = pd.read_parquet(_RAW_PATH, columns=["symbol", "close"])
    except Exception as exc:                       # noqa: BLE001
        logger.error("Benchmark gate: cannot read %s (%s)", _RAW_PATH, exc)
        return None

    if "date" not in raw.columns:
        raw = raw.reset_index()
    if "date" not in raw.columns:
        logger.error("Benchmark gate: no 'date' column or index in %s", _RAW_PATH)
        return None

    raw["date"] = pd.to_datetime(raw["date"], utc=True)
    px = raw.pivot_table(index="date", columns="symbol", values="close")
    return px.sort_index()


def _month_ends(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """Last available trading date of each calendar month, in order."""
    s = pd.Series(index, index=index)
    return sorted(s.groupby([index.year, index.month]).max())


def _sector_capped_pick(
    symbols: list[str],
    scores: np.ndarray,
    k: int = TOP_K,
    cap: int = SECTOR_CAP,
) -> list[str]:
    """
    Greedy top-k walk that skips any name whose sector is already full.

    Mirrors portfolio_orchestrator.predict_and_rank so the gate scores the
    basket that would really be bought, not the unconstrained ranking.
    """
    order = np.argsort(-np.asarray(scores, dtype=float))
    picked: list[str] = []
    per_sector: dict[str, int] = {}
    for i in order:
        sym = symbols[i]
        sector = SECTORS.get(sym, "unknown")
        if per_sector.get(sector, 0) >= cap:
            continue
        picked.append(sym)
        per_sector[sector] = per_sector.get(sector, 0) + 1
        if len(picked) == k:
            break
    return picked


def _fold_basket_months(
    test_df: pd.DataFrame,
    scores: np.ndarray,
    px: pd.DataFrame,
) -> list[tuple[pd.Timestamp, float, float]]:
    """
    Simulate the deployed monthly rebalance across one fold's test window.

    At each month-end inside the window the fold's model picks a basket;
    it is held to the next month-end and compared with equal-weighting
    every symbol that has prices at both ends.

    The closing date may fall outside the test window — that is correct
    and not leakage: the pick used only in-window information, and the
    holding period's outcome is exactly what live trading would realise.
    """
    scored = test_df[["date", "symbol"]].copy()
    scored["score"] = scores

    all_month_ends = _month_ends(px.index)
    window_ends = [m for m in all_month_ends
                   if scored["date"].min() <= m <= scored["date"].max()]

    out: list[tuple[pd.Timestamp, float, float]] = []
    for d0 in window_ends:
        later = [m for m in all_month_ends if m > d0]
        if not later or d0 not in px.index:
            continue
        d1 = later[0]
        if d1 not in px.index:
            continue

        day = scored[scored["date"] == d0]
        if day.empty:
            continue

        p0, p1 = px.loc[d0], px.loc[d1]
        tradable = [s for s in px.columns
                    if np.isfinite(p0.get(s, np.nan)) and np.isfinite(p1.get(s, np.nan))]
        if not tradable:
            continue

        day = day[day["symbol"].isin(tradable)]
        if day.empty:
            continue

        picks = _sector_capped_pick(
            day["symbol"].tolist(), day["score"].to_numpy()
        )
        if not picks:
            continue

        basket = float(np.mean([p1[s] / p0[s] - 1 for s in picks]))
        bench = float(np.mean([p1[s] / p0[s] - 1 for s in tradable]))
        out.append((d0, basket, bench))

    return out


def _benchmark_at_alignment(
    df: pd.DataFrame,
    X_df: pd.DataFrame,
    y_all: np.ndarray,
    px: pd.DataFrame,
    offset: int,
) -> tuple[float, int]:
    """
    Mean monthly excess over equal-weight, with the walk-forward started
    `offset` trading days later.

    Sliding the start reshuffles every fold boundary and the month
    composition without changing the data or the rules — the cheapest
    honest way to ask whether a result is structural or a lucky slice.

    Returns (mean excess in bps, months measured). NaN if unmeasurable.
    """
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    if offset >= len(dates):
        return float("nan"), 0

    keep = df["date"] >= dates[offset]
    d = df[keep].reset_index(drop=True)
    Xd = X_df[keep.values].reset_index(drop=True)
    yd = y_all[keep.values]

    sub_dates = pd.DatetimeIndex(sorted(d["date"].unique()))
    n = len(sub_dates)
    months: list[tuple[pd.Timestamp, float, float]] = []

    fold = 0
    while True:
        train_end = TRAIN_DAYS + fold * TEST_DAYS
        embargo_end = train_end + EMBARGO_DAYS
        test_end = embargo_end + TEST_DAYS
        if test_end > n:
            break

        train_mask = d["date"] <= sub_dates[train_end - 1]
        test_mask = ((d["date"] > sub_dates[embargo_end - 1])
                     & (d["date"] <= sub_dates[test_end - 1]))
        if not train_mask.any() or not test_mask.any():
            fold += 1
            continue

        train_df = d[train_mask]
        model = lgb.LGBMRanker(**LGBM_PARAMS)
        model.fit(
            Xd[train_mask.values],
            yd[train_mask.values],
            group=_build_groups(train_df),
            callbacks=[lgb.log_evaluation(period=-1)],
        )
        scores = model.predict(Xd[test_mask.values])
        months.extend(_fold_basket_months(d.loc[test_mask, ["date", "symbol"]], scores, px))
        fold += 1

    if not months:
        return float("nan"), 0
    excess = np.array([b - h for _, b, h in months], dtype=float)
    return float(excess.mean() * 10_000), len(months)


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

def main() -> int:
    try:
        return _main_impl()
    except Exception as exc:
        logger.exception("Trainer failed: %s", exc)
        return 1


def _main_impl() -> int:
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
    raw_feat_cols = [c for c in df.columns if c in FACTOR_COLS]

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

    # Prices for the benchmark gate. None => unmeasurable => gate fails.
    px = _load_close_matrix()
    bench_months: list[tuple[pd.Timestamp, float, float]] = []

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
        p_at_8 = _precision_at_k(test_df, scores, k=8)  # deployed basket depth

        # Benchmark gate: what this fold's model would actually have earned
        if px is not None:
            bench_months.extend(_fold_basket_months(test_df, scores, px))

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
            "precision_at_8": p_at_8,
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
        mean_p8   = results_df["precision_at_8"].mean()
        logger.info(
            "\nMean across folds │ %s=%.4f │ P@1=%.3f │ P@2=%.3f",
            ndcg_col[0], mean_ndcg, mean_p1, mean_p2,
        )
    else:
        logger.error("🚫 GATE FAILED — No NDCG column found in walk-forward results.")
        return 2

    # ── Self-approval Gate (Task 1) ──
    base_rate = float(y_all.mean())
    p1_floor = base_rate * GATE_P1_MIN_LIFT
    p2_floor = base_rate * GATE_P2_MIN_LIFT
    p8_floor = base_rate * GATE_P8_MIN_LIFT

    p1_pass = mean_p1 >= p1_floor
    p2_pass = mean_p2 >= p2_floor
    p8_pass = mean_p8 >= p8_floor
    ndcg_pass = mean_ndcg >= GATE_NDCG_MIN

    # ── Benchmark gate — lift over doing nothing ─────────────────────
    if px is None:
        bench_pass = False
        bench_excess_bps = float("nan")
        bench_beat_share = float("nan")
        bench_basket_ret = bench_bench_ret = float("nan")
        bench_excess_t = float("nan")
        logger.error(
            "Benchmark gate: prices unavailable — FAILING CLOSED. "
            "An unmeasurable model must not be promoted."
        )
    elif not bench_months:
        bench_pass = False
        bench_excess_bps = float("nan")
        bench_beat_share = float("nan")
        bench_basket_ret = bench_bench_ret = float("nan")
        bench_excess_t = float("nan")
        logger.error(
            "Benchmark gate: no held months produced — FAILING CLOSED."
        )
    else:
        basket = np.array([m[1] for m in bench_months], dtype=float)
        bench = np.array([m[2] for m in bench_months], dtype=float)
        excess = basket - bench
        bench_excess_bps = float(excess.mean() * 10_000)
        bench_beat_share = float((basket > bench).mean())
        # t on the paired monthly excess. Pairing against the benchmark is
        # already the tightest available comparison: differencing two model
        # baskets instead does NOT help, because they hold different names
        # (monthly-return correlation ~0.7) and the idiosyncratic variance
        # dominates the market variance that pairing removes.
        _sd = float(excess.std(ddof=1)) if len(excess) > 1 else float("nan")
        bench_excess_t = (
            float(excess.mean() / (_sd / np.sqrt(len(excess))))
            if _sd and np.isfinite(_sd) and _sd > 0 else float("nan")
        )
        bench_basket_ret = float(np.prod(1 + basket) - 1)
        bench_bench_ret = float(np.prod(1 + bench) - 1)

    # ── Stability across fold alignments ─────────────────────────────
    # A single measurement of this quantity is not trustworthy: see
    # GATE_BENCH_ALIGNMENTS. Re-measure at several fold starts and require
    # the result to hold at (nearly) all of them.
    alignment_results: dict[int, float] = {}
    if px is not None and bench_months:
        logger.info("\nBenchmark stability — re-measuring at %d fold alignments ...",
                    len(GATE_BENCH_ALIGNMENTS))
        for off in GATE_BENCH_ALIGNMENTS:
            bps, n_months = _benchmark_at_alignment(df, X_df, y_all, px, off)
            alignment_results[off] = bps
            logger.info("  offset %3d trading days: %+8.1f bps  (%d months)",
                        off, bps, n_months)

    if alignment_results:
        cleared = [b for b in alignment_results.values()
                   if np.isfinite(b) and b >= GATE_BENCH_MIN_EXCESS_BPS]
        pass_share = len(cleared) / len(alignment_results)
        stability_pass = pass_share >= GATE_BENCH_MIN_PASS_SHARE
    else:
        pass_share = 0.0
        stability_pass = False

    # The t-statistic is the binding constraint; stability is the guard
    # against a lucky slice. Both must hold.
    t_pass = np.isfinite(bench_excess_t) and bench_excess_t >= GATE_BENCH_MIN_T
    bench_pass = stability_pass and t_pass

    logger.info("=" * 70)
    logger.info("VALIDATION GATE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Positive Base Rate   : {base_rate:.4f} (equivalent to random prediction)")
    logger.info(f"Mean Precision@1     : {mean_p1:.4f} vs floor {p1_floor:.4f} (lift required: {GATE_P1_MIN_LIFT}x) -> {'PASS' if p1_pass else 'FAIL'}")
    logger.info(f"Mean Precision@2     : {mean_p2:.4f} vs floor {p2_floor:.4f} (lift required: {GATE_P2_MIN_LIFT}x) -> {'PASS' if p2_pass else 'FAIL'}")
    logger.info(f"Mean Precision@8     : {mean_p8:.4f} vs floor {p8_floor:.4f} (lift required: {GATE_P8_MIN_LIFT}x, deployed basket depth) -> {'PASS' if p8_pass else 'FAIL'}")
    logger.info(f"Mean NDCG            : {mean_ndcg:.4f} vs floor {GATE_NDCG_MIN:.4f} -> {'PASS' if ndcg_pass else 'FAIL'}")
    logger.info("-" * 70)
    logger.info("BENCHMARK GATE (lift over equal-weighting the universe)")
    if bench_months:
        logger.info(
            f"Held months          : {len(bench_months)} "
            f"({bench_months[0][0].date()} → {bench_months[-1][0].date()})"
        )
        logger.info(f"Basket total return  : {bench_basket_ret * 100:+.1f}%")
        logger.info(f"Equal-weight return  : {bench_bench_ret * 100:+.1f}%")
        logger.info(f"Months beating bench : {bench_beat_share * 100:.0f}%")
    logger.info(f"Mean monthly excess  : {bench_excess_bps:+.1f} bps (base alignment)")
    logger.info(
        f"Excess t-statistic   : {bench_excess_t:+.2f} vs floor "
        f"{GATE_BENCH_MIN_T:+.2f} -> {'PASS' if t_pass else 'FAIL'}   "
        f"<-- the binding constraint"
    )
    if alignment_results:
        spread = "  ".join(f"{o}d:{b:+.0f}" for o, b in sorted(alignment_results.items()))
        logger.info(f"Across alignments    : {spread}")
        logger.info(
            f"Cleared {GATE_BENCH_MIN_EXCESS_BPS:+.0f} bps at   : "
            f"{pass_share * 100:.0f}% of alignments vs required "
            f"{GATE_BENCH_MIN_PASS_SHARE * 100:.0f}% -> "
            f"{'PASS' if stability_pass else 'FAIL'}   "
            f"(~2.2 effective independent samples, not {len(alignment_results)})"
        )
    else:
        logger.info(f"Stability            : not measured -> FAIL")
    logger.info("=" * 70)

    gate_passed = p1_pass and p2_pass and p8_pass and ndcg_pass and bench_pass

    if not gate_passed and not FORCE_SAVE:
        logger.warning("=" * 70)
        logger.warning("🚫 GATE FAILED — existing model retained, nothing written")
        logger.warning("=" * 70)
        return 2

    if FORCE_SAVE and not gate_passed:
        logger.warning("=" * 70)
        logger.warning("⚠️ GATE FAILED BUT FORCE_SAVE IS ENABLED — Proceeding with promotion")
        logger.warning("=" * 70)
    else:
        logger.info("=" * 70)
        logger.info("✅ VALIDATION GATE PASSED — PROMOTING MODELS")
        logger.info("=" * 70)

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

    # ── Atomic Metadata Sidecar (Task 2) ─────────────────────────────
    metadata_path = _MODEL_PATH.parent / "v4_investor_lgbm.metadata.json"
    metadata_temp = _MODEL_PATH.parent / "v4_investor_lgbm.metadata_temp.json"
    metadata = {
        "model": "v4_investor_lgbm",
        "asset_class": "equities_longterm",
        "horizon_days": EMBARGO_DAYS,                      # 60-day forward target
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "trained_on_symbols": sorted(df["symbol"].unique().tolist()),
        "n_features": len(feature_cols),
        "data_source": "alpaca",
        "walk_forward": {
            "folds": int(fold_num),
            "mean_ndcg": round(float(mean_ndcg), 4),
            "mean_precision_at_1": round(float(mean_p1), 4),
            "mean_precision_at_2": round(float(mean_p2), 4),
            "mean_precision_at_8": round(float(mean_p8), 4),
            "positive_base_rate": round(float(base_rate), 4),
        },
        "benchmark": {
            "held_months": len(bench_months),
            "mean_monthly_excess_bps": (
                round(bench_excess_bps, 1) if bench_months else None
            ),
            "months_beating_benchmark": (
                round(bench_beat_share, 3) if bench_months else None
            ),
            "basket_total_return": (
                round(bench_basket_ret, 4) if bench_months else None
            ),
            "equal_weight_total_return": (
                round(bench_bench_ret, 4) if bench_months else None
            ),
            "floor_bps": GATE_BENCH_MIN_EXCESS_BPS,
            "excess_t": (round(bench_excess_t, 3)
                         if np.isfinite(bench_excess_t) else None),
            "floor_t": GATE_BENCH_MIN_T,
            "t_passed": bool(t_pass),
            "alignments_bps": {str(k): round(v, 1) for k, v in alignment_results.items()},
            "alignment_pass_share": round(pass_share, 3),
            "required_pass_share": GATE_BENCH_MIN_PASS_SHARE,
            "stability_passed": bool(stability_pass),
            "passed": bool(bench_pass),
        },
        "gate_passed": bool(gate_passed),
    }
    try:
        with open(metadata_temp, "w") as f:
            json.dump(metadata, f, indent=2)
        os.replace(metadata_temp, metadata_path)
        logger.info(f"[ATOMIC] Metadata saved: {metadata_path}")
    except Exception as e:
        logger.error(f"[ATOMIC] Failed to save metadata: {e}")
        if metadata_temp.exists():
            metadata_temp.unlink()
        raise

    # ── Feature importance ───────────────────────────────────────────
    importances = (
        pd.Series(final_model.feature_importances_, index=feature_cols)
        .sort_values(ascending=False)
    )
    logger.info("\nTop 15 features by gain importance:")
    for feat, gain in importances.head(15).items():
        logger.info("  %-45s  %.1f", feat, gain)

    logger.info("\nV4 walk-forward training complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
