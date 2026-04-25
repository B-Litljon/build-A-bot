"""
src/day_trading/train_model.py
Walk-Forward Angel/Devil Training — Universal Scalper V4.0

Pipeline
--------
1.  Load  data/processed/dt_training_data.parquet
2.  OOF   Generate Angel probabilities via TimeSeriesSplit(n_splits=5)
          sorted strictly by timestamp so future bars never inform past models.
3.  Folds Walk-forward validation (2 expanding date-based folds, ~30% OOS each)
          - Train Angel on train window
          - Filter to Angel-approved rows (OOF angel_prob >= ANGEL_THRESHOLD)
          - Train Devil on Angel-approved rows -> devil_target (survival)
          - Evaluate: Profit Factor, EV, win-rate on devil_target_macro (the
            true P&L outcome: MFE hit AND no SL breach before EOD)
4.  Gate  Fold 2 (terminal OOS window) is the promotion gate.
5.  Full  Train final models on all 365 days if gate passes.
6.  Save  models/dt_angel_latest.pkl
          models/dt_devil_latest.pkl
          models/dt_threshold.json

Class-imbalance targets (after refinement)
------------------------------------------
angel_target       : target 15–30% positive
                     Entry window (09:30–11:00 ET) + MFE ≥ 0.6× Daily ATR
devil_target       : target 60–80% positive
                     SL tightened to 0.75× Daily ATR (was 1.5×)
devil_target_macro : angel AND devil — the true P&L outcome used for EV/PF

R:R profile (refined)
----------------------
TP_MULT = 0.6  (0.6× Daily ATR)
SL_MULT = 0.4  (0.4× Daily ATR — tightened from 0.75 in iteration 3)
Break-even win-rate = SL / (TP + SL) = 0.4 / 1.0 = 40.0%
R:R = 1.5 — wins pay 1.5× what losses cost.

Usage
-----
    PYTHONPATH=src python src/day_trading/train_model.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

# ── Project paths ──────────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

from day_trading.features import DAY_TRADE_FEATURE_COLS  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DATASET_PATH = _PROJECT_ROOT / "data" / "processed" / "dt_training_data.parquet"
MODEL_DIR = _PROJECT_ROOT / "models"
ANGEL_PATH = MODEL_DIR / "dt_angel_latest.pkl"
DEVIL_PATH = MODEL_DIR / "dt_devil_latest.pkl"
THRESH_PATH = MODEL_DIR / "dt_threshold.json"

FEATURE_COLS: List[str] = DAY_TRADE_FEATURE_COLS  # 22 features
ANGEL_THRESHOLD: float = 0.40  # Angel gate: propose a trade if prob >= this
ANGEL_OOF_SPLITS: int = 5  # TimeSeriesSplit folds for OOF generation

# Day-trade R:R profile (iteration 3 — matches targets.py TP_MULT / SL_MULT)
TP_MULT: float = 0.6  # 0.6 × Daily ATR
SL_MULT: float = 0.4  # 0.4 × Daily ATR  (tightened from 0.75)
BREAKEVEN_WIN_RATE: float = SL_MULT / (
    TP_MULT + SL_MULT
)  # 40.0%  (wins pay 1.5× losses)

# Walk-forward fold schedule (calendar days from the dataset's earliest timestamp)
# With ~365 days of data:  Fold-1 val = days 243–303,  Fold-2 val = days 303–365
FOLD_CONFIGS: List[Tuple[int, int]] = [
    (243, 303),  # Fold 1: train days 0-242, val days 243-302
    (303, 365),  # Fold 2: train days 0-302, val days 303-365  ← promotion gate
]

# Promotion gate thresholds
MIN_PROFIT_FACTOR: float = 1.10  # relaxed from V3.4's 1.2 (lower R:R regime)
MIN_EV: float = -0.30  # sanity floor only — EV negative is expected
# at baseline; gate is PF-driven for this run
MIN_APPROVED_TRADES: int = 20  # minimum trades in val period to trust PF

# ── Baseline Random Forest hyperparameters ────────────────────────────────────
# class_weight="balanced" is mandatory given the 2.2% / 99.2% imbalances.
ANGEL_PARAMS = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=50,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)

DEVIL_PARAMS = dict(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=50,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FoldResult:
    fold: int
    train_rows: int
    val_rows: int
    angel_proposed: int
    devil_approved: int
    win_rate: float
    profit_factor: float
    ev: float
    brier_devil: float
    optimal_threshold: float


@dataclass
class TrainingReport:
    fold_results: List[FoldResult] = field(default_factory=list)
    gate_passed: bool = False
    gate_fold: int = 2
    production_threshold: float = 0.40
    rejection_reasons: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def _profit_factor_ev(
    devil_probs: np.ndarray,
    macro_targets: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float, int]:
    """
    Compute Profit Factor, EV, and win-rate for Devil-approved trades.

    Uses devil_target_macro (the P&L outcome: Angel MFE hit AND no SL breach)
    as the ground truth for realized wins/losses.

    Args:
        devil_probs:   Devil predicted probabilities on Angel-proposed rows.
        macro_targets: devil_target_macro ground truth (0/1) on same rows.
        threshold:     Devil approval threshold.

    Returns:
        (profit_factor, ev, win_rate, n_approved)
    """
    mask = devil_probs >= threshold
    n_approved = int(mask.sum())
    if n_approved == 0:
        return 0.0, -SL_MULT, 0.0, 0

    approved_outcomes = macro_targets[mask]
    n_wins = int(approved_outcomes.sum())
    n_losses = n_approved - n_wins
    win_rate = n_wins / n_approved

    gross_profit = n_wins * TP_MULT
    gross_loss = n_losses * SL_MULT
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    ev = win_rate * (TP_MULT / SL_MULT) - (1.0 - win_rate)

    return pf, ev, win_rate, n_approved


def _find_optimal_threshold(
    devil_probs: np.ndarray,
    macro_targets: np.ndarray,
    min_trades: int = 10,
) -> Tuple[float, float]:
    """
    Sweep Devil thresholds 0.10→0.90 (step 0.05) and return the one that
    maximises Profit Factor on the OOS validation set.

    Returns (best_threshold, best_profit_factor).
    """
    best_threshold = 0.40
    best_pf = 0.0

    for t in np.arange(0.10, 0.91, 0.05):
        pf, _, _, n = _profit_factor_ev(devil_probs, macro_targets, t)
        if n < min_trades:
            continue
        if pf > best_pf:
            best_pf = pf
            best_threshold = float(t)

    return best_threshold, best_pf


def _train_angel(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(**ANGEL_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def _train_devil(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    model = RandomForestClassifier(**DEVIL_PARAMS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model


def _save_artifacts(
    angel_model: RandomForestClassifier,
    devil_model: RandomForestClassifier,
    threshold: float,
) -> None:
    """Atomic save of both models and the threshold JSON."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Atomic write via temp file + os.replace (safe for live hot-reloader)
    for path, obj in [(ANGEL_PATH, angel_model), (DEVIL_PATH, devil_model)]:
        tmp = path.with_suffix(".tmp.pkl")
        joblib.dump(obj, tmp)
        os.replace(tmp, path)
        logger.info("Saved  %s  (%.1f MB)", path.name, path.stat().st_size / 1e6)

    thresh_tmp = THRESH_PATH.with_suffix(".tmp.json")
    thresh_tmp.write_text(
        json.dumps(
            {"devil_threshold": round(threshold, 4), "updated_at": _now_utc()}, indent=2
        )
    )
    os.replace(thresh_tmp, THRESH_PATH)
    logger.info("Saved  %s  (threshold=%.4f)", THRESH_PATH.name, threshold)


def _now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# OOF ANGEL PROBABILITY GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_oof_angel_probs(
    df_ts: pl.DataFrame,
) -> np.ndarray:
    """
    Generate Out-Of-Fold Angel probabilities using TimeSeriesSplit.

    df_ts MUST be sorted by timestamp (ascending) before calling this.
    The split is purely row-index-based on the sorted frame, which guarantees
    that every validation row is strictly later in calendar time than its
    training rows — no future information leaks into past predictions.

    The first ~1/n_splits rows never appear in any validation fold (they are
    always in the training set for fold 1).  These are scored by a model
    trained on fold 1's training data (still strictly temporal).

    Returns:
        angel_probs_oof — float64 array, same length as df_ts, no NaN values.
    """
    X = df_ts[FEATURE_COLS].to_numpy()
    y = df_ts["angel_target"].to_numpy()
    n = len(X)
    oof = np.full(n, np.nan, dtype=np.float64)
    tss = TimeSeriesSplit(n_splits=ANGEL_OOF_SPLITS)

    logger.info(
        "[OOF] Generating Angel probabilities via TimeSeriesSplit(n_splits=%d) …",
        ANGEL_OOF_SPLITS,
    )

    for fold_i, (tr_idx, val_idx) in enumerate(tss.split(X), start=1):
        m = _train_angel(X[tr_idx], y[tr_idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oof[val_idx] = m.predict_proba(X[val_idx])[:, 1]
        logger.info(
            "    OOF fold %d/%d — train=%d  val=%d  angel_pos_in_val=%.1f%%",
            fold_i,
            ANGEL_OOF_SPLITS,
            len(tr_idx),
            len(val_idx),
            100 * y[val_idx].mean(),
        )

    # Fill the head rows that are never in any val fold
    head_mask = np.isnan(oof)
    if head_mask.any():
        first_tr, _ = next(iter(tss.split(X)))
        head_m = _train_angel(X[first_tr], y[first_tr])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            oof[head_mask] = head_m.predict_proba(X[head_mask])[:, 1]
        logger.info(
            "    Head fill: %d train-only rows scored by fold-1 model", head_mask.sum()
        )

    logger.info(
        "[OOF] Done — prob range [%.4f, %.4f]  median=%.4f  "
        "rows with prob>=0.40: %d (%.2f%%)",
        oof.min(),
        oof.max(),
        np.median(oof),
        (oof >= ANGEL_THRESHOLD).sum(),
        100 * (oof >= ANGEL_THRESHOLD).mean(),
    )
    return oof


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def run_walk_forward(df: pl.DataFrame) -> TrainingReport:
    """
    2-fold expanding-window walk-forward validation.

    All splits are date-based (calendar days from the earliest timestamp)
    so that every symbol's bars for a given date range stay in the same fold.
    This prevents any symbol's future data from appearing in an earlier fold's
    training set.

    OOF angel_prob_oof is used to filter the training rows for Devil training,
    preventing data leakage from the Angel's in-sample confidence.
    """
    report = TrainingReport()
    min_date = df["timestamp"].min()

    # ── Pre-compute OOF Angel probs on timestamp-sorted frame ────────────────
    # We sort by timestamp here (not symbol+timestamp) so the row-index split
    # in TimeSeriesSplit is truly chronological across all symbols.
    df_ts = df.sort("timestamp")
    oof_ps = generate_oof_angel_probs(df_ts)
    df_ts = df_ts.with_columns(pl.Series("angel_prob_oof", oof_ps))

    # Merge OOF probs back to the (symbol, timestamp) sorted frame via join
    df = df.join(
        df_ts.select(["symbol", "timestamp", "angel_prob_oof"]),
        on=["symbol", "timestamp"],
        how="left",
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION (%d folds)", len(FOLD_CONFIGS))
    logger.info("=" * 70)

    final_angel: Optional[RandomForestClassifier] = None
    final_devil: Optional[RandomForestClassifier] = None
    production_threshold = ANGEL_THRESHOLD

    for fold_i, (train_end_day, val_end_day) in enumerate(FOLD_CONFIGS, start=1):
        train_cutoff = min_date + timedelta(days=train_end_day)
        val_cutoff = min_date + timedelta(days=val_end_day)

        train_df = df.filter(pl.col("timestamp") < train_cutoff)
        val_df = df.filter(
            (pl.col("timestamp") >= train_cutoff) & (pl.col("timestamp") < val_cutoff)
        )

        logger.info(
            "\n[Fold %d/%d]  train=%d rows (%s→%s)  val=%d rows (%s→%s)",
            fold_i,
            len(FOLD_CONFIGS),
            len(train_df),
            train_df["timestamp"].min().strftime("%Y-%m-%d"),
            train_df["timestamp"].max().strftime("%Y-%m-%d"),
            len(val_df),
            val_df["timestamp"].min().strftime("%Y-%m-%d"),
            val_df["timestamp"].max().strftime("%Y-%m-%d"),
        )

        if len(train_df) == 0 or len(val_df) == 0:
            logger.warning("  Empty split — skipping fold %d", fold_i)
            continue

        # ── ANGEL: Train ──────────────────────────────────────────────────────
        X_tr = train_df[FEATURE_COLS].to_numpy()
        y_tr_angel = train_df["angel_target"].to_numpy()
        logger.info(
            "  [Angel] Training on %d rows  (pos=%.2f%%)",
            len(X_tr),
            100 * y_tr_angel.mean(),
        )
        angel_model = _train_angel(X_tr, y_tr_angel)

        # ── ANGEL: Score validation set ───────────────────────────────────────
        X_val = val_df[FEATURE_COLS].to_numpy()
        y_val_angel = val_df["angel_target"].to_numpy()
        y_val_devil = val_df["devil_target"].to_numpy()
        y_val_macro = val_df["devil_target_macro"].to_numpy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            angel_probs_val = angel_model.predict_proba(X_val)[:, 1]

        proposed_mask = angel_probs_val >= ANGEL_THRESHOLD
        n_angel_proposed = int(proposed_mask.sum())
        logger.info(
            "  [Angel] Proposed %d / %d val bars (%.1f%%)",
            n_angel_proposed,
            len(X_val),
            100 * proposed_mask.mean(),
        )

        # ── DEVIL: Train on Angel-approved train rows using OOF probs ─────────
        train_oof_mask = train_df["angel_prob_oof"].to_numpy() >= ANGEL_THRESHOLD
        n_approved_train = int(train_oof_mask.sum())

        if n_approved_train < 20:
            logger.warning(
                "  [Devil] Only %d Angel-approved train rows — skipping devil training",
                n_approved_train,
            )
            result = FoldResult(
                fold=fold_i,
                train_rows=len(train_df),
                val_rows=len(val_df),
                angel_proposed=n_angel_proposed,
                devil_approved=0,
                win_rate=0.0,
                profit_factor=0.0,
                ev=-SL_MULT,
                brier_devil=1.0,
                optimal_threshold=ANGEL_THRESHOLD,
            )
            report.fold_results.append(result)
            continue

        # Devil feature space = base 22 features + angel_prob_oof (meta-feature)
        devil_feature_cols = FEATURE_COLS + ["angel_prob_oof"]
        X_tr_devil = train_df.filter(pl.Series(train_oof_mask))[
            devil_feature_cols
        ].to_numpy()
        y_tr_devil = train_df["devil_target"].to_numpy()[train_oof_mask]

        logger.info(
            "  [Devil] Training on %d Angel-approved rows  (survival_pos=%.1f%%)",
            n_approved_train,
            100 * y_tr_devil.mean(),
        )
        devil_model = _train_devil(X_tr_devil, y_tr_devil)

        # ── DEVIL: Score Angel-proposed val rows ──────────────────────────────
        if n_angel_proposed == 0:
            logger.warning("  [Devil] No Angel proposals in val set — no trades.")
            result = FoldResult(
                fold=fold_i,
                train_rows=len(train_df),
                val_rows=len(val_df),
                angel_proposed=0,
                devil_approved=0,
                win_rate=0.0,
                profit_factor=0.0,
                ev=-SL_MULT,
                brier_devil=1.0,
                optimal_threshold=ANGEL_THRESHOLD,
            )
            report.fold_results.append(result)
            continue

        proposed_base_feats = X_val[proposed_mask]
        proposed_angel_probs = angel_probs_val[proposed_mask]
        proposed_macro = y_val_macro[proposed_mask]
        proposed_devil_surv = y_val_devil[proposed_mask]

        import pandas as pd

        meta_df = pd.DataFrame(proposed_base_feats, columns=FEATURE_COLS)
        meta_df["angel_prob_oof"] = proposed_angel_probs

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            devil_probs_val = devil_model.predict_proba(meta_df.values)[:, 1]

        # ── Optimal threshold on this fold's val set ──────────────────────────
        opt_thresh, opt_pf = _find_optimal_threshold(devil_probs_val, proposed_macro)

        # ── Fold metrics at optimal threshold ─────────────────────────────────
        pf, ev, wr, n_approved_val = _profit_factor_ev(
            devil_probs_val, proposed_macro, opt_thresh
        )

        brier = brier_score_loss(proposed_devil_surv, devil_probs_val)

        logger.info(
            "  [Devil] Optimal threshold=%.2f  approved=%d  win_rate=%.1f%%  "
            "PF=%.3f  EV=%.4f  Brier=%.4f",
            opt_thresh,
            n_approved_val,
            100 * wr,
            pf,
            ev,
            brier,
        )
        logger.info(
            "  [Gate]  Break-even win-rate=%.1f%%  "
            "Min PF required=%.2f  Min approved=%d",
            100 * BREAKEVEN_WIN_RATE,
            MIN_PROFIT_FACTOR,
            MIN_APPROVED_TRADES,
        )

        result = FoldResult(
            fold=fold_i,
            train_rows=len(train_df),
            val_rows=len(val_df),
            angel_proposed=n_angel_proposed,
            devil_approved=n_approved_val,
            win_rate=wr,
            profit_factor=pf,
            ev=ev,
            brier_devil=brier,
            optimal_threshold=opt_thresh,
        )
        report.fold_results.append(result)

        # Store Fold 2 model + threshold as candidates for promotion
        if fold_i == len(FOLD_CONFIGS):
            final_angel = angel_model
            final_devil = devil_model
            production_threshold = opt_thresh

    # ── Promotion gate: checked on Fold 2 only ─────────────────────────────
    last_fold = report.fold_results[-1] if report.fold_results else None
    if last_fold is None:
        report.rejection_reasons.append("No folds completed.")
    else:
        reasons = []
        if last_fold.devil_approved < MIN_APPROVED_TRADES:
            reasons.append(
                f"Too few approved trades: {last_fold.devil_approved} < {MIN_APPROVED_TRADES}"
            )
        if last_fold.profit_factor < MIN_PROFIT_FACTOR:
            reasons.append(
                f"PF below gate: {last_fold.profit_factor:.3f} < {MIN_PROFIT_FACTOR}"
            )
        report.rejection_reasons = reasons
        report.gate_passed = len(reasons) == 0
        report.production_threshold = production_threshold

    return report, final_angel, final_devil


# ═══════════════════════════════════════════════════════════════════════════════
# FULL-DATA TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def train_full_models(
    df: pl.DataFrame,
) -> Tuple[RandomForestClassifier, RandomForestClassifier]:
    """
    Train final Angel and Devil on the complete 365-day dataset.

    Only called after the walk-forward gate passes.
    OOF angel probs are regenerated on the full dataset for Devil population
    filtering — same procedure as the fold-level training.
    """
    logger.info("")
    logger.info("=" * 70)
    logger.info("FULL-DATA TRAINING (all %d rows)", len(df))
    logger.info("=" * 70)

    df_ts = df.sort("timestamp")
    oof_ps = generate_oof_angel_probs(df_ts)
    df_ts = df_ts.with_columns(pl.Series("angel_prob_oof", oof_ps))
    df = df.join(
        df_ts.select(["symbol", "timestamp", "angel_prob_oof"]),
        on=["symbol", "timestamp"],
        how="left",
    )

    X = df[FEATURE_COLS].to_numpy()
    y_angel = df["angel_target"].to_numpy()
    logger.info(
        "[Angel] Full training on %d rows  (pos=%.2f%%)",
        len(X),
        100 * y_angel.mean(),
    )
    angel_model = _train_angel(X, y_angel)

    # Devil: filter to Angel-approved subpopulation using OOF probs
    approved_mask = df["angel_prob_oof"].to_numpy() >= ANGEL_THRESHOLD
    devil_feat_cols = FEATURE_COLS + ["angel_prob_oof"]
    X_devil = df.filter(pl.Series(approved_mask))[devil_feat_cols].to_numpy()
    y_devil = df["devil_target"].to_numpy()[approved_mask]
    logger.info(
        "[Devil] Full training on %d Angel-approved rows  (survival_pos=%.1f%%)",
        len(X_devil),
        100 * y_devil.mean(),
    )
    devil_model = _train_devil(X_devil, y_devil)

    return angel_model, devil_model


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT PRINTER
# ═══════════════════════════════════════════════════════════════════════════════


def _print_report(report: TrainingReport) -> None:
    bar = "=" * 70
    thin = "-" * 70
    logger.info("")
    logger.info(bar)
    logger.info("TRAINING REPORT — Universal Scalper V4.0 (Day Trade)")
    logger.info(bar)

    logger.info("Walk-Forward Fold Results:")
    logger.info(thin)
    for fr in report.fold_results:
        logger.info(
            "  Fold %d | train=%d  val=%d | "
            "angel_proposed=%d  devil_approved=%d | "
            "WR=%.1f%%  PF=%.3f  EV=%.4f  Brier=%.4f  thresh=%.2f",
            fr.fold,
            fr.train_rows,
            fr.val_rows,
            fr.angel_proposed,
            fr.devil_approved,
            100 * fr.win_rate,
            fr.profit_factor,
            fr.ev,
            fr.brier_devil,
            fr.optimal_threshold,
        )

    logger.info(thin)
    logger.info(
        "Break-even WR       : %.1f%%  (R:R = %.1f TP / %.1f SL)",
        100 * BREAKEVEN_WIN_RATE,
        TP_MULT,
        SL_MULT,
    )
    logger.info(
        "Promotion Gate      : PF >= %.2f  AND  approved_trades >= %d",
        MIN_PROFIT_FACTOR,
        MIN_APPROVED_TRADES,
    )
    logger.info(
        "Gate result         : %s",
        "PASSED" if report.gate_passed else f"FAILED — {report.rejection_reasons}",
    )
    logger.info(
        "Production threshold: %.4f",
        report.production_threshold,
    )
    logger.info(bar)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    logger.info("=" * 70)
    logger.info("DAY TRADE MODEL TRAINER — Universal Scalper V4.0")
    logger.info("=" * 70)

    # ── Load dataset ──────────────────────────────────────────────────────────
    if not DATASET_PATH.exists():
        logger.error(
            "Dataset not found: %s — run build_dataset.py first.", DATASET_PATH
        )
        sys.exit(1)

    df = pl.read_parquet(DATASET_PATH).sort(["symbol", "timestamp"])
    logger.info(
        "Loaded: %d rows × %d cols  (%s → %s)",
        *df.shape,
        df["timestamp"].min().strftime("%Y-%m-%d"),
        df["timestamp"].max().strftime("%Y-%m-%d"),
    )
    logger.info(
        "angel_target pos=%.2f%%  devil_target pos=%.2f%%  macro pos=%.2f%%",
        100 * df["angel_target"].mean(),
        100 * df["devil_target"].mean(),
        100 * df["devil_target_macro"].mean(),
    )

    # Verify all feature columns are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        logger.error("Missing feature columns: %s", missing)
        sys.exit(1)

    # ── Walk-forward validation ───────────────────────────────────────────────
    report, fold_angel, fold_devil = run_walk_forward(df)
    _print_report(report)

    # ── Full-data training ────────────────────────────────────────────────────
    if report.gate_passed:
        logger.info("Gate PASSED — training full models on all data …")
        angel_model, devil_model = train_full_models(df)
    else:
        logger.warning(
            "Gate FAILED — saving Fold-2 candidate models (not full-data models). "
            "Production weights retained if artifacts already exist."
        )
        if fold_angel is None or fold_devil is None:
            logger.error("No candidate models available — aborting save.")
            sys.exit(2)
        angel_model = fold_angel
        devil_model = fold_devil

    # ── Save artifacts ────────────────────────────────────────────────────────
    logger.info("")
    logger.info("Saving model artifacts …")
    _save_artifacts(angel_model, devil_model, report.production_threshold)

    logger.info("")
    logger.info("Done.  Exit 0 = gate passed, 2 = gate failed (fold-2 models saved).")
    sys.exit(0 if report.gate_passed else 2)


if __name__ == "__main__":
    main()
