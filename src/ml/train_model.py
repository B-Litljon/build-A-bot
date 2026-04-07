"""
Meta-Labeling Model Training Pipeline (Angel & Devil Architecture).

Implements a two-stage training system:
1. The Angel (Primary Model): Learns Direction (high recall)
2. The Devil (Meta Model): Learns Conviction (high precision)

Usage:
    python -m ml.train_model
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, List

import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, precision_score, roc_auc_score, recall_score

from ml.core.interfaces import BaseTrainer
from ml.trainers.v3_rf_trainer import V3RandomForestTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

_PROCESSED_DIR = Path("data/processed")
_MODEL_DIR = Path("src/ml/models")

# Meta-Labeling Configuration
ANGEL_THRESHOLD = 0.40
DEVIL_THRESHOLD = 0.50

# Columns to exclude from features (prevent data leakage)
EXCLUDE_COLS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "target",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "sma_50",
    # V3.3: HTF intermediate columns (safety net — _compute_htf_features drops these,
    # but list them here in case any code path bypasses the pipeline cleanup)
    "_htf_sma_50",
    "_htf_bb_upper",
    "_htf_bb_lower",
    "_htf_bb_middle",
    "htf_open",
    "htf_high",
    "htf_low",
    "htf_close",
    "htf_volume",
    "available_at",
]


class ModelTrainingOrchestrator:
    def __init__(self, angel_trainer: BaseTrainer, devil_trainer: BaseTrainer):
        self.angel_trainer = angel_trainer
        self.devil_trainer = devil_trainer

    def train(self, X: pd.DataFrame, y: pd.Series):
        logger.info(f"Label distribution: 0={(y == 0).sum():,}, 1={(y == 1).sum():,}")

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 1: TRAIN THE ANGEL (DIRECTION)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 1: TRAINING THE ANGEL (DIRECTION - HIGH RECALL)")
        logger.info("=" * 70)

        self.angel_trainer.train(X, y)
        logger.info("Angel model training complete.")

        # Generate Out-Of-Fold probabilities using TimeSeriesSplit to train the Devil without leakage
        logger.info(
            "Generating manual CV predictions for Meta-Labeling via TimeSeriesSplit..."
        )

        tscv = TimeSeriesSplit(n_splits=3)
        angel_cv_probs = np.zeros(len(y))

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train = y.iloc[train_index]

            # Use the interface strictly
            self.angel_trainer.train(X_train, y_train)
            fold_probs = self.angel_trainer.predict_proba(X_test)[:, 1]
            angel_cv_probs[test_index] = fold_probs

        # Refit on full dataset for the final model
        self.angel_trainer.train(X, y)

        # Angel triggers a proposed trade at a low threshold (High Recall)
        angel_cv_preds = (angel_cv_probs >= ANGEL_THRESHOLD).astype(int)
        n_trades_proposed = angel_cv_preds.sum()
        logger.info(
            f"Angel proposed {n_trades_proposed:,} trades at threshold {ANGEL_THRESHOLD}"
        )

        # Log Universal Metrics
        if len(np.unique(y)) > 1:
            auc = roc_auc_score(y, angel_cv_probs)
            logger.info(f"Angel ROC-AUC on training: {auc:.3f}")
        logger.info(f"Angel recall on training: {(angel_cv_preds == y).sum() / len(y):.3f}")

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 2: TRAIN THE DEVIL (CONVICTION)
        # ═══════════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 70)
        logger.info("STAGE 2: TRAINING THE DEVIL (CONVICTION - HIGH PRECISION)")
        logger.info("=" * 70)

        # Meta-target: 1 if Angel was right (True Positive), 0 if Angel was wrong (False Positive)
        meta_y = pd.Series((angel_cv_preds == y).astype(int), index=y.index)

        # We only train the Devil on trades the Angel actually suggested taking
        trade_indices = np.where(angel_cv_preds == 1)[0]
        logger.info(f"Training Devil on {len(trade_indices):,} Angel-proposed trades...")

        if len(trade_indices) == 0:
            raise ValueError(
                "Angel model generated 0 trades. Lower the ANGEL_THRESHOLD or check features."
            )

        # Build meta-feature set: original features + Angel's probability
        X_meta = X.iloc[trade_indices].copy()
        X_meta["angel_prob"] = angel_cv_probs[trade_indices]
        y_meta = meta_y.iloc[trade_indices]

        logger.info(
            f"Meta-label distribution: 0={(y_meta == 0).sum():,}, 1={(y_meta == 1).sum():,}"
        )

        self.devil_trainer.train(X_meta, y_meta)
        logger.info("Devil model training complete.")

        # ═══════════════════════════════════════════════════════════════════
        # EVALUATION
        # ═══════════════════════════════════════════════════════════════════
        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION: DEVIL'S FILTER PERFORMANCE")
        logger.info("=" * 70)

        # In-sample Devil evaluation on the Angel's proposed trades
        # We need a predict method, let's assume predict is available on trainer
        devil_preds = getattr(self.devil_trainer, "predict", lambda x: (self.devil_trainer.predict_proba(x)[:, 1] >= 0.5).astype(int))(X_meta)
        devil_proba = self.devil_trainer.predict_proba(X_meta)[:, 1]

        print("\nDevil Classification Report (In-Sample on Angel's Trades):")
        print(classification_report(y_meta, devil_preds, target_names=["Wrong", "Right"]))

        devil_precision = precision_score(y_meta, devil_preds)
        devil_recall = recall_score(y_meta, devil_preds)
        logger.info(f"Devil Precision: {devil_precision:.3f}")
        logger.info(f"Devil Recall: {devil_recall:.3f}")
        if len(np.unique(y_meta)) > 1:
            devil_auc = roc_auc_score(y_meta, devil_proba)
            logger.info(f"Devil ROC-AUC: {devil_auc:.3f}")

        # Calculate combined system metrics
        final_predictions = np.zeros(len(y), dtype=int)
        final_predictions[trade_indices] = devil_preds

        # Only count as final BUY if both Angel and Devil agree
        final_buys = np.where((angel_cv_preds == 1) & (final_predictions == 1))[0]
        logger.info(f"Final system would emit {len(final_buys):,} BUY signals")
        logger.info(f"Signal rate: {len(final_buys) / len(y) * 100:.3f}%")

        # Calculate what precision we'd get on those final signals
        if len(final_buys) > 0:
            final_precision = y.iloc[final_buys].mean()
            logger.info(f"Estimated Final System Precision: {final_precision:.3f}")

    def save_models(self, angel_path: Path, devil_path: Path):
        self.angel_trainer.save(str(angel_path))
        self.devil_trainer.save(str(devil_path))

        angel_size = angel_path.stat().st_size / (1024 * 1024)
        devil_size = devil_path.stat().st_size / (1024 * 1024)

        logger.info(f"Angel model saved to {angel_path} ({angel_size:.1f} MB)")
        logger.info(f"Devil model saved to {devil_path} ({devil_size:.1f} MB)")


def main():
    data_path = _PROCESSED_DIR / "training_data.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")

    logger.info(f"Loading training data from {data_path}...")
    df = pl.read_parquet(data_path)
    logger.info(f"Loaded {len(df):,} rows, {df.width} columns")

    # Get feature columns
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Convert to pandas for sklearn
    X = df[feature_cols].to_pandas()
    y = df["target"].to_pandas()

    angel_trainer = V3RandomForestTrainer(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    devil_trainer = V3RandomForestTrainer(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    orchestrator = ModelTrainingOrchestrator(angel_trainer, devil_trainer)
    orchestrator.train(X, y)

    # ═══════════════════════════════════════════════════════════════════
    # EXPORT MODELS
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("EXPORTING MODELS")
    logger.info("=" * 70)

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)

    angel_path = _MODEL_DIR / "angel_rf_model.joblib"
    devil_path = _MODEL_DIR / "devil_rf_model.joblib"

    orchestrator.save_models(angel_path, devil_path)

    logger.info("\nMeta-Labeling training pipeline complete!")


if __name__ == "__main__":
    main()
