"""
Model training pipeline for the Build-A-Bot ML signal classifier.

Loads the feature-engineered dataset from
``data/processed/training_data.parquet``, trains a Random Forest on a
temporal train/test split, evaluates it with probability threshold tuning,
and persists the artifact to ``src/ml/models/rf_model.joblib``.

Usage (from ``src/``)::

    python -m ml.train_model

The script expects the feature pipeline to have already been run so
that ``training_data.parquet`` exists.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

_SRC_DIR = Path(__file__).resolve().parent.parent  # src/
_PROJECT_ROOT = _SRC_DIR.parent  # build-A-bot/
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_MODEL_DIR = _SRC_DIR / "ml" / "models"

# Columns that carry no predictive signal or would cause data leakage
_DROP_COLS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "target",
]

# Temporal split boundary (train on everything before this date)
_SPLIT_DATE = datetime(2024, 1, 1, tzinfo=timezone.utc)

# Threshold tuning configuration
_THRESHOLD_MIN: float = 0.50
_THRESHOLD_MAX: float = 0.90
_THRESHOLD_STEP: float = 0.05
_PRECISION_TARGET: float = 0.55


class ModelTrainer:
    """
    End-to-end trainer for the RF signal classifier with probability
    threshold optimization for improved precision.

    Parameters
    ----------
    data_path : Path | str
        Path to the feature-engineered Parquet file.
    model_dir : Path | str
        Directory where the trained model artifact will be saved.
    split_date : datetime
        Rows before this date go to train, on or after go to test.
    """

    def __init__(
        self,
        data_path: Path | str = _PROCESSED_DIR / "training_data.parquet",
        model_dir: Path | str = _MODEL_DIR,
        split_date: datetime = _SPLIT_DATE,
    ):
        self._data_path = Path(data_path)
        self._model_dir = Path(model_dir)
        self._split_date = split_date

        # Populated by train()
        self._model: RandomForestClassifier | None = None
        self._feature_names: list[str] = []
        self._X_test: np.ndarray | None = None
        self._y_test: np.ndarray | None = None
        self._y_proba: np.ndarray | None = None
        self._best_threshold: float = 0.50

    # ── public API ────────────────────────────────────────────────────

    def train(self) -> RandomForestClassifier:
        """
        Load data, split temporally, fit a Random Forest, and store
        the test set for evaluation.

        Returns
        -------
        RandomForestClassifier
            The fitted model.
        """
        # ── load ──────────────────────────────────────────────────────
        logger.info("Loading %s ...", self._data_path)
        df = pl.read_parquet(self._data_path)
        logger.info("  %d rows, %d columns", len(df), df.width)

        # ── temporal split ────────────────────────────────────────────
        train_df = df.filter(pl.col("timestamp") < self._split_date)
        test_df = df.filter(pl.col("timestamp") >= self._split_date)
        logger.info(
            "  Time split @ %s  =>  train %d / test %d",
            self._split_date.date(),
            len(train_df),
            len(test_df),
        )

        # ── feature / target separation ───────────────────────────────
        self._feature_names = [c for c in df.columns if c not in _DROP_COLS]
        logger.info(
            "  Features (%d): %s", len(self._feature_names), self._feature_names
        )

        X_train = train_df.select(self._feature_names).to_numpy()
        y_train = train_df["target"].to_numpy().astype(np.int8)
        self._X_test = test_df.select(self._feature_names).to_numpy()
        self._y_test = test_df["target"].to_numpy().astype(np.int8)

        logger.info(
            "  Train label distribution: 0=%d  1=%d  (%.1f%% positive)",
            (y_train == 0).sum(),
            (y_train == 1).sum(),
            100.0 * (y_train == 1).sum() / len(y_train),
        )
        logger.info(
            "  Test  label distribution: 0=%d  1=%d  (%.1f%% positive)",
            (self._y_test == 0).sum(),
            (self._y_test == 1).sum(),
            100.0 * (self._y_test == 1).sum() / len(self._y_test),
        )

        # ── fit ───────────────────────────────────────────────────────
        logger.info("Training RandomForestClassifier (no class balancing)...")
        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        )
        self._model.fit(X_train, y_train)
        logger.info("  Training complete.")

        # ── predict probabilities for threshold tuning ─────────────────
        proba_matrix = self._model.predict_proba(self._X_test)
        self._y_proba = proba_matrix[:, 1]
        logger.info("  Generated probability scores for test set")

        # ── feature importance (quick diagnostic) ─────────────────────
        importances = sorted(
            zip(self._feature_names, self._model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
        logger.info("  Feature importance (top → bottom):")
        for name, imp in importances:
            logger.info("    %-22s  %.4f", name, imp)

        return self._model

    def find_optimal_threshold(self) -> Tuple[float, dict]:
        """
        Test probability thresholds from 0.50 to 0.90 and select the
        optimal one based on precision/recall trade-off.

        Selection Rule:
        1. Find all thresholds where Precision >= 0.55
        2. If found, select the lowest such threshold (most recall)
        3. If none achieve 0.55 precision, select threshold with highest F1

        Returns
        -------
        tuple[float, dict]
            (best_threshold, metrics_at_best_threshold)
        """
        if self._y_proba is None or self._y_test is None:
            raise RuntimeError("Call train() before find_optimal_threshold().")

        logger.info(
            "Testing probability thresholds %.2f to %.2f (step %.2f)...",
            _THRESHOLD_MIN,
            _THRESHOLD_MAX,
            _THRESHOLD_STEP,
        )
        logger.info("-" * 70)
        logger.info(f"{'Threshold':>10} {'Precision':>12} {'Recall':>10} {'F1':>10}")
        logger.info("-" * 70)

        thresholds = np.arange(
            _THRESHOLD_MIN, _THRESHOLD_MAX + _THRESHOLD_STEP, _THRESHOLD_STEP
        )
        results: list[dict] = []
        best_f1 = 0.0
        best_f1_threshold = _THRESHOLD_MIN

        for threshold in thresholds:
            # Apply threshold to get binary predictions
            y_pred = (self._y_proba >= threshold).astype(np.int8)

            # Calculate metrics
            precision = precision_score(self._y_test, y_pred)
            recall = recall_score(self._y_test, y_pred)
            f1 = f1_score(self._y_test, y_pred)

            result = {
                "threshold": round(threshold, 2),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
            results.append(result)

            logger.info(
                f"{threshold:>10.2f} {precision:>12.4f} {recall:>10.4f} {f1:>10.4f}"
            )

            # Track best F1 for fallback
            if f1 > best_f1:
                best_f1 = f1
                best_f1_threshold = threshold

        logger.info("-" * 70)

        # Selection Logic: find lowest threshold with precision >= 0.55
        qualifying = [r for r in results if r["precision"] >= _PRECISION_TARGET]

        if qualifying:
            # Select lowest threshold (most recall) that meets precision target
            best_result = min(qualifying, key=lambda x: x["threshold"])
            self._best_threshold = best_result["threshold"]
            logger.info(
                "SELECTED: Threshold %.2f (Precision=%.4f, Recall=%.4f, F1=%.4f) - "
                "Meets precision target",
                self._best_threshold,
                best_result["precision"],
                best_result["recall"],
                best_result["f1"],
            )
        else:
            # No threshold meets precision target, use highest F1
            self._best_threshold = best_f1_threshold
            best_result = next(
                r for r in results if r["threshold"] == best_f1_threshold
            )
            logger.info(
                "SELECTED: Threshold %.2f (Precision=%.4f, Recall=%.4f, F1=%.4f) - "
                "Best F1 (no threshold met precision target)",
                self._best_threshold,
                best_result["precision"],
                best_result["recall"],
                best_result["f1"],
            )

        return self._best_threshold, best_result

    def evaluate(self, threshold: float | None = None) -> str:
        """
        Print precision / recall / F1 on the held-out test set using
        the optimal or specified probability threshold.

        Parameters
        ----------
        threshold : float | None
            Probability threshold to use. If None, uses the threshold
            found by find_optimal_threshold().

        Returns
        -------
        str
            The full ``classification_report`` text.
        """
        if self._model is None or self._y_proba is None or self._y_test is None:
            raise RuntimeError("Call train() before evaluate().")

        if threshold is None:
            threshold = self._best_threshold

        # Generate predictions using threshold
        y_pred = (self._y_proba >= threshold).astype(np.int8)

        logger.info("\n" + "=" * 70)
        logger.info("FINAL EVALUATION AT THRESHOLD = %.2f", threshold)
        logger.info("=" * 70)

        report = classification_report(
            self._y_test,
            y_pred,
            target_names=["no_trade (0)", "trade (1)"],
            digits=4,
        )
        assert isinstance(report, str), (
            "classification_report should return str when output_dict=False"
        )
        logger.info("Classification Report:\n%s", report)

        cm = confusion_matrix(self._y_test, y_pred)
        logger.info(
            "Confusion Matrix:\n"
            "                 Predicted 0    Predicted 1\n"
            "  Actual 0       %10d    %10d\n"
            "  Actual 1       %10d    %10d",
            cm[0][0],
            cm[0][1],
            cm[1][0],
            cm[1][1],
        )

        # Log signal rate
        signal_rate = y_pred.sum() / len(y_pred) * 100
        logger.info("Signal Rate: %.2f%% of bars trigger trade signal", signal_rate)

        return report

    def save_model(self, filename: str = "rf_model.joblib") -> Path:
        """
        Persist the trained model to disk via ``joblib``.

        Parameters
        ----------
        filename : str
            Name of the output file inside the model directory.

        Returns
        -------
        Path
            Absolute path to the saved artifact.
        """
        if self._model is None:
            raise RuntimeError("Call train() before save_model().")

        self._model_dir.mkdir(parents=True, exist_ok=True)
        out_path = self._model_dir / filename
        joblib.dump(self._model, out_path)

        size_mb = out_path.stat().st_size / (1024 * 1024)
        logger.info("Model saved to %s (%.1f MB)", out_path, size_mb)
        return out_path


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    trainer = ModelTrainer()
    trainer.train()
    trainer.find_optimal_threshold()
    trainer.evaluate()
    trainer.save_model()


if __name__ == "__main__":
    main()
