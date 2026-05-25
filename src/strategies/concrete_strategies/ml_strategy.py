"""
Meta-Labeling Machine Learning Trading Strategy (Angel & Devil Architecture).

Implements a two-stage inference system with hot-reloading:
1. The Angel (Primary Model): Learns Direction (high recall, threshold 0.40)
2. The Devil (Meta Model): Learns Conviction (high precision, threshold 0.50)

Usage:
    from strategies.concrete_strategies.ml_strategy import MLStrategy

    strategy = MLStrategy(
        angel_path="models/angel_latest.pkl",
        devil_path="models/devil_latest.pkl",
        angel_threshold=0.40,
        devil_threshold=0.50,
        warmup_period=60
    )
"""

import json
import logging
import os
import threading
import warnings
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Optional

import numpy as np
import polars as pl

warnings.filterwarnings("ignore", message=".*join_asof.*")

from strategies.base import BaseStrategy, Signal
from core.notification_manager import NotificationManager

# CRITICAL: Import FeaturePipeline to prevent training/inference skew
from ml.feature_pipeline import FeaturePipeline
from ml.features.v3_features import V3BaseFeatures, V3HTFFeatures, V3SessionFeatures
from ml.regimes.hmm_regime import (
    HMM_OUTPUT_COLS,
    load_hmm_models,
    predict_regime_probs,
)
from ml.trainers.v3_rf_trainer import V3RandomForestTrainer

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """
    Meta-Labeling ML strategy using two-stage Angel/Devil architecture.

    The Angel (primary model) proposes trades with high recall.
    The Devil (meta model) filters false positives with high precision.

    Parameters
    ----------
    angel_path : str | Path
        Path to the Angel (primary) model joblib file.
    devil_path : str | Path
        Path to the Devil (meta) model joblib file.
    angel_threshold : float
        Probability threshold for Angel to propose a trade (default: 0.40).
    devil_threshold : float
        Probability threshold for Devil to approve a trade (default: 0.50).
    warmup_period : int
        Minimum candles required before trading (default: 260).
    """

    def __init__(
        self,
        asset_class: str = "equities",
        angel_path: str | Path = None,
        devil_path: str | Path = None,
        angel_threshold: float = 0.40,
        devil_threshold: float = 0.50,
        warmup_period: int = 260,  # default for 1m base / 5m HTF
        timeframe: int = 1,
        htf_timeframe: str = "5m",
        angel_trainer=None,
        devil_trainer=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.asset_class = asset_class
        if angel_path is None:
            angel_path = f"models/{asset_class}/angel_latest.pkl"
        if devil_path is None:
            devil_path = f"models/{asset_class}/devil_latest.pkl"

        self._reload_lock = threading.Lock()
        self.timeframe = timeframe
        self.warmup = warmup_period
        self.angel_threshold = angel_threshold
        self.devil_threshold = devil_threshold

        # Load both models
        angel_file = Path(angel_path)
        devil_file = Path(devil_path)

        if not angel_file.exists():
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            angel_file = project_root / angel_path

        if not devil_file.exists():
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            devil_file = project_root / devil_path

        # Store model paths for hot-reloading
        self.angel_path = angel_file
        self.devil_path = devil_file

        # Load models and track modification times
        logger.info(f"Loading Angel model from {angel_file}")
        self.angel_trainer = (
            angel_trainer if angel_trainer is not None else V3RandomForestTrainer()
        )
        self.angel_trainer.load(str(angel_file))
        if hasattr(self.angel_trainer, "model") and hasattr(
            self.angel_trainer.model, "n_jobs"
        ):
            self.angel_trainer.model.n_jobs = (
                1  # Prevent joblib IPC overhead on single-row inference
            )

        self.angel_mtime = os.path.getmtime(angel_file)
        logger.info(f"Angel model loaded via trainer (mtime: {self.angel_mtime})")

        logger.info(f"Loading Devil model from {devil_file}")
        self.devil_trainer = (
            devil_trainer if devil_trainer is not None else V3RandomForestTrainer()
        )
        self.devil_trainer.load(str(devil_file))
        if hasattr(self.devil_trainer, "model") and hasattr(
            self.devil_trainer.model, "n_jobs"
        ):
            self.devil_trainer.model.n_jobs = (
                1  # Prevent joblib IPC overhead on single-row inference
            )

        self.devil_mtime = os.path.getmtime(devil_file)
        logger.info(f"Devil model loaded via trainer (mtime: {self.devil_mtime})")

        # Initialize notification manager for hot-reload alerts
        self.notification_manager = NotificationManager()

        # Initialize feature pipeline (imported, not duplicated!)
        self.pipeline = FeaturePipeline(
            feature_generators=[
                V3BaseFeatures(),
                V3HTFFeatures(timeframe=htf_timeframe),
                V3SessionFeatures(),
            ]
        )

        # Feature columns (excluding absolute price columns to prevent leakage).
        # Source the schema from the trained model itself so a retrain that
        # changes the feature space (e.g. enabling the HMM regime experiment
        # via RETRAIN_USE_HMM=1) propagates here without a code edit.
        model_features = self.angel_trainer.feature_names_in_
        if model_features is None:
            raise RuntimeError(
                "Loaded Angel model exposes no feature_names_in_ — cannot "
                "establish inference schema. Re-run retrainer with a model "
                "type that records feature names (sklearn / LightGBM)."
            )
        self.feature_names = list(model_features)
        logger.info(
            "MLStrategy feature schema sourced from model: %d features",
            len(self.feature_names),
        )

        # If the model trained on HMM regime probs, load the per-symbol HMM
        # artifact persisted alongside Angel/Devil and apply it at inference.
        self.hmm_models: Optional[dict] = None
        if any(c in self.feature_names for c in HMM_OUTPUT_COLS):
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            hmm_path = project_root / "models" / self.asset_class / "hmm_latest.pkl"
            try:
                self.hmm_models = load_hmm_models(hmm_path)
                fitted = sum(1 for m in self.hmm_models.values() if m is not None)
                logger.info(
                    "MLStrategy loaded HMM regime artifact: %s (%d/%d symbols fitted)",
                    hmm_path, fitted, len(self.hmm_models),
                )
            except FileNotFoundError:
                raise RuntimeError(
                    f"Angel model expects HMM features but {hmm_path} is missing. "
                    "Re-run the retrainer with RETRAIN_USE_HMM=1 so the HMM "
                    "artifact is persisted alongside the model."
                )

        # Validate metadata sidecar
        self._validate_metadata()

        # Override devil_threshold with the value persisted by the retrainer
        # (models/threshold.json).  Must be called AFTER self.devil_threshold is
        # set above so _load_threshold() can use it as a fallback.
        self.devil_threshold = self._load_threshold()

        # Heartbeat state: every N bars per symbol, log a summary of the
        # angel_prob distribution. Lets the operator see the model is
        # actively evaluating even when no signals fire (most rejections
        # happen at logger.debug, which the project's logging setup
        # silently suppresses at INFO root level).
        self._heartbeat_window: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=30)
        )
        self._heartbeat_counter: Dict[str, int] = defaultdict(int)
        self._heartbeat_every_n_bars = int(
            os.getenv("MLSTRATEGY_HEARTBEAT_EVERY_N", "15")
        )

    def _validate_metadata(self) -> None:
        """
        Validate that the loaded model matches the expected asset class using the metadata sidecar.
        """
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        metadata_path = project_root / "models" / self.asset_class / "metadata.json"
        
        if not metadata_path.exists():
            logger.warning(
                "_validate_metadata: %s not found. Skipping distribution drift check.", 
                metadata_path
            )
            return
            
        try:
            with open(metadata_path, "r") as fh:
                data = json.load(fh)
                
            trained_class = data.get("asset_class")
            if trained_class != self.asset_class:
                raise RuntimeError(
                    f"Distribution drift detected: Strategy instantiated for asset class '{self.asset_class}', "
                    f"but model was trained on '{trained_class}'."
                )
            logger.info("_validate_metadata: passed (asset_class=%s)", trained_class)
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                raise
            logger.warning("_validate_metadata: failed to read %s (%s)", metadata_path, exc)

    def _load_threshold(self) -> float:
        """
        Load the Devil model's optimal threshold from models/<asset_class>/threshold.json.

        Written by retrainer.save_threshold() after a successful validation gate.
        Falls back to self.devil_threshold (the value passed to __init__) if the
        file is absent or corrupt.

        Returns:
            float: The production threshold for Devil approval decisions.
        """
        # Search relative to project root (4 levels up from this file in src/)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        threshold_path = project_root / "models" / self.asset_class / "threshold.json"
        if not threshold_path.exists():
            logger.warning(
                "_load_threshold: models/threshold.json not found — "
                "using constructor default devil_threshold=%.2f",
                self.devil_threshold,
            )
            return self.devil_threshold
        try:
            with open(threshold_path, "r") as fh:
                data = json.load(fh)
            threshold = float(data["devil_threshold"])
            logger.info(
                "_load_threshold: loaded production threshold=%.4f from %s",
                threshold,
                threshold_path,
            )
            return threshold
        except Exception as exc:
            logger.warning(
                "_load_threshold: failed to read %s (%s) — "
                "using constructor default devil_threshold=%.2f",
                threshold_path,
                exc,
                self.devil_threshold,
            )
            return self.devil_threshold

    @property
    def warmup_period(self) -> int:
        """Returns minimum candles required for indicators to warm up."""
        return self.warmup

    def _check_model_updates(self) -> bool:
        """
        Check for model file updates and hot-reload if necessary.

        Monitors the modification times of model files and reloads
        models in memory if they have been updated on disk.

        Returns:
            bool: True if any model was reloaded, False otherwise.
        """
        reloaded = False

        try:
            # Check Angel model
            if self.angel_path.exists():
                current_angel_mtime = os.path.getmtime(self.angel_path)
                if current_angel_mtime > self.angel_mtime:
                    logger.info(
                        f"[HOT-RELOAD] Detected new Angel model: {self.angel_path}"
                    )
                    try:
                        with self._reload_lock:
                            self.angel_trainer.load(str(self.angel_path))
                            if hasattr(self.angel_trainer, "model") and hasattr(
                                self.angel_trainer.model, "n_jobs"
                            ):
                                self.angel_trainer.model.n_jobs = 1
                            self.angel_mtime = current_angel_mtime
                        logger.info(f"[HOT-RELOAD] Angel model updated successfully")
                        reloaded = True
                    except Exception as e:
                        logger.error(f"[HOT-RELOAD] Failed to reload Angel model: {e}")

            # Check Devil model
            if self.devil_path.exists():
                current_devil_mtime = os.path.getmtime(self.devil_path)
                if current_devil_mtime > self.devil_mtime:
                    logger.info(
                        f"[HOT-RELOAD] Detected new Devil model: {self.devil_path}"
                    )
                    try:
                        self.devil_trainer.load(str(self.devil_path))
                        if hasattr(self.devil_trainer, "model") and hasattr(
                            self.devil_trainer.model, "n_jobs"
                        ):
                            self.devil_trainer.model.n_jobs = 1
                        self.devil_mtime = current_devil_mtime
                        logger.info(f"[HOT-RELOAD] Devil model updated successfully")
                        reloaded = True
                    except Exception as e:
                        logger.error(f"[HOT-RELOAD] Failed to reload Devil model: {e}")

            # Send notification if any model was reloaded
            if reloaded:
                # Also reload the threshold — a retrain always produces a new
                # threshold.json alongside the new model weights.
                old_threshold = self.devil_threshold
                self.devil_threshold = self._load_threshold()
                if self.devil_threshold != old_threshold:
                    logger.info(
                        "[HOT-RELOAD] Devil threshold updated: %.4f -> %.4f",
                        old_threshold,
                        self.devil_threshold,
                    )

                alert_message = (
                    "🔄 [HOT-RELOAD] New model weights ingested from disk. "
                    f"Angel: {self.angel_path.name}, Devil: {self.devil_path.name} "
                    f"| devil_threshold={self.devil_threshold:.4f}"
                )
                logger.critical(alert_message)
                self.notification_manager.send_system_message(alert_message)

        except Exception as e:
            logger.error(f"[HOT-RELOAD] Error checking for model updates: {e}")

        return reloaded

    def generate_signals(self, df: pl.DataFrame) -> Optional[Signal]:
        """
        Analyze single-symbol market data using two-stage Meta-Labeling.

        Stage 1: Angel proposes trades (high recall, low threshold).
        Stage 2: Devil filters false positives (high precision).

        Args:
            df: Polars DataFrame with OHLCV data for a single symbol.
                Must contain a 'symbol' column (added by callers) so the
                strategy can tag emitted signals with their instrument.

        Returns:
            base.Signal on joint Angel & Devil approval, or None.
        """
        # Check for model updates at the start of each bar processing cycle
        self._check_model_updates()

        self.validate_input(df)

        if len(df) < self.warmup_period:
            logger.debug(f"Insufficient data ({len(df)} < {self.warmup_period})")
            return None

        try:
            # Generate features using imported FeatureEngineer
            features_df = self._generate_features(df)

            if features_df is None or len(features_df) == 0:
                return None

            # Get latest bar's features for prediction. Pass a pandas
            # DataFrame (with column names) so LightGBM does not emit a
            # per-call UserWarning about missing feature names. Predictions
            # are identical either way (positional matching), but the live
            # log fills with warnings without this.
            latest_features_df = features_df[self.feature_names].tail(1)
            X_angel = latest_features_df.to_pandas()

            # Get current price for signal
            current_price = float(df["close"].tail(1)[0])

            # Resolve symbol — callers add this as a literal column before
            # invoking generate_signals (Option A design).
            symbol = str(df["symbol"].tail(1)[0]) if "symbol" in df.columns else None

            # ═══════════════════════════════════════════════════════════
            # STAGE 1: THE ANGEL (DIRECTION)
            # ═══════════════════════════════════════════════════════════
            angel_prob = self.angel_trainer.predict_proba(X_angel)[0, 1]

            # Heartbeat: track this bar's prob and periodically emit a
            # per-symbol distribution summary so silence in the logs is
            # distinguishable from a hung evaluation loop.
            heartbeat_key = symbol or "_anon"
            self._heartbeat_window[heartbeat_key].append(float(angel_prob))
            self._heartbeat_counter[heartbeat_key] += 1
            if self._heartbeat_counter[heartbeat_key] >= self._heartbeat_every_n_bars:
                probs = list(self._heartbeat_window[heartbeat_key])
                proposed = sum(1 for p in probs if p >= self.angel_threshold)
                logger.info(
                    "[%s] Heartbeat: last %d bars angel_prob "
                    "median=%.3f p75=%.3f max=%.3f | proposed=%d/%d (%.1f%%) "
                    "vs threshold=%.2f",
                    heartbeat_key,
                    len(probs),
                    float(np.median(probs)),
                    float(np.percentile(probs, 75)),
                    float(np.max(probs)),
                    proposed,
                    len(probs),
                    100.0 * proposed / len(probs),
                    self.angel_threshold,
                )
                self._heartbeat_counter[heartbeat_key] = 0

            if angel_prob < self.angel_threshold:
                logger.debug(
                    f"[{symbol}] Angel rejected | Prob: {angel_prob:.4f} < {self.angel_threshold}"
                )
                return None

            logger.debug(f"[{symbol}] Angel proposed trade | Prob: {angel_prob:.4f}")

            # ═══════════════════════════════════════════════════════════
            # STAGE 2: THE DEVIL (CONVICTION)
            # ═══════════════════════════════════════════════════════════
            # Build meta-feature frame: base features + Angel's probability,
            # preserving column names so LightGBM does not warn here either.
            X_devil = X_angel.copy()
            X_devil["angel_prob"] = angel_prob

            devil_prob = self.devil_trainer.predict_proba(X_devil)[0, 1]

            if devil_prob < self.devil_threshold:
                logger.debug(
                    f"[{symbol}] Devil veto | Angel: {angel_prob:.2f}, Devil: {devil_prob:.2f} < {self.devil_threshold}"
                )
                return None

            # Both Angel and Devil agree — emit raw ATR volatility.
            # RiskManager applies multipliers and floor checks (Path Alpha).
            natr_value = float(latest_features_df["natr_14"].to_numpy()[0])
            # TA-Lib NATR is a percentage; convert to absolute ATR
            atr_abs = (natr_value / 100.0) * current_price

            logger.info(
                f"[{symbol}] ANGEL & DEVIL AGREEMENT | "
                f"Price={current_price:.2f} | "
                f"Angel Prob: {angel_prob:.2f} | "
                f"Devil Prob: {devil_prob:.2f} | "
                f"raw_ATR={atr_abs:.4f}"
            )

            return Signal(
                direction="long",
                entry_price=current_price,
                raw_sl_distance=atr_abs,
                raw_tp_distance=atr_abs,
                metadata={
                    "symbol": symbol,
                    "angel_prob": float(angel_prob),
                    "devil_prob": float(devil_prob),
                    "atr_abs": atr_abs,
                    "timestamp": df["timestamp"].tail(1)[0],
                },
            )

        except Exception as e:
            logger.error(f"[{symbol}] Error in ML analysis: {e}", exc_info=True)
            return None

    def _generate_features(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        Generate ML features using imported FeaturePipeline.

        This method ensures zero training/inference skew by using the exact
        same feature computation logic as the training pipeline.

        Args:
            df: Raw OHLCV DataFrame.

        Returns:
            DataFrame with computed features, or None if insufficient data.
        """
        try:
            # Use imported FeaturePipeline.run(). clean_data filters its
            # null-drop subset to columns that exist in the frame, so HMM
            # cols added after this call don't interfere with base cleaning.
            features_df = self.pipeline.run(df, feature_cols=self.feature_names)

            # Append HMM regime posteriors if the loaded model expects them.
            if self.hmm_models is not None:
                features_df = predict_regime_probs(features_df, self.hmm_models)

            return features_df

        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            return None
