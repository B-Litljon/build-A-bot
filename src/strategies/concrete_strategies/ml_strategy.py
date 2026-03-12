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
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import polars as pl
import pandas as pd

from core.order_management import OrderParams
from core.signal import Signal, SignalType
from core.notification_manager import NotificationManager
from strategies.strategy import Strategy

# CRITICAL: Import FeatureEngineer to prevent training/inference skew
from ml.feature_pipeline import FeatureEngineer

logger = logging.getLogger(__name__)


class MLStrategy(Strategy):
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
        Minimum candles required before trading (default: 60).
    """

    def __init__(
        self,
        angel_path: str | Path = "src/ml/models/angel_rf_model.joblib",
        devil_path: str | Path = "src/ml/models/devil_rf_model.joblib",
        angel_threshold: float = 0.40,
        devil_threshold: float = 0.50,
        warmup_period: int = 260,  # V3.3: expanded for 5m HTF SMA-50 warm-up
    ):
        super().__init__()

        self.timeframe = 1  # 1-minute bars
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
        self.angel_model = joblib.load(angel_file)
        self.angel_model.n_jobs = (
            1  # Prevent joblib IPC overhead on single-row inference
        )
        self.angel_mtime = os.path.getmtime(angel_file)
        logger.info(
            f"Angel model loaded: {type(self.angel_model).__name__} (mtime: {self.angel_mtime})"
        )

        logger.info(f"Loading Devil model from {devil_file}")
        self.devil_model = joblib.load(devil_file)
        self.devil_model.n_jobs = (
            1  # Prevent joblib IPC overhead on single-row inference
        )
        self.devil_mtime = os.path.getmtime(devil_file)
        logger.info(
            f"Devil model loaded: {type(self.devil_model).__name__} (mtime: {self.devil_mtime})"
        )

        # Initialize notification manager for hot-reload alerts
        self.notification_manager = NotificationManager()

        # Initialize feature engineer (imported, not duplicated!)
        self.feature_engineer = FeatureEngineer()

        # Feature columns (excluding absolute price columns to prevent leakage)
        # V3.3: expanded from 10 to 14 features with multi-timeframe (5m) additions
        self.feature_names = [
            "rsi_14",
            "ppo",
            "natr_14",
            "bb_pct_b",
            "bb_width_pct",
            "price_sma50_ratio",
            "log_return",
            "hour_of_day",
            "dist_sma50",
            "vol_rel",
            # V3.3: HTF features
            "htf_rsi_14",
            "htf_trend_agreement",
            "htf_vol_rel",
            "htf_bb_pct_b",
        ]

        self.order_params = OrderParams(
            risk_percentage=0.02,
            tp_multiplier=1.005,  # 0.5% take profit
            sl_multiplier=0.998,  # 0.2% stop loss
            use_trailing_stop=False,
        )

        # Override devil_threshold with the value persisted by the retrainer
        # (models/threshold.json).  Must be called AFTER self.devil_threshold is
        # set above so _load_threshold() can use it as a fallback.
        self.devil_threshold = self._load_threshold()

    def _load_threshold(self) -> float:
        """
        Load the Devil model's optimal threshold from models/threshold.json.

        Written by retrainer.save_threshold() after a successful validation gate.
        Falls back to self.devil_threshold (the value passed to __init__) if the
        file is absent or corrupt.

        Returns:
            float: The production threshold for Devil approval decisions.
        """
        # Search relative to project root (4 levels up from this file in src/)
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        threshold_path = project_root / "models" / "threshold.json"
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
                        new_angel_model = joblib.load(self.angel_path)
                        self.angel_model = new_angel_model
                        self.angel_model.n_jobs = (
                            1  # Prevent joblib IPC overhead on single-row inference
                        )
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
                        new_devil_model = joblib.load(self.devil_path)
                        self.devil_model = new_devil_model
                        self.devil_model.n_jobs = (
                            1  # Prevent joblib IPC overhead on single-row inference
                        )
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

    def analyze(self, data: Dict[str, pl.DataFrame]) -> Tuple[List[Signal], float]:
        """
        Analyze market data using two-stage Meta-Labeling.

        Stage 1: Angel proposes trades (high recall, low threshold).
        Stage 2: Devil filters false positives (high precision).

        Args:
            data: Dict mapping symbol -> Polars DataFrame with OHLCV data.

        Returns:
            Tuple of (List of BUY signals where both Angel & Devil agree, highest Angel probability).
        """
        # Check for model updates at the start of each bar processing cycle
        self._check_model_updates()

        signals = []
        highest_angel_prob = 0.0

        for symbol, df in data.items():
            if len(df) < self.warmup_period:
                logger.debug(
                    f"{symbol}: Insufficient data ({len(df)} < {self.warmup_period})"
                )
                continue

            try:
                # Generate features using imported FeatureEngineer
                features_df = self._generate_features(df)

                if features_df is None or len(features_df) == 0:
                    continue

                # Get latest bar's features for prediction
                latest_features_df = features_df[self.feature_names].tail(1)
                latest_features = latest_features_df.to_numpy()

                # Get current price for signal
                current_price = float(df["close"].tail(1)[0])

                # ═══════════════════════════════════════════════════════════
                # STAGE 1: THE ANGEL (DIRECTION)
                # ═══════════════════════════════════════════════════════════
                angel_prob = self.angel_model.predict_proba(latest_features)[0, 1]
                highest_angel_prob = max(highest_angel_prob, angel_prob)

                if angel_prob < self.angel_threshold:
                    logger.debug(
                        f"[{symbol}] Angel rejected | Prob: {angel_prob:.4f} < {self.angel_threshold}"
                    )
                    continue

                logger.debug(
                    f"[{symbol}] Angel proposed trade | Prob: {angel_prob:.4f}"
                )

                # ═══════════════════════════════════════════════════════════
                # STAGE 2: THE DEVIL (CONVICTION)
                # ═══════════════════════════════════════════════════════════
                # Build meta-feature set: original features + Angel's probability
                import pandas as pd

                meta_features = pd.DataFrame(
                    latest_features_df.to_numpy(), columns=self.feature_names
                )
                meta_features["angel_prob"] = angel_prob

                devil_prob = self.devil_model.predict_proba(meta_features)[0, 1]

                if devil_prob < self.devil_threshold:
                    logger.debug(
                        f"[{symbol}] Devil veto | Angel: {angel_prob:.2f}, Devil: {devil_prob:.2f} < {self.devil_threshold}"
                    )
                    continue

                # Both Angel and Devil agree - emit BUY signal
                logger.info(
                    f"[{symbol}] ANGEL & DEVIL AGREEMENT | "
                    f"Price={current_price:.2f} | "
                    f"Angel Prob: {angel_prob:.2f} | "
                    f"Devil Prob: {devil_prob:.2f}"
                )

                signal = Signal(
                    symbol=symbol,
                    type=SignalType.BUY,
                    price=current_price,
                    confidence=devil_prob,
                    timestamp=df["timestamp"].tail(1)[0],
                    metadata={
                        "angel_prob": float(angel_prob),
                        "devil_prob": float(devil_prob),
                    },
                )
                signals.append(signal)

            except Exception as e:
                logger.error(f"[{symbol}] Error in ML analysis: {e}", exc_info=True)
                continue

        return signals, highest_angel_prob

    def _generate_features(self, df: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        Generate ML features using imported FeatureEngineer.

        This method ensures zero training/inference skew by using the exact
        same feature computation logic as the training pipeline.

        Args:
            df: Raw OHLCV DataFrame.

        Returns:
            DataFrame with computed features, or None if insufficient data.
        """
        try:
            # Use imported FeatureEngineer.compute_indicators()
            features_df = self.feature_engineer.compute_indicators(df)

            # Handle NaN values that may exist in warmup period
            feature_cols = [c for c in features_df.columns if c in self.feature_names]
            features_df = features_df.drop_nulls(subset=feature_cols)

            return features_df

        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            return None

    def get_order_params(self) -> OrderParams:
        """Returns order parameters for this strategy."""
        return self.order_params
