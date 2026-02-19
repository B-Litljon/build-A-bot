"""
Machine Learning-based trading strategy using the trained Random Forest model.

This strategy imports FeatureEngineer from ml.feature_pipeline to ensure
identical feature computation between training and inference, preventing
training/inference skew.

Optimization (Feb 2026):
    - Threshold: 0.48 (previously 0.50)
    - Profit Factor: 1.68
    - Win Rate: 40.3%
    - Trades: 713/year
    - Risk Profile: TP 0.5% / SL 0.2%

Usage:
    from strategies.concrete_strategies.ml_strategy import MLStrategy

    strategy = MLStrategy(
        model_path="src/ml/models/rf_model.joblib",
        threshold=0.48,  # Optimized threshold from grid search
        warmup_period=60  # Must accommodate all indicator windows
    )
"""

import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import polars as pl

from core.order_management import OrderParams
from core.signal import Signal
from strategies.strategy import Strategy

# CRITICAL: Import FeatureEngineer to prevent training/inference skew
from ml.feature_pipeline import FeatureEngineer

logger = logging.getLogger(__name__)


class MLStrategy(Strategy):
    """
    ML-powered strategy using trained Random Forest with probability threshold.

    Prevents training/inference skew by importing FeatureEngineer from the
    training pipeline rather than duplicating feature logic.

    Optimization Results (Feb 2026):
        Threshold 0.48 selected from grid search:
        - Profit Factor: 1.68
        - Win Rate: 40.3%
        - Total Trades: 713/year
        - Risk: TP 0.5% / SL 0.2%

    Parameters
    ----------
    model_path : str | Path
        Path to the trained model joblib file.
    threshold : float
        Probability threshold for trade signals (default: 0.48).
        Optimization shows 0.48 delivers PF=1.68 vs 1.46 at 0.50.
    warmup_period : int
        Minimum candles required before trading (default: 60).
    """

    def __init__(
        self,
        model_path: str | Path = "src/ml/models/rf_model.joblib",
        threshold: float = 0.48,
        warmup_period: int = 60,
    ):
        super().__init__()

        self.timeframe = 1  # 1-minute bars
        self.warmup = warmup_period
        self.threshold = threshold

        # Load trained model
        model_file = Path(model_path)
        if not model_file.exists():
            # Try relative to project root
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            model_file = project_root / model_path

        logger.info(f"Loading ML model from {model_file}")
        self.model = joblib.load(model_file)
        logger.info(f"Model loaded: {type(self.model).__name__}")

        # Initialize feature engineer (imported, not duplicated!)
        self.feature_engineer = FeatureEngineer()

        # Store expected feature order from model training
        # These are the features the model was trained on (from _DROP_COLS inverse)
        self.feature_names = [
            "rsi_14",
            "macd",
            "macd_signal",
            "macd_hist",
            "bb_upper",
            "bb_lower",
            "sma_50",
            "atr_14",
            "bb_pct_b",
            "price_sma50_ratio",
            "log_return",
            "hour_of_day",
            "vol_rel",
            "dist_sma50",
        ]

        self.order_params = OrderParams(
            risk_percentage=0.02,
            tp_multiplier=1.005,  # 0.5% take profit (based on 0.3% min gain + buffer)
            sl_multiplier=0.998,  # 0.2% stop loss (based on label criteria)
            use_trailing_stop=False,
        )

    @property
    def warmup_period(self) -> int:
        """Returns minimum candles required for indicators to warm up."""
        return self.warmup

    def analyze(self, data: Dict[str, pl.DataFrame]) -> tuple[List[Signal], float]:
        """
        Analyze market data and generate ML-based trading signals.

        Args:
            data: Dict mapping symbol -> Polars DataFrame with OHLCV data.

        Returns:
            Tuple of (List of BUY signals where model probability >= threshold, probability).
        """
        signals = []
        last_proba = 0.0

        for symbol, df in data.items():
            if len(df) < self.warmup_period:
                logger.debug(
                    f"{symbol}: Insufficient data ({len(df)} < {self.warmup_period})"
                )
                continue

            try:
                # Generate features using imported FeatureEngineer
                # This ensures identical feature computation as training!
                features_df = self._generate_features(df)

                if features_df is None or len(features_df) == 0:
                    continue

                # Get latest bar's features for prediction
                latest_features = features_df[self.feature_names].tail(1).to_numpy()

                # Predict probability of Class 1 (trade signal)
                proba = self.model.predict_proba(latest_features)[0, 1]
                last_proba = proba

                # Get current price for signal
                current_price = float(df["close"].tail(1)[0])

                # Generate signal if probability exceeds threshold
                if proba >= self.threshold:
                    logger.info(
                        f"{symbol}: ML Signal Generated | "
                        f"Price={current_price:.2f} | "
                        f"Probability={proba:.4f} | "
                        f"Threshold={self.threshold}"
                    )
                    signals.append(Signal("BUY", symbol, current_price))
                else:
                    logger.debug(
                        f"{symbol}: No signal | Probability={proba:.4f} < {self.threshold}"
                    )

            except Exception as e:
                logger.error(f"{symbol}: Error in ML analysis: {e}", exc_info=True)
                continue

        return signals, last_proba

    def _generate_features(self, df: pl.DataFrame) -> pl.DataFrame | None:
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
            # This is the SAME method used during training!
            features_df = self.feature_engineer.compute_indicators(df)

            # Handle NaN values that may exist in warmup period
            # Drop rows with any NaN in feature columns
            feature_cols = [c for c in features_df.columns if c in self.feature_names]
            features_df = features_df.drop_nulls(subset=feature_cols)

            return features_df

        except Exception as e:
            logger.error(f"Feature generation failed: {e}")
            return None

    def get_order_params(self) -> OrderParams:
        """Returns order parameters for this strategy."""
        return self.order_params
