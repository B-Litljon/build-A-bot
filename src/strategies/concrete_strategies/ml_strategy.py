"""
Meta-Labeling Machine Learning Trading Strategy (Angel & Devil Architecture).

Implements a two-stage inference system:
1. The Angel (Primary Model): Learns Direction (high recall, threshold 0.40)
2. The Devil (Meta Model): Learns Conviction (high precision, threshold 0.50)

Usage:
    from strategies.concrete_strategies.ml_strategy import MLStrategy

    strategy = MLStrategy(
        angel_path="src/ml/models/angel_rf_model.joblib",
        devil_path="src/ml/models/devil_rf_model.joblib",
        angel_threshold=0.40,
        devil_threshold=0.50,
        warmup_period=60
    )
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import polars as pl
import pandas as pd

from core.order_management import OrderParams
from core.signal import Signal, SignalType
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
        warmup_period: int = 60,
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

        logger.info(f"Loading Angel model from {angel_file}")
        self.angel_model = joblib.load(angel_file)
        logger.info(f"Angel model loaded: {type(self.angel_model).__name__}")

        logger.info(f"Loading Devil model from {devil_file}")
        self.devil_model = joblib.load(devil_file)
        logger.info(f"Devil model loaded: {type(self.devil_model).__name__}")

        # Initialize feature engineer (imported, not duplicated!)
        self.feature_engineer = FeatureEngineer()

        # Feature columns (excluding absolute price columns to prevent leakage)
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
        ]

        self.order_params = OrderParams(
            risk_percentage=0.02,
            tp_multiplier=1.005,  # 0.5% take profit
            sl_multiplier=0.998,  # 0.2% stop loss
            use_trailing_stop=False,
        )

    @property
    def warmup_period(self) -> int:
        """Returns minimum candles required for indicators to warm up."""
        return self.warmup

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
