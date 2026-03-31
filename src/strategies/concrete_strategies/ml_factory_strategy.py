"""ML Factory Strategy — Concrete implementation of BaseStrategy for the SDK.

Implements the Angel/Devil two-stage meta-labeling architecture with strict
adherence to the 18-feature Polars schema.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import polars as pl
import pandas as pd

from src.strategies.base import BaseStrategy, Signal
from src.ml.feature_pipeline import FeatureEngineer


logger = logging.getLogger(__name__)


class MLFactoryStrategy(BaseStrategy):
    """
    Concrete ML strategy implementing Angel/Devil meta-labeling.

    Inherits from BaseStrategy to integrate with the SDK architecture.
    """

    # Strict 18-feature schema (must match training pipeline)
    FEATURE_NAMES = [
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
        "htf_rsi_14",
        "htf_trend_agreement",
        "htf_vol_rel",
        "htf_bb_pct_b",
        "range_coil_10",
        "bar_body_pct",
        "bar_upper_wick_pct",
        "bar_lower_wick_pct",
    ]

    def __init__(
        self,
        angel_path: str = "models/angel_latest.pkl",
        devil_path: str = "models/devil_latest.pkl",
        angel_threshold: float = 0.40,
        devil_threshold: float = 0.52,
        **kwargs,
    ) -> None:
        """
        Initialize ML strategy with model paths and thresholds.

        Args:
            angel_path: Path to Angel model pickle file
            devil_path: Path to Devil model pickle file
            angel_threshold: Probability threshold for Angel approval
            devil_threshold: Default Devil threshold (overridden by threshold.json)
            **kwargs: Additional strategy parameters
        """
        super().__init__(**kwargs)

        self.angel_threshold = angel_threshold
        self.devil_threshold = devil_threshold

        project_root = Path(__file__).resolve().parent.parent.parent.parent
        self.angel_path = project_root / angel_path
        self.devil_path = project_root / devil_path

        logger.info("Loading Angel model from %s", self.angel_path)
        self.angel_model = joblib.load(self.angel_path)
        self.angel_model.n_jobs = 1
        logger.info("Angel model loaded: %s", type(self.angel_model).__name__)

        logger.info("Loading Devil model from %s", self.devil_path)
        self.devil_model = joblib.load(self.devil_path)
        self.devil_model.n_jobs = 1
        logger.info("Devil model loaded: %s", type(self.devil_model).__name__)

        self.feature_engineer = FeatureEngineer()
        self.devil_threshold = self._load_threshold()

        logger.info(
            "MLFactoryStrategy initialized | angel_threshold=%.2f | devil_threshold=%.4f",
            self.angel_threshold,
            self.devil_threshold,
        )

    def _load_threshold(self) -> float:
        """Load Devil threshold from models/threshold.json with fallback."""
        threshold_path = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "models"
            / "threshold.json"
        )

        if not threshold_path.exists():
            logger.warning(
                "threshold.json not found - using default devil_threshold=%.2f",
                self.devil_threshold,
            )
            return self.devil_threshold

        try:
            with open(threshold_path, "r") as fh:
                data = json.load(fh)
            threshold = float(data.get("devil_threshold", self.devil_threshold))
            logger.info(
                "Loaded devil_threshold=%.4f from %s", threshold, threshold_path
            )
            return threshold
        except Exception as exc:
            logger.warning(
                "Failed to load threshold.json (%s) - using default=%.2f",
                exc,
                self.devil_threshold,
            )
            return self.devil_threshold

    def generate_signals(self, df: pl.DataFrame) -> Optional[Signal]:
        """
        Generate trading signal from 18-feature microstructure input.

        Args:
            df: Polars DataFrame with OHLCV history (260+ bars)

        Returns:
            Signal object if Angel and Devil both approve, None otherwise
        """
        try:
            self.validate_input(df)

            features_df = self.feature_engineer.compute_indicators(df)

            if features_df is None or len(features_df) == 0:
                logger.debug("Feature computation returned empty DataFrame")
                return None

            features_df = features_df.filter(
                pl.all_horizontal(pl.col(self.FEATURE_NAMES).is_finite())
            )

            if len(features_df) == 0:
                logger.debug("All rows contain NaN/Inf after feature computation")
                return None

            latest_row = features_df.tail(1)
            current_price = float(latest_row["close"][0])
            natr_14 = float(latest_row["natr_14"][0])

            feature_matrix = latest_row.select(self.FEATURE_NAMES).to_numpy()

            angel_prob = float(self.angel_model.predict_proba(feature_matrix)[0, 1])

            if angel_prob < self.angel_threshold:
                logger.debug(
                    "Angel rejected | prob=%.4f < threshold=%.2f",
                    angel_prob,
                    self.angel_threshold,
                )
                return None

            logger.debug("Angel proposed | prob=%.4f", angel_prob)

            meta_df = pd.DataFrame(feature_matrix, columns=self.FEATURE_NAMES)
            meta_df["angel_prob"] = angel_prob

            devil_prob = float(self.devil_model.predict_proba(meta_df)[0, 1])

            if devil_prob < self.devil_threshold:
                logger.debug(
                    "Devil veto | angel=%.4f, devil=%.4f < threshold=%.4f",
                    angel_prob,
                    devil_prob,
                    self.devil_threshold,
                )
                return None

            atr_abs = (natr_14 / 100.0) * current_price

            raw_sl_distance = 0.5 * atr_abs
            raw_tp_distance = 3.0 * atr_abs

            logger.info(
                "SIGNAL | price=%.2f | angel=%.4f | devil=%.4f | "
                "natr=%.4f | atr_abs=%.4f | sl_dist=%.4f | tp_dist=%.4f",
                current_price,
                angel_prob,
                devil_prob,
                natr_14,
                atr_abs,
                raw_sl_distance,
                raw_tp_distance,
            )

            return Signal(
                direction="long",
                entry_price=current_price,
                raw_sl_distance=raw_sl_distance,
                raw_tp_distance=raw_tp_distance,
                metadata={
                    "angel_prob": angel_prob,
                    "devil_prob": devil_prob,
                    "natr_14": natr_14,
                    "atr_abs": atr_abs,
                },
            )

        except Exception as exc:
            logger.error("Error generating signals: %s", exc, exc_info=True)
            return None
