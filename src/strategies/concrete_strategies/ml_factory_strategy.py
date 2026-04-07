"""
ML Factory Strategy: Proprietary dual-model (Angel/Devil) pipeline.
Strictly requires an 18-feature Polars DataFrame for inference.
Heavily relies on a 5-minute Higher-Time-Frame (HTF) SMA-50.
"""

import logging
from typing import Dict, List, Tuple, Optional
import polars as pl
from strategies.concrete_strategies.ml_strategy import MLStrategy
from core.signal import Signal
from ml.feature_pipeline import FeaturePipeline
from ml.features.v3_features import V3BaseFeatures, V3HTFFeatures

logger = logging.getLogger(__name__)

class MLFactoryStrategy(MLStrategy):
    """
    The Brain: Implementation of the ML Factory Strategy.
    Wraps the core MLStrategy logic with Factory-specific requirements.
    """
    def __init__(self, **kwargs):
        # Ensure warmup is set to at least 260 for 5m SMA-50 support
        kwargs.setdefault("warmup_period", 260)

        # Inject framework-agnostic V3 Random Forest trainers
        from ml.trainers.v3_rf_trainer import V3RandomForestTrainer
        kwargs.setdefault("angel_trainer", V3RandomForestTrainer())
        kwargs.setdefault("devil_trainer", V3RandomForestTrainer())

        super().__init__(**kwargs)
        self.pipeline = FeaturePipeline(
            feature_generators=[V3BaseFeatures(), V3HTFFeatures(timeframe="5m")]
        )

    def analyze(self, data: Dict[str, pl.DataFrame]) -> Tuple[List[Signal], float]:
        """
        Performs inference. The warm-up sequence ensures 'data' already contains
        the necessary historical depth.
        """
        # Base MLStrategy.analyze already uses FeaturePipeline which handles
        # the 18-feature set and HTF indicators.
        return super().analyze(data)
