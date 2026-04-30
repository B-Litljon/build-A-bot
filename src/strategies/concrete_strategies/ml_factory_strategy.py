"""
ML Factory Strategy: Proprietary dual-model (Angel/Devil) pipeline.
Strictly requires an 18-feature Polars DataFrame for inference.
Heavily relies on a 5-minute Higher-Time-Frame (HTF) SMA-50.
"""

import logging
from strategies.concrete_strategies.ml_strategy import MLStrategy

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
        # Note: self.pipeline is inherited from MLStrategy and already contains
        # the identical [V3BaseFeatures(), V3HTFFeatures(timeframe="5m")] pipeline.
        # The redundant override that existed in the legacy version has been
        # removed because it added no value and could confuse readers.
