"""
ML Factory Strategy: Proprietary dual-model (Angel/Devil) pipeline.
Strictly requires an 18-feature Polars DataFrame for inference.
Heavily relies on a 5-minute Higher-Time-Frame (HTF) SMA-50.

NOTE (glossary pass, 2026-07-27): the "18-feature" claim above is STALE. The
live feature set is 22 columns (23 with cost_ratio), and MLStrategy reads its
schema off the trained model rather than assuming a count. Flagged, not changed.

A thin subclass of MLStrategy that only presets two defaults; all the actual
inference logic is inherited. Used by the Factory orchestrator.

Glossary:
    MLFactoryStrategy -- MLStrategy with Factory-appropriate defaults. It adds
        no behaviour of its own beyond the two kwargs below.
    warmup_period=260 -- minimum bars before it will trade. 260 one-minute bars
        is what a 50-period average on 5-minute bars needs to warm up
        (50 x 5 = 250, plus headroom). Trading before that means acting on
        half-computed indicators.
    angel_trainer / devil_trainer -- pre-built V3RandomForestTrainer shells.
        Despite the name these end up holding LightGBM models, because load()
        unpickles whatever is on disk; see src/ml/trainers/README.md.
    self.pipeline -- inherited from MLStrategy, NOT overridden here. An earlier
        version re-declared an identical pipeline; that was removed because a
        duplicate definition invites the two copies to drift apart.
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
