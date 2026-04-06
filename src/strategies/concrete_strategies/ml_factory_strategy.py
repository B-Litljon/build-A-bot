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

logger = logging.getLogger(__name__)

class MLFactoryStrategy(MLStrategy):
    """
    The Brain: Implementation of the ML Factory Strategy.
    Wraps the core MLStrategy logic with Factory-specific requirements.
    """
    def __init__(self, **kwargs):
        # Ensure warmup is set to at least 260 for 5m SMA-50 support
        kwargs.setdefault("warmup_period", 260)
        super().__init__(**kwargs)

    def analyze(self, data: Dict[str, pl.DataFrame]) -> Tuple[List[Signal], float]:
        """
        Performs inference. The warm-up sequence ensures 'data' already contains
        the necessary historical depth.
        """
        # Base MLStrategy.analyze already uses FeatureEngineer which handles
        # the 18-feature set and HTF indicators.
        return super().analyze(data)
