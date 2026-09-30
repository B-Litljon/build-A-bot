"""
A simple fixed-percentage training label.

STATUS: legacy, kept for ``feature_pipeline.main()`` only (verified 2026-09-30
by grep: no importer under src/core/). The LIVE training path
(``src/core/retrainer.py``) builds its own labels from volatility-scaled
brackets and does not use this.

Glossary:
    V3DirectionalTarget -- labels each bar 1 if price rises by at least
        min_gain within the next `lookahead` bars, else 0.
    _LOOKAHEAD_BARS -- 15 bars, the default window to look ahead.
    _MIN_GAIN_PCT -- 0.003, i.e. a 0.3% rise counts as a win.
    target -- the output column. Null (not 0) for the final rows where the
        future is unknown, so incomplete rows get dropped rather than being
        mislabelled as losses.

Note the difference from the production labels: this only asks whether price
EVER reached a level, ignoring whether a stop would have been hit first. The
retrainer's labels replay stops and targets bar by bar, which is why they are
harsher and more realistic.
"""

import polars as pl
from ml.core.interfaces import BaseTargetGenerator

_LOOKAHEAD_BARS = 15
_MIN_GAIN_PCT = 0.003

class V3DirectionalTarget(BaseTargetGenerator):
    """
    Generates target labels using a lookahead approach.
    Labels as 1 if the price hits a minimum gain within the lookahead window, 0 otherwise.
    """
    def __init__(self, lookahead: int = _LOOKAHEAD_BARS, min_gain: float = _MIN_GAIN_PCT):
        self.lookahead = lookahead
        self.min_gain = min_gain

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        # Legacy note (2026-09-30): used only by feature_pipeline.main(); the
        # production retrainer labels its own brackets (core/retrainer/_labels.py)
        # and does not import this class. The shift(-lookahead) must still be
        # partitioned per symbol, or a pooled multi-symbol frame would copy the
        # NEXT symbol's opening close onto this symbol's final rows — same
        # cross-symbol bleed class as the htf_vol_rel bug fixed the same day
        # in features/v3_features.py.
        if "symbol" in df.columns:
            future_close = pl.col("close").shift(-self.lookahead).over("symbol")
        else:
            future_close = pl.col("close").shift(-self.lookahead)
        df = df.with_columns(
            pl.when(future_close.is_null())
            .then(pl.lit(None, dtype=pl.Int8))
            .when(future_close > pl.col("close") * (1.0 + self.min_gain))
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("target")
        )
        return df
