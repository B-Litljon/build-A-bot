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
