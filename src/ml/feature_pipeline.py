"""
Feature engineering pipeline for ML training datasets.
Modified for Universal Scalping (multi-ticker support).
Absolute price values (MACD, ATR) replaced with normalized equivalents (PPO, NATR).
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import talib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

_RSI_PERIOD = 14
_PPO_FAST = 12
_PPO_SLOW = 26
_BB_PERIOD = 20
_BB_STD = 2
_SMA_PERIOD = 50
_NATR_PERIOD = 14
_LOOKAHEAD_BARS = 15
_MIN_GAIN_PCT = 0.003


class FeatureEngineer:
    @staticmethod
    def compute_indicators(df: pl.DataFrame) -> pl.DataFrame:
        close: np.ndarray = df["close"].to_numpy()
        high: np.ndarray = df["high"].to_numpy()
        low: np.ndarray = df["low"].to_numpy()

        # Universal Momentum
        rsi = talib.RSI(close, timeperiod=_RSI_PERIOD)
        ppo = talib.PPO(
            close, fastperiod=_PPO_FAST, slowperiod=_PPO_SLOW, matype=talib.MA_Type.SMA
        )

        # Volatility Bands
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close,
            timeperiod=_BB_PERIOD,
            nbdevup=_BB_STD,
            nbdevdn=_BB_STD,
            matype=talib.MA_Type.SMA,
        )

        # Trend
        sma_50 = talib.SMA(close, timeperiod=_SMA_PERIOD)

        # Universal Volatility
        natr = talib.NATR(high, low, close, timeperiod=_NATR_PERIOD)

        df = df.with_columns(
            pl.Series("rsi_14", rsi),
            pl.Series("ppo", ppo),
            pl.Series("bb_upper", bb_upper),
            pl.Series("bb_middle", bb_middle),
            pl.Series("bb_lower", bb_lower),
            pl.Series("sma_50", sma_50),
            pl.Series("natr_14", natr),
        )

        # Normalized derived features
        df = df.with_columns(
            (
                (pl.col("close") - pl.col("bb_lower"))
                / (pl.col("bb_upper") - pl.col("bb_lower"))
            ).alias("bb_pct_b"),
            ((pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle")).alias(
                "bb_width_pct"
            ),
            (pl.col("close") / pl.col("sma_50")).alias("price_sma50_ratio"),
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return"),
            pl.col("timestamp").dt.hour().cast(pl.Int8).alias("hour_of_day"),
            ((pl.col("close") - pl.col("sma_50")) / pl.col("sma_50")).alias(
                "dist_sma50"
            ),
        )

        df = df.with_columns(
            (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20))
            .fill_nan(1.0)
            .fill_null(1.0)
            .alias("vol_rel")
        )

        return df

    @staticmethod
    def generate_labels(
        df: pl.DataFrame,
        lookahead: int = _LOOKAHEAD_BARS,
        min_gain: float = _MIN_GAIN_PCT,
    ) -> pl.DataFrame:
        future_close = pl.col("close").shift(-lookahead)
        df = df.with_columns(
            pl.when(future_close.is_null())
            .then(pl.lit(None, dtype=pl.Int8))
            .when(future_close > pl.col("close") * (1.0 + min_gain))
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(0, dtype=pl.Int8))
            .alias("target")
        )
        return df

    @staticmethod
    def clean_data(df: pl.DataFrame) -> pl.DataFrame:
        float_cols = [
            col for col, dtype in df.schema.items() if dtype in (pl.Float64, pl.Float32)
        ]
        if float_cols:
            df = df.with_columns(
                pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
                for c in float_cols
            )
        return df.drop_nulls()

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        df = self.compute_indicators(df)
        df = self.generate_labels(df)
        df = self.clean_data(df)
        return df


def main() -> None:
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(_RAW_DIR.glob("*_1min.parquet"))
    if not raw_files:
        logger.error("No raw Parquet files found in %s", _RAW_DIR)
        return

    engineer = FeatureEngineer()
    processed_frames: List[pl.DataFrame] = []

    for path in raw_files:
        symbol = path.stem.replace("_1min", "")
        df = pl.read_parquet(path)
        df = df.with_columns(pl.lit(symbol).alias("symbol"))
        df = engineer.run(df)
        processed_frames.append(df)

    if not processed_frames:
        return

    combined = pl.concat(processed_frames, how="vertical_relaxed")
    combined = combined.sort(["symbol", "timestamp"])
    out_path = _PROCESSED_DIR / "training_data.parquet"
    combined.write_parquet(out_path)
    logger.info("Wrote %d rows to %s", len(combined), out_path)


if __name__ == "__main__":
    main()
