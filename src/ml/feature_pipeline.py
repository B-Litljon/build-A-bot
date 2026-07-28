"""
Feature engineering pipeline for ML training datasets.
Modified for Universal Scalping (multi-ticker support).
Absolute price values (MACD, ATR) replaced with normalized equivalents (PPO, NATR).

V3.3: Multi-Timeframe (MTF) extension.
  lookahead bias prevention via available_at logic implemented in V3HTFFeatures.

Abstracted to dynamic injection structure using Core Interfaces.

This is the assembly point: callers hand it a LIST of generators and it runs
them in order, then scrubs the result. The live strategy and the retrainer each
build their own list, which is how they stay in lockstep -- same generators,
same order, same cleaning.

Glossary:
    FeaturePipeline -- holds an ordered list of feature generators plus one
        optional target generator, and runs them.
    feature_generators -- ORDER MATTERS. Later generators depend on columns
        earlier ones added (V3CostFeatures raises outright if natr_14 is
        missing because it was placed before V3BaseFeatures).
    target_generator -- training only; None for live inference, where there is
        no answer to attach.
    run() -- generate everything, then clean_data. Always returns a frame ready
        to hand to a model.
    clean_data -- converts NaN AND infinity to null, then drops incomplete
        rows. Infinity is the non-obvious half: normalised features divide, and
        a perfectly flat bar makes the denominator zero. Model fitting rejects
        infinity, so it must be scrubbed rather than passed through.
    feature_cols -- when given, only these columns are checked for
        completeness. Without it a null in ANY column drops the row, including
        columns the model never looks at.
    _RAW_DIR / _PROCESSED_DIR -- data/raw/ (input, one parquet per symbol) and
        data/processed/ (output). Used only by main().
    main() -- the standalone script path: read every data/raw/*_1min.parquet,
        run a fixed pipeline, write data/processed/training_data.parquet. The
        production retrainer does NOT use this; it builds its own pipeline in
        memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from ml.core.interfaces import BaseFeatureGenerator, BaseTargetGenerator
from ml.features.v3_features import V3BaseFeatures, V3HTFFeatures
from ml.targets.v3_targets import V3DirectionalTarget

logger = logging.getLogger(__name__)

_SRC_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SRC_DIR.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"


class FeaturePipeline:
    def __init__(
        self,
        feature_generators: list[BaseFeatureGenerator],
        target_generator: Optional[BaseTargetGenerator] = None,
    ):
        self.feature_generators = feature_generators
        self.target_generator = target_generator

    def run(
        self, df: pl.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> pl.DataFrame:
        for gen in self.feature_generators:
            df = gen.generate(df)
        if self.target_generator:
            df = self.target_generator.generate(df)
        return self.clean_data(df, feature_cols=feature_cols)

    @staticmethod
    def clean_data(
        df: pl.DataFrame, feature_cols: Optional[List[str]] = None
    ) -> pl.DataFrame:
        float_cols = [
            col for col, dtype in df.schema.items() if dtype in (pl.Float64, pl.Float32)
        ]
        if float_cols:
            # Convert both NaN and ±Inf to null so drop_nulls can sweep them.
            # Inf surfaces from division-by-zero in normalized features on
            # flat-volatility bars (e.g. bb_pct_b when bb_upper==bb_lower);
            # sklearn's fit() rejects inf, so we must scrub it before training.
            df = df.with_columns(
                pl.when(pl.col(c).is_nan() | pl.col(c).is_infinite())
                .then(None)
                .otherwise(pl.col(c))
                .alias(c)
                for c in float_cols
            )
        subset = [c for c in feature_cols if c in df.columns] if feature_cols else None
        return df.drop_nulls(subset=subset) if subset else df.drop_nulls()


def main() -> None:
    # Configure logging only when run as a script — at module scope this
    # hijacked root logging for every importer (including the live bot).
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
    )
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(_RAW_DIR.glob("*_1min.parquet"))
    if not raw_files:
        logger.error("No raw Parquet files found in %s", _RAW_DIR)
        return

    # Instantiate the new FeaturePipeline with V3 configuration
    pipeline = FeaturePipeline(
        feature_generators=[
            V3BaseFeatures(),
            V3HTFFeatures(timeframe="5m"),
        ],
        target_generator=V3DirectionalTarget(),
    )

    processed_frames: List[pl.DataFrame] = []

    for path in raw_files:
        symbol = path.stem.replace("_1min", "")
        df = pl.read_parquet(path)
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

        df = pipeline.run(df)
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
