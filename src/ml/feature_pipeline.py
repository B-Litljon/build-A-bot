"""
Feature engineering pipeline for ML training datasets.

Loads raw 1-minute Parquet files from ``data/raw/``, computes
technical-indicator features via TA-Lib, generates binary target labels,
cleans NaN rows, and writes the result to
``data/processed/training_data.parquet``.

Usage (from ``src/``)::

    python -m ml.feature_pipeline

The output schema appends indicator and label columns to the raw OHLCV:

    timestamp, open, high, low, close, volume,   (original)
    symbol,                                       (tagged)
    rsi_14, macd, macd_signal, macd_hist,         (momentum)
    bb_upper, bb_lower, bb_pct_b,                 (volatility bands)
    sma_50, price_sma50_ratio,                    (trend)
    atr_14,                                       (volatility)
    log_return,                                   (returns)
    target                                        (label: 0 or 1)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import polars as pl
import talib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

_SRC_DIR = Path(__file__).resolve().parent.parent  # src/
_PROJECT_ROOT = _SRC_DIR.parent  # build-A-bot/
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"

# ── indicator hyper-parameters ────────────────────────────────────────
_RSI_PERIOD = 14
_MACD_FAST = 12
_MACD_SLOW = 26
_MACD_SIGNAL = 9
_BB_PERIOD = 20
_BB_STD = 2
_SMA_PERIOD = 50
_ATR_PERIOD = 14

# ── label hyper-parameters ────────────────────────────────────────────
_LOOKAHEAD_BARS = 15  # bars into the future for the label
_MIN_GAIN_PCT = 0.003  # 0.3 % pure gain threshold


class FeatureEngineer:
    """
    Transforms raw OHLCV Parquet files into a labeled feature matrix
    suitable for binary classification.

    The pipeline is three stages:

    1. **compute_indicators** -- attach technical features via TA-Lib.
    2. **generate_labels**    -- create the forward-looking binary target.
    3. **clean_data**         -- drop any row with a null in *any* column.
    """

    # ── 1. indicators ─────────────────────────────────────────────────

    @staticmethod
    def compute_indicators(df: pl.DataFrame) -> pl.DataFrame:
        """
        Append technical-indicator columns to *df*.

        TA-Lib requires contiguous ``float64`` numpy arrays, so each
        price series is extracted once with ``.to_numpy()``, fed through
        the C library, and re-attached as a Polars Series.

        Added columns
        -------------
        rsi_14, macd, macd_signal, macd_hist,
        bb_upper, bb_lower, bb_pct_b,
        sma_50, price_sma50_ratio,
        atr_14,
        log_return
        """
        # ── extract numpy arrays (one copy per source column) ─────────
        close: np.ndarray = df["close"].to_numpy()
        high: np.ndarray = df["high"].to_numpy()
        low: np.ndarray = df["low"].to_numpy()

        # ── RSI ───────────────────────────────────────────────────────
        rsi = talib.RSI(close, timeperiod=_RSI_PERIOD)

        # ── MACD ──────────────────────────────────────────────────────
        macd, macd_signal, macd_hist = talib.MACD(
            close,
            fastperiod=_MACD_FAST,
            slowperiod=_MACD_SLOW,
            signalperiod=_MACD_SIGNAL,
        )

        # ── Bollinger Bands ───────────────────────────────────────────
        bb_upper, _bb_middle, bb_lower = talib.BBANDS(
            close,
            timeperiod=_BB_PERIOD,
            nbdevup=_BB_STD,
            nbdevdn=_BB_STD,
            matype=0,  # Simple MA
        )

        # ── SMA 50 ───────────────────────────────────────────────────
        sma_50 = talib.SMA(close, timeperiod=_SMA_PERIOD)

        # ── ATR ──────────────────────────────────────────────────────
        atr = talib.ATR(high, low, close, timeperiod=_ATR_PERIOD)

        # ── attach all numpy results as Polars columns ────────────────
        df = df.with_columns(
            pl.Series("rsi_14", rsi),
            pl.Series("macd", macd),
            pl.Series("macd_signal", macd_signal),
            pl.Series("macd_hist", macd_hist),
            pl.Series("bb_upper", bb_upper),
            pl.Series("bb_lower", bb_lower),
            pl.Series("sma_50", sma_50),
            pl.Series("atr_14", atr),
        )

        # ── derived features (pure Polars, no numpy detour) ───────────

        # %B: where the close sits inside the bands (0 = lower, 1 = upper)
        df = df.with_columns(
            (
                (pl.col("close") - pl.col("bb_lower"))
                / (pl.col("bb_upper") - pl.col("bb_lower"))
            ).alias("bb_pct_b")
        )

        # Normalised price vs. SMA-50
        df = df.with_columns(
            (pl.col("close") / pl.col("sma_50")).alias("price_sma50_ratio")
        )

        # Log return: ln(close_t / close_{t-1})
        df = df.with_columns(
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return")
        )

        # ── NEW: Contextual features for noise filtering ─────────────────

        # Hour of day (0-23) - captures market microstructure by time
        df = df.with_columns(
            pl.col("timestamp").dt.hour().cast(pl.Int8).alias("hour_of_day")
        )

        # Relative Volume: current volume vs 20-bar rolling mean
        # Fills initial NaN window with 1.0 (neutral/typical volume)
        df = df.with_columns(
            (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20))
            .fill_nan(1.0)
            .fill_null(1.0)
            .alias("vol_rel")
        )

        # Distance from SMA-50: normalized deviation from trend
        df = df.with_columns(
            ((pl.col("close") - pl.col("sma_50")) / pl.col("sma_50")).alias(
                "dist_sma50"
            )
        )

        return df

    # ── 2. labels ─────────────────────────────────────────────────────

    @staticmethod
    def generate_labels(
        df: pl.DataFrame,
        lookahead: int = _LOOKAHEAD_BARS,
        min_gain: float = _MIN_GAIN_PCT,
    ) -> pl.DataFrame:
        """
        Append a binary ``target`` column.

        Simplified "triple barrier" for MVP:

            target = 1  if  close[t + lookahead] > close[t] * (1 + min_gain)
                     0  otherwise

        The last *lookahead* rows will have a ``null`` target (cleaned
        later by :meth:`clean_data`).

        Parameters
        ----------
        df : pl.DataFrame
            Must contain a ``close`` column.
        lookahead : int
            Number of bars to look ahead.
        min_gain : float
            Minimum fractional price gain for a positive label.
        """
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

    # ── 3. clean ──────────────────────────────────────────────────────

    @staticmethod
    def clean_data(df: pl.DataFrame) -> pl.DataFrame:
        """
        Drop every row that has a null *or* NaN in any column.

        This removes:
        - the first ~50 rows (SMA-50 warmup; TA-Lib returns IEEE NaN)
        - the last *lookahead* rows (target is null)
        - any sporadic NaN from TA-Lib edge cases

        TA-Lib writes IEEE-754 NaN into its output arrays.  Polars
        stores these as ``NaN`` (a valid float), *not* as ``null``.
        We must convert NaN -> null first so that ``drop_nulls``
        catches them.
        """
        before = len(df)

        # Identify float columns that could carry TA-Lib NaN values
        float_cols = [
            col for col, dtype in df.schema.items() if dtype in (pl.Float64, pl.Float32)
        ]
        # Convert IEEE NaN -> Polars null in every float column
        if float_cols:
            df = df.with_columns(
                pl.when(pl.col(c).is_nan()).then(None).otherwise(pl.col(c)).alias(c)
                for c in float_cols
            )

        df = df.drop_nulls()
        after = len(df)
        logger.info(
            "clean_data: dropped %d / %d rows (%.1f%%)",
            before - after,
            before,
            100.0 * (before - after) / max(before, 1),
        )
        return df

    # ── convenience: full pipeline ────────────────────────────────────

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        """Execute indicators -> labels -> clean in sequence."""
        df = self.compute_indicators(df)
        df = self.generate_labels(df)
        df = self.clean_data(df)
        return df


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    _PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(_RAW_DIR.glob("*_1min.parquet"))
    if not raw_files:
        logger.error("No raw Parquet files found in %s", _RAW_DIR)
        return

    logger.info("Found %d raw file(s): %s", len(raw_files), [f.name for f in raw_files])

    engineer = FeatureEngineer()
    processed_frames: List[pl.DataFrame] = []

    for path in raw_files:
        symbol = path.stem.replace("_1min", "")
        logger.info("Processing %s ...", symbol)

        df = pl.read_parquet(path)
        logger.info("  loaded %d rows", len(df))

        # Tag with symbol so multiple tickers can coexist in one file
        df = df.with_columns(pl.lit(symbol).alias("symbol"))

        df = engineer.run(df)
        logger.info("  %d rows after pipeline", len(df))

        processed_frames.append(df)

    if not processed_frames:
        logger.error("All symbols produced empty DataFrames. Nothing to write.")
        return

    combined = pl.concat(processed_frames, how="vertical_relaxed")
    combined = combined.sort(["symbol", "timestamp"])

    out_path = _PROCESSED_DIR / "training_data.parquet"
    combined.write_parquet(out_path)

    logger.info(
        "Wrote %d rows (%d columns) to %s (%.2f MB)",
        len(combined),
        combined.width,
        out_path,
        out_path.stat().st_size / (1024 * 1024),
    )

    # Quick sanity summary
    logger.info(
        "Label distribution:\n%s", combined["target"].value_counts().sort("target")
    )
    logger.info("Schema:\n%s", combined.schema)


if __name__ == "__main__":
    main()
