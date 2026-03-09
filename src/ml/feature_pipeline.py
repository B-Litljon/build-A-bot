"""
Feature engineering pipeline for ML training datasets.
Modified for Universal Scalping (multi-ticker support).
Absolute price values (MACD, ATR) replaced with normalized equivalents (PPO, NATR).

V3.3: Multi-Timeframe (MTF) extension.
  compute_indicators() now accepts an optional htf_timeframe parameter (default "5m").
  When set, it appends 4 higher-timeframe features via _compute_htf_features():
    htf_rsi_14, htf_trend_agreement, htf_vol_rel, htf_bb_pct_b

  Lookahead-bias prevention: each HTF bar is labeled with its `available_at`
  timestamp (bar_start + htf_period) and joined via join_asof(strategy="backward"),
  so each 1m bar only ever sees the most recently *completed* HTF bar.
"""

from __future__ import annotations

import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

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

# 1m base-indicator periods
_RSI_PERIOD = 14
_PPO_FAST = 12
_PPO_SLOW = 26
_BB_PERIOD = 20
_BB_STD = 2
_SMA_PERIOD = 50
_NATR_PERIOD = 14
_LOOKAHEAD_BARS = 15
_MIN_GAIN_PCT = 0.003

# HTF configuration (V3.3)
_HTF_TIMEFRAME = "5m"
_HTF_RSI_PERIOD = 14
_HTF_SMA_PERIOD = 50
_HTF_BB_PERIOD = 20
_HTF_BB_STD = 2


class FeatureEngineer:
    """Computes base 1m indicators plus optional higher-timeframe (HTF) features."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_indicators(
        self,
        df: pl.DataFrame,
        htf_timeframe: Optional[str] = _HTF_TIMEFRAME,
    ) -> pl.DataFrame:
        """
        Compute all ML features for the given 1m OHLCV DataFrame.

        Args:
            df:             1m OHLCV DataFrame. Must contain at minimum:
                            timestamp, open, high, low, close, volume.
                            May optionally contain a 'symbol' column for
                            multi-ticker training data.
            htf_timeframe:  HTF period string (default "5m"). Pass None to
                            skip HTF computation (backward-compatible).

        Returns:
            DataFrame enriched with 1m features and (when htf_timeframe is set)
            the 4 HTF features: htf_rsi_14, htf_trend_agreement,
            htf_vol_rel, htf_bb_pct_b.
        """
        df = self._compute_base_features(df)
        if htf_timeframe is not None:
            df = self._compute_htf_features(df, htf_timeframe)
        return df

    # ------------------------------------------------------------------
    # Static helpers that don't depend on instance config
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Private: 1m base features
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_base_features(df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute 1m technical indicators via TA-Lib.

        Produces: rsi_14, ppo, natr_14, bb_pct_b, bb_width_pct,
                  price_sma50_ratio, log_return, hour_of_day,
                  dist_sma50, vol_rel.
        Also adds intermediate columns bb_upper/middle/lower, sma_50
        (retained for derived calculations; dropped in train_model's EXCLUDE_COLS).
        """
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

    # ------------------------------------------------------------------
    # Private: HTF features (V3.3)
    # ------------------------------------------------------------------

    def _compute_htf_features(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """
        Compute higher-timeframe features and join them onto the 1m DataFrame.

        Uses the 'available_at' pattern to prevent lookahead bias:
        each HTF bar is labeled with the timestamp at which it becomes
        available (bar_start + htf_period), then joined via
        join_asof(strategy="backward") so each 1m bar only sees the most
        recently *completed* HTF bar.

        Multi-symbol path (training): group_by_dynamic(..., by="symbol") so
        bars from different tickers are never mixed into the same HTF candle.

        Single-symbol path (live inference): no 'by' grouping needed.

        Args:
            df:         1m DataFrame already enriched by _compute_base_features.
                        Must contain: timestamp, high, low, close, volume.
            timeframe:  Polars group_by_dynamic period string (e.g. "5m").

        Returns:
            df with 4 new columns appended (intermediate _htf_* columns dropped):
                htf_rsi_14, htf_trend_agreement, htf_vol_rel, htf_bb_pct_b.
        """
        has_symbol = "symbol" in df.columns

        # ── warn when buffer is too shallow for full HTF warm-up ────────────
        n_rows = (
            len(df)
            if not has_symbol
            else (
                df.group_by("symbol")
                .agg(pl.len().alias("n"))
                .select(pl.col("n").min())[0, 0]
            )
        )
        if n_rows < 250:
            logger.warning(
                "HTF features: only %d 1m bars available "
                "(need ~250 for full 5m SMA-50 warm-up). "
                "Some HTF features will be NaN.",
                n_rows,
            )

        # ── 1. Resample to HTF OHLCV bars ───────────────────────────────────
        if has_symbol:
            htf_bars = (
                df.sort(["symbol", "timestamp"])
                .group_by_dynamic("timestamp", every=timeframe, by="symbol")
                .agg(
                    pl.col("open").first().alias("htf_open"),
                    pl.col("high").max().alias("htf_high"),
                    pl.col("low").min().alias("htf_low"),
                    pl.col("close").last().alias("htf_close"),
                    pl.col("volume").sum().alias("htf_volume"),
                )
            )
        else:
            htf_bars = (
                df.sort("timestamp")
                .group_by_dynamic("timestamp", every=timeframe)
                .agg(
                    pl.col("open").first().alias("htf_open"),
                    pl.col("high").max().alias("htf_high"),
                    pl.col("low").min().alias("htf_low"),
                    pl.col("close").last().alias("htf_close"),
                    pl.col("volume").sum().alias("htf_volume"),
                )
            )

        # ── 2. Apply TA-Lib HTF indicators per symbol ────────────────────────
        def _apply_htf_talib(sym_df: pl.DataFrame) -> pl.DataFrame:
            """Apply HTF TA-Lib indicators to a single symbol's resampled bars."""
            htf_close = sym_df["htf_close"].to_numpy()
            htf_high = sym_df["htf_high"].to_numpy()
            htf_low = sym_df["htf_low"].to_numpy()

            htf_rsi = talib.RSI(htf_close, timeperiod=_HTF_RSI_PERIOD)
            htf_sma_50 = talib.SMA(htf_close, timeperiod=_HTF_SMA_PERIOD)
            htf_bb_upper, htf_bb_middle, htf_bb_lower = talib.BBANDS(
                htf_close,
                timeperiod=_HTF_BB_PERIOD,
                nbdevup=_HTF_BB_STD,
                nbdevdn=_HTF_BB_STD,
                matype=talib.MA_Type.SMA,
            )
            return sym_df.with_columns(
                pl.Series("htf_rsi_14", htf_rsi),
                pl.Series("_htf_sma_50", htf_sma_50),
                pl.Series("_htf_bb_upper", htf_bb_upper),
                pl.Series("_htf_bb_lower", htf_bb_lower),
                pl.Series("_htf_bb_middle", htf_bb_middle),
            )

        if has_symbol:
            htf_bars = pl.concat(
                [
                    _apply_htf_talib(htf_bars.filter(pl.col("symbol") == sym))
                    for sym in htf_bars["symbol"].unique().sort().to_list()
                ],
                how="vertical_relaxed",
            )
        else:
            htf_bars = _apply_htf_talib(htf_bars)

        # ── 3. Derived HTF features ──────────────────────────────────────────
        # Volume relative to its own 20-bar HTF rolling mean
        htf_bars = htf_bars.with_columns(
            (pl.col("htf_volume") / pl.col("htf_volume").rolling_mean(window_size=20))
            .fill_nan(1.0)
            .fill_null(1.0)
            .alias("htf_vol_rel")
        )

        # Bollinger %B on HTF bars
        htf_bars = htf_bars.with_columns(
            (
                (pl.col("htf_close") - pl.col("_htf_bb_lower"))
                / (pl.col("_htf_bb_upper") - pl.col("_htf_bb_lower"))
            )
            .fill_nan(0.5)
            .fill_null(0.5)
            .alias("htf_bb_pct_b")
        )

        # ── 4. available_at — THE LOOKAHEAD PREVENTION ──────────────────────
        # Parse timeframe string: "5m" → timedelta(minutes=5)
        match = re.match(r"^(\d+)([mhd])$", timeframe)
        if not match:
            raise ValueError(
                f"Invalid htf_timeframe format '{timeframe}'. "
                "Expected format: '<N>m', '<N>h', or '<N>d' (e.g. '5m')."
            )
        value, unit = int(match.group(1)), match.group(2)
        td = timedelta(
            minutes=value if unit == "m" else 0,
            hours=value if unit == "h" else 0,
            days=value if unit == "d" else 0,
        )

        # The HTF bar that *starts* at T is only available (complete) at T + td.
        # Any 1m bar whose timestamp < T + td must NOT see this HTF bar.
        htf_bars = htf_bars.with_columns(
            (pl.col("timestamp") + td).alias("available_at")
        )

        # ── 5. Select join columns ───────────────────────────────────────────
        join_cols = [
            "available_at",
            "htf_rsi_14",
            "_htf_sma_50",
            "htf_vol_rel",
            "htf_bb_pct_b",
        ]
        if has_symbol:
            join_cols = ["symbol"] + join_cols

        htf_features = htf_bars.select(join_cols).sort(
            ["symbol", "available_at"] if has_symbol else "available_at"
        )

        # ── 6. join_asof (backward) — each 1m bar sees latest completed HTF bar
        df_sorted = df.sort(["symbol", "timestamp"] if has_symbol else "timestamp")

        if has_symbol:
            df_sorted = df_sorted.join_asof(
                htf_features,
                left_on="timestamp",
                right_on="available_at",
                by="symbol",
                strategy="backward",
            )
        else:
            df_sorted = df_sorted.join_asof(
                htf_features,
                left_on="timestamp",
                right_on="available_at",
                strategy="backward",
            )

        # ── 7. htf_trend_agreement: +1 above 5m SMA-50, -1 below, 0 when NaN ─
        df_sorted = df_sorted.with_columns(
            pl.when(pl.col("_htf_sma_50").is_null() | pl.col("_htf_sma_50").is_nan())
            .then(pl.lit(0, dtype=pl.Int8))
            .when(pl.col("close") > pl.col("_htf_sma_50"))
            .then(pl.lit(1, dtype=pl.Int8))
            .otherwise(pl.lit(-1, dtype=pl.Int8))
            .alias("htf_trend_agreement")
        )

        # ── 8. Drop all intermediate columns ────────────────────────────────
        drop_cols = [
            "_htf_sma_50",
            "_htf_bb_upper",
            "_htf_bb_lower",
            "_htf_bb_middle",
            "available_at",
        ]
        existing_drops = [c for c in drop_cols if c in df_sorted.columns]
        df_sorted = df_sorted.drop(existing_drops)

        return df_sorted


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
