from __future__ import annotations

import logging
import re
from datetime import timedelta
from typing import Optional

import numpy as np
import polars as pl
import talib

from ml.core.interfaces import BaseFeatureGenerator

logger = logging.getLogger(__name__)

# 1m base-indicator periods
_RSI_PERIOD = 14
_PPO_FAST = 12
_PPO_SLOW = 26
_BB_PERIOD = 20
_BB_STD = 2
_SMA_PERIOD = 50
_NATR_PERIOD = 14

# Microstructure configuration (Phase 5)
_RANGE_COIL_PERIOD: int = 10  # bars for range compression rolling mean

class V3BaseFeatures(BaseFeatureGenerator):
    """
    Computes 1m technical indicators via TA-Lib and Phase 5 Microstructure features.
    Produces: rsi_14, ppo, natr_14, bb_pct_b, bb_width_pct,
              price_sma50_ratio, log_return, hour_of_day,
              dist_sma50, vol_rel,
              range_coil_10, bar_body_pct,
              bar_upper_wick_pct, bar_lower_wick_pct.
    """

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
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

        # ── Phase 5: Microstructure features (pure Polars, no TA-Lib) ───────
        df = df.with_columns(
            # Range Compression
            (
                (pl.col("high") - pl.col("low"))
                / (
                    (pl.col("high") - pl.col("low"))
                    .rolling_mean(window_size=_RANGE_COIL_PERIOD)
                    .fill_null(1.0)
                    + 1e-6
                )
            ).alias("range_coil_10"),
            # Body Percentage
            (
                (pl.col("close") - pl.col("open")).abs()
                / (pl.col("high") - pl.col("low") + 1e-6)
            ).alias("bar_body_pct"),
            # Upper Wick Toxicity
            (
                (pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close")))
                / (pl.col("high") - pl.col("low") + 1e-6)
            ).alias("bar_upper_wick_pct"),
            # Lower Wick Defense
            (
                (pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low"))
                / (pl.col("high") - pl.col("low") + 1e-6)
            ).alias("bar_lower_wick_pct"),
        )

        return df

class V3HTFFeatures(BaseFeatureGenerator):
    """
    Compute higher-timeframe features and join them onto the 1m DataFrame
    using the 'available_at' pattern to prevent lookahead bias.
    """

    def __init__(self, timeframe: str = "5m"):
        self.timeframe = timeframe
        self._htf_rsi_period = 14
        self._htf_sma_period = 50
        self._htf_bb_period = 20
        self._htf_bb_std = 2

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        has_symbol = "symbol" in df.columns

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
                .group_by_dynamic("timestamp", every=self.timeframe, by="symbol")
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
                .group_by_dynamic("timestamp", every=self.timeframe)
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
            htf_close = sym_df["htf_close"].to_numpy()
            htf_rsi = talib.RSI(htf_close, timeperiod=self._htf_rsi_period)
            htf_sma_50 = talib.SMA(htf_close, timeperiod=self._htf_sma_period)
            htf_bb_upper, htf_bb_middle, htf_bb_lower = talib.BBANDS(
                htf_close,
                timeperiod=self._htf_bb_period,
                nbdevup=self._htf_bb_std,
                nbdevdn=self._htf_bb_std,
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
        htf_bars = htf_bars.with_columns(
            (pl.col("htf_volume") / pl.col("htf_volume").rolling_mean(window_size=20))
            .fill_nan(1.0)
            .fill_null(1.0)
            .alias("htf_vol_rel")
        )

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
        match = re.match(r"^(\d+)([mhd])$", self.timeframe)
        if not match:
            raise ValueError(
                f"Invalid htf_timeframe format '{self.timeframe}'. "
                "Expected format: '<N>m', '<N>h', or '<N>d' (e.g. '5m')."
            )
        value, unit = int(match.group(1)), match.group(2)
        td = timedelta(
            minutes=value if unit == "m" else 0,
            hours=value if unit == "h" else 0,
            days=value if unit == "d" else 0,
        )

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

        # ── 6. join_asof (backward)
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

        # ── 7. htf_trend_agreement ───────────────────────────────────────────
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
