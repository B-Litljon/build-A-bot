"""
src/day_trading/features.py
Feature Engineering Pipeline — Universal Scalper V4.0

Three composable generators chained in this order:

    DayTradeBaseFeatures      ← TA-Lib indicators on 5-minute bars
    DayTradeDailyJoin         ← Per-session scalars from daily bars (ATR, gap, vol)
    DayTradeIntradayFeatures  ← Intraday session dynamics (VWAP, range, progress)

Wire them together with the shared FeaturePipeline from src/ml/feature_pipeline.py:

    from ml.feature_pipeline import FeaturePipeline
    from day_trading.features import (
        DayTradeBaseFeatures,
        DayTradeDailyJoin,
        DayTradeIntradayFeatures,
    )
    from day_trading.targets import DayTradeTargets
    import polars as pl

    daily_df = pl.concat([
        pl.read_parquet(f"data/raw/dt_{s}_daily.parquet")
        for s in UNIVERSE
    ])
    pipeline = FeaturePipeline(
        feature_generators=[
            DayTradeBaseFeatures(),
            DayTradeDailyJoin(daily_df),
            DayTradeIntradayFeatures(),
        ],
        target_generator=DayTradeTargets(),
    )
    df_5min = pl.concat([
        pl.read_parquet(f"data/raw/dt_{s}_5min.parquet")
        for s in UNIVERSE
    ])
    processed = pipeline.run(df_5min)
    processed.write_parquet("data/processed/training_data_dt_5m.parquet")

──────────────────────────────────────────────────────────────────────────────
LOOKAHEAD BIAS GUARANTEE — DayTradeDailyJoin
──────────────────────────────────────────────────────────────────────────────
The danger in joining daily bars to intraday bars is that a 5-minute bar at
09:35 on a Monday might receive that Monday's close-based features — values
that could not have been known until 16:00.  We prevent this via the
`available_at` pattern used by V3HTFFeatures:

  1. For each daily row, `available_at` is set to the OPEN of the NEXT
     trading day (i.e. today's close becomes available tomorrow morning).
     Concretely:

         daily_df["available_at"] = daily_df["timestamp"].shift(-1)
         # "timestamp" is the OPEN of each daily bar.  shift(-1) gives the
         # OPEN timestamp of the following day.

  2. We join the 5-minute DataFrame to the daily feature table using
     `join_asof(strategy="backward")` on (symbol, timestamp ← available_at).
     `strategy="backward"` means: for each 5m bar, find the most recent
     daily feature row whose `available_at` ≤ bar_timestamp.

  3. Consequence: the first 5-minute bar of Monday (09:30) can only see
     Friday's daily features (available_at = Monday 09:30 == first 5m bar,
     which satisfies the ≤ condition only if we set available_at to the
     exact market open).  To be safe we set available_at to noon UTC of
     the next calendar day (12:00 UTC ≈ 08:00 ET), which is strictly
     before any RTH bar but strictly after the prior session's close.

  Result: every 5-minute bar during a session receives only features that
  were finalised as of the PREVIOUS session's close.  No future data leaks.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import polars as pl
import talib

from ml.core.interfaces import BaseFeatureGenerator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TA-Lib periods for 5-minute bars
#
# At 5-minute resolution these numeric periods encode the following real-time
# durations — which are the correct windows for day-trading context:
#
#   RSI(14) : 14 × 5 min = 70 min  ≈ first 2 hours of momentum context
#   PPO(12,26): 60 / 130 min       ≈ intraday oscillator
#   BB(20)  : 100 min              ≈ 1.5-hour volatility envelope
#   SMA(50) : 250 min              ≈ full RTH session trend anchor
#   NATR(14): 70 min               ≈ intraday micro-volatility regime
#
# These are the same numeric values as V3.4 (1-minute bars) but they encode
# different temporal context.  No rescaling is needed — the model learns from
# what the numbers mean at this resolution, not from the raw period integer.
# ─────────────────────────────────────────────────────────────────────────────
_RSI_PERIOD: int = 14
_PPO_FAST: int = 12
_PPO_SLOW: int = 26
_BB_PERIOD: int = 20
_BB_STD: int = 2
_SMA_PERIOD: int = 50
_NATR_PERIOD: int = 14
_RANGE_COIL_PERIOD: int = 10

# RTH session boundaries in minutes from midnight (Eastern Time)
_RTH_OPEN_MINUTE: int = 570  # 09:30 ET
_RTH_DURATION_MINUTES: int = 390  # 09:30 → 16:00

# Bars that constitute the "first 30 minutes" of RTH.
# At 5-minute resolution: 30 min / 5 min = 6 bars.
_FIRST_30M_BARS: int = 6

# Rolling window for the 30-day average of first-30m volume.
_FIRST_30M_VOL_ROLLING_WINDOW: int = 30


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR 1 — DayTradeBaseFeatures
# ═══════════════════════════════════════════════════════════════════════════════


class DayTradeBaseFeatures(BaseFeatureGenerator):
    """
    Computes TA-Lib momentum / volatility / trend indicators and Phase-5
    microstructure features on 5-minute OHLCV bars.

    Input columns required : timestamp, open, high, low, close, volume
    New columns produced   :
        rsi_14, ppo, natr_14,
        bb_pct_b, bb_width_pct,
        price_sma50_ratio, log_return, hour_of_day (Int8),
        dist_sma50, vol_rel,
        range_coil_10, bar_body_pct,
        bar_upper_wick_pct, bar_lower_wick_pct

    Intermediate columns (prefixed with "_") are dropped before returning.
    """

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        high = df["high"].to_numpy()
        low = df["low"].to_numpy()

        # ── Momentum ──────────────────────────────────────────────────────────
        rsi = talib.RSI(close, timeperiod=_RSI_PERIOD)
        ppo = talib.PPO(
            close,
            fastperiod=_PPO_FAST,
            slowperiod=_PPO_SLOW,
            matype=talib.MA_Type.SMA,
        )

        # ── Volatility Bands ──────────────────────────────────────────────────
        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            close,
            timeperiod=_BB_PERIOD,
            nbdevup=_BB_STD,
            nbdevdn=_BB_STD,
            matype=talib.MA_Type.SMA,
        )

        # ── Trend ─────────────────────────────────────────────────────────────
        sma_50 = talib.SMA(close, timeperiod=_SMA_PERIOD)

        # ── Intraday Volatility Regime ────────────────────────────────────────
        # natr_14 on 5m measures within-session micro-volatility.
        # The primary risk-sizing unit for day trading is daily_atr_abs
        # (produced by DayTradeDailyJoin); natr_14 here is an additional
        # intraday-regime feature for the model, not the bracket calculator.
        natr = talib.NATR(high, low, close, timeperiod=_NATR_PERIOD)

        df = df.with_columns(
            pl.Series("rsi_14", rsi, nan_to_null=True),
            pl.Series("ppo", ppo, nan_to_null=True),
            pl.Series("_bb_upper", bb_upper, nan_to_null=True),
            pl.Series("_bb_mid", bb_middle, nan_to_null=True),
            pl.Series("_bb_lower", bb_lower, nan_to_null=True),
            pl.Series("_sma_50", sma_50, nan_to_null=True),
            pl.Series("natr_14", natr, nan_to_null=True),
        )

        # ── Normalised derived features ───────────────────────────────────────
        df = df.with_columns(
            (
                (pl.col("close") - pl.col("_bb_lower"))
                / (pl.col("_bb_upper") - pl.col("_bb_lower"))
            ).alias("bb_pct_b"),
            ((pl.col("_bb_upper") - pl.col("_bb_lower")) / pl.col("_bb_mid")).alias(
                "bb_width_pct"
            ),
            (pl.col("close") / pl.col("_sma_50")).alias("price_sma50_ratio"),
            ((pl.col("close") / pl.col("close").shift(1)).log()).alias("log_return"),
            pl.col("timestamp").dt.hour().cast(pl.Int8).alias("hour_of_day"),
            ((pl.col("close") - pl.col("_sma_50")) / pl.col("_sma_50")).alias(
                "dist_sma50"
            ),
        )

        # ── Volume relative to 20-bar rolling mean ────────────────────────────
        df = df.with_columns(
            (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20))
            .fill_nan(1.0)
            .fill_null(1.0)
            .alias("vol_rel")
        )

        # ── Phase-5 Microstructure (pure Polars, no TA-Lib) ──────────────────
        df = df.with_columns(
            # Range compression — detects coiling before breakouts
            (
                (pl.col("high") - pl.col("low"))
                / (
                    (pl.col("high") - pl.col("low"))
                    .rolling_mean(window_size=_RANGE_COIL_PERIOD)
                    .fill_null(1.0)
                    + 1e-6
                )
            ).alias("range_coil_10"),
            # Body conviction
            (
                (pl.col("close") - pl.col("open")).abs()
                / (pl.col("high") - pl.col("low") + 1e-6)
            ).alias("bar_body_pct"),
            # Upper wick — rejection / distribution signal
            (
                (pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close")))
                / (pl.col("high") - pl.col("low") + 1e-6)
            ).alias("bar_upper_wick_pct"),
            # Lower wick — stop-hunt / accumulation signal
            (
                (pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low"))
                / (pl.col("high") - pl.col("low") + 1e-6)
            ).alias("bar_lower_wick_pct"),
        )

        # Drop intermediate columns
        df = df.drop(["_bb_upper", "_bb_mid", "_bb_lower", "_sma_50"])

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR 2 — DayTradeDailyJoin
# ═══════════════════════════════════════════════════════════════════════════════


class DayTradeDailyJoin(BaseFeatureGenerator):
    """
    Computes per-session scalar features from daily bars and joins them onto
    the 5-minute DataFrame using a strict `available_at` lookahead guard.

    ──────────────────────────────────────────────────────────────────────────
    LOOKAHEAD BIAS PREVENTION — detailed explanation
    ──────────────────────────────────────────────────────────────────────────
    A daily bar for date D has:
        open      = price at ~09:30 ET on D
        close     = price at ~16:00 ET on D
        high/low  = session extremes on D

    We compute NATR(14), gap_pct (today's open vs yesterday's close), and
    daily_atr_abs from these bars.  None of these values are known until
    after the session on date D CLOSES at 16:00 ET.  A 5-minute bar on
    date D (any bar from 09:30–15:55) must NOT receive values derived from
    date D's close.

    The fix: `available_at` = noon UTC on date D+1.

        available_at[D] = D's next-calendar-day noon UTC
                        = timestamp(D) shifted forward by 1 calendar day,
                          then floor to date, plus 12 hours UTC
                          (12:00 UTC ≈ 08:00 ET — before the 09:30 ET open)

    When we do join_asof(strategy="backward"):
        For any 5m bar on date D with timestamp T:
            The query is: find the largest available_at ≤ T
            available_at[D]   = D+1 12:00 UTC  >  T (since T ≤ 15:55 ET on D)  → excluded
            available_at[D-1] = D   12:00 UTC  ≤  T (since T ≥ 09:30 ET on D)  → matched

    So every bar on date D receives D-1's finalised daily features.
    Monday's first bar (09:30) correctly sees Friday's daily features.
    Friday's daily features become available at Monday 12:00 UTC.

    This is structurally identical to the `available_at` pattern in
    V3HTFFeatures (src/ml/features/v3_features.py), extended to
    calendar-day (rather than 5-minute) boundaries.

    ──────────────────────────────────────────────────────────────────────────
    Input columns required (daily_df):
        symbol, timestamp (UTC datetime), open, high, low, close, volume

    New columns produced (joined onto 5m df, constant per session):
        daily_natr_14     — NATR(14) on daily bars (% volatility regime width)
        daily_atr_abs     — close × daily_natr_14 / 100  ($ risk unit)
        gap_pct           — (today_open − prev_close) / prev_close
        first_30m_vol_rel — today's first-30-min volume / 30-day rolling average
    ──────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, daily_df: pl.DataFrame) -> None:
        """
        Pre-compute all daily-level features so that `generate()` only
        needs to perform the join — no repeated computation per symbol.

        Args:
            daily_df: Combined daily OHLCV for all symbols (dt_{S}_daily.parquet).
                      Must include columns: symbol, timestamp, open, high, low,
                      close, volume.  Extra columns are harmless.
        """
        self._daily_features: pl.DataFrame = self._precompute(daily_df)

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _precompute(daily_df: pl.DataFrame) -> pl.DataFrame:
        """
        Build the join table: one row per (symbol, available_at) with the
        four daily scalar features.

        Steps:
        1. Sort by (symbol, timestamp) — essential for shift(-1) semantics.
        2. Compute NATR(14) via TA-Lib per symbol → daily_natr_14, daily_atr_abs.
        3. Compute gap_pct: shift(1) prev close within each symbol.
        4. Set available_at = noon UTC on the NEXT calendar day.
        5. Select only the columns needed for the join.
        """
        has_symbol = "symbol" in daily_df.columns
        daily_df = daily_df.sort(
            ["symbol", "timestamp"] if has_symbol else ["timestamp"]
        )

        # ── Step 2: TA-Lib NATR(14) per symbol ───────────────────────────────
        def _apply_natr(sym_df: pl.DataFrame) -> pl.DataFrame:
            high = sym_df["high"].to_numpy()
            low = sym_df["low"].to_numpy()
            close = sym_df["close"].to_numpy()
            natr = talib.NATR(high, low, close, timeperiod=14)
            return sym_df.with_columns(
                pl.Series("daily_natr_14", natr, nan_to_null=True)
            )

        if has_symbol:
            daily_df = pl.concat(
                [
                    _apply_natr(daily_df.filter(pl.col("symbol") == s))
                    for s in daily_df["symbol"].unique().sort().to_list()
                ],
                how="vertical_relaxed",
            ).sort(["symbol", "timestamp"])
        else:
            daily_df = _apply_natr(daily_df)

        # daily_atr_abs: absolute dollar ATR from NATR percentage
        daily_df = daily_df.with_columns(
            (pl.col("close") * pl.col("daily_natr_14") / 100.0).alias("daily_atr_abs")
        )

        # ── Step 3: gap_pct = (today_open − prev_close) / prev_close ──────────
        # shift(1) within each symbol gives the previous row's close.
        group_by = ["symbol"] if has_symbol else []
        if group_by:
            daily_df = daily_df.with_columns(
                pl.col("close").shift(1).over("symbol").alias("_prev_close")
            )
        else:
            daily_df = daily_df.with_columns(
                pl.col("close").shift(1).alias("_prev_close")
            )

        daily_df = daily_df.with_columns(
            ((pl.col("open") - pl.col("_prev_close")) / pl.col("_prev_close")).alias(
                "gap_pct"
            )
        )

        # ── Step 4: available_at — noon UTC on the next calendar day ──────────
        #
        # Implementation note on Polars datetime arithmetic:
        #   pl.col("timestamp").dt.date() returns a Date (no time).
        #   Adding pl.duration(days=1) increments by one calendar day.
        #   Casting back to Datetime(us, UTC) at 00:00 then adding
        #   pl.duration(hours=12) gives exactly 12:00 UTC.
        #
        #   We cannot just do `timestamp + timedelta(days=1)` on a
        #   datetime-with-time because the time component of a daily bar
        #   varies across data providers (midnight, 09:30, etc.).
        #   Flooring to date and rebuilding guarantees a clean noon-UTC anchor.
        daily_df = daily_df.with_columns(
            (
                (pl.col("timestamp").dt.date() + pl.duration(days=1)).cast(
                    pl.Datetime("us", "UTC")
                )
                + pl.duration(hours=12)
            ).alias("available_at")
        )

        # ── Step 5: Select final columns ──────────────────────────────────────
        keep = ["available_at", "daily_natr_14", "daily_atr_abs", "gap_pct"]
        if has_symbol:
            keep = ["symbol"] + keep

        return daily_df.select(keep).sort(
            ["symbol", "available_at"] if has_symbol else ["available_at"]
        )

    # ─────────────────────────────────────────────────────────────────────────
    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Join precomputed daily scalars onto the 5-minute DataFrame and
        compute first_30m_vol_rel from the 5-minute data itself.

        The asof join is the lookahead fence.  All other operations are
        purely intraday (within-session cumulative sums / group keys).
        """
        has_symbol = "symbol" in df.columns

        # ── Ensure trade_date column exists ───────────────────────────────────
        # trade_date is the Eastern-Time calendar date.  Timestamps in df are
        # UTC, so we convert to ET before extracting the date.  This avoids
        # mis-assigning late-session bars (18:00–20:00 UTC) to the wrong day.
        if "trade_date" not in df.columns:
            df = df.with_columns(
                pl.col("timestamp")
                .dt.convert_time_zone("America/New_York")
                .dt.date()
                .alias("trade_date")
            )

        df = df.sort(["symbol", "timestamp"] if has_symbol else ["timestamp"])

        # ── Asof join: backward strategy on available_at ──────────────────────
        #
        # For each 5m bar with timestamp T, this finds the most recent
        # daily feature row whose available_at ≤ T.
        # Because available_at = noon UTC on D+1, any bar on day D
        # (which ends at ~21:00 UTC = 16:00 ET) will match the
        # available_at for day D-1 (which is D's noon UTC).
        #
        # Concrete example (TSLA, Monday 2025-03-10):
        #   Bar at 14:35 UTC (09:35 ET) on Mon 2025-03-10 has T = 14:35 UTC
        #   available_at for Fri 2025-03-07 = 2025-03-08 12:00 UTC ← matches (≤ T)
        #   available_at for Mon 2025-03-10 = 2025-03-11 12:00 UTC ← excluded (> T)
        #
        # The backward strategy guarantees we never look forward.
        if has_symbol:
            df = df.join_asof(
                self._daily_features,
                left_on="timestamp",
                right_on="available_at",
                by="symbol",
                strategy="backward",
            )
        else:
            df = df.join_asof(
                self._daily_features,
                left_on="timestamp",
                right_on="available_at",
                strategy="backward",
            )

        # ── first_30m_vol_rel — computed from the 5m bars ─────────────────────
        #
        # "First 30 minutes" = the first 6 × 5-minute bars of each RTH session.
        # We rank bars chronologically within each (symbol, trade_date) group.
        # Bars with rank ≤ 6 contribute to that session's opening volume sum.
        # The sum is aggregated to a daily scalar and broadcast back.
        # A 30-day rolling mean of that daily scalar is the denominator.
        group_keys: List[str] = (
            ["symbol", "trade_date"] if has_symbol else ["trade_date"]
        )

        # Ordinal rank within each session (1 = first bar of the day)
        df = df.with_columns(
            pl.col("timestamp").rank("ordinal").over(group_keys).alias("_bar_rank")
        )

        # Mark only the first 6 bars' volume as "opening volume"
        df = df.with_columns(
            pl.when(pl.col("_bar_rank") <= _FIRST_30M_BARS)
            .then(pl.col("volume"))
            .otherwise(pl.lit(0.0))
            .alias("_open_vol_contrib")
        )

        # Aggregate opening volume per session
        session_open_vol = (
            df.group_by(group_keys)
            .agg(pl.col("_open_vol_contrib").sum().alias("_first30m_vol"))
            .sort(group_keys)
        )

        # 30-day rolling mean of opening volume per symbol
        if has_symbol:
            session_open_vol = session_open_vol.with_columns(
                pl.col("_first30m_vol")
                .rolling_mean(window_size=_FIRST_30M_VOL_ROLLING_WINDOW)
                .over("symbol")
                .alias("_avg_first30m_vol")
            )
        else:
            session_open_vol = session_open_vol.with_columns(
                pl.col("_first30m_vol")
                .rolling_mean(window_size=_FIRST_30M_VOL_ROLLING_WINDOW)
                .alias("_avg_first30m_vol")
            )

        session_open_vol = session_open_vol.with_columns(
            (pl.col("_first30m_vol") / pl.col("_avg_first30m_vol"))
            .fill_nan(1.0)
            .fill_null(1.0)
            .alias("first_30m_vol_rel")
        )

        # Join first_30m_vol_rel back to the 5m DataFrame (scalar per session)
        df = df.join(
            session_open_vol.select(group_keys + ["first_30m_vol_rel"]),
            on=group_keys,
            how="left",
        )

        # Drop internal staging columns
        df = df.drop(["_bar_rank", "_open_vol_contrib"])

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR 3 — DayTradeIntradayFeatures
# ═══════════════════════════════════════════════════════════════════════════════


class DayTradeIntradayFeatures(BaseFeatureGenerator):
    """
    Computes features that evolve during the session: VWAP, intraday trend
    vs. the day's open, range exhaustion, and session progress.

    Prerequisite: `DayTradeDailyJoin` must have already run, since
    `range_exhaustion` divides by `daily_atr_abs`.

    Input columns required : timestamp, open, high, low, close, volume,
                             daily_atr_abs (from DayTradeDailyJoin),
                             trade_date (from DayTradeDailyJoin)

    New columns produced   :
        vwap               — Intraday VWAP, resets at each session's first bar
        vwap_dist          — (close − vwap) / close  (signed % deviation)
        day_open           — Open price of the session's first 5m bar
        trend_vs_open      — (close − day_open) / day_open  (intraday drift)
        intraday_range     — session-to-date high − session-to-date low
        range_exhaustion   — intraday_range / daily_atr_abs
        session_progress   — normalised RTH position  [0.0 @ 09:30 → 1.0 @ 16:00]
    """

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        has_symbol = "symbol" in df.columns
        group_keys: List[str] = (
            ["symbol", "trade_date"] if has_symbol else ["trade_date"]
        )

        # Ensure sorted so that cumulative functions are chronological
        df = df.sort(group_keys + ["timestamp"])

        # ── VWAP (intraday, resets each session) ──────────────────────────────
        #
        # VWAP = Σ(typical_price × volume) / Σ(volume)  from bar 1 to bar i.
        # Using Polars' cum_sum().over(group_keys) gives an expanding sum
        # that resets at the boundary of each (symbol, trade_date) group.
        # This is equivalent to pandas' groupby().cumsum() but vectorised.
        df = df.with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_tp")
        )
        df = df.with_columns((pl.col("_tp") * pl.col("volume")).alias("_tp_vol"))
        df = df.with_columns(
            (
                pl.col("_tp_vol").cum_sum().over(group_keys)
                / (pl.col("volume").cum_sum().over(group_keys) + 1e-10)
            ).alias("vwap")
        )
        df = df.with_columns(
            ((pl.col("close") - pl.col("vwap")) / (pl.col("close") + 1e-10)).alias(
                "vwap_dist"
            )
        )

        # ── Intraday Trend vs. Day Open ───────────────────────────────────────
        #
        # `.first().over(group_keys)` broadcasts the session's first open
        # price to every bar in that group.  This is a pure window function —
        # no future data is accessed: the first bar's open is known at 09:30.
        df = df.with_columns(pl.col("open").first().over(group_keys).alias("day_open"))
        df = df.with_columns(
            (
                (pl.col("close") - pl.col("day_open")) / (pl.col("day_open") + 1e-10)
            ).alias("trend_vs_open")
        )

        # ── Range Exhaustion ──────────────────────────────────────────────────
        #
        # cum_max / cum_min expand session-to-date within each group.
        # These are strictly causal: bar i's value uses only bars 1..i.
        df = df.with_columns(
            pl.col("high").cum_max().over(group_keys).alias("_session_high"),
            pl.col("low").cum_min().over(group_keys).alias("_session_low"),
        )
        df = df.with_columns(
            (pl.col("_session_high") - pl.col("_session_low")).alias("intraday_range")
        )
        df = df.with_columns(
            (pl.col("intraday_range") / (pl.col("daily_atr_abs") + 1e-10)).alias(
                "range_exhaustion"
            )
        )

        # ── Session Progress ──────────────────────────────────────────────────
        #
        # Normalised position within RTH: 0.0 = 09:30 ET, 1.0 = 16:00 ET.
        # We convert timestamps to Eastern Time before extracting hour/minute
        # to handle DST transitions correctly without hard-coding a UTC offset.
        df = df.with_columns(
            pl.col("timestamp").dt.convert_time_zone("America/New_York").alias("_ts_et")
        )
        df = df.with_columns(
            (
                (
                    pl.col("_ts_et").dt.hour() * 60
                    + pl.col("_ts_et").dt.minute()
                    - _RTH_OPEN_MINUTE
                ).cast(pl.Float64)
                / _RTH_DURATION_MINUTES
            )
            .clip(0.0, 1.0)
            .alias("session_progress")
        )

        # Drop all staging columns
        df = df.drop(["_tp", "_tp_vol", "_session_high", "_session_low", "_ts_et"])

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMN MANIFEST
# ═══════════════════════════════════════════════════════════════════════════════

#: Complete ordered feature vector for model training / inference.
#: Must be kept in sync with any DayTradeRetrainer FEATURE_COLS constant.
DAY_TRADE_FEATURE_COLS: List[str] = [
    # Group A — TA-Lib base (5-minute)
    "rsi_14",
    "ppo",
    "natr_14",
    "bb_pct_b",
    "bb_width_pct",
    "price_sma50_ratio",
    "log_return",
    "hour_of_day",
    "dist_sma50",
    "vol_rel",
    # Group B — Macro-intraday context
    "vwap_dist",
    "gap_pct",
    "first_30m_vol_rel",
    "trend_vs_open",
    "range_exhaustion",
    "session_progress",
    # Group C — Microstructure (carried from V3.4)
    "range_coil_10",
    "bar_body_pct",
    "bar_upper_wick_pct",
    "bar_lower_wick_pct",
    # Group D — Daily ATR context
    "daily_natr_14",
    "daily_atr_abs",
]
