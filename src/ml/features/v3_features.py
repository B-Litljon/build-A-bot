"""
The feature generators -- every number the models actually look at.

Four pluggable generators, each adding columns to the bar DataFrame. Training
and live inference run the SAME generators in the same order; that symmetry is
the whole point, because a feature computed differently in the two places
silently poisons the model. See GLOSSARY.md (feature, NATR, HTF, lookahead).

Column names produced here must match retrainer.BASE_FEATURE_COLS exactly.

Glossary:
    V3BaseFeatures -- the 14 single-bar columns, computed per symbol so one
        instrument's history never bleeds into another's indicators.
    _RSI_PERIOD / _PPO_FAST / _PPO_SLOW / _BB_PERIOD / _BB_STD / _SMA_PERIOD /
        _NATR_PERIOD -- lookback lengths in bars for each indicator
        (14/12/26/20/2/50/14). Changing any of these invalidates every existing
        model, since the saved model expects the old distribution.
    rsi_14 -- 0-100 measure of how one-sided recent moves have been. Near 100
        means almost all recent movement was upward.
    ppo -- momentum: the gap between a fast and a slow average, as a PERCENT of
        price. The percentage is what lets one model span gold at $2000 and a
        currency pair at 1.09.
    natr_14 -- typical bar range as a percent of price. This is the volatility
        measure everything else scales from: bracket widths, the cost gate, and
        the position size all derive from it.
    bb_pct_b -- where price sits inside its recent statistical range: 0 = at
        the bottom edge, 1 = at the top, outside means an unusual move.
    bb_width_pct -- how wide that range is; small means quiet/compressed.
    price_sma50_ratio -- price divided by its 50-bar average (above 1 = above
        trend).
    dist_sma50 -- the same idea as a signed percentage distance.
    log_return -- this bar's move versus the last, in log form so gains and
        losses are symmetric.
    hour_of_day -- UTC hour 0-23.
    vol_rel -- this bar's activity versus its own recent 20-bar average. 1.0
        means normal. Defaults to 1.0 when undefined rather than 0, so a
        warm-up bar reads "normal" instead of "dead".
    _RANGE_COIL_PERIOD -- 10 bars, the window for range_coil_10.
    range_coil_10 -- this bar's range versus the recent average range. Below 1
        means the market is coiling/compressing.
    bar_body_pct -- how much of the bar is solid body versus wick. High means
        decisive directional movement.
    bar_upper_wick_pct / bar_lower_wick_pct -- fraction of the bar spent in the
        upper/lower wick. A long upper wick means price pushed up and was
        rejected -- the "stop-hunt defense" signal these were added for.

    V3SessionFeatures -- four 0/1 flags for which trading session a bar falls
        in, derived from the UTC hour. Added because the same indicator values
        mean different things at 03:00 than at 14:00.
    session_asia / session_london / session_ny -- 00-09, 07-16, 12-21 UTC.
        Deliberately overlapping, and chosen as conservative midpoints so they
        stay roughly right across daylight-saving shifts.
    session_overlap -- 12-16 UTC, when London and New York are both open. The
        busiest, most tradeable window.

    V3CostFeatures -- adds one column, cost_ratio, letting the model SEE
        trading cost instead of having cost silently filtered out behind it.
    cost_ratio -- alpha_sym * baseline_natr / natr_14, i.e. the live cost gate's
        own inequality rearranged. Higher means the toll is large relative to
        the move being targeted, so the setup is expensive. Falls when
        volatility rises above its baseline -- trading gets relatively cheaper.
    alpha_table -- {symbol: measured cost} from live measurements. Mutable on
        purpose: hot-reload swaps it in place after a retrain lands. When it is
        None or empty the generator is a NO-OP and the column never appears, so
        old models behave exactly as before this feature existed.
    default_alpha -- 0.15, used for any symbol missing from the table (warned
        once per symbol via _warned_missing).
    regime_window -- 260 bars, the window for the baseline volatility median.
    min_samples=1 -- REQUIRED, not a detail: a strict window would leave
        cost_ratio null across the whole live warm-up, every row would then be
        dropped as incomplete, and the strategy would return no signal forever.

    V3HTFFeatures -- higher-timeframe context: what the slower chart says,
        joined onto each fast bar.
    timeframe -- the slower bar size, default "5m".
    htf_rsi_14 / htf_bb_pct_b -- the same measures as above but computed on
        the slower bars.
    htf_vol_rel -- the same measure as vol_rel but on the slower bars, and
        partitioned .over("symbol") when the frame is pooled: the HTF blocks
        are concatenated per symbol, so an unpartitioned rolling mean would
        bleed the previous symbol's volume into the next symbol's warm-up.
    htf_trend_agreement -- +1 if price is above the slow 50-bar average, -1 if
        below, 0 if unknown. The cheap "am I trading with or against the bigger
        move" signal.
    available_at -- THE LOOKAHEAD GUARD, and the subtlest thing in this file. A
        5-minute bar stamped 12:00 is not FINISHED until 12:05, so using it at
        12:01 would be reading the future. Each slow bar's timestamp is pushed
        forward by one full timeframe, and the join only matches slow bars
        already available. Get this wrong and backtests look brilliant while
        live trading fails.
    join_asof(strategy="backward") -- attaches the most recent ALREADY-
        AVAILABLE slow bar to each fast bar.
    Columns prefixed "_" (_htf_sma_50, _htf_bb_upper, ...) -- intermediates,
        dropped before returning.
"""

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
        has_symbol = "symbol" in df.columns
        if has_symbol and df["symbol"].n_unique() > 1:
            # Sort to keep things deterministic and ordered
            df_sorted = df.sort(["symbol", "timestamp"])
            parts = []
            for sym in df_sorted["symbol"].unique().sort().to_list():
                sym_df = df_sorted.filter(pl.col("symbol") == sym)
                parts.append(self._generate_single_symbol(sym_df))
            return pl.concat(parts, how="vertical_relaxed")
        else:
            return self._generate_single_symbol(df)

    def _generate_single_symbol(self, df: pl.DataFrame) -> pl.DataFrame:
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
                / (pl.col("bb_upper") - pl.col("bb_lower") + 1e-9)
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

class V3SessionFeatures(BaseFeatureGenerator):
    """
    Binary session-activity indicators from the UTC hour of `timestamp`.

    Forex / metals microstructure regimes shift sharply across the three
    major trading sessions; capturing them lets the model condition on
    activity context rather than rediscover the bins from `hour_of_day`.

    Session windows (UTC, conservative midpoints across DST seasons):
        Asia/Tokyo   : 00:00 – 09:00
        London       : 07:00 – 16:00
        New York     : 12:00 – 21:00
        London/NY    : 12:00 – 16:00  (overlap — the volatility sweet spot)

    Produces (all Int8 0/1): session_asia, session_london, session_ny,
    session_overlap.
    """

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        hour = pl.col("timestamp").dt.hour()
        # closed="left" → [start, end). Hour values are 0..23 ints.
        return df.with_columns(
            hour.is_between(0, 9, closed="left").cast(pl.Int8).alias("session_asia"),
            hour.is_between(7, 16, closed="left").cast(pl.Int8).alias("session_london"),
            hour.is_between(12, 21, closed="left").cast(pl.Int8).alias("session_ny"),
            hour.is_between(12, 16, closed="left").cast(pl.Int8).alias("session_overlap"),
        )


class V3CostFeatures(BaseFeatureGenerator):
    """
    Per-instrument spread-cost feature: ``cost_ratio``.

        cost_ratio = alpha_sym * baseline_natr / natr_14

    where ``alpha_sym`` is the instrument's empirically measured spread cost
    as a fraction of a typical move (baked from live SPREAD_CALIB samples by
    scripts/bake_spread_alphas.py) and ``baseline_natr`` is the per-symbol
    rolling median of ``natr_14`` over ``regime_window`` bars.

    This is exactly the live Gate A inequality rearranged
    (``sl_dist < k_eff * alpha * baseline_ATR``), so the model sees the same
    quantity the cost gate thresholds on.  Time-varying: when current
    volatility rises above baseline, the ratio falls — trading gets cheaper.

    Symmetry contract: computed from bar data + the alpha table ONLY, in both
    training and live.  The live per-tick spread is deliberately NOT used here
    (training can never see it); the tick spread keeps its existing job as
    Gate A's hard veto input.

    Notes:
      * ``alpha_table is None`` → generate() is a NO-OP (df returned
        unchanged).  The generator is always present in pipelines; model dirs
        without a ``spread_alphas.json`` behave bit-identically to before this
        feature existed.
      * ``min_samples=1`` (expanding median below a full window) is
        REQUIRED: a strict window would null ``cost_ratio`` across the whole
        live warmup buffer, clean_data would drop every row, and the
        strategy's latest-bar staleness guard would silently return None
        forever.  It also matches the training chop-veto baseline
        (expanding-then-rolling median in retrainer._compute_chop_veto_mask).
      * TA-Lib NATR emits float NaN (not null) for its first bars; polars
        rolling aggregates skip nulls but PROPAGATE NaN, so we fill_nan(None)
        here — clean_data runs too late.
      * Known, accepted divergence from the training veto: numpy's median
        yields NaN on windows containing leading NaNs (Gate A then skips via
        isfinite) while this feature skips them; affects only roughly the
        first regime_window + NATR-period bars per symbol.
      * ``alpha_table`` is intentionally a mutable attribute — live
        hot-reload swaps it in place after a retrain lands.
    """

    def __init__(
        self,
        alpha_table: Optional[dict] = None,
        default_alpha: float = 0.15,
        regime_window: int = 260,
    ):
        self.alpha_table = alpha_table
        self.default_alpha = default_alpha
        self.regime_window = regime_window
        self._warned_missing: set = set()

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        if not self.alpha_table:
            return df
        if "natr_14" not in df.columns:
            raise ValueError(
                "V3CostFeatures requires 'natr_14' — order it after "
                "V3BaseFeatures in the pipeline."
            )

        natr = pl.col("natr_14").fill_nan(None)
        baseline = natr.rolling_median(
            window_size=self.regime_window, min_samples=1
        )
        if "symbol" in df.columns:
            # Pooled training frames are already sorted (symbol, timestamp)
            # by V3BaseFeatures; live single-symbol frames are time-ordered.
            baseline = baseline.over("symbol")
            missing = set(df["symbol"].unique().to_list()) - set(self.alpha_table)
            new_missing = missing - self._warned_missing
            if new_missing:
                logger.warning(
                    "V3CostFeatures: no alpha for %s — using default_alpha=%.4f",
                    sorted(new_missing), self.default_alpha,
                )
                self._warned_missing |= new_missing
            alpha = pl.col("symbol").replace_strict(
                self.alpha_table, default=self.default_alpha,
                return_dtype=pl.Float64,
            )
        else:
            alpha = pl.lit(self.default_alpha, dtype=pl.Float64)

        # natr_14 == 0 → Inf; clean_data nulls it and drops the row
        # (degenerate-volatility bars — correct to skip).
        return df.with_columns((alpha * baseline / natr).alias("cost_ratio"))


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
                .group_by_dynamic("timestamp", every=self.timeframe, group_by="symbol")
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
        # htf_bars above is a vertical concat of per-symbol blocks, so the
        # rolling mean MUST be partitioned per symbol: unpartitioned, symbol
        # B's first ~19 HTF bars would divide by a mean containing symbol A's
        # trailing volumes (verified 2026-09-30). The same guard pattern is
        # used by V3CostFeatures and the retrainer's angel target.
        vol_rel = (
            pl.col("htf_volume") / pl.col("htf_volume").rolling_mean(window_size=20)
        )
        if has_symbol:
            vol_rel = vol_rel.over("symbol")
        htf_bars = htf_bars.with_columns(
            vol_rel.fill_nan(1.0).fill_null(1.0).alias("htf_vol_rel")
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
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
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
