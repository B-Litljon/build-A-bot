# Day Trading Model Specification
## Universal Scalper V4.0 — Intraday Trend Engine

**Authored:** 2026-04-19
**Base System:** Universal Scalper V3.4 (Angel/Devil Meta-Labeling)
**New Paradigm:** Day Trading on 5-minute bars, End-of-Day exits, Daily ATR brackets

---

## Overview

The Universal Scalper V3.4 is optimised for ultra-short-term scalping: 1-minute bar inference, 5-bar survival targets (25 minutes), and tight ATR brackets (0.5×SL / 3.0×TP). The target for V4.0 is a fundamentally different trading regime:

| Dimension | V3.4 Scalper | V4.0 Day Trader |
|---|---|---|
| Base Timeframe | 1-minute bars | **5-minute bars** |
| HTF Timeframe | 5-minute resampled | **30-minute or Daily** |
| Holding Period | 5–45 bars (5–45 min) | **From entry to EOD (up to 78 bars)** |
| SL Multiplier | 0.5× NATR-14 (1m ATR) | **1.5× Daily ATR** |
| TP Mechanism | 3.0× ATR fixed bracket | **Max Favorable Excursion gate (1.0× Daily ATR)** |
| Angel Target | 3-bar momentum (ATR-relative) | **MFE > 1.0× Daily ATR before bell** |
| Devil Target | 5-bar SL survival | **EOD survival without hitting 1.5× Daily ATR SL** |

The feature engineering must shift from microstructure noise (wick toxicity, range compression) to macro-intraday context: **where is price relative to the session's VWAP, the opening gap, the range already consumed, and the prior day's structure?**

---

## Table of Contents

1. [Data Pipeline (5-Minute Harvester)](#1-data-pipeline-5-minute-harvester)
2. [Feature Engineering (DayTradeFeatures)](#2-feature-engineering-daytradefeatures)
3. [Target Labeling (Angel/Devil Adaptation)](#3-target-labeling-angeldevil-adaptation)
4. [Feature Vector Summary](#4-feature-vector-summary)
5. [Integration Notes](#5-integration-notes)

---

## 1. Data Pipeline (5-Minute Harvester)

**File to create:** `src/data/harvester_5m.py`

This script fetches two datasets:
- **5-minute OHLCV bars** — the base timeframe for all feature computation and inference.
- **Daily OHLCV bars** — required to compute `daily_atr`, `gap_pct`, and `first_30m_vol_rel` against a 30-day rolling average.

Both are saved to `data/raw/` as Parquet files. The daily bars are fetched with extra history (`DAYS_BACK + 30`) to ensure the 14-day NATR indicator is fully warm when training begins.

```python
"""
src/data/harvester_5m.py
Day Trade Dataset Harvester — Universal Scalper V4.0

Fetches:
  1. 5-minute OHLCV bars for the day trading universe → data/raw/{SYMBOL}_5min.parquet
  2. Daily OHLCV bars (for Daily ATR and gap computation) → data/raw/{SYMBOL}_daily.parquet

Usage:
    python -m src.data.harvester_5m

Environment Variables:
    ALPACA_API_KEY: Alpaca API key
    ALPACA_SECRET_KEY: Alpaca API secret
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import polars as pl
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s"
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Day Trading universe: high-liquidity, high-beta instruments.
# SPY/QQQ provide macro regime context. TSLA/NVDA/AAPL/AMD/MSFT provide
# individual names with strong intraday trends and tight spreads.
DAY_TRADE_UNIVERSE: List[str] = [
    "SPY",   # S&P 500 ETF — macro regime reference
    "QQQ",   # Nasdaq-100 ETF — tech regime reference
    "TSLA",  # High-beta EV — strong intraday trend character
    "NVDA",  # AI proxy — volatile, momentum-driven
    "AAPL",  # Large-cap liquid — trend + mean-reversion
    "AMD",   # High-beta semiconductor
    "MSFT",  # Large-cap tech — trend anchor
]

DAYS_BACK: int = 252          # ~1 trading year of 5-minute data
DAILY_EXTRA_HISTORY: int = 30 # Extra days for ATR(14) warm-up on daily bars

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_RAW_DIR = _PROJECT_ROOT / "data" / "raw"

TIMEFRAME_5MIN = TimeFrame(5, TimeFrameUnit.Minute)
TIMEFRAME_DAILY = TimeFrame(1, TimeFrameUnit.Day)
DATA_FEED = DataFeed.IEX


# ═══════════════════════════════════════════════════════════════════════════════
# CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

def _get_client() -> StockHistoricalDataClient:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables must be set."
        )
    return StockHistoricalDataClient(api_key, secret_key)


# ═══════════════════════════════════════════════════════════════════════════════
# FETCH HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    timeframe: TimeFrame,
    start: datetime,
    end: datetime,
) -> pl.DataFrame:
    """
    Fetch OHLCV bars for a single symbol and return a clean Polars DataFrame.

    Strips Alpaca MultiIndex and metadata columns. Ensures lowercase column names.
    Returns an empty DataFrame (not an exception) on data gaps.
    """
    import pandas as pd

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=timeframe,
        start=start,
        end=end,
        feed=DATA_FEED,
        adjustment=Adjustment.SPLIT,  # split-adjusted for continuity
    )
    bars = client.get_stock_bars(req)

    if not bars.data or symbol not in bars.data:
        logger.warning("No data returned for %s", symbol)
        return pl.DataFrame()

    # Strip MultiIndex (symbol, timestamp) → flat DataFrame
    df_pandas = bars.df.loc[symbol].reset_index()
    df_pandas.columns = [col.lower() for col in df_pandas.columns]

    # Force numpy-backed copy to clear Alpaca metadata before Polars conversion
    df_clean = pd.DataFrame(
        {col: df_pandas[col].to_numpy(dtype=None, copy=True) for col in df_pandas.columns}
    )
    df = pl.from_pandas(df_clean)
    df = df.with_columns(pl.lit(symbol).alias("symbol"))
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN HARVEST
# ═══════════════════════════════════════════════════════════════════════════════

def harvest(symbols: List[str] = DAY_TRADE_UNIVERSE) -> None:
    """
    Fetch and persist 5-minute and daily bars for all symbols.

    Outputs:
        data/raw/{SYMBOL}_5min.parquet   — 5-minute OHLCV, ~1 year
        data/raw/{SYMBOL}_daily.parquet  — Daily OHLCV, ~1 year + 30 extra days

    The daily file intentionally starts 30 days earlier than the 5-minute file
    so that daily ATR(14) is fully initialised from the first 5-minute training bar.
    """
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = _get_client()

    end_date = datetime.utcnow()
    start_5min = end_date - timedelta(days=DAYS_BACK)
    start_daily = end_date - timedelta(days=DAYS_BACK + DAILY_EXTRA_HISTORY)

    logger.info("=" * 70)
    logger.info("DAY TRADE HARVESTER — Universal Scalper V4.0")
    logger.info("=" * 70)
    logger.info("Universe : %s", ", ".join(symbols))
    logger.info("5m window: %s → %s", start_5min.date(), end_date.date())
    logger.info("Daily window: %s → %s", start_daily.date(), end_date.date())

    for symbol in symbols:
        # ── 5-minute bars ──────────────────────────────────────────────────
        df_5min = _fetch_bars(client, symbol, TIMEFRAME_5MIN, start_5min, end_date)
        if len(df_5min) == 0:
            logger.error("Skipping %s — empty 5-minute response.", symbol)
            continue

        out_5min = _RAW_DIR / f"{symbol}_5min.parquet"
        df_5min.write_parquet(out_5min)
        logger.info(
            "%-6s  5min:  %7d bars → %s",
            symbol, len(df_5min), out_5min.name,
        )

        # ── Daily bars ─────────────────────────────────────────────────────
        df_daily = _fetch_bars(client, symbol, TIMEFRAME_DAILY, start_daily, end_date)
        if len(df_daily) == 0:
            logger.warning("%-6s  daily: empty response — daily ATR will be unavailable.", symbol)
            continue

        out_daily = _RAW_DIR / f"{symbol}_daily.parquet"
        df_daily.write_parquet(out_daily)
        logger.info(
            "%-6s  daily: %7d bars → %s",
            symbol, len(df_daily), out_daily.name,
        )

    logger.info("Harvest complete.")


if __name__ == "__main__":
    harvest()
```

---

## 2. Feature Engineering (DayTradeFeatures)

**File to create:** `src/ml/features/day_trade_features.py`

The `DayTradeFeatureEngineer` is composed of three `BaseFeatureGenerator` subclasses, following the same plugin pattern as `V3BaseFeatures` and `V3HTFFeatures`. They are intended to be chained inside a `FeaturePipeline` instance.

### 2.1 Mathematical Definitions

Before the code, the precise math for each intraday feature:

#### Feature 1: VWAP Distance (`vwap_dist`)

VWAP resets at the open of each RTH session (09:30 ET). It accumulates the ratio of (Typical Price × Volume) to total Volume from bar 1 of the day to the current bar.

```
typical_price[i] = (high[i] + low[i] + close[i]) / 3

VWAP[i] = Σ(typical_price[j] × volume[j], j = open_bar..i)
           ───────────────────────────────────────────────────
           Σ(volume[j], j = open_bar..i)

vwap_dist[i] = (close[i] - VWAP[i]) / close[i]
```

- Positive → price is trading above VWAP (bullish intraday bias).
- Negative → price is trading below VWAP (bearish intraday bias / mean-reversion zone).
- Range typically ±0.5% for large-caps; ±2%+ for TSLA/NVDA on trending days.

#### Feature 2: Opening Gap Percentage (`gap_pct`)

The gap captures overnight news and futures positioning. It is constant for all bars of a given session.

```
gap_pct[date] = (first_open[date] - last_close[date - 1]) / last_close[date - 1]
```

- Positive gap → bullish open, potential gap-and-go or gap-fill setup.
- Negative gap → bearish open.
- Applied identically to every 5-minute bar of that session.

#### Feature 3: First 30-Minute Volume Relative to 30-Day Average (`first_30m_vol_rel`)

The opening 30 minutes (first 6 bars of the RTH session) are the highest-conviction activity window. Comparing today's opening volume to the rolling 30-day average quantifies whether institutional flow is elevated.

```
first_30m_vol[date] = Σ(volume[j], j = bar_1..bar_6)

avg_first_30m_vol[date] = rolling_mean(first_30m_vol, window=30 trading days)

first_30m_vol_rel[date] = first_30m_vol[date] / avg_first_30m_vol[date]
```

- > 1.5 → significant elevated opening activity (potential trend day catalyst).
- < 0.7 → quiet open (range-bound / choppy session likely).
- Applied identically to every 5-minute bar of that session (it is a daily scalar).

#### Feature 4: Intraday Trend vs. Open (`trend_vs_open`)

Tracks the current directional drift relative to where the session opened. Unlike VWAP (which reflects average traded price), this is a pure price-level comparison.

```
day_open[date] = open price of the first 5-minute bar of the session

trend_vs_open[i] = (close[i] - day_open[date(i)]) / day_open[date(i)]
```

- Positive → price is above the session open (uptrend from open).
- Negative → price is below the session open (downtrend from open).
- Combines with `gap_pct`: a positive gap + positive `trend_vs_open` signals a confirmed gap-and-go.

#### Feature 5: Range Exhaustion (`range_exhaustion`)

Compares how much of the instrument's typical daily range has already been consumed by the current intraday range. This is critical for day trading — a bar attempting a new trend entry late in the day when the range is already 120% of ATR has an unfavourable risk profile.

```
intraday_high[i] = max(high[open_bar], ..., high[i])   # session-to-date high
intraday_low[i]  = min(low[open_bar],  ..., low[i])    # session-to-date low
intraday_range[i] = intraday_high[i] - intraday_low[i]

daily_atr_abs[i] = close[i] × daily_natr_14[date(i)] / 100
     where daily_natr_14 is NATR(14) computed on daily OHLCV bars

range_exhaustion[i] = intraday_range[i] / daily_atr_abs[i]
```

- 0.0–0.5 → early session; range still expanding, trend entries viable.
- 0.5–1.0 → range developing; standard day.
- > 1.0 → extended range; day has already moved more than its average range. High-risk for new long entries.

#### Feature 6: Session Progress (`session_progress`)

A simple normalised position within the RTH session. 9:30 = 0.0, 16:00 = 1.0. The model needs to learn that entries near the open (0.0–0.3) carry different risk from entries near the close (0.8–1.0).

```
RTH_OPEN = 09:30 ET  (570 minutes from midnight)
RTH_CLOSE = 16:00 ET (960 minutes from midnight)
SESSION_DURATION = 390 minutes

minute_of_day[i] = hour[i] × 60 + minute[i]  (in ET timezone)

session_progress[i] = clamp(
    (minute_of_day[i] - 570) / 390,
    0.0, 1.0
)
```

### 2.2 Implementation

```python
"""
src/ml/features/day_trade_features.py
Day Trade Feature Engineering — Universal Scalper V4.0

Three composable BaseFeatureGenerator subclasses:

  DayTradeBaseFeatures      — Adapted TA-Lib indicators for 5-minute bars
                               (RSI-14, PPO, NATR-14, BB, SMA-50, microstructure)
  DayTradeDailyJoin         — Joins per-day scalars (daily ATR, gap_pct,
                               first_30m_vol_rel) computed from daily bars
  DayTradeIntradayFeatures  — Intraday session features computed from the
                               5-minute bars themselves (VWAP, trend_vs_open,
                               range_exhaustion, session_progress)

Usage in a FeaturePipeline:
    pipeline = FeaturePipeline(
        feature_generators=[
            DayTradeBaseFeatures(),
            DayTradeDailyJoin(daily_df),
            DayTradeIntradayFeatures(),
        ],
        target_generator=DayTradeTargets(),
    )
    processed_df = pipeline.run(raw_5min_df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import talib

from ml.core.interfaces import BaseFeatureGenerator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TA-Lib periods — tuned for 5-minute bars
# At 5-minute resolution, these cover the following real-time durations:
#   RSI-14:  14 × 5 min = 70 min  (approximately first 2 hours of session)
#   SMA-50:  50 × 5 min = 250 min (approximately the full RTH session)
#   BB-20:   20 × 5 min = 100 min (~1.5 hour volatility context)
#   NATR-14: 14 × 5 min = 70 min  (current regime volatility)
# These are intentionally the same numeric periods as V3.4 because at 5-minute
# resolution they encode the correct *temporal* context for day trading.
# ─────────────────────────────────────────────────────────────────────────────
_RSI_PERIOD: int = 14
_PPO_FAST: int = 12
_PPO_SLOW: int = 26
_BB_PERIOD: int = 20
_BB_STD: int = 2
_SMA_PERIOD: int = 50
_NATR_PERIOD: int = 14
_RANGE_COIL_PERIOD: int = 10

# RTH session boundaries in minutes-from-midnight (Eastern Time)
_RTH_OPEN_MIN: int = 570    # 09:30
_RTH_CLOSE_MIN: int = 960   # 16:00
_RTH_DURATION_MIN: int = 390


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR 1: Base TA-Lib Features (adapted for 5-minute bars)
# ═══════════════════════════════════════════════════════════════════════════════

class DayTradeBaseFeatures(BaseFeatureGenerator):
    """
    Computes standard TA-Lib indicators and microstructure features on 5-minute bars.

    This is structurally equivalent to V3BaseFeatures but operates on the 5-minute
    base timeframe rather than 1-minute.

    Output columns (all floats unless noted):
        rsi_14, ppo, natr_14, bb_pct_b, bb_width_pct,
        price_sma50_ratio, log_return, hour_of_day (Int8),
        dist_sma50, vol_rel,
        range_coil_10, bar_body_pct, bar_upper_wick_pct, bar_lower_wick_pct
    """

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        close = df["close"].to_numpy()
        high  = df["high"].to_numpy()
        low   = df["low"].to_numpy()

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

        # ── Trend Anchor ──────────────────────────────────────────────────────
        sma_50 = talib.SMA(close, timeperiod=_SMA_PERIOD)

        # ── Volatility Regime ─────────────────────────────────────────────────
        # NATR on 5m bars measures intraday micro-volatility. For day trading,
        # the true risk sizing uses daily_atr (computed in DayTradeDailyJoin),
        # but natr_14 here provides a within-session volatility signal.
        natr = talib.NATR(high, low, close, timeperiod=_NATR_PERIOD)

        df = df.with_columns(
            pl.Series("rsi_14",    rsi),
            pl.Series("ppo",       ppo),
            pl.Series("bb_upper",  bb_upper),
            pl.Series("bb_middle", bb_middle),
            pl.Series("bb_lower",  bb_lower),
            pl.Series("sma_50",    sma_50),
            pl.Series("natr_14",   natr),
        )

        # ── Derived / Normalised Features ─────────────────────────────────────
        df = df.with_columns(
            (
                (pl.col("close") - pl.col("bb_lower"))
                / (pl.col("bb_upper") - pl.col("bb_lower"))
            ).alias("bb_pct_b"),
            (
                (pl.col("bb_upper") - pl.col("bb_lower")) / pl.col("bb_middle")
            ).alias("bb_width_pct"),
            (pl.col("close") / pl.col("sma_50")).alias("price_sma50_ratio"),
            (pl.col("close") / pl.col("close").shift(1)).log().alias("log_return"),
            pl.col("timestamp").dt.hour().cast(pl.Int8).alias("hour_of_day"),
            (
                (pl.col("close") - pl.col("sma_50")) / pl.col("sma_50")
            ).alias("dist_sma50"),
        )

        df = df.with_columns(
            (pl.col("volume") / pl.col("volume").rolling_mean(window_size=20))
            .fill_nan(1.0)
            .fill_null(1.0)
            .alias("vol_rel")
        )

        # ── Phase 5 Microstructure (carried forward from V3.4) ────────────────
        df = df.with_columns(
            (
                (pl.col("high") - pl.col("low"))
                / (
                    (pl.col("high") - pl.col("low"))
                    .rolling_mean(window_size=_RANGE_COIL_PERIOD)
                    .fill_null(1.0)
                    + 1e-6
                )
            ).alias("range_coil_10"),
            (
                (pl.col("close") - pl.col("open")).abs()
                / (pl.col("high") - pl.col("low") + 1e-6)
            ).alias("bar_body_pct"),
            (
                (pl.col("high") - pl.max_horizontal(pl.col("open"), pl.col("close")))
                / (pl.col("high") - pl.col("low") + 1e-6)
            ).alias("bar_upper_wick_pct"),
            (
                (pl.min_horizontal(pl.col("open"), pl.col("close")) - pl.col("low"))
                / (pl.col("high") - pl.col("low") + 1e-6)
            ).alias("bar_lower_wick_pct"),
        )

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR 2: Daily-Level Feature Join
# ═══════════════════════════════════════════════════════════════════════════════

class DayTradeDailyJoin(BaseFeatureGenerator):
    """
    Computes per-session scalar features from daily bars and joins them onto
    the 5-minute DataFrame by trade_date.

    Requires: daily_df with columns [symbol, timestamp, open, high, low, close, volume]
              where each row is a daily OHLCV bar.

    Output columns (joined onto 5m df, constant per session):
        daily_atr_abs      — Absolute daily ATR in $ (close × NATR_14_daily / 100)
        daily_natr_14      — Percentage NATR(14) on daily bars (regime width)
        gap_pct            — (today_open - prev_day_close) / prev_day_close
        first_30m_vol_rel  — today's first-30-min vol / 30-day rolling avg

    Note on daily_atr_abs:
        This is the risk unit for ALL day-trade targets and bracket sizing.
        It replaces the intraday NATR used in V3.4 to ensure SL/TP scales
        with the full-day expected range, not a 70-minute micro-volatility window.
    """

    def __init__(self, daily_df: pl.DataFrame):
        """
        Args:
            daily_df: Daily OHLCV bars for all symbols in the training universe.
                      Must span DAYS_BACK + DAILY_EXTRA_HISTORY days for ATR warm-up.
        """
        self._daily_features = self._compute_daily_features(daily_df)

    @staticmethod
    def _compute_daily_features(daily_df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute per-day scalars from the daily bar DataFrame.

        Steps:
        1. Compute NATR(14) on daily bars per symbol → daily_natr_14, daily_atr_abs
        2. Compute gap_pct: today's open vs. yesterday's close
        3. Compute first_30m_vol placeholder (daily volume used as proxy here;
           actual first_30m_vol is computed in DayTradeIntradayFeatures from 5m data)
        """
        has_symbol = "symbol" in daily_df.columns

        def _talib_daily(sym_df: pl.DataFrame) -> pl.DataFrame:
            close = sym_df["close"].to_numpy()
            high  = sym_df["high"].to_numpy()
            low   = sym_df["low"].to_numpy()

            daily_natr = talib.NATR(high, low, close, timeperiod=14)
            sym_df = sym_df.with_columns(
                pl.Series("daily_natr_14", daily_natr)
            )
            # Absolute ATR in $ = close × NATR% / 100
            sym_df = sym_df.with_columns(
                (pl.col("close") * pl.col("daily_natr_14") / 100.0).alias("daily_atr_abs")
            )
            return sym_df

        if has_symbol:
            daily_df = pl.concat(
                [
                    _talib_daily(daily_df.filter(pl.col("symbol") == sym))
                    for sym in daily_df["symbol"].unique().sort().to_list()
                ],
                how="vertical_relaxed",
            )
        else:
            daily_df = _talib_daily(daily_df)

        # ── Gap % ──────────────────────────────────────────────────────────────
        # Shift close by 1 day within each symbol to get "prev_day_close"
        group_keys = ["symbol"] if has_symbol else []
        if group_keys:
            daily_df = daily_df.sort(["symbol", "timestamp"])
            daily_df = daily_df.with_columns(
                pl.col("close").shift(1).over("symbol").alias("prev_day_close")
            )
        else:
            daily_df = daily_df.sort("timestamp")
            daily_df = daily_df.with_columns(
                pl.col("close").shift(1).alias("prev_day_close")
            )

        daily_df = daily_df.with_columns(
            (
                (pl.col("open") - pl.col("prev_day_close")) / pl.col("prev_day_close")
            ).alias("gap_pct")
        )

        # ── Add trade_date key for joining ─────────────────────────────────────
        daily_df = daily_df.with_columns(
            pl.col("timestamp").cast(pl.Date).alias("trade_date")
        )

        select_cols = ["trade_date", "daily_natr_14", "daily_atr_abs", "gap_pct"]
        if has_symbol:
            select_cols = ["symbol"] + select_cols

        return daily_df.select(select_cols)

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Join daily scalars onto the 5-minute DataFrame by (symbol, trade_date).

        Also computes first_30m_vol_rel from the 5-minute data (requires the
        full day's bars to be present in df, i.e. process one symbol at a time
        or the full multi-symbol batch).
        """
        has_symbol = "symbol" in df.columns

        # ── Add trade_date key ─────────────────────────────────────────────────
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Date).alias("trade_date")
        )

        # ── Join daily scalars ─────────────────────────────────────────────────
        join_keys = ["symbol", "trade_date"] if has_symbol else ["trade_date"]
        df = df.join(self._daily_features, on=join_keys, how="left")

        # ── First 30-minute Volume Relative (from 5m data) ────────────────────
        # Identifies the first 6 bars of each session (09:30–10:00 ET = 30 min).
        # Bar number within the day is assigned by ranking timestamps per group.
        # first_30m_vol_rel is then broadcast to all bars of that session.
        #
        # Note: Polars' rank("ordinal") assigns sequential integers starting at 1.
        group_keys_list = ["symbol", "trade_date"] if has_symbol else ["trade_date"]

        df = df.sort(group_keys_list + ["timestamp"])
        df = df.with_columns(
            pl.col("timestamp")
            .rank("ordinal")
            .over(group_keys_list)
            .alias("_bar_num_intraday")
        )

        # Sum of volume for bars 1–6 (the first 30 minutes)
        df = df.with_columns(
            pl.when(pl.col("_bar_num_intraday") <= 6)
            .then(pl.col("volume"))
            .otherwise(0.0)
            .alias("_first30m_vol_contrib")
        )

        # Aggregate first_30m_vol per session, then broadcast back
        first_30m_agg = (
            df.group_by(group_keys_list)
            .agg(pl.col("_first30m_vol_contrib").sum().alias("_first30m_vol_today"))
        )
        df = df.join(first_30m_agg, on=group_keys_list, how="left")

        # Rolling 30-day average of first_30m_vol per symbol
        # We compute this on the daily-aggregated series, then re-join.
        rolling_keys = ["symbol"] if has_symbol else []
        daily_vol = first_30m_agg.sort(
            (["symbol", "trade_date"] if has_symbol else ["trade_date"])
        )
        if rolling_keys:
            daily_vol = daily_vol.with_columns(
                pl.col("_first30m_vol_today")
                .rolling_mean(window_size=30)
                .over("symbol")
                .alias("_avg_first30m_vol_30d")
            )
        else:
            daily_vol = daily_vol.with_columns(
                pl.col("_first30m_vol_today")
                .rolling_mean(window_size=30)
                .alias("_avg_first30m_vol_30d")
            )

        df = df.join(
            daily_vol.select(group_keys_list + ["_avg_first30m_vol_30d"]),
            on=group_keys_list,
            how="left",
        )

        df = df.with_columns(
            (pl.col("_first30m_vol_today") / pl.col("_avg_first30m_vol_30d"))
            .fill_nan(1.0)
            .fill_null(1.0)
            .alias("first_30m_vol_rel")
        )

        # ── Drop internal staging columns ──────────────────────────────────────
        df = df.drop([
            "_bar_num_intraday",
            "_first30m_vol_contrib",
            "_first30m_vol_today",
            "_avg_first30m_vol_30d",
        ])

        return df


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR 3: Intraday Session Features
# ═══════════════════════════════════════════════════════════════════════════════

class DayTradeIntradayFeatures(BaseFeatureGenerator):
    """
    Computes features that require per-bar awareness of the intraday session state.

    Requires: DayTradeDailyJoin must have already run (needs daily_atr_abs, trade_date).

    Output columns:
        vwap               — Intraday VWAP (resets each session)
        vwap_dist          — (close - vwap) / close
        day_open           — First bar's open price of the session
        trend_vs_open      — (close - day_open) / day_open
        intraday_range     — max(session high) - min(session low) to current bar
        range_exhaustion   — intraday_range / daily_atr_abs
        session_progress   — Normalised position in RTH (0.0 = 09:30, 1.0 = 16:00)
    """

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        has_symbol = "symbol" in df.columns
        group_keys = ["symbol", "trade_date"] if has_symbol else ["trade_date"]

        # ── VWAP (intraday, resets each session) ──────────────────────────────
        df = df.with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3.0).alias("_tp")
        )
        df = df.with_columns(
            (pl.col("_tp") * pl.col("volume")).alias("_tp_vol")
        )
        df = df.with_columns(
            (
                pl.col("_tp_vol").cum_sum().over(group_keys)
                / pl.col("volume").cum_sum().over(group_keys)
            ).alias("vwap")
        )
        df = df.with_columns(
            (
                (pl.col("close") - pl.col("vwap")) / pl.col("close")
            ).alias("vwap_dist")
        )

        # ── Intraday Trend vs. Day Open ───────────────────────────────────────
        df = df.with_columns(
            pl.col("open").first().over(group_keys).alias("day_open")
        )
        df = df.with_columns(
            (
                (pl.col("close") - pl.col("day_open")) / pl.col("day_open")
            ).alias("trend_vs_open")
        )

        # ── Range Exhaustion ──────────────────────────────────────────────────
        # Cumulative intraday high and low from session open to current bar
        df = df.with_columns(
            pl.col("high").cum_max().over(group_keys).alias("_intraday_high"),
            pl.col("low").cum_min().over(group_keys).alias("_intraday_low"),
        )
        df = df.with_columns(
            (pl.col("_intraday_high") - pl.col("_intraday_low")).alias("intraday_range")
        )
        df = df.with_columns(
            (
                pl.col("intraday_range") / (pl.col("daily_atr_abs") + 1e-6)
            ).alias("range_exhaustion")
        )

        # ── Session Progress ──────────────────────────────────────────────────
        # Normalised position in RTH: 0.0 = 09:30 ET, 1.0 = 16:00 ET
        # Timestamps are UTC-aware; convert to ET minutes for this calculation.
        # ET = UTC - 4 hours (EDT) or - 5 hours (EST).
        # We use a pragmatic approach: extract hour/minute from the UTC timestamp
        # and convert. For production robustness, a zoneinfo conversion is used.
        df = df.with_columns(
            (
                (
                    pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.hour() * 60
                    + pl.col("timestamp").dt.convert_time_zone("America/New_York").dt.minute()
                    - _RTH_OPEN_MIN
                ).cast(pl.Float64)
                / _RTH_DURATION_MIN
            )
            .clip(0.0, 1.0)
            .alias("session_progress")
        )

        # ── Drop internal staging columns ──────────────────────────────────────
        df = df.drop(["_tp", "_tp_vol", "_intraday_high", "_intraday_low"])

        return df
```

---

## 3. Target Labeling (Angel/Devil Adaptation)

**File to create:** `src/ml/targets/day_trade_targets.py`

The V3.4 targets are designed for 25-minute scalping. The day trade targets re-anchor to the **end-of-session** as the terminal event. The math shifts from ATR multiples based on intraday volatility to multiples based on the **Daily ATR** — the true measure of a stock's expected full-day range.

### 3.1 Mathematical Definitions

#### Angel Target: Max Favorable Excursion (MFE) > 1.0 × Daily ATR

**Rationale:** A day trade is considered "Angel-approved" if the move is large enough to represent a genuine intraday trend leg, not just noise. The threshold of 1.0 × Daily ATR ensures the trade has potential to produce a meaningful return before EOD. Asking for exactly the average daily range is the correct day-trading bar — it filters out low-potential setups.

```
daily_atr_abs[i]  =  close[i]  ×  daily_natr_14[date(i)]  /  100

eod_bar(i)  =  last 5-minute bar of the RTH session on date(i)
               (the bar whose timestamp.minute == 55 at hour == 15, ET)

MFE[i]  =  max( high[i+1], high[i+2], ..., high[eod_bar(i)] )  -  close[i]

angel_target[i]  =  1  if  MFE[i]  >=  1.0 × daily_atr_abs[i]
                 =  0  otherwise
                 =  NaN  if  i  ==  eod_bar  (no lookahead window available)
```

**Key decision — SL-first check:** The Angel target does NOT check whether the SL was hit before the MFE. It is a purely directional question: "Did the move happen at all?" The Devil handles survivability. This separation matches the V3.4 meta-labeling philosophy exactly.

#### Devil Target: EOD Survival Without Hitting 1.5 × Daily ATR Stop-Loss

**Rationale:** The Devil's question is about position survivability across the full session. A 1.5× Daily ATR stop-loss is deliberately wide — it is not intended to be triggered on normal intraday noise. Only a genuine adverse move or trend reversal should stop the trade out. The Devil is trained to identify bars where the structure is too fragile to survive the full session.

```
sl_price[i]  =  close[i]  -  1.5  ×  daily_atr_abs[i]

survived[i]  =  True  if  min( low[i+1], low[i+2], ..., low[eod_bar(i)] )  >  sl_price[i]
             =  False  if  ANY  low[j]  <=  sl_price[i]  for  j  in  (i+1 .. eod_bar(i))

devil_target[i]  =  1  if  survived[i]  is  True
                 =  0  otherwise
                 =  NaN  if  i  ==  eod_bar
```

**The Macro Target (for EV calibration only):**

As in V3.4, the Devil is *trained* on the survival target, but the **validation gate and threshold sweep** use a macro target that measures the actual P&L outcome:

```
devil_target_macro[i]  =  1  if  MFE[i]  >=  1.0 × daily_atr_abs[i]   (Angel's goal achieved)
                                AND  survived[i]  is  True              (SL never hit)
                       =  0  otherwise
```

This is the "trade worked and survived" binary — the true P&L-relevant outcome. EV is then computed as:

```
EV  =  win_rate × (TP_MULT / SL_MULT)  -  (1 - win_rate)

where:
    TP_MULT  =  1.0  (Daily ATR target)
    SL_MULT  =  1.5  (Daily ATR stop)
    R:R Ratio  =  1.0 / 1.5  =  0.667

    This is a lower R:R than V3.4's 3.0 / 0.5 = 6.0 ratio.
    Day trading compensates with higher win rates (trend continuation has >50%
    base rate on trend days) rather than extreme asymmetric payouts.
```

### 3.2 Implementation

```python
"""
src/ml/targets/day_trade_targets.py
Day Trade Target Labeling — Universal Scalper V4.0

Implements the End-of-Day (EOD) Angel/Devil target architecture:

  Angel Target  — MFE > 1.0 × Daily ATR before EOD
                  (Was the move large enough to be a real trend day?)

  Devil Target  — EOD Survival: min(session_low from entry to EOD) > SL = 1.5 × Daily ATR
                  (Does the trade survive the full session?)

  Macro Target  — Angel AND Devil both satisfied
                  (Used for EV calibration and threshold sweep only, NOT for training.)

Requires: daily_atr_abs column (populated by DayTradeDailyJoin).
"""

from __future__ import annotations

import logging

import numpy as np
import polars as pl

from ml.core.interfaces import BaseTargetGenerator

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# BRACKET PARAMETERS (Day Trade)
# ─────────────────────────────────────────────────────────────────────────────
ANGEL_MFE_MULTIPLIER: float = 1.0   # MFE must exceed 1.0 × Daily ATR
DEVIL_SL_MULTIPLIER:  float = 1.5   # SL = 1.5 × Daily ATR below entry
TP_MULT:              float = 1.0   # For EV: TP pays 1.0 R
SL_MULT:              float = 1.5   # For EV: SL costs 1.5 R (R:R = 0.667)


class DayTradeTargets(BaseTargetGenerator):
    """
    Generates three target columns for day trade model training:
        angel_target        — 1 if MFE >= ANGEL_MFE_MULTIPLIER × daily_atr_abs
        devil_target        — 1 if trade survives to EOD without hitting SL
        devil_target_macro  — 1 if both angel and devil targets are satisfied
                              (used only for EV evaluation; NOT for Devil training)

    Requires upstream columns:
        high, low, close    — from 5-minute OHLCV
        daily_atr_abs       — from DayTradeDailyJoin
        trade_date          — from DayTradeDailyJoin
        symbol              — optional; enables per-symbol processing
    """

    def __init__(
        self,
        angel_mfe_mult: float = ANGEL_MFE_MULTIPLIER,
        devil_sl_mult:  float = DEVIL_SL_MULTIPLIER,
    ):
        self.angel_mfe_mult = angel_mfe_mult
        self.devil_sl_mult  = devil_sl_mult

    def generate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute Angel, Devil, and Macro targets via bar-by-bar simulation.

        The EOD horizon varies per bar: a bar early in the session has ~77 bars
        of lookahead; a bar at 15:50 has only 1. Both are valid training samples
        as long as the target is computed correctly.
        """
        has_symbol = "symbol" in df.columns

        # Process per symbol to prevent cross-symbol lookahead
        if has_symbol:
            result_frames = []
            for sym in df["symbol"].unique().sort().to_list():
                sym_df = df.filter(pl.col("symbol") == sym)
                sym_df = self._compute_targets_for_symbol(sym_df)
                result_frames.append(sym_df)
            return pl.concat(result_frames, how="vertical_relaxed")
        else:
            return self._compute_targets_for_symbol(df)

    def _compute_targets_for_symbol(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute targets for a single symbol's 5-minute DataFrame.

        Iterates over each unique trade_date and applies the bar-by-bar
        EOD simulator to that session's bars.
        """
        df = df.sort("timestamp")
        result_frames = []

        for date in df["trade_date"].unique().sort().to_list():
            session_df = df.filter(pl.col("trade_date") == date)
            session_df = self._simulate_session(session_df)
            result_frames.append(session_df)

        return pl.concat(result_frames, how="vertical_relaxed")

    def _simulate_session(self, session_df: pl.DataFrame) -> pl.DataFrame:
        """
        Bar-by-bar EOD target simulation for a single RTH session.

        For each bar i:
          - Lookahead window = bars i+1 through the final bar of the session.
          - Angel: max(high[i+1:]) - close[i] >= angel_mfe_mult × daily_atr_abs[i]
          - Devil: min(low[i+1:]) > close[i] - devil_sl_mult × daily_atr_abs[i]
          - Macro: Angel AND Devil

        The final bar (eod_bar) receives target = NaN in all three columns —
        there is no lookahead window. These rows are dropped by FeaturePipeline.clean_data().

        Complexity: O(n²) per session. With n ≤ 78 bars per RTH session, this is
        at most 78 × 78 = 6,084 iterations per session — negligible.
        """
        close        = session_df["close"].to_numpy()
        high         = session_df["high"].to_numpy()
        low          = session_df["low"].to_numpy()
        daily_atr    = session_df["daily_atr_abs"].to_numpy()
        n            = len(close)

        angel_targets = np.full(n, np.nan, dtype=np.float32)
        devil_targets = np.full(n, np.nan, dtype=np.float32)
        macro_targets = np.full(n, np.nan, dtype=np.float32)

        for i in range(n - 1):  # last bar has no lookahead → stays NaN
            atr = daily_atr[i]
            if np.isnan(atr) or atr <= 0:
                continue  # leave as NaN; clean_data() will drop it

            entry_close = close[i]
            sl_price    = entry_close - self.devil_sl_mult * atr
            mfe_target  = entry_close + self.angel_mfe_mult * atr

            # Lookahead: bars i+1 through end of session
            future_highs = high[i + 1:]
            future_lows  = low[i + 1:]

            # Angel: did MFE exceed 1.0 × daily ATR before EOD?
            mfe            = float(np.max(future_highs)) - entry_close
            angel_hit      = mfe >= (self.angel_mfe_mult * atr)
            angel_targets[i] = np.float32(1) if angel_hit else np.float32(0)

            # Devil: did price stay above SL for the entire lookahead window?
            sl_breached    = bool(np.any(future_lows <= sl_price))
            devil_survived = not sl_breached
            devil_targets[i] = np.float32(1) if devil_survived else np.float32(0)

            # Macro: Angel AND Devil (used for EV calibration only)
            macro_targets[i] = np.float32(
                1 if (angel_hit and devil_survived) else 0
            )

        session_df = session_df.with_columns(
            pl.Series("angel_target",       angel_targets.astype("float32")),
            pl.Series("devil_target",        devil_targets.astype("float32")),
            pl.Series("devil_target_macro",  macro_targets.astype("float32")),
        )

        # Cast Int8 after NaN→null replacement (done by FeaturePipeline.clean_data)
        return session_df
```

---

## 4. Feature Vector Summary

The complete V4.0 day trading feature vector has **22 features** grouped into four families:

### Group A: TA-Lib Base (adapted for 5m)
| Feature | Description |
|---|---|
| `rsi_14` | RSI(14) on 5m bars = ~70 min momentum window |
| `ppo` | PPO(12,26) on 5m bars = 60min/130min oscillator |
| `natr_14` | NATR(14) on 5m = intraday micro-volatility regime |
| `bb_pct_b` | Bollinger %B — position within 5m bands |
| `bb_width_pct` | Bollinger width / middle band — volatility expansion |
| `price_sma50_ratio` | Close / SMA(50) on 5m = close vs 4-hour trend |
| `log_return` | Bar-over-bar log return |
| `dist_sma50` | (Close - SMA50) / SMA50 — fractional trend deviation |
| `vol_rel` | Volume / rolling_mean(20) — relative bar volume |
| `hour_of_day` | Integer hour (9–15) — intraday time effect |

### Group B: Macro-Intraday Context (new in V4.0)
| Feature | Description |
|---|---|
| `vwap_dist` | (Close - VWAP) / Close — distance from intraday fair value |
| `gap_pct` | (Today open - prev close) / prev close — overnight positioning |
| `first_30m_vol_rel` | Today's opening 30-min vol / 30-day avg — session conviction |
| `trend_vs_open` | (Close - Day Open) / Day Open — intraday directional drift |
| `range_exhaustion` | Intraday range / Daily ATR — how much of the day has moved |
| `session_progress` | Normalised RTH position [0.0, 1.0] — time-of-day risk |

### Group C: Microstructure (carried from V3.4)
| Feature | Description |
|---|---|
| `range_coil_10` | Range / rolling_mean(range, 10) — pre-breakout compression |
| `bar_body_pct` | |Close - Open| / (High - Low) — body conviction |
| `bar_upper_wick_pct` | Upper wick fraction — rejection signal |
| `bar_lower_wick_pct` | Lower wick fraction — stop-hunt defense |

### Group D: Daily ATR Context (joined from daily bars)
| Feature | Description |
|---|---|
| `daily_natr_14` | NATR(14) on daily bars — full-day volatility regime |
| `daily_atr_abs` | Close × daily_natr_14 / 100 — absolute $ risk unit |

**Total: 22 features**

### Target Summary

| Target Column | Used For | Definition |
|---|---|---|
| `angel_target` | **Angel training** | MFE ≥ 1.0 × daily_atr_abs before EOD |
| `devil_target` | **Devil training** | min(session lows) > close − 1.5 × daily_atr_abs |
| `devil_target_macro` | **EV calibration / threshold sweep only** | angel_target = 1 AND devil_target = 1 |

---

## 5. Integration Notes

### Pipeline Assembly

The three generators and target class are assembled into a `FeaturePipeline` exactly as in V3.4:

```python
from src.ml.feature_pipeline import FeaturePipeline
from src.ml.features.day_trade_features import (
    DayTradeBaseFeatures,
    DayTradeDailyJoin,
    DayTradeIntradayFeatures,
)
from src.ml.targets.day_trade_targets import DayTradeTargets
import polars as pl

daily_df = pl.concat(
    [pl.read_parquet(f"data/raw/{sym}_daily.parquet") for sym in UNIVERSE]
)

pipeline = FeaturePipeline(
    feature_generators=[
        DayTradeBaseFeatures(),
        DayTradeDailyJoin(daily_df),
        DayTradeIntradayFeatures(),
    ],
    target_generator=DayTradeTargets(),
)

raw_5min_df = pl.concat(
    [pl.read_parquet(f"data/raw/{sym}_5min.parquet") for sym in UNIVERSE]
)
processed_df = pipeline.run(raw_5min_df)
processed_df.write_parquet("data/processed/training_data_5m.parquet")
```

### Retrainer Changes (V4.0)

When adapting `retrainer.py` for the day trade model, update the following constants:

```python
# Day Trade retrainer constants (replaces V3.4 values)
TIMEFRAME         = TimeFrame(5, TimeFrameUnit.Minute)   # was: 1-minute
SL_ATR_MULTIPLIER = 1.5                                   # was: 0.5
TP_ATR_MULTIPLIER = 1.0                                   # was: 3.0
MAX_HOLD_BARS     = 78                                    # was: 45 (78 × 5m = ~6.5 hours, full RTH)
SURVIVAL_BARS     = 78                                    # was: 5 (day trade = survive to EOD)
BRIER_THRESHOLD   = 0.28                                  # tighter (higher-base-rate target)
PROFIT_FACTOR_THRESHOLD = 1.3                             # slightly higher (lower R:R requires better WR)
```

### Model Hot-Reload Compatibility

The V4.0 models use a 22-feature input vector vs. V3.4's 18-feature vector. The `MLStrategy._check_model_updates()` hot-reloader is feature-name-agnostic — it does not validate the feature count, only the file mtime. **Do not mix V3.4 and V4.0 model files in `models/`.** Store day-trade models at separate paths:

```
models/dt_angel_latest.pkl
models/dt_devil_latest.pkl
models/dt_threshold.json
```

A separate `DayTradeOrchestrator` (or a `mode` parameter on `LiveOrchestrator`) should load these paths. This keeps the two regimes fully isolated.

---

*Day Trade Model Specification — Universal Scalper V4.0 — Generated 2026-04-19*
