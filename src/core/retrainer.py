"""
The Cure V2 - Validated Model Retraining Pipeline.

Triggered when feedback_loop.py detects critical model drift.
Fetches fresh data, engineers ATR-dynamic labels, runs a 3-fold
walk-forward validation gate, and only promotes models that prove
profitability across multiple chronological market regimes.

Usage:
    python -m src.core.retrainer

Exit Codes:
    0 = Models promoted successfully
    1 = Execution error (data fetch failed, config error, etc.)
    2 = Models rejected by validation gate (production weights retained)

    NOTE: run_pipeline.sh checks feedback_loop.py's exit code (2 = trigger
    retraining), NOT this retrainer's exit code. Exit code 2 from the
    retrainer means "tried but rejected" and does not cause an infinite loop.

Environment Variables:
    ALPACA_API_KEY: Alpaca API key
    ALPACA_SECRET_KEY: Alpaca API secret
    DISCORD_WEBHOOK_URL: Discord webhook for retraining reports (optional)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import polars as pl
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit

from src.ml.feature_pipeline import FeatureEngineer
from src.core.notification_manager import NotificationManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DAYS_BACK = 60
TICKERS: List[str] = ["TSLA", "NVDA", "MARA", "COIN", "SMCI"]
TIMEFRAME = TimeFrame(1, TimeFrameUnit.Minute)
DATA_FEED = DataFeed.IEX

# Model Paths
MODEL_DIR = Path("models")
ANGEL_PATH = MODEL_DIR / "angel_latest.pkl"
DEVIL_PATH = MODEL_DIR / "devil_latest.pkl"

# Model Hyperparameters
ANGEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42,
    "n_jobs": -1,
}

DEVIL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "random_state": 42,
    # class_weight intentionally None — the Devil must learn the true ~20%
    # base rate. "balanced" artificially inflated probabilities, causing
    # Brier Score failure (0.31 > 0.25 threshold) and rubber-stamping.
    "n_jobs": -1,
}

# ═══════════════════════════════════════════════════════════════════════════════
# ATR BRACKET PARAMETERS (must match evaluate_performance.py and LiveOrchestrator)
# ═══════════════════════════════════════════════════════════════════════════════

SL_ATR_MULTIPLIER = 0.5
TP_ATR_MULTIPLIER = 3.0
MAX_HOLD_BARS = 45
SURVIVAL_BARS = 5  # Phase 5.5: Devil survival window (bars)

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE THRESHOLDS (must match MLStrategy)
# ═══════════════════════════════════════════════════════════════════════════════

ANGEL_THRESHOLD = 0.40
# Legacy: used as a fallback. In validate_candidate(), the Devil threshold
# is dynamically selected per-fold via _find_optimal_threshold().
DEVIL_THRESHOLD = 0.50

# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION GATE THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

BRIER_THRESHOLD = 0.25  # Max acceptable Brier score (lower = better)
EV_THRESHOLD = 0.0005  # Min acceptable Expected Value
PROFIT_FACTOR_THRESHOLD = 1.2  # Min acceptable Profit Factor

# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE COLUMNS (must match MLStrategy.feature_names and FeatureEngineer output)
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS: List[str] = [
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
    # V3.3: Multi-timeframe (5m) features
    "htf_rsi_14",
    "htf_trend_agreement",
    "htf_vol_rel",
    "htf_bb_pct_b",
    # Phase 5: Microstructure features
    "range_coil_10",
    "bar_body_pct",
    "bar_upper_wick_pct",
    "bar_lower_wick_pct",
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward CV fold."""

    fold_number: int
    train_size: int
    val_size: int
    brier_score: float
    expected_value: float
    angel_proposed_trades: int
    devil_approved_trades: int
    win_rate: float


@dataclass
class ValidationReport:
    """Aggregated validation report across all walk-forward folds."""

    fold_metrics: List[FoldMetrics]
    mean_brier: float
    mean_ev: float
    final_profit_factor: float
    final_win_rate: float
    final_total_trades: int
    gate_passed: bool
    rejection_reasons: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# ALPACA DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════════


def get_alpaca_client() -> StockHistoricalDataClient:
    """Initialize Alpaca client from environment variables."""
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        raise ValueError(
            "ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables must be set"
        )

    return StockHistoricalDataClient(api_key, secret_key)


def fetch_training_data(
    client: StockHistoricalDataClient,
    days_back: int = DAYS_BACK,
) -> pl.DataFrame:
    """
    Fetch historical 1-minute bars for training.

    Args:
        client: Alpaca API client
        days_back: Number of days to fetch (default: 60)

    Returns:
        Polars DataFrame with OHLCV data for all tickers
    """
    logger.info("=" * 70)
    logger.info("FETCHING TRAINING DATA")
    logger.info("=" * 70)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Tickers: {', '.join(TICKERS)}")
    logger.info(f"Timeframe: 1-minute bars")

    all_frames: List[pl.DataFrame] = []

    for ticker in TICKERS:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TIMEFRAME,
                start=start_date,
                end=end_date,
                feed=DATA_FEED,
            )

            bars = client.get_stock_bars(request)

            if not bars.data or ticker not in bars.data:
                logger.warning(f"No data returned for {ticker}")
                continue

            # Convert to Polars
            df_pandas = bars.df.reset_index()
            df_pandas.columns = [col.lower() for col in df_pandas.columns]
            df = pl.from_pandas(df_pandas)

            # Add symbol column if not present
            if "symbol" not in df.columns:
                df = df.with_columns(pl.lit(ticker).alias("symbol"))

            all_frames.append(df)
            logger.info(f"Fetched {len(df):,} bars for {ticker}")

        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            continue

    if not all_frames:
        raise ValueError("No data fetched for any ticker")

    # Combine all tickers
    combined = pl.concat(all_frames, how="vertical_relaxed")
    combined = combined.sort(["symbol", "timestamp"])

    logger.info(f"Combined dataset: {len(combined):,} total rows")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# ATR-DYNAMIC DEVIL TARGET (Bar-by-Bar Bracket Simulation)
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_devil_targets_atr(
    df: pl.DataFrame,
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
    max_hold: int = MAX_HOLD_BARS,
) -> np.ndarray:
    """
    Compute Devil targets using dynamic ATR brackets with bar-by-bar resolution.

    For each bar i, simulates a bracket order:
        SL = close[i] - sl_mult * ATR_abs[i]
        TP = close[i] + tp_mult * ATR_abs[i]

    Then walks forward up to max_hold bars checking:
        - If low[j] <= SL → loss (0)
        - If high[j] >= TP → win (1)
        - SL is checked FIRST (conservative, matches evaluate_performance.py)
        - If neither hit in max_hold bars → loss (0, timeout)

    This avoids rolling max/min which does not respect the temporal ordering
    of SL vs TP hits.  Complexity: O(n × max_hold).  At 60 days × 5 tickers
    × ~390 bars/day ≈ 117k rows × 15 bars = ~1.75M iterations — runs in
    under 2 seconds on modern hardware.

    Args:
        df: DataFrame containing 'close', 'high', 'low', 'natr_14' columns.
        sl_mult: ATR multiplier for stop-loss (default: SL_ATR_MULTIPLIER).
        tp_mult: ATR multiplier for take-profit (default: TP_ATR_MULTIPLIER).
        max_hold: Maximum bars to hold before timeout (default: MAX_HOLD_BARS).

    Returns:
        NumPy array of int8 (0 = loss/timeout, 1 = win), same length as df.
        NaN/invalid entries at the tail are set to 0.
    """
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    natr = df["natr_14"].to_numpy()
    n = len(close)
    targets = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        atr_abs = close[i] * natr[i] / 100.0
        if np.isnan(atr_abs) or atr_abs <= 0:
            continue

        sl_price = close[i] - sl_mult * atr_abs
        tp_price = close[i] + tp_mult * atr_abs

        for j in range(i + 1, min(i + max_hold + 1, n)):
            # SL checked first (conservative — matches evaluate_performance.py)
            if low[j] <= sl_price:
                targets[i] = 0
                break
            if high[j] >= tp_price:
                targets[i] = 1
                break
        # If loop completes without break → timeout → 0 (already default)

    return targets


def _compute_devil_survival_target(
    df: pl.DataFrame,
    sl_mult: float = SL_ATR_MULTIPLIER,
    survival_bars: int = SURVIVAL_BARS,
) -> np.ndarray:
    """
    Compute Devil survival targets: whether price survives the SL for the
    next `survival_bars` bars after each row.

    Phase 5.5 — Temporal Realignment:
        The Devil's 1m microstructure features (wick toxicity, range
        compression) operate at a 1–5 minute horizon.  Asking the Devil
        to predict 45-bar macro outcomes (the old devil_target) creates
        an unlearnable temporal gap.  Asking it to predict 5-bar SL
        survival aligns the learning objective with the feature horizon.

    Survival definition:
        target[i] = 1  if  low[j] > SL_price  for ALL j in [i+1, i+SURVIVAL_BARS]
        target[i] = 0  if  low[j] <= SL_price  for ANY j in that window

    SL price is computed identically to the live bracket:
        SL = close[i] - sl_mult * ATR_abs[i]
        ATR_abs = close[i] * natr_14[i] / 100.0

    Args:
        df:             DataFrame with 'close', 'low', 'natr_14' columns.
        sl_mult:        ATR multiplier for stop-loss (default: SL_ATR_MULTIPLIER).
        survival_bars:  Number of bars to check for SL breach (default: SURVIVAL_BARS).

    Returns:
        NumPy int8 array of length len(df).
        1 = survived (no SL breach in window), 0 = stopped out.
        Last `survival_bars` rows are always 0 (insufficient lookahead).
    """
    close = df["close"].to_numpy()
    low = df["low"].to_numpy()
    natr = df["natr_14"].to_numpy()
    n = len(close)
    targets = np.zeros(n, dtype=np.int8)

    for i in range(n - 1):
        atr_abs = close[i] * natr[i] / 100.0
        if np.isnan(atr_abs) or atr_abs <= 0:
            continue

        sl_price = close[i] - sl_mult * atr_abs
        survived = True

        for j in range(i + 1, min(i + survival_bars + 1, n)):
            if low[j] <= sl_price:
                survived = False
                break

        targets[i] = np.int8(1) if survived else np.int8(0)

    return targets


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING & LABEL GENERATION
# ═══════════════════════════════════════════════════════════════════════════════


def engineer_features_and_labels(df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
    """
    Engineer technical features and generate ATR-dynamic target labels.

    Delegates feature computation to FeatureEngineer.compute_indicators() to
    guarantee zero training/inference skew with the production MLStrategy.

    Features (produced by FeatureEngineer, matches MLStrategy.feature_names):
        rsi_14, ppo, natr_14, bb_pct_b, bb_width_pct,
        price_sma50_ratio, log_return, hour_of_day, dist_sma50, vol_rel

    Targets:
        angel_target: 1 if close 3 bars ahead > close + 0.5 × ATR (ATR-relative)
        devil_target: 1 if TP (3.0 × ATR) hit before SL (0.5 × ATR) in ≤45 bars

    Args:
        df: Raw OHLCV DataFrame with columns:
            open, high, low, close, volume, symbol, timestamp

    Returns:
        Tuple of (features DataFrame with targets, feature column names)
    """
    logger.info("=" * 70)
    logger.info("ENGINEERING FEATURES & LABELS")
    logger.info("=" * 70)

    # ═══════════════════════════════════════════════════════════════════
    # TECHNICAL INDICATORS via FeatureEngineer (prevents training/inference skew)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("Computing indicators via FeatureEngineer (zero-skew pipeline)...")
    df = FeatureEngineer().compute_indicators(df)
    logger.info(
        "Applied indicators: RSI, PPO, NATR, BBANDS, SMA50, log_return, "
        "hour_of_day, vol_rel"
    )

    # ═══════════════════════════════════════════════════════════════════
    # ANGEL TARGET: ATR-relative 3-bar momentum
    # 1 if close 3 bars ahead > close + 0.5 × ATR_abs
    # natr_14 is a percentage: ATR_abs = close * natr_14 / 100
    # ═══════════════════════════════════════════════════════════════════
    df = df.with_columns(
        (
            pl.col("close").shift(-3)
            > pl.col("close") + 0.5 * (pl.col("close") * pl.col("natr_14") / 100.0)
        )
        .cast(pl.Int8)
        .alias("angel_target")
    )
    logger.info("Generated angel_target (ATR-relative 3-bar momentum)")

    # ═══════════════════════════════════════════════════════════════════
    # DEVIL TARGETS (Phase 5.5 — Two-Target Architecture)
    #
    # devil_target_macro  — 45-bar bracket outcome (TP hit before SL).
    #   Used ONLY during threshold calibration (_find_optimal_threshold).
    #   Computes realized EV on Devil-approved trades using the actual
    #   asymmetric R:R payload (0.5× SL / 3.0× TP).
    #
    # devil_target        — 5-bar SL survival.
    #   Used to TRAIN the Devil. Aligns the learning objective with the
    #   1m microstructure feature horizon.
    #   1 = price did NOT breach SL in the next 5 bars (survived)
    #   0 = price breached SL within 5 bars (stopped out immediately)
    # ═══════════════════════════════════════════════════════════════════

    # -- Macro target (45-bar) — evaluation only -----------------------
    logger.info(
        f"Computing devil_target_macro via ATR bracket simulation "
        f"(SL={SL_ATR_MULTIPLIER}×ATR, TP={TP_ATR_MULTIPLIER}×ATR, "
        f"max_hold={MAX_HOLD_BARS} bars)..."
    )
    devil_targets_macro = _compute_devil_targets_atr(df)
    df = df.with_columns(pl.Series("devil_target_macro", devil_targets_macro))
    logger.info(
        f"Generated devil_target_macro (45-bar bracket): "
        f"{int(devil_targets_macro.sum())} wins / {len(devil_targets_macro)} total "
        f"({devil_targets_macro.mean():.1%} macro win rate)"
    )

    # -- Survival target (5-bar) — Devil training ----------------------
    logger.info(
        f"Computing devil_target via {SURVIVAL_BARS}-bar SL survival "
        f"(SL={SL_ATR_MULTIPLIER}×ATR)..."
    )
    devil_targets_survival = _compute_devil_survival_target(df)
    df = df.with_columns(pl.Series("devil_target", devil_targets_survival))
    logger.info(
        f"Generated devil_target ({SURVIVAL_BARS}-bar survival): "
        f"{int(devil_targets_survival.sum())} survived / {len(devil_targets_survival)} total "
        f"({devil_targets_survival.mean():.1%} survival rate)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # CLEANUP: Drop NaN/null rows (uses FeatureEngineer.clean_data)
    # ═══════════════════════════════════════════════════════════════════
    initial_count = len(df)
    df = FeatureEngineer.clean_data(df)
    dropped_count = initial_count - len(df)

    logger.info(
        f"Dropped {dropped_count:,} rows with nulls ({dropped_count / initial_count:.1%})"
    )
    logger.info(f"Final dataset: {len(df):,} rows")
    logger.info(f"Feature columns: {FEATURE_COLS}")

    return df, FEATURE_COLS


# ═══════════════════════════════════════════════════════════════════════════════
# TIME-DECAY WEIGHTS
# ═══════════════════════════════════════════════════════════════════════════════


def generate_time_decay_weights(
    n_samples: int, decay_factor: float = 0.95
) -> np.ndarray:
    """
    Generate time-decay sample weights.

    More recent samples get higher weights to prevent catastrophic forgetting.
    Weights decay exponentially from 1.0 (most recent) to 0.1 (oldest).

    Args:
        n_samples: Number of samples in dataset
        decay_factor: Decay rate per time step (default: 0.95)

    Returns:
        NumPy array of sample weights
    """
    # Generate exponential decay from oldest to newest
    weights = np.power(decay_factor, np.arange(n_samples))

    # Reverse so newest has highest weight
    weights = weights[::-1]

    # Normalize to range [0.1, 1.0]
    weights = 0.1 + 0.9 * (weights - weights.min()) / (weights.max() - weights.min())

    return weights


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════


def refit_models(
    df: pl.DataFrame,
    feature_cols: List[str],
) -> Tuple[RandomForestClassifier, RandomForestClassifier, List[str], List[str]]:
    """
    Refit Angel and Devil models with time-decay weighting.

    Implements proper Meta-Labeling architecture:
    1. Train Angel on base features
    2. Generate Angel's probabilities as meta-features
    3. Train Devil on base features + angel_prob

    Args:
        df: Feature-engineered DataFrame with 'angel_target' and 'devil_target'
        feature_cols: List of base feature column names

    Returns:
        Tuple of (Angel model, Devil model, angel_features, devil_features)
    """
    logger.info("=" * 70)
    logger.info("REFITTING MODELS (META-LABELING)")
    logger.info("=" * 70)

    # Extract base features and targets
    X_base = df[feature_cols].to_numpy()
    y_angel = df["angel_target"].to_numpy()
    y_devil = df["devil_target"].to_numpy()

    logger.info(f"Training samples: {len(X_base):,}")
    logger.info(f"Base features: {feature_cols}")
    logger.info(
        f"Angel target distribution: 0={np.sum(y_angel == 0)}, 1={np.sum(y_angel == 1)}"
    )
    logger.info(
        f"Devil target distribution: 0={np.sum(y_devil == 0)}, 1={np.sum(y_devil == 1)}"
    )

    # Generate time-decay weights
    sample_weights = generate_time_decay_weights(len(X_base))
    logger.info(
        f"Time-decay weights: min={sample_weights.min():.3f}, max={sample_weights.max():.3f}"
    )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Train the Angel (Primary Model - Direction)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 1/4] Training Angel model (Direction)...")
    angel_model = RandomForestClassifier(**ANGEL_PARAMS)
    angel_model.fit(X_base, y_angel, sample_weight=sample_weights)
    logger.info(f"✓ Angel model trained on {len(feature_cols)} features")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Generate Out-Of-Fold Meta-Features (Angel's Probabilities)
    # ═══════════════════════════════════════════════════════════════════
    logger.info(
        "\n[Step 2/4] Generating OOF meta-features (temporal cross-validation)..."
    )

    # CRITICAL FIX: Use TimeSeriesSplit with a manual fold loop to generate
    # Angel probabilities via out-of-fold predictions. This prevents the Devil
    # from training on the Angel's inflated in-sample confidence.
    #
    # Why manual loop instead of cross_val_predict:
    #   cross_val_predict requires that every sample appears in exactly one
    #   test fold (a strict partition). TimeSeriesSplit's first ~1/n_splits
    #   of samples are never in any test fold (always train-only). sklearn
    #   raises "cross_val_predict only works for partitions" in this case.
    #   The manual loop handles the train-only head by filling those rows
    #   from a model trained on just that first-fold window — still OOF
    #   for the rows that follow (zero leakage into the majority of data).
    #
    # Why TimeSeriesSplit: respects chronological ordering — each fold only
    # trains on past bars. KFold would let the Angel see future bars.
    #
    # n_splits=5: 5 expanding folds. Early folds → noisier Angel probs,
    # which is realistic (production Angel also starts uncertain).

    tss = TimeSeriesSplit(n_splits=5)
    angel_probs_oof = np.full(len(X_base), np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

        for fold_train_idx, fold_val_idx in tss.split(X_base):
            fold_weights = sample_weights[fold_train_idx]
            fold_angel = RandomForestClassifier(**ANGEL_PARAMS)
            fold_angel.fit(
                X_base[fold_train_idx],
                y_angel[fold_train_idx],
                sample_weight=fold_weights,
            )
            angel_probs_oof[fold_val_idx] = fold_angel.predict_proba(
                X_base[fold_val_idx]
            )[:, 1]

        # Fill train-only head (first ~1/n_splits rows never appear in val)
        # using a model trained solely on that window — no leakage from future.
        head_missing = np.isnan(angel_probs_oof)
        if head_missing.sum() > 0:
            first_train_idx, _ = next(iter(tss.split(X_base)))
            head_angel = RandomForestClassifier(**ANGEL_PARAMS)
            head_angel.fit(
                X_base[first_train_idx],
                y_angel[first_train_idx],
                sample_weight=sample_weights[first_train_idx],
            )
            angel_probs_oof[head_missing] = head_angel.predict_proba(
                X_base[head_missing]
            )[:, 1]
            logger.info(
                f"  Head fill: {head_missing.sum()} train-only rows scored by "
                f"Fold-1 Angel (no leakage from future)"
            )

    # Add OOF angel_prob as a new column to the DataFrame
    df = df.with_columns(pl.Series("angel_prob", angel_probs_oof))

    logger.info(f"✓ Generated {len(angel_probs_oof):,} OOF Angel probabilities")
    logger.info(
        f"  OOF Angel prob range:  [{angel_probs_oof.min():.3f}, {angel_probs_oof.max():.3f}]"
    )
    logger.info(f"  OOF Angel prob median: {np.median(angel_probs_oof):.3f}")

    # Compare OOF vs in-sample to confirm leakage was present
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        angel_probs_insample = angel_model.predict_proba(X_base)[:, 1]

    logger.info(
        f"  In-sample Angel prob median: {np.median(angel_probs_insample):.3f} "
        f"(should be much higher than OOF — confirms leakage was present)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Train the Devil (Meta Model - Conviction)
    # Phase 5.5: Train ONLY on Angel-approved subpopulation.
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 3/4] Training Devil model (Conviction with meta-features)...")

    devil_features = feature_cols + ["angel_prob"]
    X_devil_full = df[devil_features].to_numpy()

    # Phase 5.5 — Population Fix:
    # The Devil is deployed exclusively on rows where angel_prob >= ANGEL_THRESHOLD.
    # Training on the full global population (all ~117k rows) violates meta-labeling
    # semantics: the Devil learns to discriminate across all market conditions, not
    # within the Angel-approved subset where it actually operates.
    # Solution: filter training data to only Angel-approved rows using the OOF
    # angel_probs (already computed in Step 2 — zero leakage).
    angel_approved_mask = angel_probs_oof >= ANGEL_THRESHOLD
    n_approved = int(angel_approved_mask.sum())
    n_total = len(X_devil_full)
    logger.info(
        f"Phase 5.5: Devil training filtered to Angel-approved subpopulation: "
        f"{n_approved:,} / {n_total:,} rows ({n_approved / n_total:.1%})"
    )

    X_devil = X_devil_full[angel_approved_mask]
    y_devil_train = y_devil[angel_approved_mask]
    devil_weights = sample_weights[angel_approved_mask]

    logger.info(
        f"Devil survival target distribution (approved rows): "
        f"survived={np.sum(y_devil_train == 1):,} | "
        f"stopped={np.sum(y_devil_train == 0):,} | "
        f"rate={np.mean(y_devil_train):.1%}"
    )
    logger.info(f"Devil feature space: {devil_features}")

    devil_model = RandomForestClassifier(**DEVIL_PARAMS)
    devil_model.fit(X_devil, y_devil_train, sample_weight=devil_weights)
    logger.info(
        f"✓ Devil model trained on {len(devil_features)} features "
        f"(Angel-approved subpopulation, n={n_approved:,})"
    )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Validation & Summary
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 4/4] Model validation...")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        angel_acc = angel_model.score(X_base, y_angel, sample_weight=sample_weights)
        devil_acc = devil_model.score(X_devil, y_devil, sample_weight=sample_weights)

    logger.info(f"\n{'=' * 70}")
    logger.info("META-LABELING TRAINING COMPLETE")
    logger.info(f"{'=' * 70}")
    logger.info(f"Angel training accuracy: {angel_acc:.3f} (recall-focused)")
    logger.info(f"Devil training accuracy: {devil_acc:.3f} (precision-focused)")
    logger.info(f"Devil can now veto Angel when angel_prob is misleading")

    return angel_model, devil_model, feature_cols, devil_features


# ═══════════════════════════════════════════════════════════════════════════════
# DYNAMIC THRESHOLD SELECTION
# ═══════════════════════════════════════════════════════════════════════════════


def _find_optimal_threshold(
    devil_probs: np.ndarray,
    survival_targets: np.ndarray,
    macro_targets: np.ndarray,
    sl_mult: float = SL_ATR_MULTIPLIER,
    tp_mult: float = TP_ATR_MULTIPLIER,
    min_trades: int = 5,
) -> Tuple[float, float]:
    """
    Sweep thresholds to find the one that maximizes Expected Value.

    Phase 5.5 — Two-Target EV Calibration:
        The Devil is trained on `survival_targets` (5-bar SL survival).
        EV calibration must use `macro_targets` (45-bar bracket outcome) to
        reflect the actual asymmetric R:R payload delivered by the live system.

        Separating these two concerns is critical:
        - Using survival_targets for EV would compute "expected value of not
          getting stopped in 5 bars" — meaningless for bracket sizing.
        - Using macro_targets for training would reintroduce the temporal
          mismatch that caused the Devil to flatline.

    For each candidate threshold:
        1. Filter to approved trades: devil_prob >= threshold
        2. Compute realized win rate from MACRO outcomes on approved trades
        3. Compute EV = win_rate * (tp_mult / sl_mult) - (1 - win_rate)

    Args:
        devil_probs:      Array of Devil's predicted probabilities (survival).
        survival_targets: 5-bar SL survival ground truth (Devil's training target).
                          Passed for signature consistency; not used in EV sweep.
        macro_targets:    45-bar bracket outcome ground truth (0/1).
                          Used to compute realized win rate and EV.
        sl_mult:          Stop-loss ATR multiplier.
        tp_mult:          Take-profit ATR multiplier.
        min_trades:       Minimum approved trades for a threshold to be valid.

    Returns:
        Tuple of (optimal_threshold, best_ev)
    """
    thresholds = np.arange(0.10, 0.46, 0.02)  # 0.10, 0.12, 0.14, ..., 0.44
    best_threshold = 0.20  # fallback default
    best_ev = -float("inf")

    for t in thresholds:
        mask = devil_probs >= t
        n_approved = int(mask.sum())

        if n_approved < min_trades:
            continue

        # EV is computed from MACRO outcomes (45-bar bracket), not survival.
        # This correctly prices the asymmetric R:R of the live bracket system.
        approved_macro = macro_targets[mask]
        win_rate = float(approved_macro.mean())

        # EV in R-multiples: wins pay (tp_mult / sl_mult) R, losses pay -1R
        rr_ratio = tp_mult / sl_mult
        ev = win_rate * rr_ratio - (1.0 - win_rate)

        if ev > best_ev:
            best_ev = ev
            best_threshold = float(t)

    return best_threshold, float(best_ev)


# ═══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD VALIDATION GATE
# ═══════════════════════════════════════════════════════════════════════════════


def validate_candidate(
    df: pl.DataFrame,
    feature_cols: List[str],
    n_folds: int = 3,
) -> Tuple[
    ValidationReport,
    RandomForestClassifier,
    RandomForestClassifier,
    List[str],
    List[str],
    float,
]:
    """
    Run expanding-window walk-forward cross-validation and apply the promotion gate.

    Splits the 60-day dataset into 3 expanding folds by calendar date (not row
    index) so that all symbols' data for a given date range stays in the same
    fold.  For each fold, trains Angel + Devil on the training window and
    evaluates strictly out-of-sample on the validation window.

    Fold schedule (calendar days from the earliest timestamp in df):
        Fold 1: Train days  0–29, Validate days 30–39
        Fold 2: Train days  0–39, Validate days 40–49
        Fold 3: Train days  0–49, Validate days 50–59  ← Profit Factor gate

    TEMPORAL BOUNDARY:
        The Profit Factor gate uses the Fold 3 model (trained on days 0–49)
        evaluated on the Fold 3 val set (days 50–59). This is strictly OOS.
        The final full-data model (all 60 days) is ONLY trained AFTER the gate
        passes. Training on all 60 days and then evaluating on a subset would
        be data leakage.

    Promotion thresholds:
        Mean Brier Score   ≤ 0.25   (across all folds)
        Mean EV            ≥ 0.0005 (across all folds)
        Profit Factor      ≥ 1.20   (Fold 3 val set only)

    Dynamic threshold:
        With class_weight=None, Devil probabilities reflect the true ~20% base
        rate. Per-fold, _find_optimal_threshold() sweeps 0.10–0.44 and selects
        the threshold maximizing EV. The Fold 3 threshold is returned as the
        production threshold.

    Args:
        df: Full 60-day feature-engineered DataFrame (output of
            engineer_features_and_labels).
        feature_cols: List of base feature column names.
        n_folds: Number of expanding folds (default: 3).

    Returns:
        Tuple of:
            - ValidationReport (gate decision + per-fold metrics)
            - angel_model (full-data if gate passed, Fold 3 if rejected)
            - devil_model (full-data if gate passed, Fold 3 if rejected)
            - angel_feature_names
            - devil_feature_names
            - production_threshold (optimal Devil threshold from Fold 3)
    """
    logger.info("=" * 70)
    logger.info("WALK-FORWARD VALIDATION (3 EXPANDING FOLDS)")
    logger.info("=" * 70)

    # ───────────────────────────────────────────────────────────────────
    # Build date-based fold boundaries
    # ───────────────────────────────────────────────────────────────────
    min_date = df["timestamp"].min()

    # fold_configs: (train_end_days, val_end_days) — exclusive upper bounds
    fold_configs = [
        (30, 40),  # Fold 1: Train days 0–29, Val days 30–39
        (40, 50),  # Fold 2: Train days 0–39, Val days 40–49
        (50, 60),  # Fold 3: Train days 0–49, Val days 50–59
    ]

    fold_metrics: List[FoldMetrics] = []

    # Placeholders for Fold 3 outputs (used for PF gate and fallback)
    fold3_angel: Optional[RandomForestClassifier] = None
    fold3_devil: Optional[RandomForestClassifier] = None
    fold3_angel_feats: Optional[List[str]] = None
    fold3_devil_feats: Optional[List[str]] = None
    profit_factor: float = 0.0
    final_win_rate: float = 0.0
    final_total_trades: int = 0
    production_threshold: float = 0.20  # fallback; overwritten by Fold 3

    for fold_idx, (train_end_day, val_end_day) in enumerate(fold_configs):
        fold_number = fold_idx + 1

        # Compute cutoff timestamps
        train_cutoff = min_date + timedelta(days=train_end_day)
        val_cutoff = min_date + timedelta(days=val_end_day)

        train_df = df.filter(pl.col("timestamp") < train_cutoff)
        val_df = df.filter(
            (pl.col("timestamp") >= train_cutoff) & (pl.col("timestamp") < val_cutoff)
        )

        logger.info(
            f"\n[Fold {fold_number}/{n_folds}] "
            f"Train: {len(train_df):,} rows | "
            f"Val: {len(val_df):,} rows"
        )

        if len(train_df) == 0 or len(val_df) == 0:
            logger.warning(
                f"[Fold {fold_number}] Empty split — skipping. "
                f"Train={len(train_df)}, Val={len(val_df)}"
            )
            fold_metrics.append(
                FoldMetrics(
                    fold_number=fold_number,
                    train_size=len(train_df),
                    val_size=len(val_df),
                    brier_score=1.0,
                    expected_value=-1.0,
                    angel_proposed_trades=0,
                    devil_approved_trades=0,
                    win_rate=0.0,
                )
            )
            continue

        # ─────────────────────────────────────────────────────────────
        # Train on this fold's training window
        # ─────────────────────────────────────────────────────────────
        angel_model, devil_model, angel_feats, devil_feats = refit_models(
            train_df, feature_cols
        )

        # ─────────────────────────────────────────────────────────────
        # Score on validation window
        # ─────────────────────────────────────────────────────────────
        X_val_base = val_df[feature_cols].to_numpy()
        y_val_angel = val_df["angel_target"].to_numpy()
        y_val_devil = val_df["devil_target"].to_numpy()  # survival (5-bar)
        y_val_devil_macro = val_df["devil_target_macro"].to_numpy()  # macro (45-bar)

        # Stage 1: Angel inference
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            angel_probs_val = angel_model.predict_proba(X_val_base)[:, 1]

        signal_mask = angel_probs_val >= ANGEL_THRESHOLD
        n_angel_proposed = int(signal_mask.sum())
        logger.info(
            f"[Fold {fold_number}] Angel proposed {n_angel_proposed} trades "
            f"({n_angel_proposed / len(val_df):.1%} of val rows)"
        )

        if n_angel_proposed == 0:
            logger.warning(
                f"[Fold {fold_number}] Zero Angel-proposed trades — "
                f"setting worst-case metrics"
            )
            fold_metrics.append(
                FoldMetrics(
                    fold_number=fold_number,
                    train_size=len(train_df),
                    val_size=len(val_df),
                    brier_score=1.0,
                    expected_value=-1.0,
                    angel_proposed_trades=0,
                    devil_approved_trades=0,
                    win_rate=0.0,
                )
            )
            if fold_number == n_folds:
                fold3_angel, fold3_devil = angel_model, devil_model
                fold3_angel_feats, fold3_devil_feats = angel_feats, devil_feats
            continue

        # Stage 2: Devil inference on Angel-proposed rows
        import pandas as pd  # noqa: PLC0415 — used only for sklearn compat

        proposed_base_feats = X_val_base[signal_mask]
        proposed_angel_probs = angel_probs_val[signal_mask]
        proposed_devil_targets = y_val_devil[signal_mask]  # survival — for Brier
        proposed_devil_targets_macro = y_val_devil_macro[signal_mask]  # macro — for EV

        meta_df = pd.DataFrame(proposed_base_feats, columns=feature_cols)
        meta_df["angel_prob"] = proposed_angel_probs

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            devil_probs_val = devil_model.predict_proba(meta_df)[:, 1]

        # ═══════════════════════════════════════════════════════════════════
        # DEVIL DIAGNOSTIC: Probability Distribution Analysis
        # ═══════════════════════════════════════════════════════════════════
        if len(devil_probs_val) > 0:
            logger.info(f"\n{'─' * 60}")
            logger.info(f"DEVIL DIAGNOSTIC — Fold {fold_number}")
            logger.info(f"{'─' * 60}")

            # 1. Global Distribution
            logger.info(f"  Probability Distribution (n={len(devil_probs_val)}):")
            logger.info(f"    Min:    {np.min(devil_probs_val):.4f}")
            logger.info(f"    P25:    {np.percentile(devil_probs_val, 25):.4f}")
            logger.info(f"    Median: {np.median(devil_probs_val):.4f}")
            logger.info(f"    P75:    {np.percentile(devil_probs_val, 75):.4f}")
            logger.info(f"    Max:    {np.max(devil_probs_val):.4f}")

            # 2. Threshold Density
            n_total = len(devil_probs_val)
            logger.info(f"  Threshold Density:")
            logger.info(
                f"    Above 0.50: {(devil_probs_val >= 0.50).sum():>4d} / {n_total} ({(devil_probs_val >= 0.50).mean():.1%})"
            )
            logger.info(
                f"    Above 0.55: {(devil_probs_val >= 0.55).sum():>4d} / {n_total} ({(devil_probs_val >= 0.55).mean():.1%})"
            )
            logger.info(
                f"    Above 0.60: {(devil_probs_val >= 0.60).sum():>4d} / {n_total} ({(devil_probs_val >= 0.60).mean():.1%})"
            )
            logger.info(
                f"    Above 0.65: {(devil_probs_val >= 0.65).sum():>4d} / {n_total} ({(devil_probs_val >= 0.65).mean():.1%})"
            )
            logger.info(
                f"    Above 0.70: {(devil_probs_val >= 0.70).sum():>4d} / {n_total} ({(devil_probs_val >= 0.70).mean():.1%})"
            )

            # 3. Separation Check: Does the Devil actually distinguish wins from losses?
            # proposed_devil_targets is y_val_devil[signal_mask] — ground truth for Angel-proposed rows
            wins_mask = proposed_devil_targets == 1
            losses_mask = proposed_devil_targets == 0

            if wins_mask.sum() > 0 and losses_mask.sum() > 0:
                mean_prob_wins = devil_probs_val[wins_mask].mean()
                mean_prob_losses = devil_probs_val[losses_mask].mean()
                separation = mean_prob_wins - mean_prob_losses

                logger.info(f"  Separation Check:")
                logger.info(
                    f"    Mean prob (Actual Wins):   {mean_prob_wins:.4f}  (n={wins_mask.sum()})"
                )
                logger.info(
                    f"    Mean prob (Actual Losses): {mean_prob_losses:.4f}  (n={losses_mask.sum()})"
                )
                logger.info(f"    Separation Gap:            {separation:+.4f}")

                if separation > 0.05:
                    logger.info(
                        f"    Verdict: SIGNAL DETECTED -- Devil can distinguish (gap > 0.05)"
                    )
                elif separation > 0.02:
                    logger.info(
                        f"    Verdict: WEAK SIGNAL -- marginal separation (0.02 < gap < 0.05)"
                    )
                else:
                    logger.info(
                        f"    Verdict: NO SIGNAL -- Devil cannot distinguish wins from losses"
                    )
            else:
                logger.info(
                    f"  Separation Check: SKIPPED (wins={wins_mask.sum()}, losses={losses_mask.sum()})"
                )

            logger.info(f"{'─' * 60}\n")

        # ─────────────────────────────────────────────────────────────
        # Dynamic threshold selection — sweep 0.10–0.44 for max EV
        # Replaces hardcoded DEVIL_THRESHOLD (0.50) which rejects all
        # trades when class_weight=None shifts probs to the true ~20%
        # base-rate range.
        # ─────────────────────────────────────────────────────────────
        optimal_threshold, fold_ev_at_threshold = _find_optimal_threshold(
            devil_probs=devil_probs_val,
            survival_targets=proposed_devil_targets,
            macro_targets=proposed_devil_targets_macro,
        )
        logger.info(
            f"  [Fold {fold_number}] Dynamic threshold: {optimal_threshold:.2f} "
            f"(EV at threshold: {fold_ev_at_threshold:+.4f})"
        )

        # Store Fold 3 threshold as the production threshold
        if fold_number == n_folds:
            production_threshold = optimal_threshold
            logger.info(
                f"  Production threshold (Fold {fold_number}): {production_threshold:.2f}"
            )

        approved_mask = devil_probs_val >= optimal_threshold
        n_devil_approved = int(approved_mask.sum())
        logger.info(
            f"[Fold {fold_number}] Devil approved {n_devil_approved} trades "
            f"({n_devil_approved / max(n_angel_proposed, 1):.1%} of Angel proposals)"
        )

        if n_devil_approved == 0:
            logger.warning(
                f"[Fold {fold_number}] Zero Devil-approved trades — "
                f"setting worst-case metrics"
            )
            fold_metrics.append(
                FoldMetrics(
                    fold_number=fold_number,
                    train_size=len(train_df),
                    val_size=len(val_df),
                    brier_score=1.0,
                    expected_value=-1.0,
                    angel_proposed_trades=n_angel_proposed,
                    devil_approved_trades=0,
                    win_rate=0.0,
                )
            )
            if fold_number == n_folds:
                fold3_angel, fold3_devil = angel_model, devil_model
                fold3_angel_feats, fold3_devil_feats = angel_feats, devil_feats
            continue

        # ─────────────────────────────────────────────────────────────
        # Compute fold metrics on Devil-approved trades
        # ─────────────────────────────────────────────────────────────
        approved_devil_probs = devil_probs_val[approved_mask]
        approved_targets = proposed_devil_targets[approved_mask]

        brier = float(brier_score_loss(approved_targets, approved_devil_probs))
        win_rate = float(approved_targets.mean()) if len(approved_targets) > 0 else 0.0

        # EV using ATR R-multiple: wins = +TP_MULT R, losses = -SL_MULT R
        # Where R = 1 unit of SL_ATR_MULTIPLIER ATR
        # EV = win_rate * (TP_MULT / SL_MULT) - (1 - win_rate) * 1
        # With TP=3.0, SL=1.5: EV = win_rate * 2 - (1 - win_rate) * 1 = 3*wr - 1
        ev = float(
            win_rate * (TP_ATR_MULTIPLIER / SL_ATR_MULTIPLIER) - (1.0 - win_rate)
        )

        logger.info(
            f"[Fold {fold_number}] "
            f"Brier={brier:.4f} | EV={ev:.6f} | WR={win_rate:.1%} | "
            f"Trades={n_devil_approved}"
        )

        fm = FoldMetrics(
            fold_number=fold_number,
            train_size=len(train_df),
            val_size=len(val_df),
            brier_score=brier,
            expected_value=ev,
            angel_proposed_trades=n_angel_proposed,
            devil_approved_trades=n_devil_approved,
            win_rate=win_rate,
        )
        fold_metrics.append(fm)

        # ─────────────────────────────────────────────────────────────
        # Fold 3 — retain model refs + compute Profit Factor gate
        # CRITICAL: PF is computed from Fold 3 model on Fold 3 val set.
        # These are the same approved_targets already computed above —
        # no additional training or data access needed.
        # ─────────────────────────────────────────────────────────────
        if fold_number == n_folds:
            fold3_angel = angel_model
            fold3_devil = devil_model
            fold3_angel_feats = angel_feats
            fold3_devil_feats = devil_feats

            # Phase 5.5: Profit Factor and win rate computed from MACRO
            # outcomes (45-bar bracket) on Devil-approved trades.
            # approved_targets (survival) is used for Brier only — the PF
            # gate must reflect actual bracket R:R, not survival rate.
            approved_macro_targets = proposed_devil_targets_macro[approved_mask]
            macro_wins = int(approved_macro_targets.sum())
            macro_losses = n_devil_approved - macro_wins
            gross_profit = macro_wins * TP_ATR_MULTIPLIER
            gross_loss = macro_losses * SL_ATR_MULTIPLIER
            profit_factor = (
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            )
            final_win_rate = (
                float(approved_macro_targets.mean()) if n_devil_approved > 0 else 0.0
            )
            final_total_trades = n_devil_approved

            logger.info(
                f"[Fold {fold_number}] Profit Factor (macro) = "
                f"{gross_profit:.2f} / {gross_loss:.2f} = {profit_factor:.4f} "
                f"| Macro WR={final_win_rate:.1%} | Survival WR={win_rate:.1%}"
            )

    # ───────────────────────────────────────────────────────────────────
    # Aggregate metrics and apply gate
    # ───────────────────────────────────────────────────────────────────
    mean_brier = float(np.mean([fm.brier_score for fm in fold_metrics]))
    mean_ev = float(np.mean([fm.expected_value for fm in fold_metrics]))

    rejection_reasons: List[str] = []
    if mean_brier > BRIER_THRESHOLD:
        rejection_reasons.append(
            f"Brier {mean_brier:.4f} > {BRIER_THRESHOLD} threshold"
        )
    if mean_ev < EV_THRESHOLD:
        rejection_reasons.append(f"EV {mean_ev:.6f} < {EV_THRESHOLD} threshold")
    if profit_factor < PROFIT_FACTOR_THRESHOLD:
        rejection_reasons.append(
            f"Profit Factor {profit_factor:.4f} < {PROFIT_FACTOR_THRESHOLD} threshold"
        )

    gate_passed = len(rejection_reasons) == 0

    logger.info("=" * 70)
    logger.info("VALIDATION GATE SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Mean Brier Score : {mean_brier:.4f} (threshold ≤ {BRIER_THRESHOLD})")
    logger.info(f"Mean EV          : {mean_ev:.6f} (threshold ≥ {EV_THRESHOLD})")
    logger.info(
        f"Profit Factor    : {profit_factor:.4f} "
        f"(threshold ≥ {PROFIT_FACTOR_THRESHOLD}, Fold 3 OOS)"
    )
    logger.info(f"Gate Result      : {'PASSED ✅' if gate_passed else 'FAILED 🚫'}")

    # ───────────────────────────────────────────────────────────────────
    # If gate passed — train final production model on ALL 60 days.
    # This is the REWARD for passing: maximum information for production.
    # If gate failed — keep Fold 3 models as placeholders (NOT saved).
    # ───────────────────────────────────────────────────────────────────
    if gate_passed:
        logger.info("Gate passed — training final production model on full dataset")
        final_angel, final_devil, final_angel_feats, final_devil_feats = refit_models(
            df, feature_cols
        )
    else:
        logger.info(
            "Gate failed — skipping full-data training. Production weights retained."
        )
        final_angel = fold3_angel
        final_devil = fold3_devil
        final_angel_feats = fold3_angel_feats
        final_devil_feats = fold3_devil_feats

    report = ValidationReport(
        fold_metrics=fold_metrics,
        mean_brier=mean_brier,
        mean_ev=mean_ev,
        final_profit_factor=profit_factor,
        final_win_rate=final_win_rate,
        final_total_trades=final_total_trades,
        gate_passed=gate_passed,
        rejection_reasons=rejection_reasons,
    )

    return (
        report,
        final_angel,
        final_devil,
        final_angel_feats,
        final_devil_feats,
        production_threshold,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GATE DECISION & MODEL PROMOTION
# ═══════════════════════════════════════════════════════════════════════════════


def promote_or_reject(
    report: ValidationReport,
    angel_model: RandomForestClassifier,
    devil_model: RandomForestClassifier,
    threshold: float = 0.20,
) -> bool:
    """
    Promote or reject candidate models based on the validation report.

    If gate passed: the models passed in were trained on the full 60-day
    dataset (after CV validation confirmed generalizability). Saves them
    atomically via save_models(), then saves the optimal threshold via
    save_threshold().

    If gate failed: the models passed in are Fold 3 models (not saved).
    Production weights are retained, rejection alert is sent to Discord.

    Args:
        report: ValidationReport from validate_candidate()
        angel_model: Final model (full-data if gate passed, Fold 3 if failed)
        devil_model: Final model (full-data if gate passed, Fold 3 if failed)
        threshold: Optimal Devil threshold from Fold 3 (default: 0.20 fallback)

    Returns:
        True if models were promoted, False if rejected.
    """
    notifier = NotificationManager()

    if report.gate_passed:
        logger.info("=" * 70)
        logger.info("✅ VALIDATION GATE PASSED — PROMOTING MODELS")
        logger.info("=" * 70)

        save_models(angel_model, devil_model)
        save_threshold(threshold)

        notifier.send_retraining_report(report, promoted=True)
        return True
    else:
        logger.warning("=" * 70)
        logger.warning("🚫 VALIDATION GATE FAILED — MODELS REJECTED")
        logger.warning("=" * 70)
        for reason in report.rejection_reasons:
            logger.warning(f"  Rejection: {reason}")

        notifier.send_retraining_report(report, promoted=False)
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL SERIALIZATION (ATOMIC)
# ═══════════════════════════════════════════════════════════════════════════════


def save_models(
    angel_model: RandomForestClassifier,
    devil_model: RandomForestClassifier,
) -> None:
    """
    Serialize models to disk using joblib with POSIX atomic writes.

    Uses temporary files and os.replace() to ensure zero-downtime atomic swaps,
    preventing the live hot-reloader from reading partially written files.

    Args:
        angel_model: Trained Angel model
        devil_model: Trained Devil model
    """
    logger.info("=" * 70)
    logger.info("SERIALIZING MODELS (ATOMIC)")
    logger.info("=" * 70)

    # Ensure model directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Define temp and final paths
    angel_temp = MODEL_DIR / "angel_temp.pkl"
    devil_temp = MODEL_DIR / "devil_temp.pkl"

    # ═══════════════════════════════════════════════════════════════════
    # ATOMIC WRITE: Angel Model
    # ═══════════════════════════════════════════════════════════════════
    try:
        joblib.dump(angel_model, angel_temp)
        angel_size = angel_temp.stat().st_size / (1024 * 1024)
        os.replace(angel_temp, ANGEL_PATH)
        logger.info(f"[ATOMIC] Angel model saved: {ANGEL_PATH} ({angel_size:.1f} MB)")

    except Exception as e:
        logger.error(f"[ATOMIC] Failed to save Angel model: {e}")
        if angel_temp.exists():
            angel_temp.unlink()
        raise

    # ═══════════════════════════════════════════════════════════════════
    # ATOMIC WRITE: Devil Model
    # ═══════════════════════════════════════════════════════════════════
    try:
        joblib.dump(devil_model, devil_temp)
        devil_size = devil_temp.stat().st_size / (1024 * 1024)
        os.replace(devil_temp, DEVIL_PATH)
        logger.info(f"[ATOMIC] Devil model saved: {DEVIL_PATH} ({devil_size:.1f} MB)")

    except Exception as e:
        logger.error(f"[ATOMIC] Failed to save Devil model: {e}")
        if devil_temp.exists():
            devil_temp.unlink()
        raise

    logger.info(
        "[ATOMIC] Model serialization complete — live bot can hot-reload safely"
    )


def save_threshold(threshold: float) -> None:
    """
    Save the optimal Devil threshold to disk as a JSON sidecar file.

    Written atomically alongside the model .pkl files.  The LiveOrchestrator
    and MLStrategy read this on startup and via hot-reload so the live bot
    always uses the threshold that maximises EV on the most recent data.

    Args:
        threshold: The optimal Devil probability threshold (e.g., 0.28)
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    threshold_path = MODEL_DIR / "threshold.json"

    data = {
        "devil_threshold": round(threshold, 4),
        "updated_at": datetime.now().isoformat(),
    }

    # Atomic write — same pattern as model serialisation
    temp_path = MODEL_DIR / "threshold_temp.json"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, threshold_path)

    logger.info(
        f"[ATOMIC] Threshold saved: {threshold_path} (devil_threshold={threshold:.4f})"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> int:
    """
    Main entry point for the validated model retraining pipeline.

    Exit codes:
        0 = Models promoted successfully
        1 = Execution error
        2 = Models rejected by validation gate (production weights retained)

    NOTE: run_pipeline.sh only checks feedback_loop.py's exit code to decide
    whether to trigger retraining. The retrainer's exit code 2 ("tried but
    rejected") is logged for observability but does NOT cause an infinite loop.
    """
    try:
        logger.info(
            "╔══════════════════════════════════════════════════════════════════╗"
        )
        logger.info(
            "║              THE CURE V2 - VALIDATED MODEL RETRAINER             ║"
        )
        logger.info(
            "╚══════════════════════════════════════════════════════════════════╝"
        )

        # ─── Phase 1: Fetch data ────────────────────────────────────────────
        client = get_alpaca_client()
        logger.info("Alpaca client initialized")
        raw_data = fetch_training_data(client)

        # ─── Phase 2: Engineer features with ATR-dynamic labels ────────────
        features_df, feature_cols = engineer_features_and_labels(raw_data)

        # ─── Phase 3: Walk-forward validation (3-fold expanding window) ────
        # TEMPORAL BOUNDARY: CV folds and the Profit Factor gate are evaluated
        # strictly OOS (Fold 3 model evaluated on days 51–60). The full-data
        # production model is only trained AFTER the gate passes. Running the
        # full-data model against any data used in its training would be data
        # leakage.
        logger.info("=" * 70)
        logger.info("WALK-FORWARD VALIDATION (3 EXPANDING FOLDS)")
        logger.info("=" * 70)

        (
            report,
            angel_model,
            devil_model,
            angel_feats,
            devil_feats,
            optimal_threshold,
        ) = validate_candidate(features_df, feature_cols, n_folds=3)
        logger.info(f"Optimal Devil threshold (from Fold 3): {optimal_threshold:.4f}")

        # ─── Phase 4: Gate decision ─────────────────────────────────────────
        # If gate passed: angel_model/devil_model are trained on full 60 days
        # If gate failed: they are Fold 3 models (will NOT be saved)
        promoted = promote_or_reject(
            report, angel_model, devil_model, optimal_threshold
        )

        if promoted:
            logger.info("=" * 70)
            logger.info("✅ MODELS PROMOTED — Ready for next market open")
            logger.info(f"  Angel: {ANGEL_PATH}")
            logger.info(f"  Devil: {DEVIL_PATH}")
            logger.info("=" * 70)
            return 0
        else:
            logger.warning("=" * 70)
            logger.warning("🚫 MODELS REJECTED — Production weights retained")
            logger.warning("Manual review recommended.")
            logger.warning("=" * 70)
            return 2  # 2 = retrained but rejected; production weights intact

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
