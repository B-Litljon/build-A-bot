"""
The Cure - Automated Model Retraining Pipeline.

Triggered when feedback_loop.py detects critical model drift.
Fetches fresh data, applies time-decay weighting, and refits
Angel/Devil Random Forest models.

Usage:
    python -m src.core.retrainer

Environment Variables:
    ALPACA_API_KEY: Alpaca API key
    ALPACA_SECRET_KEY: Alpaca API secret
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import polars as pl
import talib
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed
from sklearn.ensemble import RandomForestClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
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
    "class_weight": "balanced",
    "n_jobs": -1,
}


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


def engineer_features_and_labels(df: pl.DataFrame) -> Tuple[pl.DataFrame, List[str]]:
    """
    Engineer technical features and generate target labels using TA-Lib + Polars.

    Features:
        - rsi_14: 14-period RSI
        - macd: MACD line
        - macd_signal: MACD signal line
        - bb_upper: Bollinger Bands upper band
        - bb_lower: Bollinger Bands lower band

    Targets:
        - angel_target: 1 if close 3 bars ahead > current close + 0.1%
        - devil_target: 1 if +0.5% TP hit before -0.2% SL in next 15 bars

    Args:
        df: Raw OHLCV DataFrame with columns: open, high, low, close, volume, symbol, timestamp

    Returns:
        Tuple of (features DataFrame, feature column names)
    """
    logger.info("=" * 70)
    logger.info("ENGINEERING FEATURES & LABELS")
    logger.info("=" * 70)

    # Convert Polars columns to numpy for talib (native compatibility)
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()

    # ═══════════════════════════════════════════════════════════════════
    # TECHNICAL INDICATORS (TA-Lib)
    # ═══════════════════════════════════════════════════════════════════

    # RSI (14-period)
    rsi = talib.RSI(close, timeperiod=14)

    # MACD (12, 26, 9)
    macd, macd_signal, macd_hist = talib.MACD(
        close, fastperiod=12, slowperiod=26, signalperiod=9
    )

    # Bollinger Bands (20-period, 2 std dev)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=talib.MA_Type.SMA
    )

    # Add features to DataFrame
    df = df.with_columns(
        [
            pl.Series("rsi_14", rsi),
            pl.Series("macd", macd),
            pl.Series("macd_signal", macd_signal),
            pl.Series("bb_upper", bb_upper),
            pl.Series("bb_lower", bb_lower),
        ]
    )

    logger.info("Applied TA-Lib indicators: RSI, MACD, BBANDS")

    # ═══════════════════════════════════════════════════════════════════
    # TARGET GENERATION (Lookahead Logic)
    # ═══════════════════════════════════════════════════════════════════

    # Angel Target: Momentum shift (close 3 bars ahead > current + 0.1%)
    df = df.with_columns(
        [
            (pl.col("close").shift(-3) > pl.col("close") * 1.001)
            .cast(pl.Int8)
            .alias("angel_target")
        ]
    )

    # Devil Target: Bracket resolution (TP +0.5% hit before SL -0.2%)
    # Lookahead window: 15 bars
    lookahead = 15

    df = df.with_columns(
        [
            # Calculate forward rolling max (highest high in next N bars)
            pl.col("high")
            .rolling_max(window_size=lookahead, min_periods=1)
            .shift(-lookahead)
            .alias("forward_high_max"),
            # Calculate forward rolling min (lowest low in next N bars)
            pl.col("low")
            .rolling_min(window_size=lookahead, min_periods=1)
            .shift(-lookahead)
            .alias("forward_low_min"),
        ]
    )

    # Devil target: TP hit (+0.5%) AND SL not hit (-0.2%)
    df = df.with_columns(
        [
            (
                (pl.col("forward_high_max") >= pl.col("close") * 1.005)
                & (pl.col("forward_low_min") > pl.col("close") * 0.998)
            )
            .cast(pl.Int8)
            .alias("devil_target")
        ]
    )

    # Drop intermediate columns
    df = df.drop(["forward_high_max", "forward_low_min"])

    logger.info(
        f"Generated targets: angel_target (3-bar), devil_target ({lookahead}-bar bracket)"
    )

    # ═══════════════════════════════════════════════════════════════════
    # CLEANUP: Remove rows with nulls
    # ═══════════════════════════════════════════════════════════════════

    initial_count = len(df)
    df = df.drop_nulls()
    dropped_count = initial_count - len(df)

    logger.info(
        f"Dropped {dropped_count:,} rows with nulls ({dropped_count / initial_count:.1%})"
    )
    logger.info(f"Final dataset: {len(df):,} rows")

    # Define feature columns for model training
    feature_cols = ["rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower"]

    logger.info(f"Feature columns: {feature_cols}")

    return df, feature_cols


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
        df: Feature-engineered DataFrame
        feature_cols: List of feature column names

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
    # STEP 2: Generate Meta-Features (Angel's Probabilities)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 2/4] Generating meta-features (Angel's probabilities)...")

    # Get Angel's prediction probabilities on training data
    # Suppress sklearn warning about feature names during training inference
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        angel_probs_train = angel_model.predict_proba(X_base)[:, 1]

    # Add angel_prob as a new column to the DataFrame
    df = df.with_columns(pl.Series("angel_prob", angel_probs_train))
    logger.info(f"✓ Generated {len(angel_probs_train):,} Angel probabilities")
    logger.info(
        f"  Angel prob range: [{angel_probs_train.min():.3f}, {angel_probs_train.max():.3f}]"
    )

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Train the Devil (Meta Model - Conviction)
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 3/4] Training Devil model (Conviction with meta-features)...")

    # Devil sees base features + Angel's probability
    devil_features = feature_cols + ["angel_prob"]
    X_devil = df[devil_features].to_numpy()

    logger.info(f"Devil feature space: {devil_features}")

    devil_model = RandomForestClassifier(**DEVIL_PARAMS)
    devil_model.fit(X_devil, y_devil, sample_weight=sample_weights)
    logger.info(f"✓ Devil model trained on {len(devil_features)} features")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: Validation & Summary
    # ═══════════════════════════════════════════════════════════════════
    logger.info("\n[Step 4/4] Model validation...")

    # Quick validation
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
        # Write to temp file first
        joblib.dump(angel_model, angel_temp)
        angel_size = angel_temp.stat().st_size / (1024 * 1024)

        # Atomic swap: instant replacement at OS level
        os.replace(angel_temp, ANGEL_PATH)

        logger.info(f"[ATOMIC] Angel model saved: {ANGEL_PATH} ({angel_size:.1f} MB)")

    except Exception as e:
        logger.error(f"[ATOMIC] Failed to save Angel model: {e}")
        # Clean up temp file if it exists
        if angel_temp.exists():
            angel_temp.unlink()
        raise

    # ═══════════════════════════════════════════════════════════════════
    # ATOMIC WRITE: Devil Model
    # ═══════════════════════════════════════════════════════════════════
    try:
        # Write to temp file first
        joblib.dump(devil_model, devil_temp)
        devil_size = devil_temp.stat().st_size / (1024 * 1024)

        # Atomic swap: instant replacement at OS level
        os.replace(devil_temp, DEVIL_PATH)

        logger.info(f"[ATOMIC] Devil model saved: {DEVIL_PATH} ({devil_size:.1f} MB)")

    except Exception as e:
        logger.error(f"[ATOMIC] Failed to save Devil model: {e}")
        # Clean up temp file if it exists
        if devil_temp.exists():
            devil_temp.unlink()
        raise

    logger.info(
        "[ATOMIC] Model serialization complete - Live bot can hot-reload safely"
    )


def main():
    """Main entry point for model retraining."""
    try:
        logger.info(
            "╔══════════════════════════════════════════════════════════════════╗"
        )
        logger.info(
            "║                    THE CURE - MODEL RETRAINER                    ║"
        )
        logger.info(
            "╚══════════════════════════════════════════════════════════════════╝"
        )

        # Initialize Alpaca client
        client = get_alpaca_client()
        logger.info("Alpaca client initialized")

        # Fetch fresh training data
        raw_data = fetch_training_data(client)

        # Engineer features and labels
        features_df, feature_cols = engineer_features_and_labels(raw_data)

        # Refit models with time-decay weighting
        angel_model, devil_model, angel_features, devil_features = refit_models(
            features_df, feature_cols
        )

        # Save models
        save_models(angel_model, devil_model)

        # Success message
        logger.info("=" * 70)
        logger.info("✅ MODELS SUCCESSFULLY REFIT AND SERIALIZED")
        logger.info("=" * 70)
        logger.info("Models are ready for the next live market open.")
        logger.info(f"  - Angel: {ANGEL_PATH}")
        logger.info(f"  - Devil: {DEVIL_PATH}")
        logger.info("\n🎯 The system is cured and ready for deployment!")

        return 0

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1

    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
