"""
Out-of-Sample (OOS) Replay Testing Harness for Universal Scalper v3.0.

Simulates live market firehose conditions using Parquet files for backtesting
the Angel/Devil dual-model architecture.

Usage:
    python -m src.replay_test

Features:
    - MockAlpacaProvider: Mimics real-time DataFeed.IEX structure
    - High-speed replay loop with cross-sectional bar streaming
    - In-memory signal accumulation (no row-by-row disk writes)
    - Batch CSV export of signal ledger
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import joblib
import numpy as np
import polars as pl

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ml.feature_pipeline import FeatureEngineer
from src.strategies.concrete_strategies.ml_strategy import MLStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = Path("data/oos_bars.parquet")
LEDGER_PATH = Path("data/signal_ledger.csv")
ANGEL_MODEL_PATH = Path("src/ml/models/angel_rf_model.joblib")
DEVIL_MODEL_PATH = Path("src/ml/models/devil_rf_model.joblib")

# Thresholds (must match training configuration)
ANGEL_THRESHOLD = 0.40
DEVIL_THRESHOLD = 0.50
WARMUP_PERIOD = 60

# Feature names (must match MLStrategy)
FEATURE_NAMES = [
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
]


@dataclass
class Bar:
    """Represents a single OHLCV bar."""

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class MockAlpacaProvider:
    """
    Mock provider that mimics Alpaca's DataFeed.IEX streaming structure.

    Loads Parquet data and yields timestamp-grouped bars to simulate
    real-time cross-sectional market data (all tickers per minute).
    """

    def __init__(self, data_path: Path):
        """
        Initialize the mock provider.

        Args:
            data_path: Path to the Parquet file containing OOS bars.
        """
        self.data_path = data_path
        self.data: Optional[pl.DataFrame] = None

        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        self._load_data()

    def _load_data(self) -> None:
        """Load and prepare the Parquet data."""
        logger.info(f"Loading OOS data from {self.data_path}...")

        self.data = pl.read_parquet(self.data_path)

        # Ensure timestamp column is datetime
        if self.data["timestamp"].dtype != pl.Datetime:
            self.data = self.data.with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Sort by timestamp to ensure chronological order
        self.data = self.data.sort("timestamp")

        logger.info(f"Loaded {len(self.data):,} bars")
        logger.info(f"Symbols: {self.data['symbol'].unique().to_list()}")
        logger.info(
            f"Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}"
        )

    def stream_bars(self) -> Iterator[Tuple[datetime, pl.DataFrame]]:
        """
        Stream bars grouped by timestamp (cross-sectional slice).

        Yields:
            Tuple of (timestamp, DataFrame) where DataFrame contains
            all ticker bars for that specific minute.
        """
        if self.data is None:
            raise RuntimeError("Data not loaded. Call _load_data() first.")

        # Group by timestamp with maintained order
        for timestamp, group in self.data.group_by("timestamp", maintain_order=True):
            yield timestamp, group

    def get_symbol_data(self, symbol: str) -> pl.DataFrame:
        """
        Get all historical bars for a specific symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            DataFrame with all bars for the symbol.
        """
        if self.data is None:
            return pl.DataFrame()
        return self.data.filter(pl.col("symbol") == symbol).sort("timestamp")


class ReplayHarness:
    """
        OOS Testing Harness that simulates live trading conditions.

        Accumulates signals in memory and performs single batch write
    to disk for optimal performance.
    """

    def __init__(
        self,
        provider: MockAlpacaProvider,
        angel_model_path: Path,
        devil_model_path: Path,
    ):
        """
        Initialize the replay harness.

        Args:
            provider: MockAlpacaProvider instance.
            angel_model_path: Path to Angel model joblib file.
            devil_model_path: Path to Devil model joblib file.
        """
        self.provider = provider
        self.feature_engineer = FeatureEngineer()

        # Load models
        logger.info("Loading Angel model...")
        self.angel_model = joblib.load(angel_model_path)
        logger.info("Loading Devil model...")
        self.devil_model = joblib.load(devil_model_path)

        # Track historical data per symbol for feature calculation
        self.symbol_history: Dict[str, pl.DataFrame] = {}

        # In-memory signal ledger (no row-by-row disk writes)
        self.signal_ledger: List[Dict] = []

        # Statistics
        self.bars_processed = 0
        self.signals_generated = 0

    def _update_symbol_history(self, symbol: str, bar: pl.DataFrame) -> None:
        """Append new bar to symbol's history, maintaining rolling window."""
        if symbol not in self.symbol_history:
            self.symbol_history[symbol] = bar
        else:
            # Append and keep last N bars for warmup
            self.symbol_history[symbol] = pl.concat(
                [self.symbol_history[symbol], bar], how="vertical_relaxed"
            ).tail(WARMUP_PERIOD * 2)  # Keep extra buffer

    def _generate_features(self, symbol: str) -> Optional[pl.DataFrame]:
        """Generate features for a symbol's current history."""
        if symbol not in self.symbol_history:
            return None

        df = self.symbol_history[symbol]

        if len(df) < WARMUP_PERIOD:
            return None

        try:
            features_df = self.feature_engineer.compute_indicators(df)

            # Filter to feature columns only
            feature_cols = [c for c in features_df.columns if c in FEATURE_NAMES]
            features_df = features_df.drop_nulls(subset=feature_cols)

            return features_df
        except Exception as e:
            logger.error(f"Feature generation failed for {symbol}: {e}")
            return None

    def _run_inference(
        self,
        symbol: str,
        close_price: float,
        timestamp: datetime,
    ) -> Optional[Dict]:
        """
        Run Angel/Devil inference on current symbol state.

        Returns:
            Signal dict if both models agree, None otherwise.
        """
        features_df = self._generate_features(symbol)

        if features_df is None or len(features_df) == 0:
            return None

        # Get latest features
        latest_features = features_df[FEATURE_NAMES].tail(1).to_numpy()

        # Stage 1: Angel (Direction)
        angel_prob = self.angel_model.predict_proba(latest_features)[0, 1]

        if angel_prob < ANGEL_THRESHOLD:
            return {
                "timestamp": timestamp,
                "symbol": symbol,
                "close_price": close_price,
                "angel_prob": float(angel_prob),
                "devil_prob": None,
                "action": "REJECT_ANGEL",
            }

        # Stage 2: Devil (Conviction)
        import pandas as pd

        meta_features = pd.DataFrame(latest_features, columns=FEATURE_NAMES)
        meta_features["angel_prob"] = angel_prob

        devil_prob = self.devil_model.predict_proba(meta_features)[0, 1]

        if devil_prob < DEVIL_THRESHOLD:
            return {
                "timestamp": timestamp,
                "symbol": symbol,
                "close_price": close_price,
                "angel_prob": float(angel_prob),
                "devil_prob": float(devil_prob),
                "action": "REJECT_DEVIL",
            }

        # Both agree - BUY signal
        self.signals_generated += 1
        return {
            "timestamp": timestamp,
            "symbol": symbol,
            "close_price": close_price,
            "angel_prob": float(angel_prob),
            "devil_prob": float(devil_prob),
            "action": "BUY",
        }

    def run(self) -> None:
        """
        Execute the replay simulation.

        Processes all bars from the provider, accumulates signals in memory,
        and performs a single batch write to disk.
        """
        logger.info("=" * 70)
        logger.info("OOS REPLAY HARNESS v3.0")
        logger.info("=" * 70)
        logger.info(f"Angel threshold: {ANGEL_THRESHOLD}")
        logger.info(f"Devil threshold: {DEVIL_THRESHOLD}")
        logger.info(f"Warmup period: {WARMUP_PERIOD} bars")

        # Main replay loop
        for timestamp, bar_group in self.provider.stream_bars():
            self.bars_processed += len(bar_group)

            # Process each bar in the cross-sectional slice
            for row in bar_group.iter_rows(named=True):
                symbol = row["symbol"]

                # Create single-row DataFrame for this bar
                bar_df = pl.DataFrame([row])

                # Update symbol history
                self._update_symbol_history(symbol, bar_df)

                # Run inference
                signal = self._run_inference(
                    symbol=symbol,
                    close_price=row["close"],
                    timestamp=timestamp,
                )

                if signal:
                    self.signal_ledger.append(signal)

            # Progress logging every 1000 bars
            if self.bars_processed % 1000 == 0:
                logger.info(
                    f"Processed {self.bars_processed:,} bars | "
                    f"Signals: {self.signals_generated}"
                )

        logger.info(f"Replay complete. Total bars: {self.bars_processed:,}")
        logger.info(f"Total signals: {self.signals_generated}")

    def save_ledger(self, output_path: Path) -> None:
        """
        Save the signal ledger to CSV.

        Performs single batch write after accumulation (no row-by-row disk I/O).

        Args:
            output_path: Path to save the ledger CSV.
        """
        if not self.signal_ledger:
            logger.warning("No signals to save")
            return

        logger.info(f"Saving {len(self.signal_ledger):,} signals to {output_path}...")

        # Convert list of dicts to Polars DataFrame (single batch operation)
        ledger_df = pl.DataFrame(self.signal_ledger)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Single batch write to CSV
        ledger_df.write_csv(output_path)

        file_size = output_path.stat().st_size / 1024  # KB
        logger.info(f"Ledger saved: {file_size:.2f} KB")

        # Summary statistics
        buy_signals = ledger_df.filter(pl.col("action") == "BUY")
        logger.info(f"BUY signals: {len(buy_signals):,}")
        logger.info(
            f"Angel rejections: {len(ledger_df.filter(pl.col('action') == 'REJECT_ANGEL')):,}"
        )
        logger.info(
            f"Devil rejections: {len(ledger_df.filter(pl.col('action') == 'REJECT_DEVIL')):,}"
        )


def main():
    """Main entry point for OOS replay testing."""
    # Verify required files exist
    if not DATA_PATH.exists():
        logger.error(f"OOS data not found: {DATA_PATH}")
        logger.error("Run: python -m src.data.harvester")
        return 1

    if not ANGEL_MODEL_PATH.exists():
        logger.error(f"Angel model not found: {ANGEL_MODEL_PATH}")
        logger.error("Run: python -m src.ml.train_model")
        return 1

    if not DEVIL_MODEL_PATH.exists():
        logger.error(f"Devil model not found: {DEVIL_MODEL_PATH}")
        logger.error("Run: python -m src.ml.train_model")
        return 1

    # Initialize provider
    provider = MockAlpacaProvider(DATA_PATH)

    # Initialize harness
    harness = ReplayHarness(
        provider=provider,
        angel_model_path=ANGEL_MODEL_PATH,
        devil_model_path=DEVIL_MODEL_PATH,
    )

    # Run simulation
    harness.run()

    # Save results
    harness.save_ledger(LEDGER_PATH)

    logger.info("=" * 70)
    logger.info("OOS TESTING COMPLETE")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
