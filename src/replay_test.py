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

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
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
LEDGER_PATH = Path("data/signal_ledger.parquet")
ANGEL_MODEL_PATH = Path("src/ml/models/angel_rf_model.joblib")
DEVIL_MODEL_PATH = Path("src/ml/models/devil_rf_model.joblib")

# Thresholds (must match training configuration)
ANGEL_THRESHOLD = 0.40
DEVIL_THRESHOLD = 0.50
WARMUP_PERIOD = 260  # V3.3: expanded for 5m HTF SMA-50 warm-up (250 bars minimum)

# Feature names (must match MLStrategy and retrainer.FEATURE_COLS)
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
    # V3.3: HTF features
    "htf_rsi_14",
    "htf_trend_agreement",
    "htf_vol_rel",
    "htf_bb_pct_b",
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

        # ═══════════════════════════════════════════════════════════════════
        # Dynamic Devil threshold — read from models/threshold.json so replay
        # uses exactly the same threshold the retrainer selected.  Falls back
        # to the module-level DEVIL_THRESHOLD constant if file is absent.
        # ═══════════════════════════════════════════════════════════════════
        project_root = Path(__file__).resolve().parent.parent
        threshold_path = project_root / "models" / "threshold.json"
        self.devil_threshold: float = DEVIL_THRESHOLD  # start with module default
        if threshold_path.exists():
            try:
                with open(threshold_path, "r") as _fh:
                    _data = json.load(_fh)
                self.devil_threshold = float(_data["threshold"])
                logger.info(
                    "Dynamic Devil threshold loaded: %.4f (from %s)",
                    self.devil_threshold,
                    threshold_path,
                )
            except Exception as _exc:
                logger.warning(
                    "Could not read %s (%s) — using fallback DEVIL_THRESHOLD=%.2f",
                    threshold_path,
                    _exc,
                    DEVIL_THRESHOLD,
                )
        else:
            logger.warning(
                "models/threshold.json not found — using fallback DEVIL_THRESHOLD=%.2f",
                DEVIL_THRESHOLD,
            )

        # ═══════════════════════════════════════════════════════════════════
        # IRONCLAD ALIGNMENT: Capture official feature order from models
        # This ensures NumPy arrays match the exact column order from training
        # ═══════════════════════════════════════════════════════════════════
        self.angel_features = list(self.angel_model.feature_names_in_)
        self.devil_features = list(self.devil_model.feature_names_in_)
        logger.info(f"Angel expects features: {self.angel_features}")
        logger.info(f"Devil expects features: {self.devil_features}")

        # Verify Devil's last feature is angel_prob (meta-labeling check)
        if self.devil_features[-1] != "angel_prob":
            logger.warning(
                "⚠️  Devil's last feature is not 'angel_prob' - meta-labeling may be misconfigured"
            )

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

        Uses IRONCLAD ALIGNMENT: Features are selected by name in exact training order
        before NumPy conversion to prevent misalignment errors.

        Returns:
            Signal dict if both models agree, None otherwise.
        """
        import warnings

        features_df = self._generate_features(symbol)

        if features_df is None or len(features_df) == 0:
            return None

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 1: THE ANGEL (DIRECTION)
        # ═══════════════════════════════════════════════════════════════════

        # IRONCLAD ALIGNMENT: Select features in exact order Angel expects
        # This prevents column misalignment even if DataFrame columns are shuffled
        angel_input = features_df.select(self.angel_features).tail(1).to_numpy()

        # Suppress sklearn warning after alignment is guaranteed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            angel_prob = self.angel_model.predict_proba(angel_input)[0, 1]

        if angel_prob < ANGEL_THRESHOLD:
            return None  # Rejection - not appended to ledger

        # ═══════════════════════════════════════════════════════════════════
        # STAGE 2: THE DEVIL (CONVICTION)
        # ═══════════════════════════════════════════════════════════════════

        # IRONCLAD ALIGNMENT: Build Devil input with meta-feature
        # Devil expects: [base_features..., angel_prob]
        devil_input = features_df.select(self.angel_features).tail(1).to_numpy()

        # CRITICAL: Append angel_prob as the final column (meta-labeling requirement)
        devil_input = np.column_stack([devil_input, np.array([[angel_prob]])])

        # Verify input shape matches Devil's expectations
        if devil_input.shape[1] != len(self.devil_features):
            logger.error(
                f"Feature count mismatch: Devil expects {len(self.devil_features)} "
                f"features, got {devil_input.shape[1]}"
            )
            return None

        # Suppress sklearn warning after alignment is guaranteed
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            devil_prob = self.devil_model.predict_proba(devil_input)[0, 1]

        if devil_prob < self.devil_threshold:
            return None  # Rejection - not appended to ledger

        # Both agree - BUY signal
        self.signals_generated += 1
        # STRICT SCHEMA: Preserve native datetime[μs, UTC] from source data
        # timestamp is already timezone-aware datetime from row["timestamp"]
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
        logger.info(
            f"Devil threshold: {self.devil_threshold} (module default: {DEVIL_THRESHOLD})"
        )
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
                    close_price=float(row["close"]),
                    # FIX: Extract native Python datetime from row to avoid Polars grouping key artifacts
                    timestamp=row["timestamp"],
                )

                # Only append valid BUY signals to ledger (rejections are discarded)
                if signal and signal.get("action") == "BUY":
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
        Save the signal ledger to Parquet with strict schema preservation.

        Logic Ledger:
        - Input: List of signal dictionaries with timezone-aware datetime objects
        - Process: Construct Polars DataFrame (schema inferred from native types)
        - Output: Parquet file with datetime[μs, UTC] for timestamp column
        - Strict type matching ensures seamless join with oos_bars.parquet
        """
        if not self.signal_ledger:
            logger.warning("No signals to save")
            return

        logger.info(f"Saving {len(self.signal_ledger):,} signals to {output_path}...")

        # STRICT SCHEMA: Build DataFrame preserving native Python types
        # Polars will infer datetime[μs, UTC] from timezone-aware datetime objects
        ledger_df = pl.DataFrame(self.signal_ledger)

        # Defensive cast: guarantee timestamp matches oos_bars schema (μs, UTC)
        # Prevents silent join failures if upstream ever changes resolution (ns vs μs)
        if "timestamp" in ledger_df.columns:
            ledger_df = ledger_df.with_columns(
                pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
            )

        # Schema verification for debugging
        logger.debug(f"Ledger schema: {ledger_df.schema}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to Parquet - preserves full type information including timezone
        ledger_df.write_parquet(output_path)

        file_size = output_path.stat().st_size / 1024  # KB
        logger.info(f"Ledger saved: {file_size:.2f} KB")

        # Summary statistics
        buy_signals = ledger_df.filter(pl.col("action") == "BUY")
        logger.info(f"BUY signals: {len(buy_signals):,}")


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
