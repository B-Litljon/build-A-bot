"""
Drift Evaluator - Phase 2 of the Reinforcement Feedback Loop.

Evaluates the Devil model's precision and probability calibration
to detect concept drift in the out-of-sample performance.

Usage:
    python -m src.core.feedback_loop

Inputs:
    - data/resolved_ledger.csv (resolved trades with outcomes)

Output:
    - Terminal metrics summary
    - Discord alert (if drift detected)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, log_loss

# Import notification manager for drift alerts
from src.core.notification_manager import NotificationManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
RESOLVED_PATH = Path("data/resolved_ledger.csv")

# Bracket Parameters for EV Calculation
# NOTE: These are legacy static-percentage brackets from the pre-V3.2 era.
# The live system now uses ATR-dynamic brackets (SL=0.5×ATR, TP=3.0×ATR, Hold=45).
# These constants are retained here for backward compatibility with resolved_ledger.csv
# data produced by src/core/resolver.py, which also uses static percentage brackets.
# They are superseded by the ATR-dynamic constants in retrainer.py / evaluate_performance.py.
TAKE_PROFIT = 0.005  # +0.5% (legacy static bracket — superseded by ATR-dynamic)
STOP_LOSS = 0.002  # -0.2% (legacy static bracket — superseded by ATR-dynamic)

# Drift Detection Thresholds
CRITICAL_BRIER = 0.25
MINIMUM_EV = 0.0005  # 0.05% minimum expected return


@dataclass
class PerformanceMetrics:
    """Container for model performance metrics."""

    win_rate: float
    expected_value: float
    brier_score: float
    log_loss: float
    total_trades: int
    wins: int
    losses: int


class DriftEvaluator:
    """
        Evaluates model performance on resolved OOS trades.

        Calculates win rate, expected value, Brier score, and log loss
    to detect concept drift in the Devil model's calibration.
    """

    def __init__(self, resolved_path: Path):
        """
        Initialize the drift evaluator.

        Args:
            resolved_path: Path to resolved_ledger.csv
        """
        self.resolved_path = resolved_path
        self.data: Optional[pl.DataFrame] = None
        self.metrics: Optional[PerformanceMetrics] = None
        self.notification_manager = NotificationManager()

        self._load_data()

    def _load_data(self) -> None:
        """Load and validate resolved trade data."""
        logger.info("=" * 70)
        logger.info("DRIFT EVALUATOR - PHASE 2")
        logger.info("=" * 70)

        if not self.resolved_path.exists():
            raise FileNotFoundError(f"Resolved ledger not found: {self.resolved_path}")

        logger.info(f"Loading resolved trades from {self.resolved_path}...")
        self.data = pl.read_csv(self.resolved_path)

        # Convert timestamp columns to datetime
        self.data = self.data.with_columns(
            [
                pl.col("timestamp").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f"),
                pl.col("exit_time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f"),
            ]
        )

        logger.info(f"Loaded {len(self.data):,} resolved trades")
        logger.info(f"Symbols: {self.data['symbol'].unique().to_list()}")

    def calculate_win_rate(self) -> Tuple[float, int, int]:
        """
        Calculate win rate from outcomes.

        Returns:
            Tuple of (win_rate, wins, losses)
        """
        outcomes = self.data["outcome"].to_numpy()
        wins = int(np.sum(outcomes == 1))
        losses = int(np.sum(outcomes == 0))
        total = wins + losses

        if total == 0:
            return 0.0, 0, 0

        win_rate = wins / total
        return win_rate, wins, losses

    def calculate_expected_value(self) -> float:
        """
        Calculate expected value per trade.

        Uses +0.5% for wins, -0.2% for losses.

        Returns:
            Average return per trade
        """
        outcomes = self.data["outcome"].to_numpy()
        wins = np.sum(outcomes == 1)
        losses = np.sum(outcomes == 0)
        total = len(outcomes)

        if total == 0:
            return 0.0

        # EV = (wins * TP + losses * SL) / total
        ev = (wins * TAKE_PROFIT - losses * STOP_LOSS) / total
        return ev

    def calculate_brier_score(self) -> float:
        """
        Calculate Brier score for probability calibration.

        Measures how well the Devil's probabilities match actual outcomes.
        Lower is better (perfectly calibrated = 0.0).

        Returns:
            Brier score
        """
        y_true = self.data["outcome"].to_numpy()
        y_prob = self.data["devil_prob"].to_numpy()

        # Ensure we have valid probabilities
        if len(y_prob) == 0 or np.any((y_prob < 0) | (y_prob > 1)):
            logger.warning("Invalid probabilities detected")
            return 1.0

        return brier_score_loss(y_true, y_prob)

    def calculate_log_loss(self) -> float:
        """
        Calculate log loss (cross-entropy) for probability calibration.

        Returns:
            Log loss
        """
        y_true = self.data["outcome"].to_numpy()
        y_prob = self.data["devil_prob"].to_numpy()

        # Clip probabilities to avoid log(0)
        y_prob = np.clip(y_prob, 1e-15, 1 - 1e-15)

        return log_loss(y_true, y_prob)

    def evaluate(self) -> PerformanceMetrics:
        """
        Run full evaluation and calculate all metrics.

        Returns:
            PerformanceMetrics dataclass with all results
        """
        logger.info("=" * 70)
        logger.info("CALCULATING METRICS")
        logger.info("=" * 70)

        # Win Rate
        win_rate, wins, losses = self.calculate_win_rate()
        logger.info(f"Win Rate: {win_rate:.2%} ({wins}/{wins + losses})")

        # Expected Value
        ev = self.calculate_expected_value()
        logger.info(f"Expected Value: {ev:.4f} ({ev * 100:.2f}% per trade)")

        # Brier Score
        brier = self.calculate_brier_score()
        logger.info(f"Brier Score: {brier:.4f} (lower is better)")

        # Log Loss
        ll = self.calculate_log_loss()
        logger.info(f"Log Loss: {ll:.4f}")

        self.metrics = PerformanceMetrics(
            win_rate=win_rate,
            expected_value=ev,
            brier_score=brier,
            log_loss=ll,
            total_trades=wins + losses,
            wins=wins,
            losses=losses,
        )

        return self.metrics

    def check_drift(self) -> Tuple[bool, str]:
        """
        Check if model has drifted beyond critical thresholds.

        Returns:
            Tuple of (is_drifted, reason)
        """
        if self.metrics is None:
            raise RuntimeError("Must run evaluate() before check_drift()")

        drift_reasons = []

        # Check Brier Score
        if self.metrics.brier_score > CRITICAL_BRIER:
            drift_reasons.append(
                f"Brier Score {self.metrics.brier_score:.4f} > {CRITICAL_BRIER}"
            )

        # Check Expected Value
        if self.metrics.expected_value < 0:
            drift_reasons.append(
                f"EV {self.metrics.expected_value:.4f} < 0 (negative returns)"
            )
        elif self.metrics.expected_value < MINIMUM_EV:
            drift_reasons.append(
                f"EV {self.metrics.expected_value:.4f} < {MINIMUM_EV} (below minimum)"
            )

        is_drifted = len(drift_reasons) > 0
        reason = "; ".join(drift_reasons) if is_drifted else "No drift detected"

        return is_drifted, reason

    def print_summary(self) -> None:
        """Print a clean, readable summary of all metrics."""
        if self.metrics is None:
            logger.error("No metrics available. Run evaluate() first.")
            return

        logger.info("=" * 70)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 70)

        # Print metrics table
        print(f"\n{'=' * 70}")
        print(f"{'METRIC':<30} {'VALUE':>20} {'THRESHOLD':>15}")
        print(f"{'=' * 70}")

        # Win Rate
        status = "✅" if self.metrics.win_rate > 0.5 else "⚠️"
        print(f"{status} {'Win Rate':<28} {self.metrics.win_rate:>19.2%} {'> 50%':>15}")

        # Expected Value
        ev_status = "✅" if self.metrics.expected_value >= MINIMUM_EV else "⚠️"
        print(
            f"{ev_status} {'Expected Value':<28} {self.metrics.expected_value:>19.4f} {f'>= {MINIMUM_EV}':>15}"
        )

        # Brier Score
        brier_status = "✅" if self.metrics.brier_score <= CRITICAL_BRIER else "🔴"
        print(
            f"{brier_status} {'Brier Score':<28} {self.metrics.brier_score:>19.4f} {f'<= {CRITICAL_BRIER}':>15}"
        )

        # Log Loss
        print(f"  {'Log Loss':<28} {self.metrics.log_loss:>19.4f} {'N/A':>15}")

        # Trade count
        print(f"{'=' * 70}")
        print(f"  {'Total Trades':<28} {self.metrics.total_trades:>19,}")
        print(f"  {'Wins':<28} {self.metrics.wins:>19,}")
        print(f"  {'Losses':<28} {self.metrics.losses:>19,}")
        print(f"{'=' * 70}\n")

    def trigger_alert(self, reason: str) -> None:
        """
        Send drift alert to Discord via notification manager.

        Args:
            reason: Description of why drift was triggered
        """
        if self.metrics is None:
            return

        metrics_dict = {
            "win_rate": self.metrics.win_rate,
            "expected_value": self.metrics.expected_value,
            "brier_score": self.metrics.brier_score,
            "log_loss": self.metrics.log_loss,
            "total_trades": self.metrics.total_trades,
            "wins": self.metrics.wins,
            "losses": self.metrics.losses,
            "reason": reason,
        }

        logger.error("=" * 70)
        logger.error("🚨 DRIFT ALERT TRIGGERED 🚨")
        logger.error("=" * 70)
        logger.error(f"Reason: {reason}")

        # Send to Discord
        self.notification_manager.send_drift_alert(metrics_dict)

    def run(self) -> int:
        """
        Execute the full drift evaluation pipeline.

        Returns:
            Exit code:
                0 = Healthy (model within parameters)
                1 = Error (execution failure)
                2 = Critical drift detected (triggers retraining)
        """
        try:
            # Calculate metrics
            self.evaluate()

            # Print summary
            self.print_summary()

            # Check for drift
            is_drifted, reason = self.check_drift()

            if is_drifted:
                self.trigger_alert(reason)
                logger.info("=" * 70)
                logger.info("STATUS: 🔴 CRITICAL - Model drift detected")
                logger.info("=" * 70)
                return 2  # Critical drift - triggers retraining
            else:
                logger.info("=" * 70)
                logger.info("STATUS: ✅ HEALTHY - Model within parameters")
                logger.info("=" * 70)
                return 0  # Healthy - pipeline complete

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            return 1  # Error in execution


def main():
    """Main entry point for drift evaluation."""
    try:
        evaluator = DriftEvaluator(RESOLVED_PATH)
        return evaluator.run()

    except FileNotFoundError as e:
        logger.error(f"Missing required file: {e}")
        logger.error("Ensure you have run:")
        logger.error("  1. python -m src.data.harvester")
        logger.error("  2. python -m src.replay_test")
        logger.error("  3. python -m src.core.resolver")
        return 1

    except Exception as e:
        logger.error(f"Drift evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
