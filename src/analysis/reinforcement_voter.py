"""
Reinforcement Voter - Model Drift Detection Across Market Regimes

Analyzes Angel/Devil model performance across ATR-based volatility regimes
to detect calibration drift and recommend safety switches.

Usage:
    python -m src.analysis.reinforcement_voter

Outputs:
    - data/drift_report.json: Regime-specific drift analysis
    - Console: Human-readable drift summary
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
EVALUATION_PATH = Path("data/evaluation_results.parquet")
SIGNAL_LEDGER_PATH = Path("data/signal_ledger.parquet")
BARS_PATH = Path("data/oos_bars.parquet")
OUTPUT_PATH = Path("data/drift_report.json")

# Drift thresholds
CALIBRATION_TOLERANCE = 0.20  # 20% deviation triggers safety switch


@dataclass
class RegimeMetrics:
    """Performance metrics for a single volatility regime."""

    regime: str
    trade_count: int
    actual_win_rate: float
    mean_devil_conviction: float
    calibration_gap: float  # Positive = over-confident, Negative = under-confident
    safety_switch_triggered: bool


@dataclass
class DriftReport:
    """Complete drift analysis report."""

    total_trades: int
    overall_win_rate: float
    overall_mean_conviction: float
    regimes: List[RegimeMetrics]
    safety_switch_recommended: bool
    summary: str


def load_evaluation_data() -> pl.DataFrame:
    """
    Load evaluation results with trade outcomes from Parquet.

    Logic Ledger:
    - Input: evaluation_results.parquet from Phase 3 (contains resolved trades)
    - Schema: timestamp[μs, UTC], symbol, exit_type, pnl_r, devil_prob, atr, etc.
    - Output: DataFrame ready for regime segmentation
    """
    logger.info("Loading evaluation results...")

    if not EVALUATION_PATH.exists():
        raise FileNotFoundError(f"Evaluation data not found: {EVALUATION_PATH}")

    # STRICT SCHEMA: Load from Parquet preserves datetime[μs, UTC] types
    df = pl.read_parquet(EVALUATION_PATH)

    logger.debug(f"Evaluation schema: {df.schema}")

    # Ensure required columns exist
    required_cols = ["devil_prob", "exit_type", "pnl_r"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    logger.info(f"Loaded {len(df):,} evaluated trades")
    return df


def calculate_atr_regimes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Segment trades into ATR-based volatility regimes using pl.cut().

    Logic Ledger:
    - Input: DataFrame with 'atr' column (from evaluation results)
    - Process: Calculate percentiles, create bins using pl.cut()
    - Output: DataFrame with 'atr_regime' column (Low/Normal/High)
    """
    logger.info("Segmenting trades into ATR-based volatility regimes...")

    # Check if ATR column exists
    atr_col = None
    for col in ["atr", "natr_14", "atr_14"]:
        if col in df.columns:
            atr_col = col
            break

    if atr_col is None:
        logger.warning("No ATR column found. Using PnL volatility as proxy.")
        # Use absolute PnL as volatility proxy
        df = df.with_columns([pl.col("pnl_r").abs().alias("volatility_proxy")])
        atr_col = "volatility_proxy"

    # Calculate percentiles for binning
    atr_stats = df.select(
        [
            pl.col(atr_col).quantile(0.33).alias("p33"),
            pl.col(atr_col).quantile(0.67).alias("p67"),
            pl.col(atr_col).max().alias("max_val"),
        ]
    ).to_dict()

    p33 = atr_stats["p33"][0]
    p67 = atr_stats["p67"][0]
    max_val = atr_stats["max_val"][0]

    logger.info(
        f"ATR Regime thresholds - Low: <{p33:.4f}, Normal: {p33:.4f}-{p67:.4f}, High: >{p67:.4f}"
    )

    # Create bins using pl.cut() - breaks are inclusive of left edge
    # Bin edges: [-inf, p33], (p33, p67], (p67, +inf]
    df = df.with_columns(
        [
            pl.col(atr_col)
            .cut(breaks=[p33, p67], labels=["Low", "Normal", "High"])
            .alias("atr_regime")
        ]
    )

    # Log regime distribution
    regime_counts = df.group_by("atr_regime").agg(pl.len().alias("count"))
    for row in regime_counts.iter_rows(named=True):
        logger.info(f"  {row['atr_regime']} volatility: {row['count']:,} trades")

    return df


def analyze_regime_drift(df: pl.DataFrame) -> List[RegimeMetrics]:
    """
    Calculate drift metrics per volatility regime.

    Logic Ledger:
    - Group by atr_regime
    - For each regime: calculate actual win rate vs mean devil conviction
    - Calibration gap = conviction - win rate (positive = over-confident)
    - Safety switch if gap > CALIBRATION_TOLERANCE
    """
    logger.info("Analyzing model drift per regime...")

    regime_metrics = []

    # Define win conditions
    df = df.with_columns(
        [
            pl.when(pl.col("exit_type").is_in(["WIN", "TIME_WIN"]))
            .then(1)
            .otherwise(0)
            .alias("is_win")
        ]
    )

    # Aggregate by regime
    regime_stats = (
        df.group_by("atr_regime")
        .agg(
            [
                pl.len().alias("trade_count"),
                pl.mean("is_win").alias("actual_win_rate"),
                pl.mean("devil_prob").alias("mean_devil_conviction"),
            ]
        )
        .sort("atr_regime")
    )

    for row in regime_stats.iter_rows(named=True):
        regime = row["atr_regime"]
        trade_count = row["trade_count"]
        actual_win_rate = row["actual_win_rate"]
        mean_conviction = row["mean_devil_conviction"]

        # Calibration gap: positive means over-confident
        calibration_gap = mean_conviction - actual_win_rate

        # Safety switch: triggered if model is over-confident beyond tolerance
        safety_switch = calibration_gap > CALIBRATION_TOLERANCE

        metrics = RegimeMetrics(
            regime=regime,
            trade_count=trade_count,
            actual_win_rate=actual_win_rate,
            mean_devil_conviction=mean_conviction,
            calibration_gap=calibration_gap,
            safety_switch_triggered=safety_switch,
        )
        regime_metrics.append(metrics)

        logger.info(
            f"  {regime:>6} | Trades: {trade_count:>4} | "
            f"Win: {actual_win_rate:.1%} | Conviction: {mean_conviction:.1%} | "
            f"Gap: {calibration_gap:+.1%} | {'ALERT' if safety_switch else 'OK'}"
        )

    return regime_metrics


def generate_drift_report(
    df: pl.DataFrame, regime_metrics: List[RegimeMetrics]
) -> DriftReport:
    """Generate comprehensive drift report."""

    # Overall statistics
    total_trades = len(df)

    # Calculate overall win rate
    wins = df.filter(pl.col("exit_type").is_in(["WIN", "TIME_WIN"])).height
    overall_win_rate = wins / total_trades if total_trades > 0 else 0.0

    # Calculate overall mean conviction
    overall_mean_conviction = df.select(pl.mean("devil_prob")).to_series()[0]

    # Determine if any safety switch is triggered
    safety_switch_recommended = any(m.safety_switch_triggered for m in regime_metrics)

    # Generate summary
    if safety_switch_recommended:
        triggered_regimes = [
            m.regime for m in regime_metrics if m.safety_switch_triggered
        ]
        summary = (
            f"SAFETY SWITCH TRIGGERED in {', '.join(triggered_regimes)} volatility regime(s). "
            f"Devil model over-confident by >{CALIBRATION_TOLERANCE:.0%}. Recommend temporary halt."
        )
    else:
        summary = (
            f"Model calibration within tolerance across all regimes. "
            f"Overall win rate: {overall_win_rate:.1%}, Mean conviction: {overall_mean_conviction:.1%}"
        )

    return DriftReport(
        total_trades=total_trades,
        overall_win_rate=overall_win_rate,
        overall_mean_conviction=overall_mean_conviction,
        regimes=regime_metrics,
        safety_switch_recommended=safety_switch_recommended,
        summary=summary,
    )


def save_drift_report(report: DriftReport) -> None:
    """Save drift report to JSON."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Convert dataclasses to dict for JSON serialization
    report_dict = {
        "total_trades": report.total_trades,
        "overall_win_rate": report.overall_win_rate,
        "overall_mean_conviction": report.overall_mean_conviction,
        "safety_switch_recommended": report.safety_switch_recommended,
        "summary": report.summary,
        "regimes": [
            {
                "regime": m.regime,
                "trade_count": m.trade_count,
                "actual_win_rate": m.actual_win_rate,
                "mean_devil_conviction": m.mean_devil_conviction,
                "calibration_gap": m.calibration_gap,
                "safety_switch_triggered": m.safety_switch_triggered,
            }
            for m in report.regimes
        ],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report_dict, f, indent=2)

    logger.info(f"Drift report saved to {OUTPUT_PATH}")


def print_drift_summary(report: DriftReport) -> None:
    """Print formatted drift analysis summary."""
    print("\n" + "=" * 70)
    print("MODEL DRIFT ANALYSIS - VOLATILITY REGIME BREAKDOWN")
    print("=" * 70)

    print(f"\n{'Overall Statistics':<30} {'Value':>35}")
    print("-" * 70)
    print(f"{'Total Trades':<30} {report.total_trades:>35,}")
    print(f"{'Overall Win Rate':<30} {report.overall_win_rate:>34.1%}")
    print(f"{'Mean Devil Conviction':<30} {report.overall_mean_conviction:>34.1%}")

    print(
        f"\n{'Regime':<10} {'Trades':<8} {'Win Rate':<10} {'Conviction':<12} {'Gap':<10} {'Status':<8}"
    )
    print("-" * 70)

    for m in report.regimes:
        status = "ALERT" if m.safety_switch_triggered else "OK"
        print(
            f"{m.regime:<10} {m.trade_count:<8} {m.actual_win_rate:<10.1%} "
            f"{m.mean_devil_conviction:<12.1%} {m.calibration_gap:<+10.1%} {status:<8}"
        )

    print("=" * 70)

    if report.safety_switch_recommended:
        print(f"\n STATUS: 🔴 {report.summary}")
    else:
        print(f"\n STATUS: ✅ {report.summary}")

    print("=" * 70 + "\n")


def main() -> int:
    """
    Main entry point for reinforcement voter.

    Logic Ledger:
    1. Load evaluation_results.parquet (Phase 3 output with resolved trades)
    2. Segment trades into ATR-based volatility regimes using pl.cut()
    3. Calculate drift: Devil conviction vs actual win rate per regime
    4. Trigger safety switch if calibration gap > 20%
    5. Output drift_report.json for pipeline orchestration
    """
    try:
        logger.info("=" * 70)
        logger.info("REINFORCEMENT VOTER - REGIME DRIFT ANALYSIS")
        logger.info("=" * 70)
        logger.info("STRICT SCHEMA MODE: Loading Parquet with preserved datetime types")

        # Load evaluation data (Parquet preserves schema including datetime[μs, UTC])
        df = load_evaluation_data()

        # Segment into ATR-based volatility regimes
        df = calculate_atr_regimes(df)

        # Analyze model drift per regime
        regime_metrics = analyze_regime_drift(df)

        # Generate comprehensive drift report
        report = generate_drift_report(df, regime_metrics)

        # Save JSON report and print console summary
        save_drift_report(report)
        print_drift_summary(report)

        logger.info("Analysis complete.")

        # Exit code protocol for pipeline integration:
        # 0 = Healthy (no drift detected)
        # 1 = Error
        # 2 = Safety switch triggered (critical drift - recommend halt)
        return 2 if report.safety_switch_recommended else 0

    except FileNotFoundError as e:
        logger.error(f"Missing required file: {e}")
        logger.error("Pipeline execution order:")
        logger.error("  1. python -m src.data.harvester")
        logger.error("  2. python -m src.replay_test")
        logger.error("  3. python -m src.evaluate_performance")
        logger.error("  4. python -m src.analysis.reinforcement_voter")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
