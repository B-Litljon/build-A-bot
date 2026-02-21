"""
Performance Evaluator - Phase 3: Dynamic ATR + Time-Decay Backtest.

Vectorized evaluation of trading signals using Polars.
Calculates Win Rate, Net Profit, Max Drawdown, and Profit Factor.

Usage:
    python -m src.evaluate_performance

Inputs:
    - data/oos_bars.parquet: 1-minute OHLCV + ATR data
    - data/signal_ledger.csv: Trading signals

Exit Logic:
    - Entry: Close price of signal bar
    - SL: Entry - (1.5 * ATR)
    - TP: Entry + (3.0 * ATR)
    - Max Hold: 15 bars
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
BARS_PATH = Path("data/oos_bars.parquet")
LEDGER_PATH = Path("data/signal_ledger.parquet")
OUTPUT_PATH = Path("data/evaluation_results.parquet")

# Exit Parameters
SL_MULTIPLIER = 1.5
TP_MULTIPLIER = 3.0
MAX_HOLD_BARS = 15


@dataclass
class PerformanceMetrics:
    """Container for backtest performance metrics."""

    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_net_profit: float  # In R-multiples
    max_drawdown: float
    profit_factor: float
    avg_trade: float
    avg_winner: float
    avg_loser: float


def load_data() -> tuple[pl.DataFrame, pl.DataFrame, str]:
    """Load price bars and signal ledger."""
    logger.info("=" * 70)
    logger.info("PHASE 3: PERFORMANCE EVALUATION")
    logger.info("=" * 70)

    if not BARS_PATH.exists():
        raise FileNotFoundError(f"Bars data not found: {BARS_PATH}")
    if not LEDGER_PATH.exists():
        raise FileNotFoundError(f"Signal ledger not found: {LEDGER_PATH}")

    # Load bars
    logger.info(f"Loading bars from {BARS_PATH}...")
    bars_df = pl.read_parquet(BARS_PATH)

    # Ensure required columns exist
    required_cols = ["timestamp", "symbol", "open", "high", "low", "close"]
    missing_cols = [c for c in required_cols if c not in bars_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in bars: {missing_cols}")

    # Check for ATR column (try natr_14 first, then atr, then calculate)
    atr_col = None
    for col in ["natr_14", "atr_14", "atr"]:
        if col in bars_df.columns:
            atr_col = col
            break

    if atr_col is None:
        logger.warning("No ATR column found, calculating from high/low/close...")
        # Simple ATR calculation using 14-period average true range
        bars_df = bars_df.with_columns(
            [(pl.col("high") - pl.col("low")).alias("range")]
        )
        bars_df = bars_df.with_columns(
            [pl.col("range").rolling_mean(window_size=14).alias("atr_calc")]
        )
        atr_col = "atr_calc"

    logger.info(f"Using ATR column: {atr_col}")
    logger.info(f"Loaded {len(bars_df):,} bars")

    # Load signals from Parquet (strict schema preservation)
    logger.info(f"Loading signals from {LEDGER_PATH}...")
    signals_df = pl.read_parquet(LEDGER_PATH)

    # Only keep essential columns
    essential_cols = [
        "timestamp",
        "symbol",
        "close_price",
        "angel_prob",
        "devil_prob",
        "action",
    ]
    available_cols = [c for c in essential_cols if c in signals_df.columns]
    signals_df = signals_df.select(available_cols)

    # STRICT SCHEMA: Timestamps already datetime[μs, UTC] from Parquet
    # No string parsing required - schema matches oos_bars.parquet exactly
    logger.debug(f"Signals schema: {signals_df.schema}")

    # Filter to BUY signals only
    if "action" in signals_df.columns:
        signals_df = signals_df.filter(pl.col("action") == "BUY")

    logger.info(f"Loaded {len(signals_df):,} BUY signals")

    return bars_df, signals_df, atr_col


def vectorized_backtest(
    bars_df: pl.DataFrame, signals_df: pl.DataFrame, atr_col: str
) -> pl.DataFrame:
    """
    Perform fully vectorized backtest using Polars.

    For each signal, looks ahead up to MAX_HOLD_BARS to find first exit.
    """
    logger.info("=" * 70)
    logger.info("RUNNING VECTORIZED BACKTEST")
    logger.info("=" * 70)

    # Ensure bars are sorted by symbol and timestamp
    bars_df = bars_df.sort(["symbol", "timestamp"])

    # Join signals with bars to get entry context
    # Create entry DataFrame with signal + entry bar data
    entries = signals_df.join(
        bars_df.select(["timestamp", "symbol", "close", "high", "low", atr_col]),
        on=["timestamp", "symbol"],
        how="left",
    ).rename(
        {
            "close": "entry_price",
            atr_col: "atr",
            "high": "entry_high",
            "low": "entry_low",
        }
    )

    # Calculate SL and TP levels
    entries = entries.with_columns(
        [
            (pl.col("entry_price") - (SL_MULTIPLIER * pl.col("atr"))).alias("sl_price"),
            (pl.col("entry_price") + (TP_MULTIPLIER * pl.col("atr"))).alias("tp_price"),
        ]
    )

    logger.info(f"Processing {len(entries):,} entries with ATR-based exits...")

    # Create lookahead windows using shift operations
    # We'll build a matrix of future bars for each entry
    results = []

    # Process each symbol separately to avoid cross-symbol contamination
    for symbol in entries["symbol"].unique().to_list():
        symbol_entries = entries.filter(pl.col("symbol") == symbol)
        symbol_bars = bars_df.filter(pl.col("symbol") == symbol).sort("timestamp")

        if len(symbol_bars) == 0:
            continue

        # For each entry, we need to look ahead up to MAX_HOLD_BARS
        # Vectorized approach: create shifted columns for future highs/lows
        for i in range(1, MAX_HOLD_BARS + 1):
            symbol_bars = symbol_bars.with_columns(
                [
                    pl.col("high").shift(-i).alias(f"future_high_{i}"),
                    pl.col("low").shift(-i).alias(f"future_low_{i}"),
                    pl.col("close").shift(-i).alias(f"future_close_{i}"),
                ]
            )

        # Join entries with bars to get lookahead context
        symbol_results = symbol_entries.join(
            symbol_bars.drop(["symbol"]),  # Drop symbol to avoid duplicate
            left_on="timestamp",
            right_on="timestamp",
            how="left",
        )

        results.append(symbol_results)

    if not results:
        raise ValueError("No results generated from backtest")

    # Combine all symbol results
    all_results = pl.concat(results, how="vertical_relaxed")

    # Vectorized exit detection
    # For each row, find the first bar where high >= TP or low <= SL
    exit_bars = []
    exit_prices = []
    exit_types = []

    # Process each trade to find exit
    for row in all_results.iter_rows(named=True):
        entry_price = row["entry_price"]
        sl_price = row["sl_price"]
        tp_price = row["tp_price"]

        exit_bar = None
        exit_price = None
        exit_type = None

        for i in range(1, MAX_HOLD_BARS + 1):
            future_high = row.get(f"future_high_{i}")
            future_low = row.get(f"future_low_{i}")
            future_close = row.get(f"future_close_{i}")

            if future_high is None or future_low is None:
                # Ran out of data
                break

            # Check if TP hit (Win)
            if future_high >= tp_price:
                exit_bar = i
                exit_price = tp_price
                exit_type = "WIN"
                break

            # Check if SL hit (Loss)
            if future_low <= sl_price:
                exit_bar = i
                exit_price = sl_price
                exit_type = "LOSS"
                break

        # If no exit found within MAX_HOLD_BARS, exit at last bar
        if exit_bar is None:
            exit_bar = MAX_HOLD_BARS
            future_close = row.get(f"future_close_{MAX_HOLD_BARS}")
            exit_price = future_close if future_close is not None else entry_price

            # Determine if time exit is win or loss
            if exit_price > entry_price:
                exit_type = "TIME_WIN"
            else:
                exit_type = "TIME_LOSS"

        exit_bars.append(exit_bar)
        exit_prices.append(exit_price)
        exit_types.append(exit_type)

    # Add exit information to results
    # Step 1: Attach raw exit data as new columns
    all_results = all_results.with_columns(
        [
            pl.Series("exit_bar", exit_bars),
            pl.Series("exit_price", exit_prices),
            pl.Series("exit_type", exit_types),
        ]
    )

    # Step 2: Compute derived P&L columns using pl.col() references
    # (exit_price must exist as a column before it can be referenced)
    all_results = all_results.with_columns(
        [
            (pl.col("exit_price") - pl.col("entry_price")).alias("pnl"),
            ((pl.col("exit_price") - pl.col("entry_price")) / pl.col("atr")).alias(
                "pnl_r"
            ),
        ]
    )

    # Clean up - drop future columns
    future_cols = [c for c in all_results.columns if c.startswith("future_")]
    all_results = all_results.drop(future_cols)

    return all_results


def calculate_metrics(trades_df: pl.DataFrame) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics."""

    total_trades = len(trades_df)

    if total_trades == 0:
        return PerformanceMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # Win/Loss classification
    wins_df = trades_df.filter(pl.col("exit_type").is_in(["WIN", "TIME_WIN"]))
    losses_df = trades_df.filter(pl.col("exit_type").is_in(["LOSS", "TIME_LOSS"]))

    wins = len(wins_df)
    losses = len(losses_df)
    win_rate = wins / total_trades if total_trades > 0 else 0.0

    # P&L metrics (in R-multiples)
    pnl_r = trades_df["pnl_r"].to_numpy()
    total_net_profit = float(np.sum(pnl_r))
    avg_trade = float(np.mean(pnl_r))

    if wins > 0:
        avg_winner = float(np.mean(wins_df["pnl_r"].to_numpy()))
    else:
        avg_winner = 0.0

    if losses > 0:
        avg_loser = float(np.mean(losses_df["pnl_r"].to_numpy()))
    else:
        avg_loser = 0.0

    # Profit Factor
    gross_profit = float(np.sum(np.maximum(pnl_r, 0)))
    gross_loss = abs(float(np.sum(np.minimum(pnl_r, 0))))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max Drawdown (cumulative P&L approach)
    cumulative = np.cumsum(pnl_r)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return PerformanceMetrics(
        total_trades=total_trades,
        wins=wins,
        losses=losses,
        win_rate=win_rate,
        total_net_profit=total_net_profit,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        avg_trade=avg_trade,
        avg_winner=avg_winner,
        avg_loser=avg_loser,
    )


def print_summary(metrics: PerformanceMetrics, trades_df: pl.DataFrame) -> None:
    """Print formatted performance summary."""

    logger.info("=" * 70)
    logger.info("BACKTEST RESULTS SUMMARY")
    logger.info("=" * 70)

    print(f"\n{'=' * 70}")
    print(f"{'METRIC':<30} {'VALUE':>35}")
    print(f"{'=' * 70}")

    print(f"{'Total Trades':<30} {metrics.total_trades:>35,}")
    print(f"{'Win Rate':<30} {metrics.win_rate:>34.1%}")
    print(f"{'Wins':<30} {metrics.wins:>35,}")
    print(f"{'Losses':<30} {metrics.losses:>35,}")
    print(f"{'=' * 70}")

    print(f"{'Total Net Profit (R)':<30} {metrics.total_net_profit:>+35.2f}")
    print(f"{'Avg Trade (R)':<30} {metrics.avg_trade:>+35.2f}")
    print(f"{'Avg Winner (R)':<30} {metrics.avg_winner:>+35.2f}")
    print(f"{'Avg Loser (R)':<30} {metrics.avg_loser:>+35.2f}")
    print(f"{'=' * 70}")

    print(f"{'Max Drawdown (R)':<30} {metrics.max_drawdown:>35.2f}")
    print(f"{'Profit Factor':<30} {metrics.profit_factor:>35.2f}")

    # Exit type breakdown
    if len(trades_df) > 0:
        print(f"\n{'=' * 70}")
        print(f"{'EXIT TYPE BREAKDOWN':^70}")
        print(f"{'=' * 70}")

        exit_breakdown = (
            trades_df.group_by("exit_type")
            .agg(pl.count().alias("count"), pl.mean("pnl_r").alias("avg_pnl_r"))
            .sort("count", descending=True)
        )

        for row in exit_breakdown.iter_rows(named=True):
            print(
                f"{row['exit_type']:<30} {row['count']:>20,} ({row['avg_pnl_r']:>+10.2f} R)"
            )

    print(f"{'=' * 70}\n")

    # Health assessment
    if metrics.win_rate > 0.5 and metrics.total_net_profit > 0:
        logger.info("✅ STRATEGY HEALTH: PROFITABLE")
    elif metrics.total_net_profit > 0:
        logger.info("⚠️  STRATEGY HEALTH: PROFITABLE BUT LOW WIN RATE")
    else:
        logger.info("🔴 STRATEGY HEALTH: UNPROFITABLE - REVIEW NEEDED")


def main() -> int:
    """Main entry point for performance evaluation."""
    try:
        # Load data
        bars_df, signals_df, atr_col = load_data()

        # Run vectorized backtest
        trades_df = vectorized_backtest(bars_df, signals_df, atr_col)

        # Calculate metrics
        metrics = calculate_metrics(trades_df)

        # Print summary
        print_summary(metrics, trades_df)

        # Save detailed results to Parquet (preserves schema for downstream analysis)
        trades_df.write_parquet(OUTPUT_PATH)
        logger.info(f"Detailed results saved to {OUTPUT_PATH}")

        logger.info("=" * 70)
        logger.info("PHASE 3 COMPLETE")
        logger.info("=" * 70)

        return 0

    except FileNotFoundError as e:
        logger.error(f"Missing required file: {e}")
        logger.error("Ensure you have run:")
        logger.error("  1. python -m src.data.harvester")
        logger.error("  2. python -m src.replay_test")
        return 1

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
