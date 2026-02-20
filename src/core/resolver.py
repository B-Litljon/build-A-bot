"""
Trade Resolver - Phase 1 of the Reinforcement Feedback Loop.

Deterministically maps BUY signals to ground-truth Win/Loss outcomes
using strict bracket order simulation (+0.5% TP, -0.2% SL).

Usage:
    python -m src.core.resolver

Inputs:
    - data/signal_ledger.csv (BUY signals from replay_test.py)
    - data/oos_bars.parquet (1-minute historical bars)

Output:
    - data/resolved_ledger.csv (resolved trades with outcomes)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
LEDGER_PATH = Path("data/signal_ledger.csv")
BARS_PATH = Path("data/oos_bars.parquet")
OUTPUT_PATH = Path("data/resolved_ledger.csv")

# Bracket Order Parameters
TP_MULTIPLIER = 1.005  # +0.5%
SL_MULTIPLIER = 0.998  # -0.2%


@dataclass
class TradeOutcome:
    """Represents the resolved outcome of a trade."""

    take_profit_target: float
    stop_loss_target: float
    exit_price: float
    exit_time: datetime
    time_in_trade_mins: int
    outcome: int  # 1 = Win, 0 = Loss


class TradeResolver:
    """
    Resolves BUY signals to Win/Loss outcomes using historical bar data.

    Implements conservative execution logic:
    1. Check Stop Loss first (if low <= sl_price)
    2. If not hit, check Take Profit (elif high >= tp_price)
    3. EOD fallback: Close at final available price if neither hits
    """

    def __init__(self, ledger_path: Path, bars_path: Path):
        """
        Initialize the resolver.

        Args:
            ledger_path: Path to signal_ledger.csv
            bars_path: Path to oos_bars.parquet
        """
        self.ledger_path = ledger_path
        self.bars_path = bars_path
        self.ledger: Optional[pl.DataFrame] = None
        self.bars: Optional[pl.DataFrame] = None

        self._load_data()

    def _load_data(self) -> None:
        """Load ledger and bars data with proper datetime handling."""
        logger.info("=" * 70)
        logger.info("TRADE RESOLVER - PHASE 1")
        logger.info("=" * 70)

        # Load ledger
        if not self.ledger_path.exists():
            raise FileNotFoundError(f"Ledger not found: {self.ledger_path}")

        logger.info(f"Loading ledger from {self.ledger_path}...")
        self.ledger = pl.read_csv(self.ledger_path)

        # Convert timestamp to datetime
        self.ledger = self.ledger.with_columns(
            pl.col("timestamp")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
            .alias("timestamp")
        )

        # Filter to BUY signals only (these are the actual trades)
        self.ledger = self.ledger.filter(pl.col("action") == "BUY")

        logger.info(f"Loaded {len(self.ledger):,} BUY signals")

        # Load bars
        if not self.bars_path.exists():
            raise FileNotFoundError(f"Bars data not found: {self.bars_path}")

        logger.info(f"Loading bars from {self.bars_path}...")
        self.bars = pl.read_parquet(self.bars_path)

        # Ensure timestamp is datetime
        if self.bars["timestamp"].dtype != pl.Datetime:
            self.bars = self.bars.with_columns(pl.col("timestamp").cast(pl.Datetime))

        # Sort bars by symbol and timestamp for efficient lookup
        self.bars = self.bars.sort(["symbol", "timestamp"])

        logger.info(f"Loaded {len(self.bars):,} bars")
        logger.info(f"Symbols: {self.bars['symbol'].unique().to_list()}")

    def _resolve_trade(
        self,
        entry_time: datetime,
        symbol: str,
        entry_price: float,
    ) -> TradeOutcome:
        """
        Resolve a single trade using conservative bracket order logic.

        Args:
            entry_time: Trade entry timestamp
            symbol: Ticker symbol
            entry_price: Entry price

        Returns:
            TradeOutcome with resolved exit details
        """
        # Calculate bracket levels
        tp_price = entry_price * TP_MULTIPLIER
        sl_price = entry_price * SL_MULTIPLIER

        # Filter future bars for this symbol
        symbol_bars = self.bars.filter(
            (pl.col("symbol") == symbol) & (pl.col("timestamp") > entry_time)
        )

        if len(symbol_bars) == 0:
            # No future bars available - treat as immediate close at entry
            logger.warning(f"No future bars for {symbol} after {entry_time}")
            return TradeOutcome(
                take_profit_target=tp_price,
                stop_loss_target=sl_price,
                exit_price=entry_price,
                exit_time=entry_time,
                time_in_trade_mins=0,
                outcome=0,  # Loss (no profit)
            )

        # Iterate through future bars (conservative execution)
        for bar in symbol_bars.iter_rows(named=True):
            bar_low = bar["low"]
            bar_high = bar["high"]
            bar_close = bar["close"]
            bar_time = bar["timestamp"]

            # Check Stop Loss first (conservative)
            if bar_low <= sl_price:
                return TradeOutcome(
                    take_profit_target=tp_price,
                    stop_loss_target=sl_price,
                    exit_price=sl_price,
                    exit_time=bar_time,
                    time_in_trade_mins=self._calculate_duration(entry_time, bar_time),
                    outcome=0,  # Loss
                )

            # Then check Take Profit
            elif bar_high >= tp_price:
                return TradeOutcome(
                    take_profit_target=tp_price,
                    stop_loss_target=sl_price,
                    exit_price=tp_price,
                    exit_time=bar_time,
                    time_in_trade_mins=self._calculate_duration(entry_time, bar_time),
                    outcome=1,  # Win
                )

        # EOD Fallback: Neither TP nor SL hit, close at final price
        final_bar = symbol_bars.tail(1).to_dicts()[0]
        final_price = final_bar["close"]
        final_time = final_bar["timestamp"]

        # Win if final price > entry, Loss otherwise
        outcome = 1 if final_price > entry_price else 0

        return TradeOutcome(
            take_profit_target=tp_price,
            stop_loss_target=sl_price,
            exit_price=final_price,
            exit_time=final_time,
            time_in_trade_mins=self._calculate_duration(entry_time, final_time),
            outcome=outcome,
        )

    @staticmethod
    def _calculate_duration(start: datetime, end: datetime) -> int:
        """Calculate time in trade in minutes."""
        duration = end - start
        return int(duration.total_seconds() / 60)

    def resolve_all(self) -> pl.DataFrame:
        """
        Resolve all trades in the ledger.

        Returns:
            DataFrame with resolved trade outcomes
        """
        logger.info("=" * 70)
        logger.info("RESOLVING TRADES")
        logger.info("=" * 70)
        logger.info(
            f"TP Multiplier: {TP_MULTIPLIER} (+{(TP_MULTIPLIER - 1) * 100:.1f}%)"
        )
        logger.info(
            f"SL Multiplier: {SL_MULTIPLIER} ({(SL_MULTIPLIER - 1) * 100:.1f}%)"
        )

        resolved_rows = []
        total = len(self.ledger)

        for idx, row in enumerate(self.ledger.iter_rows(named=True)):
            entry_time = row["timestamp"]
            symbol = row["symbol"]
            entry_price = row["close_price"]
            angel_prob = row["angel_prob"]
            devil_prob = row["devil_prob"]

            # Resolve the trade
            outcome = self._resolve_trade(entry_time, symbol, entry_price)

            # Build resolved row
            resolved_row = {
                "timestamp": entry_time,
                "symbol": symbol,
                "entry_price": entry_price,
                "angel_prob": angel_prob,
                "devil_prob": devil_prob,
                "action": "BUY",
                "take_profit_target": outcome.take_profit_target,
                "stop_loss_target": outcome.stop_loss_target,
                "exit_price": outcome.exit_price,
                "exit_time": outcome.exit_time,
                "time_in_trade_mins": outcome.time_in_trade_mins,
                "outcome": outcome.outcome,
            }

            resolved_rows.append(resolved_row)

            # Progress logging
            if (idx + 1) % 100 == 0 or idx == total - 1:
                logger.info(f"Resolved {idx + 1:,} / {total:,} trades")

        # Convert to DataFrame
        resolved_df = pl.DataFrame(resolved_rows)

        # Calculate summary statistics
        wins = len(resolved_df.filter(pl.col("outcome") == 1))
        losses = len(resolved_df.filter(pl.col("outcome") == 0))
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        logger.info("=" * 70)
        logger.info("RESOLUTION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total trades: {len(resolved_df):,}")
        logger.info(f"Wins: {wins:,}")
        logger.info(f"Losses: {losses:,}")
        logger.info(f"Win rate: {win_rate:.2%}")
        logger.info(
            f"Avg time in trade: {resolved_df['time_in_trade_mins'].mean():.1f} mins"
        )

        return resolved_df

    def save(self, df: pl.DataFrame, output_path: Path) -> None:
        """
        Save resolved ledger to CSV.

        Args:
            df: Resolved trades DataFrame
            output_path: Path to save CSV
        """
        logger.info(f"Saving resolved ledger to {output_path}...")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to CSV
        df.write_csv(output_path)

        file_size = output_path.stat().st_size / 1024  # KB
        logger.info(f"Saved: {file_size:.2f} KB")


def main():
    """Main entry point for trade resolution."""
    try:
        # Initialize resolver
        resolver = TradeResolver(LEDGER_PATH, BARS_PATH)

        # Resolve all trades
        resolved_df = resolver.resolve_all()

        # Save results
        resolver.save(resolved_df, OUTPUT_PATH)

        logger.info("=" * 70)
        logger.info("TRADE RESOLVER COMPLETE")
        logger.info(f"Output: {OUTPUT_PATH}")
        logger.info("=" * 70)

        return 0

    except FileNotFoundError as e:
        logger.error(f"Missing required file: {e}")
        logger.error("Ensure you have run:")
        logger.error("  1. python -m src.data.harvester")
        logger.error("  2. python -m src.replay_test")
        return 1

    except Exception as e:
        logger.error(f"Resolution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
