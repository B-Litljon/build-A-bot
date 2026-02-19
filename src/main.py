#!/usr/bin/env python3
"""
Build-A-Bot: ML Strategy Paper Trading Executive

Live trading loop for the optimized ML Strategy (Threshold 0.48).
Uses Alpaca Paper Trading with bracket orders for automatic TP/SL.

Environment Variables:
    ALPACA_API_KEY: Alpaca API key
    ALPACA_SECRET_KEY: Alpaca API secret

Usage:
    python src/main.py
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from dotenv import load_dotenv

# Ensure src is in the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import polars as pl
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce

from strategies.concrete_strategies.ml_strategy import MLStrategy
from data.alpaca_provider import AlpacaProvider
from core.signal import SignalType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── CONFIGURATION ───────────────────────────────────────────────────────────
SYMBOL = "SPY"
TIMEFRAME_MINUTES = 1
PAPER_MODE = True
HISTORY_BARS = 100

# Risk Parameters (from optimization)
TP_PERCENT = 0.005  # 0.5%
SL_PERCENT = 0.002  # 0.2%


def load_credentials() -> tuple[str, str]:
    """Load Alpaca API credentials from environment."""
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("alpaca_key")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("alpaca_secret")

    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY "
            "in your .env file or environment."
        )

    return api_key, secret_key


def wait_for_next_minute() -> None:
    """Wait until the start of the next minute to prevent drift."""
    now = datetime.now(timezone.utc)
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    sleep_seconds = (next_minute - now).total_seconds()

    if sleep_seconds > 0:
        logger.debug(f"Syncing... waiting {sleep_seconds:.1f}s for next minute")
        time.sleep(sleep_seconds)


def get_historical_data(
    provider: AlpacaProvider, symbol: str, limit: int = 100
) -> pl.DataFrame:
    """Fetch recent historical bars for strategy warmup."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=limit * 2)  # Buffer for weekends/holidays

    df = provider.get_historical_bars(symbol, TIMEFRAME_MINUTES, start, end)

    if len(df) < 60:
        logger.warning(f"Only {len(df)} bars retrieved (need 60 for warmup)")

    return df.tail(limit)


def check_position(trading_client: TradingClient, symbol: str) -> bool:
    """Check if we already hold a position in the symbol."""
    try:
        position = trading_client.get_open_position(symbol)
        return position is not None
    except Exception:
        return False


def submit_bracket_order(
    trading_client: TradingClient, symbol: str, qty: float, entry_price: float
) -> Optional[str]:
    """
    Submit a bracket order: Market Buy + TP @ +0.5% + SL @ -0.2%

    Returns order ID if successful, None otherwise.
    """
    try:
        tp_price = round(entry_price * (1 + TP_PERCENT), 2)
        sl_price = round(entry_price * (1 - SL_PERCENT), 2)

        order_request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            take_profit=TakeProfitRequest(limit_price=tp_price),
            stop_loss=StopLossRequest(stop_price=sl_price),
        )

        order = trading_client.submit_order(order_request)
        logger.info(f"🚀 SNIPER ENTRY: Bought {symbol} @ ${entry_price:.2f}")
        logger.info(
            f"   Bracket: TP ${tp_price:.2f} (+{TP_PERCENT * 100:.1f}%) | SL ${sl_price:.2f} (-{SL_PERCENT * 100:.1f}%)"
        )

        return order.id

    except Exception as e:
        logger.error(f"Failed to submit bracket order: {e}")
        return None


def run_live_loop(
    strategy: MLStrategy, provider: AlpacaProvider, trading_client: TradingClient
) -> None:
    """
    Main trading loop.

    Runs indefinitely until interrupted.
    """
    logger.info("=" * 60)
    logger.info("LIVE TRADING LOOP STARTED")
    logger.info(f"Symbol: {SYMBOL} | Angel/Devil Mode")
    logger.info(f"Risk: TP {TP_PERCENT * 100:.1f}% / SL {SL_PERCENT * 100:.1f}%")
    logger.info(f"Mode: {'PAPER' if PAPER_MODE else 'LIVE'}")
    logger.info("=" * 60)

    cycle_count = 0

    while True:
        try:
            cycle_count += 1

            # Step 1: Sync to next minute
            wait_for_next_minute()

            # Step 2: Fetch historical data
            df = get_historical_data(provider, SYMBOL, HISTORY_BARS)

            if len(df) < strategy.warmup_period:
                logger.warning(f"Insufficient data ({len(df)} bars), skipping cycle...")
                continue

            # Step 3: Generate signal
            result = strategy.analyze({SYMBOL: df})

            # Handle tuple return (signals, probability) or list return
            if isinstance(result, tuple):
                signals, prob = result
            else:
                signals = result
                prob = 0.0

            # Get current price for logging
            current_price = float(df["close"].tail(1)[0])

            # Step 4: Execute
            if signals and signals[0].type == SignalType.BUY:
                logger.info(
                    f"📊 SIGNAL DETECTED: {SYMBOL} @ ${current_price:.2f} | Prob: {prob:.2f}"
                )

                # Check if already in position
                if check_position(trading_client, SYMBOL):
                    logger.info("   Position already open, skipping...")
                    continue

                # Calculate position size (2% risk)
                account = trading_client.get_account()
                buying_power = float(account.buying_power)
                position_value = buying_power * 0.02  # Risk 2% of buying power
                qty = round(position_value / current_price, 3)

                if qty < 0.001:
                    logger.warning(f"   Calculated qty too small ({qty}), skipping...")
                    continue

                # Submit bracket order
                order_id = submit_bracket_order(
                    trading_client, SYMBOL, qty, current_price
                )

                if order_id:
                    logger.info(f"   Order submitted: {order_id}")

            else:
                # Low verbosity log for HOLD with probability
                if cycle_count % 10 == 0:  # Log every 10 cycles
                    logger.info(
                        f"💤 HOLD | Price: ${current_price:.2f} | Prob: {prob:.2f} (Angel: {strategy.angel_threshold}) | Status: Waiting..."
                    )
                else:
                    logger.debug(
                        f"   Values: Prob {prob:.2f} < {strategy.angel_threshold} | Action: Waiting..."
                    )

        except KeyboardInterrupt:
            logger.info("=" * 60)
            logger.info("Bot stopped by user (KeyboardInterrupt)")
            logger.info("=" * 60)
            break

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=False)
            logger.info("Retrying in 5 seconds...")
            time.sleep(5)


def main() -> None:
    """
    Entry point: Initialize components and start live trading loop.
    """
    logger.info("=" * 60)
    logger.info("Build-A-Bot ML Strategy - Paper Trading")
    logger.info("=" * 60)

    try:
        # Load credentials
        api_key, secret_key = load_credentials()
        logger.info("✓ Credentials loaded")

        # Initialize strategy (Meta-Labeling dual-model architecture)
        strategy = MLStrategy(
            angel_path="src/ml/models/angel_rf_model.joblib",
            devil_path="src/ml/models/devil_rf_model.joblib",
            angel_threshold=0.40,
            devil_threshold=0.50,
            warmup_period=60,
        )
        logger.info(f"✓ Strategy initialized (Angel/Devil Mode)")

        # Initialize data provider
        provider = AlpacaProvider(api_key, secret_key, paper=PAPER_MODE)
        logger.info(f"✓ Data provider connected (Alpaca)")

        # Initialize trading client
        trading_client = TradingClient(api_key, secret_key, paper=PAPER_MODE)
        account = trading_client.get_account()
        logger.info(f"✓ Trading client ready")
        logger.info(f"   Account: ${float(account.equity):,.2f} equity")
        logger.info(f"   Buying Power: ${float(account.buying_power):,.2f}")

        # Start live loop
        run_live_loop(strategy, provider, trading_client)

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
