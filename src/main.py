#!/usr/bin/env python3
"""
Build-A-Bot: Universal Scalper Executive
Transitioned from Static Sniper to Universal Hunter (Multi-Ticker).
Integrates DiscoveryService and Angel/Devil Meta-Labeling.
"""

import os
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict

from dotenv import load_dotenv

# Correct pathing: Ensure the 'src' directory is the root for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import polars as pl
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce

from core.signal import SignalType
from strategies.concrete_strategies.ml_strategy import MLStrategy
from data.alpaca_provider import AlpacaProvider
from data.discovery import DiscoveryService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── CONFIGURATION ───────────────────────────────────────────────────────────
MAX_BASKET_SIZE = 10
TIMEFRAME_MINUTES = 1
PAPER_MODE = True
HISTORY_BARS = 100

# Risk Parameters
TP_PERCENT = 0.005  # 0.5%
SL_PERCENT = 0.002  # 0.2%


def load_credentials() -> tuple[str, str]:
    load_dotenv()
    api_key = os.getenv("ALPACA_API_KEY") or os.getenv("alpaca_key")
    secret_key = os.getenv("ALPACA_SECRET_KEY") or os.getenv("alpaca_secret")
    if not api_key or not secret_key:
        raise ValueError("Alpaca credentials missing.")
    return api_key, secret_key


def wait_for_next_minute() -> None:
    now = datetime.now(timezone.utc)
    next_minute = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    sleep_seconds = (next_minute - now).total_seconds()
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)


def get_multi_historical_data(
    provider: AlpacaProvider, symbols: List[str], limit: int = 100
) -> Dict[str, pl.DataFrame]:
    """Fetch recent historical bars for the entire basket."""
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=limit * 2)

    data_dict = {}
    for symbol in symbols:
        df = provider.get_historical_bars(symbol, TIMEFRAME_MINUTES, start, end)
        if not df.is_empty():
            data_dict[symbol] = df.tail(limit)
    return data_dict


def check_position(trading_client: TradingClient, symbol: str) -> bool:
    try:
        position = trading_client.get_open_position(symbol)
        return position is not None
    except Exception:
        return False


def submit_bracket_order(
    trading_client: TradingClient, symbol: str, qty: float, entry_price: float
) -> Optional[str]:
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
        logger.info(
            f"🚀 ENTRY: {symbol} @ ${entry_price:.2f} (TP: ${tp_price} | SL: ${sl_price})"
        )
        return order.id
    except Exception as e:
        logger.error(f"Failed to submit order for {symbol}: {e}")
        return None


def run_live_loop(
    strategy: MLStrategy,
    provider: AlpacaProvider,
    trading_client: TradingClient,
    active_basket: List[str],
) -> None:
    logger.info("=" * 60)
    logger.info("UNIVERSAL HUNTER LOOP STARTED")
    logger.info(f"Basket: {active_basket}")
    logger.info(
        f"Risk: TP {TP_PERCENT * 100:.1f}% / SL {SL_PERCENT * 100:.1f}% | Mode: PAPER"
    )
    logger.info("=" * 60)

    cycle_count = 0
    while True:
        try:
            cycle_count += 1
            wait_for_next_minute()

            # 1. Bulk Fetch
            data_dict = get_multi_historical_data(provider, active_basket, HISTORY_BARS)

            # 2. Iterate Basket
            for symbol in active_basket:
                df = data_dict.get(symbol)
                if df is None or len(df) < strategy.warmup_period:
                    if cycle_count % 10 == 0:
                        bar_count = len(df) if df is not None else 0
                        logger.warning(
                            f"[{symbol}] Insufficient data: {bar_count}/{strategy.warmup_period} bars. Skipping."
                        )
                    continue

                # 3. Analyze (Angel & Devil)
                result = strategy.analyze({symbol: df})
                signals, prob = result if isinstance(result, tuple) else (result, 0.0)
                current_price = float(df["close"].tail(1)[0])

                if signals and signals[0].type == SignalType.BUY:
                    if check_position(trading_client, symbol):
                        continue

                    account = trading_client.get_account()
                    qty = round((float(account.buying_power) * 0.02) / current_price, 3)

                    if qty >= 0.001:
                        submit_bracket_order(trading_client, symbol, qty, current_price)

                elif cycle_count % 10 == 0:
                    logger.info(
                        f"💤 {symbol} | Price: ${current_price:.2f} | Angel Prob: {prob:.2f}"
                    )

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Loop error: {e}")
            time.sleep(5)


def main() -> None:
    logger.info("Build-A-Bot Universal Scalper Initialization")
    try:
        api_key, secret_key = load_credentials()

        # Discovery Phase
        discovery = DiscoveryService(api_key, secret_key, paper=PAPER_MODE)
        active_basket = discovery.get_in_play_tickers(top_n=MAX_BASKET_SIZE)
        if not active_basket:
            logger.warning("No tickers found, defaulting to SPY/QQQ")
            active_basket = ["BTC/USD"]

        # Strategy Init (Angel/Devil)
        strategy = MLStrategy(
            angel_path="src/ml/models/angel_rf_model.joblib",
            devil_path="src/ml/models/devil_rf_model.joblib",
            angel_threshold=0.40,
            devil_threshold=0.50,
            warmup_period=60,
        )

        provider = AlpacaProvider(api_key, secret_key, paper=PAPER_MODE)
        trading_client = TradingClient(api_key, secret_key, paper=PAPER_MODE)

        # TEMPORARY OVERRIDE: Force Crypto to verify live execution after-hours
        logger.info(
            "Override active: Switching basket to Crypto for 24/7 data verification."
        )
        active_basket = ["BTC/USD", "ETH/USD"]

        run_live_loop(strategy, provider, trading_client, active_basket)

    except Exception as e:
        logger.critical(f"Fatal: {e}", exc_info=True)


if __name__ == "__main__":
    main()
