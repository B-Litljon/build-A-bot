#!/usr/bin/env python3
"""Factory Ignition Script — Local paper trading entry point.

Executes the SDK-decoupled trading bot on Alpaca paper trading with crypto pairs.
"""

import sys
import os

# Inject src/ into path before any other imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import asyncio
import logging

from dotenv import load_dotenv

from execution.risk_manager import RiskManager
from strategies.concrete_strategies.ml_factory_strategy import MLFactoryStrategy
from execution.factory_orchestrator import FactoryOrchestrator


# Simple MarketDataFeed implementation for Alpaca crypto
try:
    from alpaca.data.live.crypto import CryptoDataStream
    from alpaca.data.historical.crypto import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    class AlpacaCryptoFeed:
        """Alpaca crypto WebSocket feed implementing MarketDataFeed interface."""

        def __init__(self, api_key: str, secret_key: str, symbols: list[str]) -> None:
            self.api_key = api_key
            self.secret_key = secret_key
            self.symbols = symbols
            self._stream: CryptoDataStream | None = None

        async def run(self, callback: callable) -> None:
            """Start WebSocket stream and invoke callback for each bar."""
            self._stream = CryptoDataStream(self.api_key, self.secret_key)

            async def _on_bar(bar) -> None:
                bar_dict = {
                    "symbol": bar.symbol,
                    "timestamp": bar.timestamp,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
                await callback(bar_dict)

            self._stream.subscribe_bars(_on_bar, *self.symbols)
            await self._stream._run_forever()

except ImportError:
    # Fallback mock for development without Alpaca SDK
    class AlpacaCryptoFeed:
        """Mock crypto feed for local development."""

        def __init__(self, api_key: str, secret_key: str, symbols: list[str]) -> None:
            self.symbols = symbols
            logger.warning("Alpaca SDK not installed - using mock feed")

        async def run(self, callback: callable) -> None:
            """Mock stream with synthetic bars."""
            from datetime import datetime, timezone
            import random

            logger.info("Starting mock crypto feed for symbols: %s", self.symbols)

            base_prices = {"BTC/USD": 45000.0, "ETH/USD": 3000.0}

            while True:
                await asyncio.sleep(5)  # 5-second bar interval for testing

                for symbol in self.symbols:
                    base = base_prices.get(symbol, 100.0)
                    price = base * (1 + random.uniform(-0.001, 0.001))
                    base_prices[symbol] = price

                    bar_dict = {
                        "symbol": symbol,
                        "timestamp": datetime.now(timezone.utc),
                        "open": price * 0.999,
                        "high": price * 1.001,
                        "low": price * 0.998,
                        "close": price,
                        "volume": random.uniform(0.1, 1.0),
                    }
                    await callback(bar_dict)


def setup_logging() -> None:
    """Configure root logger to INFO level with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


async def main() -> None:
    """Main execution entry point."""
    load_dotenv()
    setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("=== Build-A-Bot Factory SDK | Paper Trading Mode ===")

    # Alpaca credentials from environment
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

    if not api_key or not secret_key:
        logger.warning("ALPACA_API_KEY or ALPACA_SECRET_KEY not set - using mock mode")

    # Trading symbols (crypto pairs)
    symbols = ["BTC/USD", "ETH/USD"]

    # Instantiate SDK components
    feed = AlpacaCryptoFeed(api_key, secret_key, symbols)
    strategy = MLFactoryStrategy()
    risk_manager = RiskManager(atr_multiplier=0.5)

    # Wire components into orchestrator
    orchestrator = FactoryOrchestrator(
        feed=feed,
        strategy=strategy,
        risk_manager=risk_manager,
        symbols=symbols,
        paper=True,
    )

    logger.info("Starting orchestrator - press Ctrl+C to exit")

    try:
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received - shutting down gracefully")
    except Exception as exc:
        logger.error("Fatal error: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    asyncio.run(main())
