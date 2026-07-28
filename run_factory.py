"""
Launcher for the Factory path (Alpaca crypto) -- FactoryOrchestrator.

Root-level entry point. Its real job is the sys.path bootstrap: it prepends
src/ so that bare module names ("execution.factory_orchestrator") resolve, then
assembles the four pieces and hands control to the orchestrator.

The assembly is the useful thing to read -- it shows the whole dependency shape
of a running bot in one place:

    AlpacaCryptoFeed   (where bars come from)
    MLFactoryStrategy  (what decides)
    RiskManager        (what sizes and vetoes)
    FactoryOrchestrator(what sequences and executes)

Glossary:
    _SRC_DIR -- the src/ directory, inserted at the front of sys.path. This is
        why modules elsewhere import as "data.feed" rather than "src.data.feed";
        the entry point decides which convention holds.
    load_dotenv -- reads the .env file, which is where broker credentials live.
    main() -- the async entry point; constructs the four objects above and
        awaits the orchestrator's run loop.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Path bootstrap
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from execution.factory_orchestrator import FactoryOrchestrator
from strategies.concrete_strategies.ml_factory_strategy import MLFactoryStrategy
from execution.risk_manager import RiskManager
from data.feed import AlpacaCryptoFeed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

async def main():
    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    if not api_key or not secret_key:
        print("Error: ALPACA_API_KEY or ALPACA_SECRET_KEY not set.")
        return

    # Initialize components
    strategy = MLFactoryStrategy()
    risk_manager = RiskManager()
    feed = AlpacaCryptoFeed(api_key, secret_key)

    symbols = ["BTC/USD", "ETH/USD"]

    orchestrator = FactoryOrchestrator(
        symbols=symbols,
        api_key=api_key,
        secret_key=secret_key,
        strategy=strategy,
        risk_manager=risk_manager,
        feed=feed,
        paper=True
    )

    print("--- Build-A-Bot Factory SDK Booting ---")
    await orchestrator.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
