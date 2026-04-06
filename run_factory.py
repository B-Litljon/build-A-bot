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

    symbols = ["BTC/USD", "ETH/USD"]

    orchestrator = FactoryOrchestrator(
        symbols=symbols,
        api_key=api_key,
        secret_key=secret_key,
        strategy=strategy,
        risk_manager=risk_manager,
        paper=True
    )

    print("--- Build-A-Bot Factory SDK Booting ---")
    await orchestrator.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
