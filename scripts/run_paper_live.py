"""
Maiden Voyage — Live paper trading runner for FactoryOrchestrator.

Usage:
    pipenv run python scripts/run_paper_live.py

Requires:
    .env with ALPACA_API_KEY and ALPACA_SECRET_KEY set to paper credentials.

Shutdown:
    Ctrl-C or SIGTERM — orchestrator drains gracefully before exit.
"""

import asyncio
import logging
import os
import sys
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ── Logging — millisecond-precise for slippage and latency auditing ───────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(name)-35s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).resolve().parent.parent
            / f"logs/paper_live_{__import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        ),
    ],
)

# Silence noisy third-party loggers — keep our signal-path clean
logging.getLogger("alpaca").setLevel(logging.WARNING)
logging.getLogger("websocket").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Suppress cosmetic warning spam from Polars join_asof and Sklearn feature names
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*does not have valid feature names.*",
)
warnings.filterwarnings("ignore", message=".*join_asof.*")

# ── Imports (after sys.path is set) ──────────────────────────────────────────
from data.feed import AlpacaCryptoFeed
from execution.factory_orchestrator import FactoryOrchestrator
from execution.risk_manager import RiskManager, RiskProfile
from strategies.concrete_strategies.ml_strategy import MLStrategy

# ── Module-level handles for graceful shutdown from sync context ─────────────
_orchestrator = None
_feed = None

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOLS = ["BTC/USD", "ETH/USD"]
PAPER = True

ANGEL_PATH = Path(__file__).resolve().parent.parent / "models/angel_latest.pkl"
DEVIL_PATH = Path(__file__).resolve().parent.parent / "models/devil_latest.pkl"


def _require_env(key: str) -> str:
    val = os.environ.get(key)
    if not val:
        logger.critical(f"Required environment variable '{key}' is not set. Aborting.")
        sys.exit(1)
    return val


async def main():
    global _orchestrator, _feed

    api_key = _require_env("ALPACA_API_KEY")
    secret_key = _require_env("ALPACA_SECRET_KEY")

    logger.info("=" * 70)
    logger.info("  Build-A-Bot  |  Maiden Voyage  |  Paper Trading Session")
    logger.info(f"  Symbols      : {SYMBOLS}")
    logger.info(f"  Angel model  : {ANGEL_PATH}")
    logger.info(f"  Devil model  : {DEVIL_PATH}")
    logger.info(f"  Paper mode   : {PAPER}")
    logger.info("=" * 70)

    # ── Instantiation ─────────────────────────────────────────────────────────
    logger.info("Initializing RiskManager...")
    risk_manager = RiskManager(RiskProfile())
    logger.info(
        f"RiskProfile → sl_mult={risk_manager.profile.sl_atr_multiplier}, "
        f"tp_mult={risk_manager.profile.tp_atr_multiplier}, "
        f"min_sl_pct={risk_manager.profile.min_sl_pct}, "
        f"risk_per_trade={risk_manager.profile.risk_per_trade}, "
        f"max_notional=${risk_manager.profile.max_notional_cap:,.0f}"
    )

    logger.info("Initializing MLStrategy (Angel + Devil)...")
    strategy = MLStrategy(
        angel_path=str(ANGEL_PATH),
        devil_path=str(DEVIL_PATH),
    )

    logger.info("Initializing AlpacaCryptoFeed...")
    feed = AlpacaCryptoFeed(api_key=api_key, secret_key=secret_key)
    _feed = feed

    logger.info("Initializing FactoryOrchestrator...")
    orchestrator = FactoryOrchestrator(
        symbols=SYMBOLS,
        api_key=api_key,
        secret_key=secret_key,
        strategy=strategy,
        risk_manager=risk_manager,
        feed=feed,
        paper=PAPER,
    )
    _orchestrator = orchestrator

    logger.info("All components initialized. Handing control to orchestrator...")
    logger.info("  → Warming up history (300 min lookback)...")
    logger.info("  → Subscribing to live WebSocket streams...")
    logger.info("  → Watchdog loop starting (1s poll)...")
    logger.info("  → Press Ctrl-C or send SIGTERM to shut down gracefully.")
    logger.info("-" * 70)

    try:
        await orchestrator.run()
    except asyncio.CancelledError:
        logger.info("Shutdown signal received, cleaning up...")
        try:
            await asyncio.wait_for(feed.stop(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.info("Shutdown timeout reached, forcing exit")
        raise
    except asyncio.TimeoutError:
        logger.info("Shutdown timeout reached, forcing exit")
    finally:
        logger.info("Orchestrator exited cleanly.")


async def _shutdown():
    """Sync-context cleanup helper; runs in a fresh event loop after KeyboardInterrupt."""
    if _feed is not None:
        await asyncio.wait_for(_feed.stop(), timeout=3.0)


if __name__ == "__main__":
    # Ensure logs/ directory exists
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested, cleaning up...")
        try:
            asyncio.run(_shutdown())
        except asyncio.TimeoutError:
            logger.info("Shutdown timeout reached, forcing exit")
