#!/usr/bin/env python3
"""
run_oanda.py — V5 OANDA Forex Scalper Launcher
===============================================

Root-level entry point for the V5 Angel/Devil meta-labeling scalper on
OANDA v20 forex.  Controls sys.path injection, then constructs and runs
the :class:`OandaScalperOrchestrator`.

Usage:
    python3 run_oanda.py                      # default EUR/USD
    python3 run_oanda.py --symbols GBP/USD    # override basket
    OANDA_UNITS=500 python3 run_oanda.py      # override position size
"""

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before ANY src/ imports.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Now safe to import from src/ using bare module names
# ---------------------------------------------------------------------------
from data.oanda_provider import OandaMarketProvider  # noqa: E402
from execution.oanda_order_manager import OandaOrderManager  # noqa: E402
from execution.oanda_scalper_orchestrator import (  # noqa: E402
    OandaScalperOrchestrator,
)
from execution.risk_manager import RiskManager  # noqa: E402
from strategies.concrete_strategies.ml_strategy import MLStrategy  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["EUR/USD"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_oanda.py",
        description="V5 OANDA Forex Scalper — Angel/Devil Meta-Labeling",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=os.getenv("OANDA_SYMBOLS", ",".join(DEFAULT_SYMBOLS)),
        help="Comma-separated instrument list (default: EUR/USD)",
    )
    parser.add_argument(
        "--units",
        type=int,
        default=int(os.getenv("OANDA_UNITS", "1000")),
        help="Units per trade (default: 1000)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=os.getenv("OANDA_ENV", "practice"),
        choices=["practice", "live"],
        help="OANDA environment (default: practice)",
    )
    parser.add_argument(
        "--no-flatten",
        action="store_true",
        default=False,
        help="Disable automatic position flatten on SIGINT/SIGTERM",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        default=False,
        help="Headless mode (plain logging, no Rich UI)",
    )
    return parser.parse_args()


def _configure_logging(daemon: bool) -> None:
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    if daemon:
        logging.basicConfig(level=logging.INFO, format=fmt, datefmt=datefmt)
    else:
        logging.basicConfig(level=logging.DEBUG, format=fmt, datefmt=datefmt)


async def _main() -> None:
    args = _parse_args()
    _configure_logging(args.daemon)

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        symbols = list(DEFAULT_SYMBOLS)

    # ── initialise components ──
    provider = OandaMarketProvider(environment=args.env)
    order_manager = OandaOrderManager(environment=args.env)
    strategy = MLStrategy()
    risk_manager = RiskManager()

    orchestrator = OandaScalperOrchestrator(
        symbols=symbols,
        provider=provider,
        strategy=strategy,
        order_manager=order_manager,
        risk_manager=risk_manager,
        units_per_trade=args.units,
        flatten_on_exit=not args.no_flatten,
    )

    logger.info("--- V5 OANDA Scalper Booting | env=%s symbols=%s ---", args.env, symbols)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(_main())
