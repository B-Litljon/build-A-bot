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
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
from execution.risk_manager import RiskManager, RiskProfile  # noqa: E402
from strategies.concrete_strategies.ml_strategy import MLStrategy  # noqa: E402

logger = logging.getLogger(__name__)

# Last-resort fallback only — the real default is the trained basket from
# models/forex/metadata.json (see _trained_basket).
FALLBACK_SYMBOLS = ["EUR/USD"]

_METADATA_PATH = Path(__file__).resolve().parent / "models" / "forex" / "metadata.json"


def _trained_basket() -> list[str]:
    """
    Instruments the promoted model was trained on.

    Used as the default basket so launching with no --symbols never trades
    an out-of-distribution pair (the model has only seen these).
    """
    try:
        with open(_METADATA_PATH) as fh:
            symbols = json.load(fh).get("trained_on_symbols") or []
        if symbols:
            return list(symbols)
    except Exception as e:
        logger.warning("Could not read trained basket from %s: %s", _METADATA_PATH, e)
    return list(FALLBACK_SYMBOLS)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_oanda.py",
        description="V5 OANDA Forex Scalper — Angel/Devil Meta-Labeling",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=os.getenv("OANDA_SYMBOLS", ",".join(_trained_basket())),
        help=(
            "Comma-separated instrument list (default: the trained basket "
            "from models/forex/metadata.json)"
        ),
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
    parser.add_argument(
        "--granularity",
        type=int,
        default=1,
        help="Stream granularity/timeframe in minutes (default: 1)",
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
        symbols = _trained_basket()

    # Warn loudly when trading instruments the model has never seen.
    basket = {s.replace("/", "_").upper() for s in _trained_basket()}
    for s in symbols:
        if s.replace("/", "_").upper() not in basket:
            logger.warning(
                "Symbol %s is NOT in the trained basket %s — the model is "
                "out of distribution on it",
                s,
                sorted(basket),
            )

    # ── initialise components ──
    provider = OandaMarketProvider(
        environment=args.env,
        stream_granularity_minutes=args.granularity,
    )
    order_manager = OandaOrderManager(environment=args.env)
    
    htf_tf = "30m" if args.granularity == 5 else "5m"
    warmup_pd = 300 if args.granularity == 5 else 260
    strategy = MLStrategy(
        asset_class="forex",
        timeframe=args.granularity,
        htf_timeframe=htf_tf,
        warmup_period=warmup_pd,
    )
    
    # Forex volatility is a fraction of Equities. Use a derived 2.0 pips stop-loss floor
    # so the chop filter doesn't reject everything.
    # Set round_precision=5 since Forex pairs are quoted to 5 decimal places natively.
    risk_profile = RiskProfile.for_asset_class("forex")
    risk_manager = RiskManager(profile=risk_profile)

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
