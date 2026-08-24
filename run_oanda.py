#!/usr/bin/env python3
"""
run_oanda.py — V5 OANDA Forex Bot Launcher
===============================================

Root-level entry point for the V5 Angel/Devil meta-labeling forex bot on
OANDA v20 forex.  Controls sys.path injection, then constructs and runs
the :class:`OandaForexOrchestrator`.

Usage:
    python3 run_oanda.py                      # default EUR/USD
    python3 run_oanda.py --symbols GBP/USD    # override basket
    OANDA_UNITS=500 python3 run_oanda.py      # override position size

⚠️ THIS IS THE LAUNCHER FOR THE CURRENTLY-RUNNING BOT. The live M15 soak is
`run_oanda.py --daemon --env practice --granularity 15`, kept alive by
soak_watchdog.sh (cron, every 5 minutes). To stop it, `touch soak.off` BEFORE
killing the process, or the watchdog resurrects it within 5 minutes.

Glossary:
    _SRC_DIR -- src/, prepended to sys.path so bare module names resolve.
    FALLBACK_SYMBOLS -- ["EUR/USD"], used only when nothing else specifies a
        basket. The real basket normally comes from the model's metadata.
    _MODEL_DIR -- which model directory to load; this is what selects between
        models/forex and a side candidate like models/forex_m15.
    _METADATA_PATH -- metadata.json in that directory. Read so the bot trades
        the instruments and timeframe the model was actually TRAINED on rather
        than whatever the command line happens to say.
    _GRANULARITY_PROFILES -- maps bar size to (higher-timeframe, warm-up bars),
        matching scripts/probe_model.py. Picking the wrong pair here would feed
        the model differently-computed features than it trained on.
    --granularity -- bar size in minutes (15 for the current soak).
    --env -- "practice" (paper money) or "live" (real). Defaults to practice.
    --daemon -- headless mode; log to file, no interactive display.
    OANDA_UNITS -- position size override.
    _configure_logging -- sets the root format/level, then installs
        core.log_filters so a single oversized broker error (Cloudflare HTML
        during OANDA maintenance) cannot flood the log. See LOG_MAX_CHARS.
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
from core import events  # noqa: E402
from core import log_filters  # noqa: E402
from data.oanda_provider import OandaMarketProvider  # noqa: E402
from execution.oanda_order_manager import OandaOrderManager  # noqa: E402
from execution.oanda_forex_orchestrator import (  # noqa: E402
    OandaForexOrchestrator,
)
from execution.risk_manager import RiskManager, RiskProfile  # noqa: E402
from strategies.concrete_strategies.ml_strategy import MLStrategy  # noqa: E402

logger = logging.getLogger(__name__)

# Last-resort fallback only — the real default is the trained basket from
# models/forex/metadata.json (see _trained_basket).
FALLBACK_SYMBOLS = ["EUR/USD"]

# Model artifact directory. OANDA_MODEL_DIR redirects the whole artifact set
# (pkls + metadata.json + threshold.json) to a side model — e.g. models/forex_m15
# — without touching the promoted models/forex. Mirrors RETRAIN_MODEL_DIR on
# the training side.
_MODEL_DIR = Path(__file__).resolve().parent / (
    os.getenv("OANDA_MODEL_DIR", "").strip() or "models/forex"
)
_METADATA_PATH = _MODEL_DIR / "metadata.json"

# Per-granularity strategy profile: base minutes → (HTF resample string, warmup bars).
# Warmup must cover the HTF SMA-50: 260 M1 bars = 52 5m bars; 300 M5 bars = 50 30m
# bars; 260 M15 bars = 65 1h bars.
_GRANULARITY_PROFILES: dict[int, tuple[str, int]] = {
    1: ("5m", 260),
    5: ("30m", 300),
    15: ("1h", 260),
}


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


def _trained_timeframe() -> int | None:
    """Bar granularity (minutes) the promoted model was trained on, if known."""
    try:
        with open(_METADATA_PATH) as fh:
            return json.load(fh).get("timeframe_minutes")
    except Exception as e:
        logger.warning("Could not read trained timeframe from %s: %s", _METADATA_PATH, e)
        return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_oanda.py",
        description="V5 OANDA Forex Bot — Angel/Devil Meta-Labeling",
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
        choices=sorted(_GRANULARITY_PROFILES),
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

    # Broker errors arrive as HTML from Cloudflare during OANDA maintenance
    # (~96 KB each). Unfiltered, a weekend of reconnects wrote a 447 MB log.
    # Attach to the root handlers so oandapyV20's own logger is covered too.
    log_filters.install()


async def _main() -> None:
    args = _parse_args()
    _configure_logging(args.daemon)

    # Structured telemetry sink (logs/events-*.jsonl + logs/status.json).
    # Best-effort by construction; EVENTS_ENABLED=0 turns it off entirely.
    events.configure()

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

    # Warn loudly when streaming at a granularity the model wasn't trained on.
    trained_tf = _trained_timeframe()
    if trained_tf is not None and args.granularity != trained_tf:
        logger.warning(
            "Streaming at %d-minute granularity but the model was trained on "
            "%d-minute bars — the model is out of distribution on this timeframe",
            args.granularity,
            trained_tf,
        )

    # ── initialise components ──
    provider = OandaMarketProvider(
        environment=args.env,
        stream_granularity_minutes=args.granularity,
    )
    order_manager = OandaOrderManager(environment=args.env)
    
    # Forex volatility is a fraction of Equities. Use a derived 2.0 pips stop-loss floor
    # so the chop filter doesn't reject everything.
    # Set round_precision=5 since Forex pairs are quoted to 5 decimal places natively.
    risk_profile = RiskProfile.for_asset_class("forex")

    htf_tf, warmup_pd = _GRANULARITY_PROFILES[args.granularity]
    strategy = MLStrategy(
        asset_class="forex",
        angel_path=_MODEL_DIR / "angel_latest.pkl",
        devil_path=_MODEL_DIR / "devil_latest.pkl",
        timeframe=args.granularity,
        htf_timeframe=htf_tf,
        warmup_period=warmup_pd,
        # cost_ratio feature baseline must use the same window as the live
        # regime gate — one source of truth for both sides.
        regime_window=risk_profile.regime_window,
    )

    # Per-instrument spread alphas shipped with the model (spread_alphas.json,
    # written by the retrainer on gate pass). Used by Gate A's stale-spread
    # proxy branch; fresh tick spreads always win. Absent → flat env alpha.
    alpha_overrides = None
    try:
        with open(_MODEL_DIR / "spread_alphas.json") as fh:
            alpha_overrides = json.load(fh).get("alphas") or None
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("Could not read spread_alphas.json: %s", e)
    logger.info(
        "Gate A spread proxy mode: %s",
        f"per-instrument ({len(alpha_overrides)} alphas)"
        if alpha_overrides
        else f"flat alpha={risk_profile.spread_atr_alpha}",
    )
    risk_manager = RiskManager(profile=risk_profile, alpha_overrides=alpha_overrides)

    orchestrator = OandaForexOrchestrator(
        symbols=symbols,
        provider=provider,
        strategy=strategy,
        order_manager=order_manager,
        risk_manager=risk_manager,
        units_per_trade=args.units,
        flatten_on_exit=not args.no_flatten,
    )

    logger.info("--- V5 OANDA Forex Bot Booting | env=%s symbols=%s ---", args.env, symbols)
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(_main())
