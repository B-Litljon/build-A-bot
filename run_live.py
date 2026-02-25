#!/usr/bin/env python3
"""
run_live.py — Universal Scalper V3.1 Live Paper-Trading Launcher
================================================================

Root-level entry point that controls sys.path injection, then delegates
entirely to the LiveOrchestrator daemon inside src/.

Usage:
    python3 run_live.py                        # interactive Rich dashboard
    python3 run_live.py --daemon               # headless / systemd mode
    SYMBOLS="TSLA,NVDA" python3 run_live.py    # override basket via env

This script is intentionally thin.  All trading logic lives in:
    src/execution/live_orchestrator.py
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before ANY src/ imports.
# Inserts the absolute path to src/ at the front of sys.path so that bare
# module names (core, ml, utils, strategies, etc.) resolve correctly.
# This mirrors the pattern used by src/main.py and root main.py.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# ---------------------------------------------------------------------------
# Now safe to import from src/ using bare module names
# ---------------------------------------------------------------------------
from execution.live_orchestrator import LiveOrchestrator, DEFAULT_SYMBOLS  # noqa: E402


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    --daemon
        Enables headless mode: bypasses all Rich UI components (Live dashboard,
        Progress bars) and configures plain stdout logging so that systemd /
        journald receives clean, ANSI-free log lines.
    """
    parser = argparse.ArgumentParser(
        prog="run_live.py",
        description="Universal Scalper V3.1 — Live Paper-Trading Launcher",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        default=False,
        help=(
            "Run in headless daemon mode (no Rich UI). "
            "Intended for systemd user-service deployment. "
            "Logs are written as plain text to stdout for journald."
        ),
    )
    return parser.parse_args()


def _resolve_symbols() -> list[str]:
    """
    Read the symbol basket from the SYMBOLS env var (comma-separated),
    falling back to the orchestrator's DEFAULT_SYMBOLS constant.

    Example:
        SYMBOLS="TSLA,NVDA,MARA" python3 run_live.py
    """
    raw = os.getenv("SYMBOLS", "").strip()
    if raw:
        return [s.strip().upper() for s in raw.split(",") if s.strip()]
    return list(DEFAULT_SYMBOLS)


async def _main() -> None:
    args = _parse_args()
    symbols = _resolve_symbols()
    orchestrator = LiveOrchestrator(
        symbols=symbols,
        paper=True,
        daemon_mode=args.daemon,
    )
    await orchestrator.run()


if __name__ == "__main__":
    asyncio.run(_main())
