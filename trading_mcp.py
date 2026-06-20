#!/usr/bin/env python3
"""Read-only MCP server exposing V5 OANDA soak observability as typed tools.

This wraps the status / calibration / gate commands we otherwise run by hand on
every check-in, so ANY MCP client (Claude Code, OpenCode, an Agent-SDK app) can
call them as first-class tools instead of ad-hoc bash.

READ-ONLY by design: it only reads the pidfile, the soak log, and `ps`. It never
starts/stops the bot, retrains, promotes, or touches the account — so it cannot
affect the running soak. Control tools (with confirmation gating) are a later step.

Run:    python trading_mcp.py            # stdio transport
Deps:   pip install "mcp[cli]"
Env:    SOAK_PIDFILE (default /tmp/soak.pid), SOAK_LOGPATH_FILE
        (default /tmp/soak_logpath), SOAK_REPO_DIR (default the repo path) —
        the logpath file stores a path relative to the repo, so we resolve it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("trading-ops")

PID_FILE = os.getenv("SOAK_PIDFILE", "/tmp/soak.pid")
LOGPATH_FILE = os.getenv("SOAK_LOGPATH_FILE", "/tmp/soak_logpath")
REPO_DIR = os.getenv("SOAK_REPO_DIR", "/mnt/storage/mystuf/development/build-A-bot")

# Cost gate passes iff sl_dist >= k_eff * spread; with sl_mult=1.0, k_eff=1.5
# that is alpha_emp <= 1/1.5. Below this an instrument is tradeable on M1.
TRADEABLE_ALPHA_MAX = 1.0 / 1.5


def _first_line(path: str) -> str | None:
    try:
        text = Path(path).read_text().strip()
        return text.splitlines()[0].strip() if text else None
    except (OSError, IndexError):
        return None


def _soak_pid() -> int | None:
    raw = _first_line(PID_FILE)
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _log_path() -> str | None:
    raw = _first_line(LOGPATH_FILE)
    if not raw:
        return None
    return raw if os.path.isabs(raw) else os.path.join(REPO_DIR, raw)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True       # exists, owned by another user
    except OSError:
        return False
    return True


def _ps_field(pid: int, fmt: str) -> str | None:
    try:
        out = subprocess.run(
            ["ps", "-o", f"{fmt}=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except (subprocess.SubprocessError, OSError):
        return None


def _tail(path: str, n: int) -> list[str]:
    try:
        with open(path, "r", errors="replace") as fh:
            return fh.readlines()[-n:]
    except OSError:
        return []


# ── plain helpers (decorator-independent, so they're directly testable) ──
def _status() -> dict:
    pid, log = _soak_pid(), _log_path()
    info: dict = {"pidfile": PID_FILE, "pid": pid, "log_path": log,
                  "alive": bool(pid and _pid_alive(pid))}
    if info["alive"]:
        info["uptime"] = _ps_field(pid, "etime")
        rss = _ps_field(pid, "rss")
        info["mem_mb"] = round(int(rss) / 1024, 1) if rss and rss.isdigit() else None
        info["stat"] = _ps_field(pid, "stat")
    if log and Path(log).exists():
        age_s = time.time() - Path(log).stat().st_mtime
        info["last_log_age_min"] = round(age_s / 60, 1)
    return info


_CALIB_RE = re.compile(
    r"SPREAD_CALIB (\w+) \| n=(\d+) med_spread_pct=([\d.]+).*?"
    r"med_baseline_natr=([\d.]+) alpha_emp=([\d.]+)"
)


def _calibration() -> dict:
    log = _log_path()
    if not log:
        return {"error": "no soak log path found"}
    latest: dict[str, dict] = {}
    for line in _tail(log, 8000):
        m = _CALIB_RE.search(line)
        if m:
            inst, n, spread, natr, alpha = m.groups()
            latest[inst] = {
                "n": int(n),
                "alpha_emp": float(alpha),
                "med_spread_pct": float(spread),
                "med_baseline_natr": float(natr),
                "tradeable_on_m1": float(alpha) <= TRADEABLE_ALPHA_MAX,
            }
    return latest or {"note": "no SPREAD_CALIB readings yet (first prints ~1h in)"}


_REJECT_RE = re.compile(
    r"Bracket rejected \(\w+ gate\) \| spread=(\d+) regime=(\d+) time=(\d+) "
    r"\(\d+ / (\d+) Devil-approved vetoed"
)


def _gate_activity() -> dict:
    log = _log_path()
    if not log:
        return {"error": "no soak log path found"}
    lines = _tail(log, 20000)
    tally = {"spread": 0, "regime": 0, "time": 0, "devil_approved": 0}
    recent: list[str] = []
    for line in lines:
        m = _REJECT_RE.search(line)
        if m:
            sp, rg, tm, approved = m.groups()
            tally = {"spread": int(sp), "regime": int(rg), "time": int(tm),
                     "devil_approved": int(approved)}
            recent.append(line.strip()[:170])
        elif "veto:" in line and "Gate" in line:
            recent.append(line.strip()[:170])
    trades = sum(1 for ln in lines
                 if re.search(r"\b(OPENED|FILLED|position opened)\b", ln))
    return {"veto_tally": tally, "executed_trades": trades,
            "recent_events": recent[-8:]}


# ── MCP tools (return JSON strings, per the FastMCP `-> str` convention) ──
@mcp.tool()
def soak_status() -> str:
    """Live health of the V5 OANDA soak: alive/dead, pid, uptime, memory, and how
    stale the log is (minutes since last write). Read-only. Returns JSON."""
    return json.dumps(_status(), indent=2)


@mcp.tool()
def calibration() -> str:
    """Latest per-instrument spread-calibration from the soak — the alpha_emp
    (spread/ATR) value per instrument, with `tradeable_on_m1` (alpha <= 0.667).
    These are the numbers that decide whether an instrument can clear the cost
    gate on 1-minute bars. Read-only. Returns JSON keyed by instrument."""
    return json.dumps(_calibration(), indent=2)


@mcp.tool()
def gate_activity() -> str:
    """Risk-gate activity from the soak: cumulative veto tally
    (spread/regime/time gates), Devil-approved signal count, executed-trade count,
    and the most recent veto/gate events. Read-only. Returns JSON."""
    return json.dumps(_gate_activity(), indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
