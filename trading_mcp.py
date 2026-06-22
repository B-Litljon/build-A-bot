#!/usr/bin/env python3
"""Read-only MCP server exposing V5 OANDA soak observability as typed tools.

This wraps the status / calibration / gate commands we otherwise run by hand on
every check-in, so ANY MCP client (Claude Code, OpenCode, an Agent-SDK app) can
call them as first-class tools instead of ad-hoc bash.

Tools — read-only: soak_status, calibration, gate_activity, retrain_status (they
only read the pidfile, the soak/retrain logs, and `ps`). Control: start_soak,
stop_soak — each guarded by a TWO-STEP confirm token (first call returns a preview
+ token; call again with that token, same args, to actually act). The control
tools touch ONLY the practice-account soak process (start/stop); they never trade,
retrain, or promote, and start_soak refuses to double-start.

Run:    python trading_mcp.py            # stdio transport
Deps:   pip install "mcp[cli]"
Env:    SOAK_PIDFILE (default /tmp/soak.pid), SOAK_LOGPATH_FILE
        (default /tmp/soak_logpath), SOAK_REPO_DIR (default the repo path) —
        the logpath file stores a path relative to the repo, so we resolve it.
"""
from __future__ import annotations

import glob
import json
import os
import re
import secrets
import signal
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


# ── retrain log parsing (read-only) ──────────────────────────────────────────
def _latest_retrain_log() -> str | None:
    paths = glob.glob(os.path.join(REPO_DIR, "logs", "retrain_*.log"))
    return max(paths, key=os.path.getmtime) if paths else None


def _retrain_status() -> dict:
    """Parse the most recent retrain log into its REAL verdict + gate numbers.

    The retrainer's process exit code can be masked by a wrapper (a trailing echo
    once reported 0 for a rejected run), so we read the log — which is the truth.
    """
    log = _latest_retrain_log()
    if not log:
        return {"note": "no retrain logs found (logs/retrain_*.log)"}
    text = Path(log).read_text(errors="replace")

    def grab(pat, cast=str):
        m = re.search(pat, text)
        if not m:
            return None
        try:
            return cast(m.group(1).strip())
        except (ValueError, TypeError):
            return None

    info = {
        "log": os.path.relpath(log, REPO_DIR),
        "symbols": grab(r"Symbols:\s*(.+)"),
        "date_range": grab(r"Date range:\s*(.+)"),
        "mean_brier": grab(r"Mean Brier Score\s*:\s*([\d.]+)", float),
        "mean_ev": grab(r"Mean EV\s*:\s*([\d.]+)", float),
        "profit_factor": grab(r"Profit Factor\s*:\s*([\d.]+)", float),
        "pooled_oos_trades": grab(r"Pooled OOS Trades:\s*(\d+)", int),
        "gate_result": grab(r"Gate Result\s*:\s*(\w+)"),
        "complete": bool(re.search(r"MODELS (PROMOTED|REJECTED)", text)),
    }
    if "MODELS PROMOTED" in text:
        info["verdict"] = "PROMOTED"
        info["saved_in"] = grab(r"Models saved in:\s*(.+)")
    elif "MODELS REJECTED" in text:
        info["verdict"] = "REJECTED"
        info["rejection_reasons"] = [r.strip() for r in re.findall(r"Rejection:\s*(.+)", text)]
    else:
        info["verdict"] = "INCOMPLETE/UNKNOWN (run may still be in progress)"
    return info


# ── two-step confirmation for control tools ──────────────────────────────────
# Tokens are tied to the action AND its args, single-use, and short-lived, so a
# stray re-call can't fire and a start-token can't be replayed as a stop.
_CONFIRM_TTL_S = 120
_pending: dict[str, dict] = {}   # action -> {"token", "expires", "args"}


def _issue_confirmation(action: str, args: dict, preview: str) -> dict:
    token = secrets.token_hex(3)
    _pending[action] = {"token": token, "expires": time.time() + _CONFIRM_TTL_S, "args": args}
    return {
        "status": "confirmation_required",
        "action": action,
        "preview": preview,
        "confirm_token": token,
        "expires_in_s": _CONFIRM_TTL_S,
        "next_step": f'call {action} again with confirm_token="{token}" (same args) within {_CONFIRM_TTL_S}s',
    }


def _consume_confirmation(action: str, token: str, args: dict) -> tuple[bool, str]:
    rec = _pending.get(action)
    if not rec:
        return False, "no pending confirmation — call once without a token to get one"
    if time.time() > rec["expires"]:
        _pending.pop(action, None)
        return False, "confirm_token expired — request a fresh one"
    if token != rec["token"]:
        return False, "confirm_token does not match the pending one"
    if args != rec["args"]:
        return False, "args changed since the token was issued — request a new token"
    _pending.pop(action, None)   # one-time use
    return True, "ok"


# ── soak process control (practice account only) ─────────────────────────────
def _spawn_soak(symbols: str) -> dict:
    """Launch run_soak.sh detached; wait briefly for the fresh pid/log to appear."""
    old_log = _first_line(LOGPATH_FILE)
    cmd = ["bash", "run_soak.sh"] + ([symbols] if symbols else [])
    subprocess.Popen(
        cmd, cwd=REPO_DIR,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.time() + 6
    while time.time() < deadline:
        new_log, pid = _first_line(LOGPATH_FILE), _soak_pid()
        if new_log and new_log != old_log and pid:
            return {"pid": pid, "log": _log_path()}
        time.sleep(0.2)
    return {"pid": _soak_pid(), "log": _log_path(),
            "warning": "launch issued but pidfile didn't refresh in 6s — verify with soak_status"}


def _start_soak(symbols: str = "", confirm_token: str = "") -> dict:
    pid = _soak_pid()
    if pid and _pid_alive(pid):
        return {"status": "refused", "reason": f"a soak is already running (pid {pid}) — stop it first"}
    args = {"symbols": (symbols or "").strip()}
    if not confirm_token:
        basket = args["symbols"] or "full trained basket"
        preview = f"Launch run_soak.sh on the PRACTICE account, symbols = {basket}."
        return _issue_confirmation("start_soak", args, preview)
    ok, msg = _consume_confirmation("start_soak", confirm_token, args)
    if not ok:
        return {"status": "rejected", "reason": msg}
    started = _spawn_soak(args["symbols"])
    return {"status": "started", "symbols": args["symbols"] or "full trained basket", **started}


def _stop_soak(confirm_token: str = "") -> dict:
    pid = _soak_pid()
    if not (pid and _pid_alive(pid)):
        return {"status": "noop", "reason": "no soak is currently running"}
    args = {"pid": pid}
    if not confirm_token:
        preview = (f"SIGTERM soak pid {pid} (uptime {_ps_field(pid, 'etime')}) — "
                   "it flattens positions and shuts down cleanly.")
        return _issue_confirmation("stop_soak", args, preview)
    ok, msg = _consume_confirmation("stop_soak", confirm_token, args)
    if not ok:
        return {"status": "rejected", "reason": msg}
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        return {"status": "error", "reason": f"kill failed: {e}"}
    return {"status": "stopping", "pid": pid, "note": "SIGTERM sent — call soak_status to confirm shutdown"}


# ── MCP tools: read + (confirm-gated) control ────────────────────────────────
@mcp.tool()
def retrain_status() -> str:
    """Most recent retrain's REAL verdict + numbers, parsed from the log: gate
    PASSED/FAILED, mean Brier, mean EV, Profit Factor, pooled OOS trades, and any
    rejection reasons. The process exit code can mislead — this reads the log,
    which is the source of truth. Read-only. Returns JSON."""
    return json.dumps(_retrain_status(), indent=2)


@mcp.tool()
def start_soak(symbols: str = "", confirm_token: str = "") -> str:
    """Start the V5 paper soak on the PRACTICE account. `symbols`: comma-separated
    (e.g. "XAU_USD,XAG_USD"); empty = full trained basket. TWO-STEP: call once with
    no token to get a preview + confirm_token, then call again with that token (and
    the same symbols) to actually launch. Refuses if a soak is already running.
    Returns JSON."""
    return json.dumps(_start_soak(symbols, confirm_token), indent=2)


@mcp.tool()
def stop_soak(confirm_token: str = "") -> str:
    """Stop the running soak (SIGTERM by pidfile; the bot flattens positions on the
    way down). TWO-STEP: call once with no token for a preview + confirm_token, then
    again with that token to actually stop. No-ops if nothing is running. Returns
    JSON."""
    return json.dumps(_stop_soak(confirm_token), indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
