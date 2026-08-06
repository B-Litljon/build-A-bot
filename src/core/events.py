"""
Structured event sink — the machine-readable half of the soak's output.

The soak log is written for a human reading `tail`. Everything downstream of it
(`trading_mcp.py`, and now the dashboard API) has had to recover structure with
regexes, and several facts were never in the log at all: exit prices, realized
P&L, per-bar probabilities. This module writes those facts as JSON Lines, one
object per line, so consumers parse data instead of prose.

⚠️ THIS RUNS INSIDE THE LIVE TRADING PROCESS. Three rules follow from that, in
priority order:

1. **It never raises.** Every public function swallows its own exceptions. A
   telemetry bug must never reach the order path.
2. **It never blocks the caller.** ``emit`` puts a tuple on a bounded queue and
   returns; a daemon thread does all file I/O. If the queue is full the event
   is DROPPED and counted -- losing telemetry is always the right trade against
   stalling a bar.
3. **It is never called from the tick path.** ``_on_tick`` must return in <50 µs
   with no I/O. There are no calls to this module there, and
   ``tests/test_events.py`` asserts that at source level.

See GLOSSARY.md (angel/devil, bracket, gate, sealed bar).

Glossary:
    emit -- enqueue one event; ``ts`` (UTC ISO-8601) is stamped automatically.
    write_status -- enqueue a full state snapshot, written atomically (temp
        file + os.replace) because a reader may open it at any instant and
        must never see a half-written file. Same pattern model artifacts use.
    configure -- called once at startup; sets the output directory and the
        on/off switch. Telemetry is OPT-IN: until this is called every entry
        point is a no-op. That is deliberate -- importing this module must
        never cause a test run, a backtest, or an ad-hoc script to start
        writing files into the live bot's logs/ directory.
    EVENTS_ENABLED / EVENTS_DIR -- env switches. Setting ENABLED=0 makes every
        entry point a no-op, which is the escape hatch if this ever misbehaves
        in production.
    STATUS_FILENAME -- "status.json", the snapshot of right-now.
    _QUEUE_MAXSIZE -- 1000 events of slack. At the observed rate (~100/day) this
        is unreachable except during a pathological burst, which is exactly when
        dropping is correct.
    _writer_loop -- the daemon thread: owns the file handle, rotates it when
        the UTC date changes, and treats every item as independently failable.
    _dropped / _errors -- counters, surfaced by stats() so a consumer can tell
        "quiet" from "broken".
    _SENTINEL -- the shutdown token that ends the writer loop.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

STATUS_FILENAME = "status.json"
_QUEUE_MAXSIZE = 1000
_SENTINEL = ("__stop__", None)

_queue: "Optional[queue.Queue[Tuple[str, Any]]]" = None
_thread: Optional[threading.Thread] = None
_lock = threading.Lock()

# Opt-in: nothing is written until configure() runs. Any process that has not
# asked for telemetry (tests, backtests, scripts importing a strategy) stays
# silent instead of appending to the live bot's logs.
_enabled: bool = False
_dir: Path = Path("logs")
_dropped: int = 0
_errors: int = 0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def configure(log_dir: "str | os.PathLike | None" = None,
              enabled: Optional[bool] = None) -> None:
    """
    Set the output directory and on/off switch. Safe to call more than once.

    Reads EVENTS_DIR / EVENTS_ENABLED when the arguments are omitted, so the
    live bot can be configured entirely from the environment.
    """
    global _enabled, _dir
    try:
        with _lock:
            if enabled is None:
                enabled = os.getenv("EVENTS_ENABLED", "1").strip() not in (
                    "0", "false", "False", "no", ""
                )
            _enabled = bool(enabled)
            if log_dir is None:
                log_dir = os.getenv("EVENTS_DIR", "logs")
            _dir = Path(log_dir)
            if _enabled:
                _dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # A telemetry setup failure must not stop the bot from starting.
        _enabled = False


def _ensure_writer() -> "Optional[queue.Queue[Tuple[str, Any]]]":
    """Start the writer thread on first use. Returns None when disabled."""
    global _queue, _thread
    if not _enabled:
        return None
    with _lock:
        if _queue is None:
            _queue = queue.Queue(maxsize=_QUEUE_MAXSIZE)
        if _thread is None or not _thread.is_alive():
            _thread = threading.Thread(
                target=_writer_loop, name="events-writer", daemon=True
            )
            _thread.start()
    return _queue


def _put(item: Tuple[str, Any]) -> None:
    """Enqueue without blocking; drop and count if the queue is full."""
    global _dropped
    q = _ensure_writer()
    if q is None:
        return
    try:
        q.put_nowait(item)
    except queue.Full:
        _dropped += 1


def emit(ev: str, **fields: Any) -> None:
    """Record one event. Returns immediately; never raises."""
    try:
        if not _enabled:
            return
        payload: Dict[str, Any] = {"ts": _utc_now_iso(), "ev": ev}
        payload.update(fields)
        _put(("event", payload))
    except Exception:
        pass


def write_status(payload: Dict[str, Any]) -> None:
    """Replace the status snapshot. Returns immediately; never raises."""
    try:
        if not _enabled:
            return
        snapshot = dict(payload)
        snapshot.setdefault("ts", _utc_now_iso())
        _put(("status", snapshot))
    except Exception:
        pass


def stats() -> Dict[str, Any]:
    """Counters, so a consumer can distinguish 'quiet' from 'broken'."""
    return {
        "enabled": _enabled,
        "dir": str(_dir),
        "dropped": _dropped,
        "errors": _errors,
        "queued": _queue.qsize() if _queue is not None else 0,
    }


def events_path(when: Optional[datetime] = None) -> Path:
    """Path of the JSONL file for ``when`` (default: now, UTC)."""
    day = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%d")
    return _dir / f"events-{day}.jsonl"


def status_path() -> Path:
    return _dir / STATUS_FILENAME


def _write_status_atomic(snapshot: Dict[str, Any]) -> None:
    """
    Temp file + os.replace, never a partial overwrite.

    A dashboard polls this file on its own schedule; a reader that catches a
    half-written JSON object would report nonsense about a live trading
    process. os.replace is atomic within a filesystem, so a reader sees either
    the old snapshot or the new one.
    """
    target = status_path()
    tmp = target.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(snapshot, fh, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def _writer_loop() -> None:
    """Daemon thread: the only place this module touches the filesystem."""
    global _errors
    handle = None
    handle_day: Optional[str] = None
    while True:
        try:
            kind, payload = _queue.get()  # type: ignore[union-attr]
        except Exception:
            return
        if kind == "__stop__":
            break
        try:
            if kind == "status":
                _write_status_atomic(payload)
            else:
                # Rotate on the event's own UTC date, so a line never lands in
                # yesterday's file after midnight.
                day = str(payload.get("ts", ""))[:10]
                if handle is None or day != handle_day:
                    if handle is not None:
                        handle.close()
                    path = _dir / f"events-{day}.jsonl"
                    handle = open(path, "a")
                    handle_day = day
                handle.write(json.dumps(payload, default=str) + "\n")
                handle.flush()
        except Exception:
            # One bad event must not kill the writer for every later event.
            _errors += 1
    if handle is not None:
        try:
            handle.close()
        except Exception:
            pass


def shutdown(timeout: float = 2.0) -> None:
    """Drain and stop the writer. Best-effort; never raises."""
    try:
        if _queue is None or _thread is None:
            return
        _queue.put_nowait(_SENTINEL)
        _thread.join(timeout=timeout)
    except Exception:
        pass


atexit.register(shutdown)
