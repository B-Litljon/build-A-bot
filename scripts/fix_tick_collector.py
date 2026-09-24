#!/usr/bin/env python3
"""
fix_tick_collector.py — Audit B's deployable artifact: OANDA quote streamer
that captures bid/ask ticks in the ±5 min window around the London WM/R and
Tokyo Nakane fixes and writes them to daily parquet.

**SHIPS UNINSTALLED.** Do NOT run this; do NOT install the accompanying
systemd unit. It exists so the tick-level question the coarse M15 audit
cannot answer ("does the spread actually widen into the fix?") becomes
measurable once Brandon reviews the coarse study and opts in.

Stream path: ``OandaMarketProvider.subscribe(symbols, callback,
tick_callback=...)`` — the provider's raw-tick hook fires synchronously on
every PRICE message; this collector's tick callback only appends to an
in-memory buffer when the tick's UTC time lies inside a fix window, so it
comfortably meets the provider's <50 µs / no-blocking-I/O contract. The
parquet flush is done by a separate timer thread, never in the callback.

Latency: ``source_latency_ms`` is (collector receive time) − (OANDA tick
timestamp), i.e. transport + queueing delay as seen by this process; it is
NOT a measure of OANDA-internal latency.

Output: ``data/ticks/fix_ticks_YYYY-MM-DD.parquet`` written atomically
(temp + fsync + os.replace) via ``lab.fix_collector`` per UTC day.

Reconnect: the OANDA stream is wrapped in an exponential-backoff retry loop
(1 s → 60 s cap, full jitter). A terminated stream is re-subscribed and
re-run; a KeyboardInterrupt / SIGTERM flushes and exits.

Env:
    OANDA_API_KEY, OANDA_ACCOUNT_ID   (required; practice environment)
    OANDA_ENV                         (default "practice")
    FIX_TICKS_DIR                     (default data/ticks/ under cwd)
    FIX_WINDOW_SECONDS                (default 300 = ±5 min)

CLI: no args; run under systemd (see systemd/fix-tick-collector.service,
uninstalled template) or by hand for a smoke test.
"""

from __future__ import annotations

import logging
import os
import random
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

# The collector runs from the repo root; src/ is on sys.path via the unit's
# Environment=PYTHONPATH. For hand runs from the repo root:
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lab.fix_audit import FIAT_PAIRS, london_fix_utc, tokyo_fix_utc
from lab.fix_collector import FixTickBuffer
from data.oanda_provider import OandaMarketProvider
import data.oanda_provider as _op

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("fix_tick_collector")

FIAT_OANDA = list(FIAT_PAIRS)  # EUR_USD-style OANDA symbols
DEFAULT_WINDOW_SECONDS = 300
FLUSH_INTERVAL_SECONDS = 60
BACKOFF_MIN_S = 1.0
BACKOFF_MAX_S = 60.0


def _fix_windows_today(now_utc: datetime, window_s: int) -> List[tuple]:
    """
    Today's and tomorrow's fix windows as (start_utc, end_utc) pairs, so a
    collector started at 23:59 already knows the 00:00+ windows. Each fix
    produces one window [fix−window, fix+window].
    """
    days = [now_utc, now_utc + timedelta(days=1)]
    out: List[tuple] = []
    for d in days:
        if d.weekday() < 5:
            w = london_fix_utc(d)
            out.append((w - timedelta(seconds=window_s), w + timedelta(seconds=window_s)))
        t = tokyo_fix_utc(d)
        out.append((t - timedelta(seconds=window_s), t + timedelta(seconds=window_s)))
    return out


class Collector:
    def __init__(self) -> None:
        self._dest = Path(os.getenv("FIX_TICKS_DIR", "data/ticks"))
        self._window_s = int(os.getenv("FIX_WINDOW_SECONDS", str(DEFAULT_WINDOW_SECONDS)))
        self._buf = FixTickBuffer()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._provider = None

    # ── tick path (provider callback: <50 µs, no I/O) ────────────────
    def on_tick(self, symbol: str, bid: float, ask: float) -> None:
        now = datetime.now(timezone.utc)
        if not hasattr(self, "_windows") or self._windows_day != now.date():
            self._windows = _fix_windows_today(now, self._window_s)
            self._windows_day = now.date()
        in_window = any(a <= now <= b for a, b in self._windows)
        if not in_window:
            return
        # Latency: the provider's tick_callback signature carries no
        # timestamp. We recover the tick's own OANDA time by wrapping the
        # provider's _handle_tick (set up in run()), which stores the
        # current tick's ts in `self._last_tick_ts` before delegating.
        ts_tick = getattr(self, "_last_tick_ts", None)
        latency_ms = (
            (now - ts_tick).total_seconds() * 1000.0 if ts_tick is not None else 0.0
        )
        with self._lock:
            self._buf.add(now, symbol, bid, ask, latency_ms)

    # ── flush timer ──────────────────────────────────────────────────
    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(FLUSH_INTERVAL_SECONDS)
            self.flush_all(flush_today=False)

    def flush_all(self, *, flush_today: bool) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            days = [d for d in self._buf.pending_days() if flush_today or d < today]
            for d in days:
                try:
                    p = self._buf.flush_day(d, self._dest)
                    log.info("flushed %s -> %s", d, p)
                except Exception:
                    log.exception("flush failed for %s", d)

    # ── stream supervision ───────────────────────────────────────────
    def run(self) -> None:
        backoff = BACKOFF_MIN_S
        flusher = threading.Thread(target=self._flush_loop, daemon=True)
        flusher.start()

        while not self._stop.is_set():
            try:
                provider = OandaMarketProvider(
                    environment=os.getenv("OANDA_ENV", "practice"),
                )
                # Wrap _handle_tick so the tick callback above sees the
                # tick's own OANDA timestamp (source_latency_ms). The wrap
                # only stamps `self._last_tick_ts` and delegates; it adds
                # ~1 µs and keeps the callback contract.
                # TODO: confirm stream callback signature at runtime — if a
                # future provider version passes the tick timestamp directly
                # to tick_callback, drop this wrap and use it.
                collector_self = self
                orig_handle = provider._handle_tick

                def _stamped_handle(msg, _orig=orig_handle, _c=collector_self):
                    try:
                        if msg.get("type") == "PRICE":
                            _c._last_tick_ts = _op._parse_iso(msg["time"])
                    except Exception:
                        pass
                    _orig(msg)

                provider._handle_tick = _stamped_handle  # type: ignore[assignment]
                self._provider = provider
                provider.subscribe(FIAT_OANDA, lambda _bar: None, tick_callback=self.on_tick)
                log.info("streaming %s, fix window ±%ds", FIAT_OANDA, self._window_s)
                backoff = BACKOFF_MIN_S
                provider.run_stream()  # blocks until stop/error
            except KeyboardInterrupt:
                break
            except Exception:
                if self._stop.is_set():
                    break
                sleep_s = min(BACKOFF_MAX_S, backoff) * (0.5 + random.random())
                log.exception("stream dropped; reconnecting in %.1fs", sleep_s)
                self._stop.wait(sleep_s)
                backoff = min(BACKOFF_MAX_S, backoff * 2)

        self._stop.set()
        if self._provider is not None:
            try:
                self._provider.stop_stream()
            except Exception:
                pass
        self.flush_all(flush_today=True)

    def stop(self, *_args) -> None:
        self._stop.set()
        if self._provider is not None:
            try:
                self._provider.stop_stream()
            except Exception:
                pass


def main() -> int:
    if not os.getenv("OANDA_API_KEY") or not os.getenv("OANDA_ACCOUNT_ID"):
        log.error("OANDA_API_KEY / OANDA_ACCOUNT_ID must be set")
        return 1
    c = Collector()
    signal.signal(signal.SIGTERM, c.stop)
    signal.signal(signal.SIGINT, c.stop)
    c.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
