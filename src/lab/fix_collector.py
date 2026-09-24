"""
Fix tick capture buffer + atomic daily parquet writer for Audit B.

Split from ``scripts/fix_tick_collector.py`` so the on-disk contract is
testable without a live OANDA connection: the collector script owns the
stream/reconnect machinery, this module owns the tick buffer, the daily
slicing, and the atomic write convention (temp file + fsync + os.replace —
the same pattern as ``src/core/events.py:_write_status_atomic`` and
``src/lab/frames.py:_save_frame``).

Glossary:
    FixTickBuffer -- an in-memory list of fix-window tick rows, grouped by
        UTC day. ``add(timestamp_utc, symbol, bid, ask, latency_ms)`` records
        one tick; ``flush_day(day, dest_dir)`` writes and clears one day's
        rows. The buffer is deliberately dumb — no filtering logic lives here
        (the collector filters to ±5 min around each fix before calling add).
    TICK_COLUMNS -- the parquet schema: ``[timestamp_utc, symbol, bid, ask,
        mid, source_latency_ms]``. mid = (bid+ask)/2 is computed here so a
        reader never has to re-derive it.
    write_ticks_atomic(day_rows, path) -- the atomic write: writes to
        ``<path>.tmp``, fsyncs, then ``os.replace``s onto ``path``. A reader
        sees either the previous complete file or the new complete file,
        never a partial one; no ``.tmp`` is left behind on success.

Research-only. Nothing here places orders or touches the live soak.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

TICK_COLUMNS: Tuple[str, ...] = (
    "timestamp_utc", "symbol", "bid", "ask", "mid", "source_latency_ms",
)


class FixTickBuffer:
    """In-memory tick rows grouped by UTC day, with atomic daily flush."""

    def __init__(self) -> None:
        self._rows: Dict[str, List[tuple]] = {}

    def add(
        self,
        timestamp_utc: datetime,
        symbol: str,
        bid: float,
        ask: float,
        source_latency_ms: float,
    ) -> None:
        """Record one tick. ``timestamp_utc`` must be tz-aware UTC."""
        if timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware (UTC)")
        day = timestamp_utc.astimezone(timezone.utc).strftime("%Y-%m-%d")
        mid = (bid + ask) / 2.0
        self._rows.setdefault(day, []).append(
            (timestamp_utc, symbol, float(bid), float(ask), mid,
             float(source_latency_ms))
        )

    def pending_days(self) -> List[str]:
        return sorted(self._rows)

    def flush_day(self, day: str, dest_dir: Path | str) -> Path:
        """
        Atomically write ``dest_dir/fix_ticks_<day>.parquet`` from the
        buffered rows for ``day`` and clear them from the buffer.
        """
        rows = self._rows.pop(day, [])
        path = Path(dest_dir) / f"fix_ticks_{day}.parquet"
        write_ticks_atomic(rows, path)
        return path


def write_ticks_atomic(rows: List[tuple], path: Path | str) -> Path:
    """
    Write tick rows to ``path`` as parquet with exactly TICK_COLUMNS,
    atomically: temp file in the destination dir, fsync, then os.replace.
    An empty ``rows`` still writes the (header-only) file so a day with no
    fix-window ticks is distinguishable from a missing flush.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        ts = [r[0] for r in rows]
        sym = [r[1] for r in rows]
        bid = [r[2] for r in rows]
        ask = [r[3] for r in rows]
        mid = [r[4] for r in rows]
        lat = [r[5] for r in rows]
    else:
        ts, sym, bid, ask, mid, lat = [], [], [], [], [], []

    table = pa.table(
        {
            "timestamp_utc": pa.array(ts, type=pa.timestamp("us", tz="UTC")),
            "symbol": pa.array(sym, type=pa.string()),
            "bid": pa.array(bid, type=pa.float64()),
            "ask": pa.array(ask, type=pa.float64()),
            "mid": pa.array(mid, type=pa.float64()),
            "source_latency_ms": pa.array(lat, type=pa.float64()),
        }
    )
    # guarantee column order
    table = table.select(list(TICK_COLUMNS))

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as fh:
            pq.write_table(table, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        # never leave a .tmp behind on failure
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path
