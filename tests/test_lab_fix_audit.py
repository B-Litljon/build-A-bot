"""Tests for lab.fix_audit and lab.fix_collector (Audit B)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from lab.fix_audit import (
    bar_containing,
    is_gotobi_day,
    london_fix_utc,
    measure_fix,
    run_b,
    tokyo_fix_utc,
)
from lab.fix_collector import TICK_COLUMNS, FixTickBuffer, write_ticks_atomic


# ── DST-correct fix instants (brief §5.4 #1) ───────────────────────────────


def test_london_fix_dst_boundary():
    # 2026: BST starts last Sunday of March (2026-03-29), ends last Sunday
    # of October (2026-10-25).
    jan = london_fix_utc(datetime(2026, 1, 15, tzinfo=timezone.utc))
    jul = london_fix_utc(datetime(2026, 7, 15, tzinfo=timezone.utc))
    assert jan.hour == 16 and jan.minute == 0  # GMT: 16:00 London = 16:00 UTC
    assert jul.hour == 15 and jul.minute == 0  # BST: 16:00 London = 15:00 UTC
    # the boundary days themselves
    assert london_fix_utc(datetime(2026, 3, 27, tzinfo=timezone.utc)).hour == 16
    assert london_fix_utc(datetime(2026, 3, 30, tzinfo=timezone.utc)).hour == 15
    assert london_fix_utc(datetime(2026, 10, 23, tzinfo=timezone.utc)).hour == 15
    assert london_fix_utc(datetime(2026, 10, 26, tzinfo=timezone.utc)).hour == 16


def test_tokyo_fix_constant_0055_utc():
    for month in (1, 4, 8, 12):
        t = tokyo_fix_utc(datetime(2026, month, 10, tzinfo=timezone.utc))
        assert (t.hour, t.minute) == (0, 55)


def test_gotobi_days():
    assert is_gotobi_day(datetime(2026, 1, 5, tzinfo=timezone.utc))
    assert is_gotobi_day(datetime(2026, 1, 10, tzinfo=timezone.utc))
    assert is_gotobi_day(datetime(2026, 1, 25, tzinfo=timezone.utc))
    assert is_gotobi_day(datetime(2026, 1, 31, tzinfo=timezone.utc))  # month-end
    assert not is_gotobi_day(datetime(2026, 1, 6, tzinfo=timezone.utc))
    assert not is_gotobi_day(datetime(2026, 1, 30, tzinfo=timezone.utc))


# ── bar_containing ─────────────────────────────────────────────────────────


def _grid(day: str, n_bars: int):
    base = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
    return [base + __import__("datetime").timedelta(minutes=15 * i) for i in range(n_bars)]


def test_bar_containing_finds_fix_bar():
    ts = _grid("2026-07-15", 96)  # full day of M15 bars from 00:00
    inst = datetime(2026, 7, 15, 15, 0, tzinfo=timezone.utc)  # London fix (BST)
    i = bar_containing(ts, inst)
    assert i is not None
    assert ts[i].hour == 15 and ts[i].minute == 0
    # inside the 15:00 bar → still the 15:00 bar
    inst2 = datetime(2026, 7, 15, 15, 7, 30, tzinfo=timezone.utc)
    assert bar_containing(ts, inst2) == i
    # outside the grid → None
    assert bar_containing(ts, datetime(2026, 7, 20, 15, 0, tzinfo=timezone.utc)) is None


# ── collector file format & atomic rename (brief §5.4 #2) ──────────────────


def test_collector_parquet_schema_and_no_tmp(tmp_path: Path):
    buf = FixTickBuffer()
    t0 = datetime(2026, 9, 24, 14, 55, tzinfo=timezone.utc)
    buf.add(t0, "EUR_JPY", 186.90, 186.92, 12.0)
    buf.add(t0.replace(minute=56), "EUR_JPY", 186.91, 186.93, 9.0)
    p = buf.flush_day("2026-09-24", tmp_path)

    assert p.name == "fix_ticks_2026-09-24.parquet"
    assert p.exists()
    assert not list(tmp_path.glob("*.tmp"))  # atomic rename left no .tmp

    import pyarrow.parquet as pq

    table = pq.read_table(p)
    assert table.column_names == list(TICK_COLUMNS)
    assert table.num_rows == 2
    # mid is the true mid
    assert table.column("mid")[0].as_py() == pytest.approx((186.90 + 186.92) / 2)


def test_collector_buffer_rejects_naive_timestamp(tmp_path: Path):
    buf = FixTickBuffer()
    with pytest.raises(ValueError):
        buf.add(datetime(2026, 9, 24, 14, 55), "EUR_JPY", 1.0, 1.01, 0.0)  # naive


def test_write_ticks_atomic_empty_rows(tmp_path: Path):
    p = write_ticks_atomic([], tmp_path / "fix_ticks_2026-09-24.parquet")
    import pyarrow.parquet as pq

    table = pq.read_table(p)
    assert table.column_names == list(TICK_COLUMNS)
    assert table.num_rows == 0


# ── coarse-study determinism (brief §5.4 #3) ───────────────────────────────


def _mk_m15_pair(days: int = 220):
    """Deterministic synthetic M15 series: 96 bars per weekday for `days`
    weekdays, each weekday's bars stamped 00:00..23:45 UTC."""
    from datetime import timedelta

    rows = []
    start = datetime(2026, 1, 5, tzinfo=timezone.utc)  # a Monday
    g = np.random.default_rng(42)
    d = start
    counted = 0
    while counted < days:
        if d.weekday() < 5:
            base = d.replace(hour=0, minute=0, second=0, microsecond=0)
            for b in range(96):
                rows.append(base + timedelta(minutes=15 * b))
            counted += 1
        d += timedelta(days=1)
    ts = rows
    n = len(ts)
    c = 150.0 * np.exp(np.cumsum(g.normal(0, 3e-4, n)))
    o = np.roll(c, 1); o[0] = c[0]
    h = np.maximum(o, c) * (1 + abs(g.normal(0, 2e-4, n)))
    l = np.minimum(o, c) * (1 - abs(g.normal(0, 2e-4, n)))
    return ts, o, h, l, c


def test_coarse_study_deterministic():
    bars = {"EUR_JPY": _mk_m15_pair()}
    r1 = run_b(bars)
    r2 = run_b(bars)

    def pack(s):
        keys = (s.n_fix_days, s.n_nonfix_days, s.p99_absret_fix,
                s.p99_absret_nonfix, s.p99_range_fix, s.sign_flip_rate)
        return tuple(None if (isinstance(v, float) and v != v) else v for v in keys)

    assert pack(r1.wmr["EUR_JPY"]) == pack(r2.wmr["EUR_JPY"])
    assert pack(r1.tokyo["EUR_JPY"]) == pack(r2.tokyo["EUR_JPY"])
    assert r1.verdict == r2.verdict


def test_measure_fix_counts_fix_and_nonfix():
    ts, o, h, l, c = _mk_m15_pair(60)
    # London: every weekday is a fix day, so the non-fix reference at the
    # 16:00-London bar on weekdays is empty by construction — the honest
    # "no reference cohort" case.
    stats = measure_fix(
        ts, o, h, l, c, fix_utc_fn=london_fix_utc,
        day_filter=lambda d: d.weekday() < 5,
    )
    assert stats.n_fix_days > 0
    assert stats.n_nonfix_days == 0
    # Tokyo Gotobi: fix days sparse, non-fix reference plentiful.
    stats_t = measure_fix(
        ts, o, h, l, c, fix_utc_fn=tokyo_fix_utc, day_filter=is_gotobi_day,
    )
    assert stats_t.n_fix_days > 0
    assert stats_t.n_nonfix_days > stats_t.n_fix_days
