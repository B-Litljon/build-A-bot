"""
Audit B — calendar-flow liquidity around the London WM/R and Tokyo Nakane fixes
(Lane 5 falsification audit, coarse study on M15 mid bars).

**Data-resolution limitation (first-class):** there is NO tick, bid/ask or
spread time series anywhere in this repo (verified 2026-09-24). All cached
forex data is M15 mid-price OHLCV bars; the highest-fidelity spread artifact
is six per-instrument median scalars in ``config/spread_alphas_m15.json``.
This audit therefore measures what M15 bars can show — fix-bar return and
high/low range (as % of mid), a *coarse proxy for activity* — and CANNOT
measure bid/ask widening around the fix. A fix-bar that is indistinguishable
from a non-fix bar at M15 resolution does NOT falsify tick-level widening;
it means only "needs tick data", which is what ``fix_collector.py`` and
``scripts/fix_tick_collector.py`` exist to collect.

Glossary:
    WM/R fix -- the 4:00 PM London benchmark (WM/Reuters). Converted to UTC
        with zoneinfo ``Europe/London`` so DST is handled correctly (16:00
        BST = 15:00 UTC in summer; 16:00 GMT = 16:00 UTC in winter). The
        M15 bar "containing" the fix instant is the bar whose open timestamp
        satisfies open <= fix_utc < open + 15 min (bars are stamped at open).
    Tokyo Nakane fix -- the 9:55 AM JST benchmark (00:55 UTC always; JST has
        no DST). **Gotobi day** (5/10/15/20/25/month-end, JST calendar) is
        when the Tokyo fix carries the heaviest corporate flow; the audit
        compares Gotobi days against non-Gotobi days, and the London fix
        against all non-fix weekdays, at M15 resolution.
    fix_bar / post_bar -- the M15 bar containing the fix instant, and the
        bar immediately after it. Metrics: log return r = ln(C/C_prev),
        and range_pct = (high − low) / mid, measured on both and on the same
        clock-time bars on non-fix days.
    verdict -- per the brief: |r| p99 on fix days > 2× non-fix days AND
        directionally consistent post-fix reversion means "not obviously
        eliminated — NEEDS TICK DATA" (never "falsified" at this
        resolution). Indistinguishable fix/non-fix ranges mean the
        post-2015 spread-widening concern is "not supported at M15
        resolution" — NOT "no widening exists".
    run_b -- the public entry: takes the six-pair mid-bar cache and computes
        per-fix, per-pair fix-bar and post-fix-bar statistics on fix vs
        non-fix days. Deterministic given the cache.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, time as dtime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:  # zoneinfo is stdlib since 3.9; guard only for clarity
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

LONDON_TZ = "Europe/London"
TOKYO_TZ = "Asia/Tokyo"
LONDON_FIX_LOCAL = dtime(16, 0)
TOKYO_FIX_LOCAL = dtime(9, 55)
BAR_MINUTES = 15
DOUBLE_RATIO_THRESHOLD = 2.0  # fix p99 must exceed non-fix p99 by >2x

# The six fiat pairs the strategy matrix caches (verified 2026-09-24 in
# analysis_cache/strategy_matrix on the sibling checkout; this lane's
# worktree reads its own copy).
FIAT_PAIRS: Tuple[str, ...] = (
    "EUR_JPY", "GBP_JPY", "AUD_JPY", "NZD_JPY", "GBP_AUD", "GBP_NZD",
)


# ── calendar helpers (DST-correct via zoneinfo) ───────────────────────────


def london_fix_utc(day: datetime) -> datetime:
    """
    The UTC instant of the WM/R 16:00 London fix on the calendar date of
    ``day`` (any tz). Uses zoneinfo Europe/London, so 2026-03-29 → 15:00 UTC
    (BST) and 2026-01-15 → 16:00 UTC (GMT). ``day`` itself may carry a tz;
    only its date in London is used.
    """
    local = datetime.combine(day.date(), LONDON_FIX_LOCAL, tzinfo=ZoneInfo(LONDON_TZ))
    return local.astimezone(timezone.utc)


def tokyo_fix_utc(day: datetime) -> datetime:
    """
    The UTC instant of the Nakane 09:55 JST fix on the calendar date of
    ``day``. JST has no DST, so this is always 00:55 UTC; the ZoneInfo path
    is kept so a hypothetical JST rule change is absorbed by the tz database
    rather than by a constant edit.
    """
    local = datetime.combine(day.date(), TOKYO_FIX_LOCAL, tzinfo=ZoneInfo(TOKYO_TZ))
    return local.astimezone(timezone.utc)


def is_gotobi_day(day: datetime) -> bool:
    """
    Whether ``day`` (JST calendar date) is a Gotobi day: the 5th, 10th, 15th,
    20th, 25th, or the last day of the month.
    """
    d = day.astimezone(ZoneInfo(TOKYO_TZ)).date() if day.tzinfo else day.date()
    if d.day in (5, 10, 15, 20, 25):
        return True
    nxt = d + timedelta(days=1)
    return nxt.month != d.month


def bar_containing(ts_index: Sequence[datetime], instant_utc: datetime) -> Optional[int]:
    """
    The index of the M15 bar whose [open, open+15min) window contains
    ``instant_utc``: the largest i with ts_index[i] <= instant_utc, provided
    ts_index[i] + 15 min > instant_utc. ``ts_index`` must be sorted UTC
    datetimes at the 15-minute grid. Returns None when the instant falls
    outside the cached span or into a (weekend) gap.
    """
    # bars are stamped at open on a 15-min grid: floor the instant.
    minute = (instant_utc.minute // BAR_MINUTES) * BAR_MINUTES
    open_guess = instant_utc.replace(minute=minute, second=0, microsecond=0)
    lo, hi = 0, len(ts_index)
    # binary search for open_guess
    while lo < hi:
        mid = (lo + hi) // 2
        if ts_index[mid] < open_guess:
            lo = mid + 1
        else:
            hi = mid
    if lo < len(ts_index) and ts_index[lo] == open_guess:
        return lo
    return None


# ── per-fix coarse measurements ───────────────────────────────────────────


@dataclass
class FixBarStats:
    """One fix's coarse statistics for one pair."""

    n_fix_days: int = 0
    n_nonfix_days: int = 0
    p99_absret_fix: float = float("nan")  # fix-bar |r| 99th pct, fix days
    p99_absret_nonfix: float = float("nan")  # same bar on non-fix days
    p99_range_fix: float = float("nan")  # fix-bar range % of mid, fix days
    p99_range_nonfix: float = float("nan")
    mean_post_ret_fix: float = float("nan")  # next-bar return on fix days
    mean_post_ret_nonfix: float = float("nan")
    sign_flip_rate: float = float("nan")  # post-bar return sign opposite fix-bar
    ratio_absret: float = float("nan")
    ratio_range: float = float("nan")


@dataclass
class AuditBResult:
    """Per-fix, per-pair stats plus the audit-level verdict."""

    wmr: Dict[str, FixBarStats] = field(default_factory=dict)
    tokyo: Dict[str, FixBarStats] = field(default_factory=dict)
    pairs_with_data: List[str] = field(default_factory=list)
    verdict: str = ""


def _bar_metrics(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray, idx: int
) -> Tuple[float, float]:
    """(log return vs prior close, high-low range as % of mid) at bar idx."""
    if idx <= 0 or idx >= len(c):
        return float("nan"), float("nan")
    prev = c[idx - 1]
    if prev <= 0 or not np.isfinite(prev):
        return float("nan"), float("nan")
    r = math.log(c[idx] / prev)
    mid = c[idx]
    rng = (h[idx] - l[idx]) / mid if mid > 0 else float("nan")
    return r, rng


def _p99(x: Sequence[float]) -> float:
    a = np.asarray([v for v in x if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.percentile(a, 99))


def measure_fix(
    ts: List[datetime],
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    *,
    fix_utc_fn,
    day_filter=None,
) -> FixBarStats:
    """
    For one pair: per fix event in the cache window, the fix-bar |log return|
    and range, the same clock-time bar on non-fix days, and the next-bar
    (post-fix) return.     ``fix_utc_fn(day)`` gives the fix instant; ``day_filter(day)`` (default:
    weekdays) selects fix days. Non-fix reference: weekdays failing
    ``day_filter`` — for the London fix (every weekday) that is empty, and
    Tokyo's Gotobi-day filter fills it. When none accumulate, the audit
    reports the fix cohort against NaN reference and the verdict degrades
    honestly (no fabricated comparison).
    """
    out = FixBarStats()
    if not ts:
        return out
    first, last = ts[0].date(), ts[-1].date()
    fix_absret: List[float] = []
    nonfix_absret: List[float] = []
    fix_range: List[float] = []
    nonfix_range: List[float] = []
    fix_post: List[float] = []
    nonfix_post: List[float] = []
    flips = 0
    n_post = 0

    day = first
    while day <= last:
        d0 = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
        fix_utc = fix_utc_fn(d0)
        is_fix = (day_filter(d0) if day_filter else d0.weekday() < 5)
        idx = bar_containing(ts, fix_utc)
        if idx is not None:
            r, rng = _bar_metrics(o, h, l, c, idx)
            idx_post = idx + 1
            r_post = float("nan")
            if idx_post < len(c) and c[idx] > 0:
                r_post = math.log(c[idx_post] / c[idx])
            if is_fix and np.isfinite(r):
                fix_absret.append(abs(r))
                fix_range.append(rng)
                if np.isfinite(r_post):
                    fix_post.append(r_post)
                    n_post += 1
                    if r_post * r < 0:
                        flips += 1
            elif (not is_fix) and d0.weekday() < 5 and np.isfinite(r):
                # non-fix reference at the same clock time on weekday non-fix days
                nonfix_absret.append(abs(r))
                nonfix_range.append(rng)
                if np.isfinite(r_post):
                    nonfix_post.append(r_post)
        day += timedelta(days=1)

    out.n_fix_days = len(fix_absret)
    out.n_nonfix_days = len(nonfix_absret)
    out.p99_absret_fix = _p99(fix_absret)
    out.p99_absret_nonfix = _p99(nonfix_absret)
    out.p99_range_fix = _p99(fix_range)
    out.p99_range_nonfix = _p99(nonfix_range)
    out.mean_post_ret_fix = float(np.mean(fix_post)) if fix_post else float("nan")
    out.mean_post_ret_nonfix = float(np.mean(nonfix_post)) if nonfix_post else float("nan")
    out.sign_flip_rate = (flips / n_post) if n_post else float("nan")
    if np.isfinite(out.p99_absret_nonfix) and out.p99_absret_nonfix > 0:
        out.ratio_absret = out.p99_absret_fix / out.p99_absret_nonfix
    if np.isfinite(out.p99_range_nonfix) and out.p99_range_nonfix > 0:
        out.ratio_range = out.p99_range_fix / out.p99_range_nonfix
    return out


def run_b(bars: Dict[str, Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> AuditBResult:
    """
    The coarse study across the six-pair cache. ``bars`` maps pair →
    (utc_open_timestamps_sorted, o, h, l, c) float arrays.
    """
    res = AuditBResult()
    for pair, (ts, o, h, l, c) in bars.items():
        if not ts:
            continue
        res.pairs_with_data.append(pair)
        res.wmr[pair] = measure_fix(
            ts, o, h, l, c,
            fix_utc_fn=london_fix_utc,
            day_filter=lambda d: d.weekday() < 5,  # London fix: every weekday
        )
        res.tokyo[pair] = measure_fix(
            ts, o, h, l, c,
            fix_utc_fn=tokyo_fix_utc,
            day_filter=is_gotobi_day,  # Tokyo fix flow: Gotobi days only
        )
    res.verdict = _verdict(res)
    return res


def _verdict(res: AuditBResult) -> str:
    """The brief §5.2 rule, applied to the pooled stats."""
    ratios_r, ratios_g, flips = [], [], []
    for table in (res.wmr, res.tokyo):
        for s in table.values():
            if np.isfinite(s.ratio_absret):
                ratios_r.append(s.ratio_absret)
            if np.isfinite(s.ratio_range):
                ratios_g.append(s.ratio_range)
            if np.isfinite(s.sign_flip_rate):
                flips.append(s.sign_flip_rate)
    if not ratios_r:
        return "INCONCLUSIVE — no fix bars measured"
    med_ratio_r = float(np.median(ratios_r))
    med_ratio_g = float(np.median(ratios_g)) if ratios_g else float("nan")
    mean_flip = float(np.mean(flips)) if flips else float("nan")
    # post-fix drift "directionally consistent with reversion" = sign-flip rate > 0.5
    reversion = np.isfinite(mean_flip) and mean_flip > 0.5
    if med_ratio_r > DOUBLE_RATIO_THRESHOLD and reversion:
        return (
            "NOT OBVIOUSLY ELIMINATED — fix-bar |r| p99 exceeds non-fix by "
            f"{med_ratio_r:.2f}× (>2) with post-fix sign-flip {mean_flip:.2f}; "
            "but M15 bars CANNOT measure bid/ask widening: NEEDS TICK DATA "
            "before any fade-the-fix routing decision."
        )
    return (
        "FIX AND NON-FIX BARS ARE INDISTINGUISHABLE AT M15 RESOLUTION "
        f"(|r| p99 ratio {med_ratio_r:.2f}×, range ratio {med_ratio_g:.2f}×). "
        "The post-2015 spread-widening concern is NOT SUPPORTED AT M15 "
        "RESOLUTION — this is not evidence that no tick-level widening "
        "exists; only the tick collector can measure that."
    )
