"""
Audit C — session-scale VWAP reversion at London open (Lane 5 falsification
audit).

Hypothesis: during the 08:00–09:00 UTC London open hour, M15 forex price
deviations from the session VWAP larger than k·σ_session mean-revert. The
strategy fades such deviations with exits driven by **session inventory
imbalance** — the running signed volume imbalance Σ sign(r)·V within the
session — exiting when imbalance crosses zero, at a protective stop
(k·σ re-extension), or at the hard 09:00 UTC session close, whichever comes
first. Exits are anchored on inventory structure, not arbitrary targets.

Glossary:
    SESSION_START / SESSION_END -- 08:00 and 09:00 UTC, fixed (the London
        open hour; UTC so no DST ambiguity). Four M15 bars per session
        (08:00, 08:15, 08:30, 08:45 opens).
    VWAP_t -- cumulative session VWAP (Σ P·V / Σ V) from session open,
        re-anchored daily. Mid price (close) is used for P.
    sigma_session_t -- realized std of within-session log returns up to bar t
        (in price units, σ_P = σ_r × VWAP_t, so the deviation and stop are
        comparable in price). Uses only bars <= t (leakage guard).
    entry -- at bar t, if |C_t − VWAP_t| > k·σ_session: fade — SHORT when
        price is above VWAP, LONG when below.
    stop -- protective: the deviation re-extends so that the signed deviation
        moves a further k·σ AWAY from VWAP vs its value at entry
        (dev − dev_entry > sign(dev_entry)·k·σ). Exit at the close of the
        triggering bar. Stop distance in price = k·σ exactly, so a stop-out
        loses exactly 1R and R is well-defined.
    inventory imbalance -- running Σ sign(r_i)·V_i within the session
        (sign of bar i's log return × its volume). Exit at the close of the
        first bar AFTER entry where the cumulative imbalance crosses zero
        (sign flips vs its value at entry) — the position's net session
        inventory is judged neutralised. Uses only bars <= t.
    session-end close -- any trade still open at the 09:00 UTC session close
        is force-closed at the session's last bar close.
    R -- a trade's PnL normalised by its own risk = k·σ at entry (price
        units). R = dir·(exit − entry) / (k·σ_entry). Sizing intent is 1%
        NAV risk per trade (units ∝ 0.01·NAV/(k·σ)), so a stop-out loses
        ≈1% NAV.
    toll -- 0.25R per round trip, subtracted from each trade's gross R
        (consistent with the repo's per-trade spread-toll convention; at the
        2×ATR live geometry the M15 toll measured ≈ 0.25R, see
        recons/2026-09-14_session-evidence-and-options.md). The gate is on
        NET EV per trade: net EV > +0.25R required, i.e. gross EV > 0.50R.
    k sweep -- k ∈ {1.5, 2.0, 2.5, 3.0}; the gate requires positive net
        expectancy across ALL k (independent of k), plus DSR > 0.95,
        PBO < 0.50, HLZ t > 3.0 (n_trials = 4). If exactly one k clears,
        the verdict flags overfit.

Leakage guards (brief §6.3):
    * σ_session and VWAP at t use only session bars <= t;
    * exit imbalance at t uses only bars <= t;
    * VWAP re-anchors every session — no cross-session lookback.

Research-only; run_c() consumes the M15 mid-bar cache and produces the
verdict record.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from lab import stats as _stats

SESSION_START_H = 8
SESSION_END_H = 9
K_SWEEP: Tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
RISK_PER_TRADE = 0.01  # 1% NAV
TOLL_R = 0.25  # per round trip, in R
N_TRIALS = len(K_SWEEP)
MIN_BARS_TO_MEASURE = 2  # need ≥2 session bars before a σ exists


# ── per-bar session state ─────────────────────────────────────────────────


@dataclass
class _SessionDay:
    """One trading day's 08:00–09:00 UTC session bar indices (opens at
    08:00, 08:15, 08:30, 08:45 — the 08:45 bar closes the session; the
    session-end close is its close price)."""

    indices: List[int] = field(default_factory=list)


def _session_indices(ts: Sequence[datetime]) -> List[_SessionDay]:
    """Group bar indices into session days (bars with open in [08:00, 09:00))."""
    days: Dict[Tuple[int, int, int], _SessionDay] = {}
    for i, t in enumerate(ts):
        if t.hour == SESSION_START_H and t.minute in (0, 15, 30, 45):
            key = (t.year, t.month, t.day)
            days.setdefault(key, _SessionDay()).indices.append(i)
    out = list(days.values())
    out.sort(key=lambda s: s.indices[0])
    return out


@dataclass
class Trade:
    """One completed fade trade."""

    k: float
    entry_idx: int
    exit_idx: int
    direction: int  # +1 long (fade a dip), −1 short (fade a pop)
    entry_price: float
    exit_price: float
    stop_distance_price: float  # |entry − VWAP_entry| + k·σ at entry, in price
    exit_reason: str  # "imbalance" | "stop" | "session_end"
    pnl_price: float
    gross_R: float
    net_R: float


@dataclass
class KSweepResult:
    k: float
    trades: List[Trade] = field(default_factory=list)
    mean_net_R: float = 0.0  # winsorised at ±winsor_R before gating
    raw_mean_net_R: float = 0.0  # untouched by winsorisation — audit only
    t_stat_net: float = 0.0
    hlz_adj_t: float = 0.0
    dsr: float = 0.0
    pbo: float = 0.5
    positive: bool = False


@dataclass
class AuditCResult:
    sweeps: Dict[float, KSweepResult] = field(default_factory=dict)
    sessions_measured: int = 0
    verdict: str = ""
    gate_pass: bool = False
    overfit_flag: bool = False


# ── core per-session simulation ───────────────────────────────────────────


def simulate_session(
    ts: Sequence[datetime],
    c: np.ndarray,
    v: np.ndarray,
    sess: _SessionDay,
    k: float,
) -> List[Trade]:
    """
    Walk one session bar by bar for fade parameter k. Returns any completed
    trades (at most one open at a time; once a trade closes the session is
    over for that k — one fade per session, the conservative convention).

    All state updates are causal: at bar i the VWAP, σ and imbalance use
    only session bars with index ≤ i, and entries are evaluated at the bar
    close with the exit evaluated from the NEXT bar onward.
    """
    idxs = sess.indices
    if len(idxs) < MIN_BARS_TO_MEASURE:
        return []

    cum_pv = 0.0
    cum_v = 0.0
    rets: List[float] = []
    imbalance = 0.0  # Σ sign(r)·V over session bars so far

    in_trade = False
    direction = 0
    entry_price = 0.0
    entry_vwap = 0.0
    entry_idx = -1
    entry_sigma_price = 0.0
    entry_imbalance = 0.0
    trades: List[Trade] = []

    prev_close = float(c[idxs[0]])

    for pos, i in enumerate(idxs):
        price = float(c[i])
        vol = float(v[i]) if i < len(v) else 0.0

        # update session aggregates with THIS bar first (they include bar i,
        # consistent with "uses only bars <= t")
        cum_pv += price * vol
        cum_v += vol
        vwap = cum_pv / cum_v if cum_v > 0 else price

        r = 0.0
        if prev_close > 0 and price > 0 and pos > 0:
            r = math.log(price / prev_close)
        if pos > 0:  # first bar gets no return contribution
            rets.append(r)
            if r != 0:
                imbalance += math.copysign(1.0, r) * vol

        sigma_r = float(np.std(rets, ddof=1)) if len(rets) >= 2 else 0.0
        sigma_price = sigma_r * vwap if vwap > 0 else 0.0

        dev = price - vwap

        if in_trade:
            # ── exits, in priority order: session end, stop, imbalance.
            # The stop uses the current signed deviation: entry at dev d0
            # with stop at dev = d0 + sign(d0)·k·σ (deviation re-extends a
            # further k·σ away from VWAP in the adverse direction).
            # Stop distance in price = k·σ exactly, so a stop-out loses
            # exactly 1R.
            is_last = pos == len(idxs) - 1
            exit_now = False
            reason = ""
            adverse = (dev - entry_dev) * math.copysign(1.0, entry_dev if entry_dev else 1.0)
            if is_last:
                exit_now = True
                reason = "session_end"
            elif adverse > k * entry_sigma_price:
                exit_now = True
                reason = "stop"
            elif (
                pos >= 1
                and i != entry_idx
                and entry_imbalance != 0
                and imbalance * entry_imbalance < 0  # strict sign flip
            ):
                exit_now = True
                reason = "imbalance"

            if exit_now:
                stop_dist = k * entry_sigma_price  # stop distance = k·σ exactly
                pnl = direction * (price - entry_price)
                gross_r = pnl / stop_dist if stop_dist > 0 else 0.0
                trades.append(
                    Trade(
                        k=k,
                        entry_idx=entry_idx,
                        exit_idx=i,
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=price,
                        stop_distance_price=stop_dist,
                        exit_reason=reason,
                        pnl_price=pnl,
                        gross_R=gross_r,
                        net_R=gross_r - TOLL_R,
                    )
                )
                in_trade = False
                break  # one fade per session

        else:
            # ── entry (only when not already in a trade, and not on the
            # final bar of the session — no time for a fade to work)
            if pos < len(idxs) - 1 and sigma_price > 0 and abs(dev) > k * sigma_price:
                in_trade = True
                direction = -1 if dev > 0 else 1
                entry_price = price
                entry_vwap = vwap
                entry_idx = i
                entry_dev = dev
                entry_sigma_price = sigma_price
                entry_imbalance = imbalance

        prev_close = price

    # a trade opened on the last bar would never be measured; none allowed
    return trades


def run_c(
    ts: Sequence[datetime],
    close: np.ndarray,
    volume: np.ndarray,
    *,
    k_sweep: Sequence[float] = K_SWEEP,
    winsor_R: float = 5.0,
) -> AuditCResult:
    """
    The full k-sweep across every session in the cache. ``winsor_R`` clips
    each trade's net R to ±winsor_R before the stats — the raw R distribution
    is dominated by a handful of session_end trades whose entry σ was tiny
    (stop distance ≈ 0), and winsorising at ±5 reports the robust core of the
    distribution without pretending a 1700σ outlier is a 1700R win. Both raw
    and winsorised numbers are carried on the result so the report can quote
    them side by side.
    """
    res = AuditCResult()
    sessions = _session_indices(ts)
    res.sessions_measured = len(sessions)

    for k in k_sweep:
        ks = KSweepResult(k=k)
        for sess in sessions:
            ks.trades.extend(simulate_session(ts, close, volume, sess, k))
        res.sweeps[k] = ks

    # gates per k on the winsorised net R
    series_by_k: Dict[float, np.ndarray] = {}
    for k, ks in res.sweeps.items():
        raw = np.array([t.net_R for t in ks.trades], dtype=float)
        raw_mean = float(raw.mean()) if raw.size else 0.0
        net = np.clip(raw, -winsor_R, winsor_R) if raw.size else raw
        n = net.size
        if n >= 10:
            m = float(net.mean())
            sd = float(net.std(ddof=1))
            t_stat = m / sd * math.sqrt(n) if sd > 0 else 0.0
            ks.mean_net_R = m
            ks.t_stat_net = t_stat
            ks.hlz_adj_t = _stats.hlz_haircut_sharpe(t_stat, N_TRIALS)
            skew = _skew(net)
            kurt = _kurt(net)
            ks.dsr = _stats.deflated_sharpe_ratio(
                _stats.sharpe_ratio(net, periods_per_year=252.0), N_TRIALS, n, skew, kurt
            )
            series_by_k[k] = net
            ks.positive = m > 0
            ks.raw_mean_net_R = raw_mean
        else:
            ks.positive = False

    # CSCV PBO needs a T×N matrix; columns are the four k series truncated
    # to the shortest length.
    if series_by_k:
        min_len = min(a.size for a in series_by_k.values())
        if min_len >= 8:
            mat = np.column_stack([a[:min_len] for a in series_by_k.values()])
            for k, ks in res.sweeps.items():
                ks.pbo = _stats.cscv_pbo(mat)

    active = [ks for ks in res.sweeps.values() if ks.trades]
    clearing = [
        k for k, ks in res.sweeps.items()
        if ks.trades
        and ks.mean_net_R > TOLL_R
        and ks.dsr > 0.95
        and ks.pbo < 0.50
        and ks.hlz_adj_t > 3.0
    ]
    n_independent_ok = sum(1 for ks in active if ks.positive)
    res.overfit_flag = (len(clearing) == 1) or (n_independent_ok == 1 and len(active) > 1)
    res.gate_pass = (
        len(clearing) == len(active)
        and len(clearing) > 1
        and not res.overfit_flag
    )

    if not active:
        res.verdict = "INCONCLUSIVE — no session produced a fade signal"
    elif res.gate_pass:
        res.verdict = (
            f"PASS — net EV > +0.25R for every k in {tuple(sorted(res.sweeps))}, "
            "expectancy independent of k"
        )
    elif res.overfit_flag:
        k0 = clearing[0] if clearing else max(res.sweeps, key=lambda k: res.sweeps[k].mean_net_R)
        res.verdict = (
            f"FAIL — only k={k0} clears ({n_independent_ok}/{len(active)} k-positive, "
            f"{len(clearing)}/{len(active)} k-clear); single-k success is overfit, "
            "expectancy is NOT independent of k"
        )
    else:
        best = max(active, key=lambda s: s.mean_net_R)
        res.verdict = (
            f"FAIL — no k clears the +0.25R net bar and stats gates "
            f"(best winsorised net EV {best.mean_net_R:+.3f}R at k={best.k}; "
            f"k-positive {n_independent_ok}/{len(active)})"
        )
    return res


def _skew(x: np.ndarray) -> float:
    n = x.size
    if n < 3:
        return 0.0
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 0.0
    return float(n / ((n - 1) * (n - 2)) * (((x - m) / s) ** 3).sum())


def _kurt(x: np.ndarray) -> float:
    n = x.size
    if n < 4:
        return 3.0
    m = x.mean()
    s = x.std(ddof=1)
    if s == 0:
        return 3.0
    g2 = (((x - m) / s) ** 4).mean() - 3.0
    return float(((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6.0) + 3.0)
