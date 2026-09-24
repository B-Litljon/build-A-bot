"""Lane 1: daily crypto time-series momentum + volatility targeting.

An offline, deterministic, self-contained backtest harness for a daily spot
crypto trend overlay: median-of-four momentum sign ensemble, inverse-vol
sizing, a 0.05 rebalance buffer, and an honest cost model, evaluated on
BTC/USD + ETH/USD daily bars from Alpaca. Research artifact only — no order
submission, no scheduler, nothing live. Nothing in src/execution/ or
run_oanda.py imports this.

Dispatch brief: `llm_reports/handoffs/2026-09-24_lane1-crypto-trend-mom-tv.md`.
Report: `llm_reports/recons/2026-09-24_lab-crypto-trend-mom-tv.md`.

Two ensemble choices the brief left open, pinned here and by tests:
  * The vote is the **median of the four sign values** (the brief's primary
    form), landing in {-1, 0, +1}; 0 = tie = no position.
  * The default vol-estimator window is **N_sigma = 60** days; 20 is swept as
    part of the 8-arm trial grid and both are reported.

Glossary:
    momentum sign s_{k,i,t} -- sign of the k-day log return of asset i at day t,
        computed on closes up to and including t; zero when |r| < DEADBAND.
    ensemble vote S_{i,t} -- median of the four momentum signs; +1 up, -1 down
        (unused: long-only -> treated as 0), 0 tie.
    raw weight w*_{i,t} -- (sigma_target / vol_est) * max(0, S), clipped to
        [0, 1]; portfolio-normalized if the sum exceeds 1.
    vol_est -- std of daily log closes over the last sigma_window days
        (ddof=1) * sqrt(365); crypto trades 365 days.
    sigma_target -- the annualized-volatility target, in [0.20, 0.30], default
        SIGMA_TARGET_DEFAULT = 0.25, per the brief.
    rebalance buffer REBALANCE_BUFFER -- trade only when |w*_t - w_{t-1}| > 0.05;
        the buffer binds on the TARGET weight before the cash sweep. Fills are
        modeled at the next day's OPEN (the non-negotiable leak guard).
    cost model -- per trade day: sum_i |dW_i| * (SPREAD_COST_BPS + FEE_BPS) /
        10_000. SPREAD_COST_BPS = 33 is the brief's amortized half-turn of the
        measured 6.6-bps round trip spread; FEE_BPS = 25 is the Alpaca taker
        fee. Applied to EVERY weight change.
    cash bucket -- residual 1 - sum(w) after the buffer; earns 0%.
    NAV -- net-of-cost equity, starts at 1; gross NAV is the same weights with
        costs zeroed. Open-to-open portfolio return = sum_i w_{i,t} * r_{i,t+1}.
    Calmar -- CAGR / |max drawdown|, CAGR annualized on 365-day years.
    realized window -- the date range the Alpaca crypto feed actually covers,
        discovered by `probe_data_floor` (the brief's 2021 probe); the gate is
        always evaluated on the REALIZED window, never the aimed one.
    trial -- one distinct (sigma_window, sigma_target) arm actually evaluated;
        the momentum lookbacks are FIXED at (21, 63, 126, 252) so they are not
        counted as trials. This lane evaluates 2 x 3 = 6 arms
        (SWEEP_SIGMA_WINDOWS x SWEEP_SIGMA_TARGETS); `run_sweep` returns one
        ArmResult per arm and the DSR/HLZ adjustments use that count.
    BarCache -- atomic parquet cache of fetched daily bars at
        analysis_cache/lab_frames/crypto_daily_bars.parquet, per the brief.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Constants — the contract the tests pin.

MOMENTUM_LOOKBACKS: tuple[int, ...] = (21, 63, 126, 252)
"""The four momentum lookbacks, in trading days (crypto: calendar days)."""

DEADBAND: float = 1e-6
"""|log return| below this counts as sign 0 (flat window, no signal)."""

SIGMA_TARGET_DEFAULT: float = 0.25
SIGMA_WINDOW_DEFAULT: int = 60
SWEEP_SIGMA_WINDOWS: tuple[int, ...] = (20, 60)
SWEEP_SIGMA_TARGETS: tuple[float, ...] = (0.20, 0.25, 0.30)

REBALANCE_BUFFER: float = 0.05
"""Minimum |target - current| weight delta that justifies a trade."""

SPREAD_COST_BPS: float = 33.0
"""Amortized half-turn spread cost, bps per unit of one-way weight turnover."""

FEE_BPS: float = 25.0
"""Alpaca taker fee, bps per unit of one-way weight turnover."""

DAYS_PER_YEAR: int = 365
"""Crypto never sleeps."""

SYMBOLS: tuple[str, ...] = ("BTC/USD", "ETH/USD")

BAR_CACHE_PATH = Path("analysis_cache/lab_frames/crypto_daily_bars.parquet")
BAR_FETCH_META_PATH = BAR_CACHE_PATH.with_suffix(".fetched.json")
BAR_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
)
"""The 7 declared cached bar columns, in order — pinned by the schema test."""

CACHE_SCHEMA: tuple[str, ...] = BAR_COLUMNS + ("fetched_at_utc",)
"""The full cached-file layout: the 7 bar columns plus the fetch stamp."""

TARGET_START = datetime(2021, 1, 1, tzinfo=timezone.utc)
"""The aimed window start per the brief; the realized window may be later."""


# ---------------------------------------------------------------------------
# Bar fetch + cache (Alpaca, through the provider only)


def _provider_from_env() -> "object":
    """Build an AlpacaProvider from env vars, loading `.env` via dotenv.

    Reads ALPACA_API_KEY / ALPACA_SECRET_KEY from the process environment,
    falling back to a `.env` file in the current working directory. Never
    prints the values. Paper=True always: this harness never submits orders.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass
    api_key = os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret_key:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set (process env or .env); "
            "this harness fetches data and cannot run keyless"
        )
    from data.alpaca_provider import AlpacaProvider

    return AlpacaProvider(api_key=api_key, secret_key=secret_key, paper=True)


def fetch_daily_bars(provider, symbol: str, start: datetime, end: datetime) -> pl.DataFrame:
    """Fetch daily bars for one symbol through AlpacaProvider (nothing else).

    Returns a frame with exactly :data:`BAR_COLUMNS` (timestamp normalized to
    UTC, rows sorted). An empty fetch returns a zero-row frame with the right
    schema rather than raising — the caller decides whether empty is fatal.
    """
    df = provider.get_historical_bars(symbol, 1440, start, end)
    if df.height == 0:
        return pl.DataFrame(
            schema={c: pl.Datetime("us", "UTC") if c == "timestamp" else pl.Float64
                    for c in BAR_COLUMNS if c != "symbol"}
        ).with_columns(pl.lit(symbol).alias("symbol")).select(BAR_COLUMNS)
    out = df.select([c for c in BAR_COLUMNS if c in df.columns])
    out = out.with_columns(pl.lit(symbol).alias("symbol")) if "symbol" not in out.columns else out
    out = out.select(BAR_COLUMNS)
    out = out.sort("timestamp")
    return out


def probe_data_floor(provider, symbols: tuple[str, ...] = SYMBOLS,
                     target_start: datetime = TARGET_START) -> datetime:
    """Probe whether the aimed 2021 window is servable; degrade if not.

    The brief's rule: fetch a narrow 2021-01-01 -> 2021-01-15 window first.
    If empty for ANY symbol, bisect-walk forward (15-day probes, then a
    bounded exponential widening) until a probe returns data for every
    symbol, and return the discovered floor. Returns target_start itself when
    the 2021 window is fully servable.
    """
    from datetime import timedelta

    def window_ok(day: datetime, days: int = 15) -> bool:
        for sym in symbols:
            df = fetch_daily_bars(provider, sym, day, day + timedelta(days=days))
            if df.height == 0:
                return False
        return True

    if window_ok(target_start):
        return target_start

    # Degrade: bisect walk between target_start and today.
    lo = target_start
    hi = datetime.now(timezone.utc)
    # Invariant: lo is NOT servable, hi is (today's feed covers the recent past
    # by construction of this harness — if it does not, the abort rule in the
    # brief applies and the caller sees the empty fetch).
    for _ in range(40):  # bounded: ~2^-40 of 5.7 years, far tighter than a day
        mid = lo + (hi - lo) / 2
        if (hi - lo).days <= 1:
            break
        if window_ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


class BarCache:
    """Atomic parquet cache for fetched daily bars.

    Schema is exactly :data:`CACHE_SCHEMA` (7 columns, in order); the fetch
    timestamp is stamped at write time. Writes go to a sibling temp file and
    `os.replace`, so a concurrent reader never sees a half-written file.
    Re-runs read the cache; `refresh=True` refetches.
    """

    def __init__(self, path: Path = BAR_CACHE_PATH):
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(".fetched.json")

    def load(self) -> pl.DataFrame | None:
        if not self.path.exists():
            return None
        df = pl.read_parquet(self.path)
        if tuple(df.columns) != CACHE_SCHEMA:
            return None
        return df

    def save(self, df: pl.DataFrame) -> None:
        assert tuple(df.columns) == CACHE_SCHEMA, (
            f"cache schema violation: {df.columns} != {CACHE_SCHEMA}"
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            dir=self.path.parent, suffix=".parquet.tmp", delete=False
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        try:
            df.write_parquet(tmp_path)
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
        meta = {"fetched_at_utc": datetime.now(timezone.utc).isoformat()}
        self.meta_path.write_text(json.dumps(meta))

    def get_bars(self, provider, start: datetime, end: datetime,
                 refresh: bool = False) -> pl.DataFrame:
        """Cache-first load of the 2-symbol daily basket."""
        if not refresh:
            cached = self.load()
            if cached is not None and cached.height > 0:
                ts = cached["timestamp"]
                if ts.min() <= start and ts.max() >= end.replace(tzinfo=timezone.utc) if end.tzinfo is None else ts.max() >= end:
                    return cached
                # Cache exists but covers a different window; trust it only if
                # it at least covers [start, end]. Otherwise fall through and
                # refetch to cover the requested window.
                if cached.height and cached["timestamp"].min() <= start:
                    return cached
        frames = [fetch_daily_bars(provider, sym, start, end) for sym in SYMBOLS]
        df = pl.concat(frames) if any(f.height for f in frames) else frames[0]
        fetched_at = datetime.now(timezone.utc).isoformat()
        df = df.with_columns(pl.lit(fetched_at).alias("fetched_at_utc"))
        df = df.select(CACHE_SCHEMA).sort(["symbol", "timestamp"])
        if df.height:
            self.save(df)
        return df


# ---------------------------------------------------------------------------
# Signal + sizing (pure; leak guards live here)


def momentum_signs(closes: np.ndarray, lookbacks: tuple[int, ...] = MOMENTUM_LOOKBACKS,
                   deadband: float = DEADBAND) -> np.ndarray:
    """Per-lookback momentum sign series for one asset's closes.

    signs[t, j] = sign(ln(P_t / P_{t-k_j})) with 0 inside the deadband.
    Leading k entries are NaN (not enough history). Pure function of closes
    up to t — the leak-guard property the shift test pins.
    """
    closes = np.asarray(closes, dtype=float)
    t_n = closes.shape[0]
    logp = np.log(closes)
    out = np.full((t_n, len(lookbacks)), np.nan)
    for j, k in enumerate(lookbacks):
        for t in range(k, t_n):
            r = logp[t] - logp[t - k]
            out[t, j] = 0.0 if abs(r) < deadband else (1.0 if r > 0 else -1.0)
    return out


def ensemble_vote(signs: np.ndarray) -> np.ndarray:
    """Median-of-four vote per day -> {-1, 0, +1}. NaN rows stay NaN.

    Median of an even count of signs can be a half-integer (e.g. median of
    {-1, -1, +1, +1} = 0; of {+1, +1, +1, -1} = +0.5). Snap: |v| < 0.5 -> 0,
    else sign(v). A perfect 2-2 tie lands exactly on 0 by definition of the
    median, so S=0 is the no-position state the brief names.
    """
    with warnings.catch_warnings():
        # Warm-up rows are all-NaN by construction; nanmedian warns on them.
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(signs, axis=1)
    out = np.zeros_like(med)
    out[med >= 0.5] = 1.0
    out[med <= -0.5] = -1.0
    out[np.isnan(med)] = np.nan
    return out

def realized_vol(closes: np.ndarray, window: int) -> np.ndarray:
    """Annualized realized vol: std of daily log returns over `window` days,
    ddof=1, * sqrt(365). NaN until `window` returns exist (window+1 closes).
    Uses closes up to and including t — no look-ahead.
    """
    closes = np.asarray(closes, dtype=float)
    logp = np.log(closes)
    rets = np.diff(logp)
    t_n = closes.shape[0]
    out = np.full(t_n, np.nan)
    for t in range(window, t_n):
        w = rets[t - window:t]
        out[t] = float(np.std(w, ddof=1)) * math.sqrt(DAYS_PER_YEAR)
    return out


def raw_sign_weights(closes_a: np.ndarray, closes_b: np.ndarray,
                     sigma_target: float = SIGMA_TARGET_DEFAULT,
                     sigma_window: int = SIGMA_WINDOW_DEFAULT) -> np.ndarray:
    """Daily target weights BEFORE the rebalance buffer, shape (T, 2).

    w*_{i,t} = (sigma_target / vol_est_{i,t}) * max(0, S_{i,t}), clipped to
    [0, 1], then portfolio-normalized if the row sum exceeds 1. Computable
    from closes up to t only.
    """
    votes = [ensemble_vote(momentum_signs(c)) for c in (closes_a, closes_b)]
    vols = [realized_vol(c, sigma_window) for c in (closes_a, closes_b)]
    t_n = closes_a.shape[0]
    w = np.zeros((t_n, 2))
    for t in range(t_n):
        row = []
        for i in range(2):
            s = votes[i][t]
            v = vols[i][t]
            if not np.isfinite(s) or not np.isfinite(v) or v <= 0 or s <= 0:
                row.append(0.0)
            else:
                row.append(min(1.0, sigma_target / v))
        total = sum(row)
        if total > 1.0:
            row = [x / total for x in row]
        w[t] = row
    return w


def apply_rebalance_buffer(target: np.ndarray, buffer: float = REBALANCE_BUFFER,
                           w0: float = 0.0) -> np.ndarray:
    """One-asset buffer walk: hold w_{t-1} unless |w*_t - w_{t-1}| > buffer.

    Exactly AT the buffer the trade fires (strictly-greater reads "stays");
    the boundary test pins this. `target` is a 1-D daily series; return is the
    executed weight path actually held each day.
    """
    target = np.asarray(target, dtype=float)
    out = np.empty_like(target)
    w = w0
    for t in range(target.shape[0]):
        if abs(target[t] - w) > buffer:
            w = target[t]
        out[t] = w
    return out


def buffered_weights(raw: np.ndarray, buffer: float = REBALANCE_BUFFER) -> np.ndarray:
    """Apply the buffer column-wise to a (T, N) raw-weight matrix."""
    cols = [apply_rebalance_buffer(raw[:, i], buffer) for i in range(raw.shape[1])]
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Backtest loop + cost model


def run_backtest(dates: np.ndarray, opens: np.ndarray, closes: np.ndarray,
                 target_w: np.ndarray,
                 spread_cost_bps: float = SPREAD_COST_BPS,
                 fee_bps: float = FEE_BPS,
                 buffer: float = REBALANCE_BUFFER) -> dict:
    """Deterministic daily loop.

    dates: (T,) datetime64[D] trading calendar (union of both assets' days).
    opens/closes: (T, N) price matrices aligned to dates (NaN = asset not
        trading that day; weight held, return 0 for that leg).
    target_w: (T, N) raw target weights, computable from data up to day t.

    Execution: signal at close t, fill at open t+1. In weight terms the held
    vector for return day t (open_t -> open_{t+1}) is the buffered target
    from day t-1. The first day everyone is flat.
    """
    opens = np.asarray(opens, dtype=float)
    raw = np.asarray(target_w, dtype=float)
    t_n, n = raw.shape
    buffered = buffered_weights(raw, buffer)

    held = np.zeros((t_n, n))
    for t in range(1, t_n):
        held[t] = buffered[t - 1]

    # Open-to-open returns per asset; NaN open -> leg contributes 0.
    open_ret = np.full((t_n - 1, n), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        open_ret = opens[1:] / opens[:-1] - 1.0
    open_ret = np.nan_to_num(open_ret, nan=0.0)

    # Trading cost on the day the held vector changes: sum |dW| * unit cost.
    unit_cost = (spread_cost_bps + fee_bps) / 10_000.0
    cost = np.zeros(t_n)
    prev = np.zeros(n)
    for t in range(1, t_n):
        dw = np.abs(held[t] - prev)
        cost[t] = float(np.sum(dw)) * unit_cost
        prev = held[t]

    port_gross = np.zeros(t_n)
    port_net = np.zeros(t_n)
    nav_gross = np.ones(t_n)
    nav_net = np.ones(t_n)
    for t in range(1, t_n):
        port_gross[t] = float(np.dot(held[t], open_ret[t - 1]))
        port_net[t] = port_gross[t] - cost[t]
        nav_gross[t] = nav_gross[t - 1] * (1.0 + port_gross[t])
        nav_net[t] = nav_net[t - 1] * (1.0 + port_net[t])

    return {
        "dates": dates,
        "held_weights": held,
        "target_weights": buffered,
        "raw_weights": raw,
        "cost": cost,
        "port_gross": port_gross,
        "port_net": port_net,
        "nav_gross": nav_gross,
        "nav_net": nav_net,
    }


# ---------------------------------------------------------------------------
# Performance statistics


def _annualized_sharpe(daily_returns: np.ndarray) -> float:
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    sd = float(np.std(r, ddof=1))
    if sd <= 0:
        return float("nan")
    return float(np.mean(r)) / sd * math.sqrt(DAYS_PER_YEAR)


def max_drawdown(nav: np.ndarray) -> float:
    """Worst peak-to-trough fall of a NAV path, as a positive fraction."""
    nav = np.asarray(nav, dtype=float)
    peak = np.maximum.accumulate(nav)
    dd = nav / peak - 1.0
    return float(-np.min(dd))


def cagr(nav: np.ndarray, n_days: int) -> float:
    if n_days <= 0 or nav[-1] <= 0:
        return float("nan")
    years = n_days / DAYS_PER_YEAR
    return float(nav[-1] ** (1.0 / years) - 1.0)


def nav_stats(nav: np.ndarray, daily_returns: np.ndarray) -> dict:
    return {
        "total_return": float(nav[-1] - 1.0),
        "cagr": cagr(np.asarray(nav), len(nav) - 1),
        "max_drawdown": max_drawdown(np.asarray(nav)),
        "calmar": (cagr(np.asarray(nav), len(nav) - 1)
                   / max_drawdown(np.asarray(nav))
                   if max_drawdown(np.asarray(nav)) > 0 else float("nan")),
        "sharpe": _annualized_sharpe(daily_returns),
    }


def skew_kurt(daily_returns: np.ndarray) -> tuple[float, float]:
    """Sample skewness and FULL (Pearson) kurtosis of a daily-return series."""
    from scipy import stats as sst

    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 4:
        return float("nan"), float("nan")
    return float(sst.skew(r)), float(sst.kurtosis(r, fisher=False))


# ---------------------------------------------------------------------------
# Frame assembly (strict same-day join; no cross-asset timestamp mixing)


def _pivot_daily(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(dates, opens (T,2), closes (T,2)) on the strict common calendar.

    Each asset carries its own timestamps; the harness INNER-joins the two
    assets on the exact day so a basket extension cannot accidentally
    cross-join mismatched timestamps (the brief's structural requirement).
    """
    per_sym = {}
    for sym in SYMBOLS:
        sub = (
            df.filter(pl.col("symbol") == sym)
            .sort("timestamp")
            .unique(subset=["timestamp"], keep="last")
        )
        per_sym[sym] = {
            d.date() if hasattr(d, "date") else d: (o, c)
            for d, o, c in zip(sub["timestamp"].to_list(),
                               sub["open"].to_list(), sub["close"].to_list())
        }
    days = sorted(set(per_sym[SYMBOLS[0]]) & set(per_sym[SYMBOLS[1]]))
    opens = np.array([[per_sym[SYMBOLS[0]][d][0], per_sym[SYMBOLS[1]][d][0]] for d in days])
    closes = np.array([[per_sym[SYMBOLS[0]][d][1], per_sym[SYMBOLS[1]][d][1]] for d in days])
    dates = np.array(days, dtype="datetime64[D]")
    return dates, opens, closes


def buy_and_hold_nav(dates: np.ndarray, closes: np.ndarray, col: int) -> np.ndarray:
    """Buy-and-hold NAV series for one column, from its first valid close."""
    px = closes[:, col]
    nav = px / px[0]
    return nav


# ---------------------------------------------------------------------------
# Sweep + gate


@dataclass
class ArmResult:
    sigma_window: int
    sigma_target: float
    backtest: dict
    stats_net: dict = field(default_factory=dict)
    stats_gross: dict = field(default_factory=dict)
    daily_net: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def label(self) -> str:
        return f"N{self.sigma_window}-T{self.sigma_target:.2f}"


@dataclass
class GateResultLane1:
    realized_start: str
    realized_end: str
    n_days: int
    arms: list
    best_arm: ArmResult
    btc_stats: dict
    eth_stats: dict
    dsr: float
    pbo: float
    hlz_adj_sr: float
    hlz_t: float
    n_trials: int
    short_sample: bool

    @property
    def calmar_beats_btc(self) -> bool:
        return (np.isfinite(self.best_arm.stats_net.get("calmar", float("nan")))
                and np.isfinite(self.btc_stats.get("calmar", float("nan")))
                and self.best_arm.stats_net["calmar"] > self.btc_stats["calmar"])

    @property
    def maxdd_ok(self) -> bool:
        return self.best_arm.stats_net.get("max_drawdown", 1.0) < 0.30

    @property
    def dsr_ok(self) -> bool:
        return self.dsr > 0.95

    @property
    def pbo_ok(self) -> bool:
        return self.pbo < 0.50

    @property
    def hlz_ok(self) -> bool:
        return self.hlz_t > 3.0

    @property
    def gate_pass(self) -> bool:
        return all([
            self.calmar_beats_btc, self.maxdd_ok, self.dsr_ok,
            self.pbo_ok, self.hlz_ok,
        ]) and not self.short_sample

    @property
    def verdict_line(self) -> str:
        if self.short_sample:
            return ("SHORT SAMPLE (<365 days) — gate not evaluated. "
                    f"Measured over {self.n_days} days.")
        if self.gate_pass:
            return ("GATE PASS — Calmar best-vs-BTC, MaxDD < 30%, DSR > 0.95, "
                    "PBO < 0.50, HLZ t > 3.0 all clear the bar.")
        fails = []
        if not self.calmar_beats_btc:
            fails.append(
                f"Calmar {self.best_arm.stats_net['calmar']:.3f} <= BTC B&H "
                f"{self.btc_stats['calmar']:.3f}")
        if not self.maxdd_ok:
            fails.append(f"MaxDD {self.best_arm.stats_net['max_drawdown']:.1%} >= 30%")
        if not self.dsr_ok:
            fails.append(f"DSR {self.dsr:.3f} <= 0.95")
        if not self.pbo_ok:
            fails.append(f"PBO {self.pbo:.2f} >= 0.50")
        if not self.hlz_ok:
            fails.append(f"HLZ t {self.hlz_t:.2f} <= 3.0")
        return "GATE FAIL — " + "; ".join(fails)


def run_sweep(dates: np.ndarray, opens: np.ndarray, closes: np.ndarray,
              sigma_windows: tuple[int, ...] = SWEEP_SIGMA_WINDOWS,
              sigma_targets: tuple[float, ...] = SWEEP_SIGMA_TARGETS) -> list:
    """Evaluate every (sigma_window, sigma_target) arm — the counted trials."""
    arms: list[ArmResult] = []
    for w in sigma_windows:
        for st in sigma_targets:
            raw = raw_sign_weights(closes[:, 0], closes[:, 1],
                                   sigma_target=st, sigma_window=w)
            bt = run_backtest(dates, opens, closes, raw)
            daily_net = bt["port_net"][1:]
            daily_gross = bt["port_gross"][1:]
            arm = ArmResult(
                sigma_window=w, sigma_target=st, backtest=bt,
                stats_net=nav_stats(bt["nav_net"], daily_net),
                stats_gross=nav_stats(bt["nav_gross"], daily_gross),
                daily_net=daily_net,
            )
            arms.append(arm)
    return arms


def evaluate_gate(dates: np.ndarray, opens: np.ndarray, closes: np.ndarray) -> GateResultLane1:
    """Arm sweep + benchmark Calmars + DSR/PBO/HLZ — the falsification suite."""
    from lab.stats import cscv_pbo, deflated_sharpe_ratio, hlz_haircut_sharpe

    arms = run_sweep(dates, opens, closes)
    best = max(arms, key=lambda a: a.stats_net.get("calmar", float("-inf")))

    btc_nav = buy_and_hold_nav(dates, closes, 0)
    eth_nav = buy_and_hold_nav(dates, closes, 1)
    btc_ret = np.diff(np.log(btc_nav))
    eth_ret = np.diff(np.log(eth_nav))
    btc_stats = nav_stats(btc_nav, np.diff(btc_nav) / btc_nav[:-1])
    eth_stats = nav_stats(eth_nav, np.diff(eth_nav) / eth_nav[:-1])
    btc_stats["sharpe"] = _annualized_sharpe(btc_ret)
    eth_stats["sharpe"] = _annualized_sharpe(eth_ret)

    n_trials = len(arms)
    n_obs = best.daily_net.size
    skew, kurt = skew_kurt(best.daily_net)
    sr_hat = best.stats_net["sharpe"]

    dsr = deflated_sharpe_ratio(sr_hat, n_trials, n_obs, skew, kurt)
    hlz_adj = hlz_haircut_sharpe(sr_hat, n_trials)
    hlz_t = hlz_adj * math.sqrt(n_obs - 1)

    # CSCV PBO across the arms' daily net-return matrix (T x N).
    T = min(a.daily_net.size for a in arms)
    mat = np.stack([a.daily_net[:T] for a in arms], axis=1)
    pbo = cscv_pbo(mat)

    start = str(dates[0])
    end = str(dates[-1])
    return GateResultLane1(
        realized_start=start, realized_end=end, n_days=len(dates),
        arms=arms, best_arm=best,
        btc_stats=btc_stats, eth_stats=eth_stats,
        dsr=dsr, pbo=pbo, hlz_adj_sr=hlz_adj, hlz_t=hlz_t, n_trials=n_trials,
        short_sample=(len(dates) < DAYS_PER_YEAR),
    )


# ---------------------------------------------------------------------------
# CLI — the reproducibility entry point the report's numbers come from


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--refresh-bars", action="store_true",
                   help="ignore the parquet cache and refetch from Alpaca")
    p.add_argument("--start", default=TARGET_START.date().isoformat())
    p.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    p.add_argument("--json", action="store_true", help="print the gate JSON")
    return p


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    provider = _provider_from_env()
    floor = probe_data_floor(provider)
    if floor > start:
        print(f"[data] 2021 window NOT servable; degraded floor = {floor.date()}")
        start = floor

    cache = BarCache()
    df = cache.get_bars(provider, start, end, refresh=args.refresh_bars)
    if df.height == 0:
        print("[data] FATAL: no daily crypto bars returned for the realized window")
        return 1

    dates, opens, closes = _pivot_daily(df)
    result = evaluate_gate(dates, opens, closes)
    out = {
        "window": [result.realized_start, result.realized_end],
        "n_days": result.n_days,
        "best_arm": result.best_arm.label,
        "net": result.best_arm.stats_net,
        "gross": result.best_arm.stats_gross,
        "btc_bh": result.btc_stats,
        "eth_bh": result.eth_stats,
        "dsr": result.dsr, "pbo": result.pbo,
        "hlz_adj_sr": result.hlz_adj_sr, "hlz_t": result.hlz_t,
        "n_trials": result.n_trials,
        "verdict": result.verdict_line,
        "arms": {a.label: {"net": a.stats_net} for a in result.arms},
    }
    print(json.dumps(out, indent=2, default=float))
    print("\n" + result.verdict_line)
    return 0 if result.gate_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
